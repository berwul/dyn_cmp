# Copyright (c) 2026, ABB Schweiz AG
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import time

import numpy as np

from aux.math import interpolate_equidistant
from planners.dcmp.controller import SafeCorridorMPC
from planners.dcmp.corridor_solver import CorridorSolver


def index_compute_largets_margin_bubble(p, path_centers, path_radii):
    dists = np.linalg.norm(path_centers - p, axis=-1)
    diff = path_radii - dists
    return diff.argmax()


def compute_goal_state(cs, rs, path_centers, p_g, return_index=False):
    c, r = cs[-1], rs[-1]
    if np.linalg.norm(p_g - c) <= r:
        p_g_v = p_g
        i_max = cs.shape[0]
    else:
        dists = np.linalg.norm(path_centers - c, axis=-1)
        mask = dists <= r
        indxs, = np.nonzero(mask)
        i_max = indxs.max()
        p_g_v = path_centers[i_max]
    if return_index:
        return np.r_[p_g_v, np.zeros_like(p_g_v)], i_max
    else:
        return np.r_[p_g_v, np.zeros_like(p_g_v)]


class DynamicCorridorTracking:

    def __init__(self, nscdf, u_lim=None, v_lim=None, dt=1e-2, add_slacks=False):
        self.nscdf = nscdf
        self.path_centers = np.empty((0, ))
        self.path_track = np.empty((0, ))
        self.path_radii = np.empty((0, ))
        self.path_radii_t = np.empty((0, ))
        self.cnt_failed = 0

        self.times = np.zeros(4, )
        self.cont = SafeCorridorMPC.load_default(
            nr_dof=3,
            u_lim=u_lim,
            v_lim=v_lim,
            dt=dt,
            add_slacks=add_slacks,
            scaling_safety=0.1
        )
        H, H_stop = self.cont.N, self.cont.N_stop
        n_x, n_p = self.cont.M_x, self.cont.M_p

        self.X = np.zeros((H + 1, n_x))
        self.U = np.zeros((H, n_p))

        self.X_stop = np.zeros((H_stop + 1, n_x))
        self.U_stop = np.zeros((H_stop, n_p))

        self.x = np.zeros(n_x)
        self.x_g_v = np.zeros(n_x)
        self.x_g_em_v = np.zeros(n_x)

        self.cs, self.rs = np.zeros((H + 1, n_p)), np.zeros(H + 1)

        self.status_braking = False
        self.is_first = True
        self.corr_solver = None
        self.is_forward_direction = True

    def set_path(self, path, delta=0.05):
        self.path_centers = interpolate_equidistant(path, delta=delta)
        self.path_radii = np.r_[[self.nscdf.sdf(p) for p in self.path_centers]]
        self.path_radii_t = self.path_radii.copy()
        self.corr_solver = CorridorSolver(self.path_centers)

    def initialize(self, p_s, p_g):
        H, H_stop = self.cont.N, self.cont.N_stop
        n_x, n_p = self.cont.M_x, self.cont.M_p

        self.X[:, :n_p] = p_s
        self.X_stop[:, :n_p] = p_s

        self.U = np.zeros((H, n_p))
        self.U_stop = np.zeros((H_stop, n_p))
        self.x = np.zeros(n_x)
        self.x_g_v = np.zeros(n_x)
        self.x_g_em_v = np.zeros(n_x)
        self.cs, self.rs = np.zeros((H + 1, n_p)), np.zeros(H + 1)
        self.is_forward_direction = np.isclose(p_g, self.path_centers[-1]).all()
        self.set_tracking_path(self.path_radii)

    def clear(self):
        self.times = np.ones(4, ) * np.nan

    def observe(self, dims_t):
        self.nscdf.set_dims(dims_t)
        time_s = time.time()
        path_radii_dyn = np.r_[[self.nscdf.sdf(p) for p in self.path_centers]]
        self.path_radii_t = np.minimum(path_radii_dyn, self.path_radii)
        self.times[1] = time.time() - time_s
        time_s = time.time()
        self.set_tracking_path(self.path_radii_t)
        self.times[2] = time.time() - time_s

    def set_tracking_path(self, path_radii_t, margin_corr = 0.05, inter_corr = 0.005):
        path_track = self.corr_solver.solve(np.maximum(path_radii_t - margin_corr, 0))
        self.path_track = interpolate_equidistant(path_track, delta=inter_corr)
        if not self.is_forward_direction:
            self.path_track = self.path_track[::-1]

    def get_dynamics(self):
        return self.cont.A, self.cont.B

    def shift(self):
        for Z in (self.X, self.X_stop, self.U, self.U_stop):
            Z[:-1] = Z[1:]

    def plan(self, x):
        self.x = x
        p_g = self.path_track[-1]
        path_centers = self.path_centers
        path_radii_t = self.path_radii_t
        n_x, n_p = self.cont.M_x, self.cont.M_p
        nscdf = self.nscdf
        cont = self.cont
        cs, rs = self.cs, self.rs
        H_stop = self.cont.N_stop

        self.shift()

        r_t = nscdf.sdf(x[:n_p])
        is_coll_free = r_t > 0.00
        status = False
        self.cont_status = True
        if is_coll_free:
            time_s = time.time()
            indxs_traj = []
            for k, p in enumerate(self.X[:, :n_p]):
                i_max = index_compute_largets_margin_bubble(p, path_centers, path_radii_t)
                cs[k] = path_centers[i_max]
                rs[k] = path_radii_t[i_max]
                indxs_traj.append(i_max)
            p = self.X_stop[0, :n_p]
            i_max = index_compute_largets_margin_bubble(p, path_centers, path_radii_t)
            c_stop, r_stop = path_centers[i_max], path_radii_t[i_max]
            try:
                x_g_v, i_max_goal = compute_goal_state(
                    cs, rs, self.path_track, p_g, return_index=True
                )
                self.times[3] = time.time() - time_s
                mask = np.linalg.norm(self.path_centers - c_stop, axis=-1) < r_stop
                indxs, = np.nonzero(mask)
                p_stop = self.path_centers[indxs[-1]]
                x_g_em_v = np.r_[p_stop, np.zeros_like(p_stop)]
                time_s = time.time()
                status, X, U, X_stop, U_stop = cont.solve(
                    x, x_g_v, cs.T, rs, c_stop, r_stop, x_g_em_v
                )
                self.times[0] = time.time() - time_s
                self.cont_status = status
                if self.is_first:
                    self.times[0] = np.nan
                    self.is_first = False
            except:
                pass
        if status:
            # MPC problem solved successfully
            self.status_braking = False
            self.cnt_failed = 0
            self.X = X
            self.U = U
            self.X_stop = X_stop
            self.U_stop = U_stop
            # For debugging
            self.x_g_v = x_g_v
            self.x_g_em_v = x_g_em_v
        else:
            # If failed, overwrite perf trajectory with em.stop
            self.status_braking = True
            self.cnt_failed += 1
            self.X[:H_stop+1] = self.X_stop
            self.X[H_stop+1:] = self.X_stop[-1]
            self.U[:H_stop] = self.U_stop
            self.U[H_stop:] = self.U_stop[-1]
            # For debugging
            self.x_g_v = self.X[-1]
            self.x_g_em_v = self.X[-1]
        u = self.U[0]
        self.path_radii_t = path_radii_t
        return u

    def get_simulation_data(self):
        return {
            "x": self.x,
            "X": self.X,
            "X_stop": self.X_stop,
            "path_track": self.path_track,
            "x_g_v": self.x_g_v,
            "x_g_em_v": self.x_g_em_v
        }

    def get_input_data(self):
        return {
            "cs": self.cont.cs.value.copy(),
            "rs": self.cont.rs.value.copy(),
            "c_stop": self.cont.c_stop.value.copy(),
            "r_stop": self.cont.r_stop.value.copy(),
            "x": self.cont.x_start.value.copy(),
            "x_g": self.cont.x_g.value.copy(),
            "x_g_stop": self.cont.x_g_stop.value.copy(),
        }

    def get_status_data(self):
        return {
            "status_breaking": self.status_braking,
            "status_controller": self.cont_status
        }

    def get_performance_data(self):
        return {
            "times": self.times.copy()
        }
