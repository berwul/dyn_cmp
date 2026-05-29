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
from planners.dpcmp.controller import SafeCorridorMPC
from planners.dpcmp.corridor_solver import CorridorSolver


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

    def __init__(
            self,
            cc,
            u_lim=None,
            v_lim=None,
            dt=1e-2,
            H=10,
    ):
        self.cc = cc
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
            H=H,
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
        self.cs_stop, self.rs_stop = np.zeros((H_stop + 1, n_p)), np.zeros(H_stop + 1)

        self.status_braking = False
        self.is_first = True
        self.corr_solver = None
        self.is_forward_direction = True
        self.predict = True
        self.path_radii_t_1_old = []
        self.path_radii_t_1 = []
        self.path_radii_min = None
        self.x_planned = None
        
    def set_path_radii_min(self, path_radii_min):
        self.path_radii_min = path_radii_min
        self.set_tracking_path(self.path_radii_min, margin_corr=0.01)
        self.path_track_safe = self.path_track.copy()

    def set_path(self, path, delta=0.05):
        self.path_centers = interpolate_equidistant(path, delta=delta)
        dims = self.cc.dims
        self.path_radii = self.cc.sdf_batch(self.path_centers, dims)
        self.path_radii_t = self.path_radii.copy()
        self.corr_solver = CorridorSolver(self.path_centers)

    def initialize(self, p_s, p_g):
        H, H_stop = self.cont.N, self.cont.N_stop
        n_x, n_p = self.cont.M_x, self.cont.M_p
        self.X[:] = 0
        self.X_stop[:] = 0
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

    def observe(self, human, cnt, dt):
        nr_pred = self.cont.N + 1
        path_radii_t = []
        dims_t_prev = human.get_dims_at_time((cnt - 1) * dt)
        dims_t = human.get_dims_at_time(cnt * dt)
        dims_vel_est = (dims_t - dims_t_prev) / dt
        time_s = time.time()
        for i in range(nr_pred):
            # Use GT
            # dims_t_i = human.get_dims_at_time((cnt + i) * dt)
            # Predict future
            dims_t_i = dims_vel_est * i * dt + dims_t
            # Conservative estimation
            dims_t_i[:, -1] = dims_t_i[:, -1] *  (1 + i * 0.01)
            path_radii_dyn = self.cc.sdf_batch(self.path_centers, dims_t_i)
            r_k = np.minimum(path_radii_dyn, self.path_radii)
            path_radii_t.append(r_k)
        self.path_radii_t = np.vstack(path_radii_t)
        self.times[1] = time.time() - time_s
        time_s = time.time()
        self.set_tracking_path(self.path_radii_t[-1], margin_corr=0.05)
        self.times[2] = time.time() - time_s

    def set_tracking_path(self, path_radii_t, margin_corr = 0.05, inter_corr = 0.005):
        path_track = self.corr_solver.solve(np.maximum(path_radii_t - margin_corr, 0))
        self.path_track = interpolate_equidistant(path_track, delta=inter_corr)

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
        cont = self.cont
        cs, rs = self.cs, self.rs
        cs_stop, rs_stop = self.cs_stop, self.rs_stop
        H_stop = self.cont.N_stop

        self.shift()
        status = False
        self.cont_status = True
        time_s = time.time()
        M = self.X_stop.shape[0]
        self.i_max = []
        for k in range(H_stop + 1):
            p = self.X_stop[k, :n_p]
            if k >= H_stop:
                i_max = index_compute_largets_margin_bubble(p, path_centers, self.path_radii_min)
                cs[k] = path_centers[i_max]
                rs[k] = path_radii_t[-1][i_max]
                cs_stop[k] = path_centers[i_max]
                rs_stop[k] = self.path_radii_min[i_max]
            else:
                i_max = index_compute_largets_margin_bubble(p, path_centers, path_radii_t[k])
                cs[k] = path_centers[i_max]
                rs[k] = path_radii_t[k][i_max]
                cs_stop[k] = path_centers[i_max]
                rs_stop[k] = path_radii_t[k][i_max]
            self.i_max.append(i_max)
        try:
            x_g_v, i_max_goal = compute_goal_state(
                cs, rs, self.path_track, p_g, return_index=True
            )
            x_g_v_em, i_max_goal_em = compute_goal_state(
                cs_stop, rs_stop, self.path_track_safe, p_g, return_index=True
            )
            self.times[3] = time.time() - time_s
            time_s = time.time()
            status, X, U, X_stop, U_stop = cont.solve(
                x.copy(), x_g_v.copy(), cs.T.copy(), rs.copy(), cs_stop.T, rs_stop.copy(),  x_g_v_em.copy()
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
            self.X = X.copy()
            self.U = U.copy()
            self.X_stop = X_stop.copy()
            self.U_stop = U_stop.copy()
            # For debugging
            self.x_g_v = x_g_v.copy()
            self.x_g_em_v = x_g_v_em.copy()
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
        return u

    def get_simulation_data(self):
        return {
            "x": self.x,
            "X": self.X,
            "U": self.U,
            "X_stop": self.X_stop,
            "path_track": self.path_track,
            "x_g_v": self.x_g_v,
            "x_g_em_v": self.x_g_em_v
        }

    def get_input_data(self):
        return {
            "cs": self.cont.cs.value.copy(),
            "rs": self.cont.rs.value.copy(),
            "cs_stop": self.cont.cs_stop.value.copy(),
            "rs_stop": self.cont.rs_stop.value.copy(),
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
