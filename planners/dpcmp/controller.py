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

from aux.math import get_linear_double_integrator_discrete_dynamics

import cvxpy as cp
import numpy as np




class SafeCorridorMPC:

    def __init__(
            self,
            A,
            B,
            Q,
            Q_e,
            R,
            N,
            u_lim,
            v_lim=1.0,
    ):
        self.compiled = False
        self.A = A
        _, self.M_x = A.shape
        self.M_p = int(self.M_x / 2)
        self.B = B
        self.Q = Q
        self.Q_e = Q_e
        self.R = R
        self.N = N
        self.N_stop = N
        N_stop = N
        self.u_lim = u_lim
        _, M_x = A.shape
        _, M_u = B.shape
        self.M_u = M_u
        m = int(M_x / 2)
        self.N = N
        self.X = cp.Variable((M_x, N + 1), name="X")
        self.U = cp.Variable((M_u, N), name="U")
        self.X_stop = cp.Variable((M_x, N_stop + 1), name="X_stop")
        self.U_stop = cp.Variable((M_u, N_stop), name="U_stop")
        self.x_start = cp.Parameter(M_x, name="x_start")
        self.x_g = cp.Parameter(M_x, name="x_g")
        self.x_g_stop = cp.Parameter(M_x, name="x_g_stop")
        self.cs = cp.Parameter((m, N + 1), name="cs")
        self.rs = cp.Parameter(N + 1, name="rs")
        self.cs_stop = cp.Parameter((m, N_stop + 1), name="cs_stop")
        self.rs_stop = cp.Parameter(N_stop + 1, name="rs_stop")

        j_lower = -np.ones(m) * 2.
        j_upper = np.ones(m) * 2.

        objective_perf = 0
        for i in range(N):
            objective_perf += cp.quad_form(self.X[:, i] - self.X[:, -1], Q) + cp.quad_form(self.U[:, i], R)
        objective_perf += cp.quad_form(self.X[:, -1] - self.x_g, Q_e)
        objective_safety = 0
        for i in range(N_stop):
            objective_safety += cp.quad_form(self.X_stop[:, i] - self.X_stop[:, -1], Q) + cp.quad_form(
                self.U_stop[:, i], R)
        objective_safety += cp.quad_form(self.X_stop[:, -1] - self.x_g_stop, Q_e)

        objective = objective_safety +  objective_perf

        box_lims = np.vstack([np.eye(m), -np.eye(m)])
        const = [
            self.X[:, 1:] == A @ self.X[:, :-1] + B @ self.U,
            self.X[:, 0] == self.x_start,

            self.X[self.M_p:, -1] == 0,
            self.U[:, -1] == np.zeros(self.M_p, ),
            self.X_stop[:, 1:] == A @ self.X_stop[:, :-1] + B @ self.U_stop,
            self.X_stop[:, :2] == self.X[:, :2],
            self.X_stop[self.M_p:, -1] == 0, # np.zeros(m, ),
            self.U_stop[:, -1] == np.zeros(self.M_p, ),

            self.X[:self.M_p, :] >= j_lower[:, None],
            self.X[:self.M_p, :] <= j_upper[:, None],
            self.X_stop[:self.M_p, :] <= j_upper[:, None],
            self.X_stop[:self.M_p, :] >= j_lower[:, None],

            box_lims @ self.X[m:, :] <= v_lim,
            box_lims @ self.X_stop[m:, :] <= v_lim,
            box_lims @ self.U <= u_lim,
            box_lims @ self.U_stop <= u_lim,
        ]
        const += [
            cp.norm(self.X_stop[:self.M_p] - self.cs_stop, axis=0) <= self.rs_stop,
            cp.norm(self.X[:self.M_p] - self.cs, axis=0) <= self.rs
        ]
        self.problem = cp.Problem(cp.Minimize(objective), const)
        self.solver = cp.CLARABEL

    def set_solver(self, solver):
        self.solver = solver

    def solve(self, x, x_g, cs, rs, c_stop, r_stop, x_g_stop):
        self.x_start.value = x
        self.x_g.value = x_g
        self.x_g_stop.value = x_g_stop

        self.cs.value = cs
        self.rs.value = rs
        self.cs_stop.value = c_stop
        self.rs_stop.value = r_stop
        try:
            if self.compiled:
                self.loss = self.problem.solve(method='CPG', verbose=False)
                failure = not self.problem.status.startswith("1")
            else:
                self.loss = self.problem.solve(solver=self.solver, verbose=False)
                failure = self.X.value is None # or self.problem.status.endswith("inaccurate")
        except:
            failure = True
        if failure:
            X = np.zeros((self.N + 1, self.M_x))
            U = np.zeros((self.N, self.M_u))
            return False, X, U, X, U
        else:
            return True, self.X.value.T, self.U.value.T, self.X_stop.value.T, self.U_stop.value.T

    @classmethod
    def load_default(
            cls,
            nr_dof=3,
            H=10,
            dt = 1e-2,
            u_lim=None,
            v_lim=None
    ):
        # scaling = 0.01
        A, B = get_linear_double_integrator_discrete_dynamics(nr_dof=nr_dof, dt=dt)
        _, n_x = A.shape
        _, n_u = B.shape

        n_p = int(n_x / 2)
        Q = np.eye(n_x) * 10
        Q[n_p:, n_p:] *= 0.01
        Q_e = np.eye(n_x) * 1e4
        R = np.eye(n_u) * 1e-2
        return cls(
            A,
            B,
            Q,
            Q_e,
            R,
            H,
            u_lim=u_lim,
            v_lim=v_lim
        )

