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


import cvxpy as cp
import numpy as np

from aux.math import get_linear_double_integrator_discrete_dynamics


class SafeCorridorMPC:

    def __init__(
            self,
            A,
            B,
            Q,
            Q_e,
            R,
            N,
            N_stop,
            u_lim,
            v_lim=1.0,
            scaling_safety=1.0,
            add_slacks=False,
            config_lims=None
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
        self.N_stop = N_stop
        self.u_lim = u_lim
        _, M_x = A.shape
        _, M_u = B.shape
        self.M_u = M_u
        m = int(M_x / 2)
        self.N = N
        self.X = cp.Variable((M_x, N + 1))
        self.U = cp.Variable((M_u, N))
        self.X_stop = cp.Variable((M_x, N_stop + 1))
        self.U_stop = cp.Variable((M_u, N_stop))
        self.x_start = cp.Parameter(M_x)
        self.x_g = cp.Parameter(M_x)

        self.x_g_stop = cp.Parameter(M_x)

        self.cs = cp.Parameter((m, N + 1))
        self.rs = cp.Parameter(N + 1)
        self.c_stop = cp.Parameter((m, ))
        self.r_stop = cp.Parameter()
        objective = 0

        for i in range(N):
            objective += cp.quad_form(self.X[:, i] - self.X[:, -1], Q) + cp.quad_form(self.U[:, i], R) * (self.u_lim ** 2)
        objective += cp.quad_form(self.X[:, -1] - self.x_g, Q_e)

        scaling = scaling_safety
        for i in range(N_stop):
            objective += cp.quad_form(self.X_stop[:, i] - self.X_stop[:, -1], scaling * Q) + cp.quad_form(self.U_stop[:, i], R) * (self.u_lim ** 2)
        objective += cp.quad_form(self.X_stop[:, -1] - self.x_g_stop, scaling * Q_e)

        if add_slacks:
            self.slacks = cp.Variable(N + 1)
            objective += self.slacks.sum() * 1e3
        else:
            self.slacks = None


        if config_lims is None:
            j_lower = -np.ones(m) * 2.
            j_upper = np.ones(m) * 2.
        else:
            j_lower, j_upper = config_lims

        box_lims = np.vstack([np.eye(m), -np.eye(m)])

        const = [
            self.X[:, 1:] == A @ self.X[:, :-1] + B @ (self.U * self.u_lim),
            self.X[:, 0] == self.x_start,

            self.X[self.M_p:, -1] == np.zeros(m, ),

            self.X_stop[:, 1:] == A @ self.X_stop[:, :-1] + B @ (self.U_stop * self.u_lim),
            self.X_stop[:, :2] == self.X[:, :2],

            self.X_stop[self.M_p:, -1] == np.zeros(m, ),
            self.U_stop[:, -1] == np.zeros(self.M_p, ),

            self.X[:self.M_p, :] >= j_lower[:, None],
            self.X[:self.M_p, :] <= j_upper[:, None],

            self.X_stop[:self.M_p, :] <= 2.,
            self.X_stop[:self.M_p, :] >= -2.,

            box_lims @ self.X[m:, :] <= v_lim,
            box_lims @ self.X_stop[m:, :] <= v_lim,
            box_lims @ self.U <= 1.,  # u_lim,
            box_lims @ self.U_stop <= 1., #u_lim,
        ]


        if add_slacks:
            const += [
                cp.norm(self.X_stop[:self.M_p] - self.c_stop[:, None], axis=0) <= self.r_stop, # + self.slacks[0],
                cp.norm(self.X[:self.M_p] - self.cs, axis=0) <= self.rs + self.slacks,
                self.slacks >= 0
            ]
        else:
            const += [
                cp.norm(self.X_stop[:self.M_p] - self.c_stop[:, None], axis=0) <= self.r_stop,
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
        self.c_stop.value = c_stop
        self.r_stop.value = r_stop
        try:
            if self.compiled:
                self.loss = self.problem.solve(method='CPG', verbose=False)
                failure = not self.problem.status.startswith("1")
            else:
                self.loss = self.problem.solve(solver=self.solver)
                failure = self.X.value is None or self.problem.status.endswith("inaccurate")
        except:
            failure = True
        if failure:
            X = np.zeros((self.N + 1, self.M_x))
            U = np.zeros((self.N, self.M_u))
            return False, X, U, X, U
        else:
            return True, self.X.value.T, self.U.value.T * self.u_lim, self.X_stop.value.T, self.U_stop.value.T * self.u_lim

    @classmethod
    def load_default(
            cls,
            nr_dof=3,
            H=20,
            H_stop=10,
            dt = 1e-2,
            u_lim=10.,
            v_lim=1.,
            add_slacks=False,
            scaling_safety=1.0
    ):
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
            H_stop,
            u_lim=u_lim,
            v_lim=v_lim,
            add_slacks=add_slacks,
            scaling_safety=scaling_safety
        )
