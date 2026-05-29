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
import cvxpy as cp


class CorridorSolver:

    def __init__(self, cs):
        m, n = cs.shape
        P = cp.Variable((m, n))
        C = cs
        R = cp.Parameter((m,))
        objective = cp.norm2(P[:-1] - P[1:], axis=1).sum()
        ball_distance = cp.norm2(C - P, axis=1)
        constraints = [
            P[0] == cs[0],
            P[-1] == cs[-1],
            ball_distance <= R
        ]
        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        self.P = P
        self.R = R
        self.time_solve = 0

    def solve(self, rs):
        self.R.value = rs
        time_s = time.time()
        self.problem.solve(solver=cp.CLARABEL)
        self.time_solve = time.time() - time_s
        return self.P.value
