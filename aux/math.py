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


import control as ct
from control import matlab
import numpy as np


def interpolate_equidistant(path, delta=0.1, return_s=False):
    distance = np.linalg.norm(path[1:] - path[:-1], axis=-1)
    distance_acc = np.r_[0, np.cumsum(distance)]
    s = distance_acc / distance_acc[-1]
    dist = distance_acc[-1]
    nr_pnts = int(np.ceil(dist / delta) + 1)
    ss = np.linspace(0, 1, nr_pnts)
    if return_s:
        return ss, np.vstack([np.interp(ss, s, th) for th in path.T]).T
    else:
        return np.vstack([np.interp(ss, s, th) for th in path.T]).T


def get_linear_double_integrator_discrete_dynamics(nr_dof, dt, method="zoh"):
    m = nr_dof
    I_2 = np.eye(m)
    O_2 = np.zeros((m, m))
    A_c = np.block(
        [
            [O_2, I_2],
            [O_2, O_2]]
    )
    B_c = np.block([
        [O_2],
        [I_2]
    ])
    C = np.block([I_2, I_2])
    D = I_2
    sys = matlab.ss(A_c, B_c, C, D)
    sys_d = ct.sample_system(sys, dt, method=method)
    A, B = sys_d.A, sys_d.B
    return A, B


