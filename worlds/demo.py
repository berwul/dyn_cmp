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


import numpy as np
import trimesh


class StaticObstacles:

    def __init__(self):
        p_s = np.r_[0.3, 0.0, 0.0]
        p_e = np.r_[0.5, 0.0, 0.0]
        N = 3
        s = np.linspace(0, 1, N)
        s = s[:, None]
        r = 0.075
        centers = p_s * (1-s) + p_e * s
        self.dims = np.c_[centers, r * np.ones(N)]
        self.obstacles = [
            trimesh.creation.icosphere(
                radius=r
            ).apply_translation(c)
            for (*c, r) in self.dims
        ]


class DynamicObstacle:

    def __init__(self, v=0.2, r_body=0.075):
        self.v = v
        self.p_s = np.r_[0.4, 0, 0.05]
        self.p_e = np.r_[0.4, 0, 0.3]
        self.r_direction = np.linalg.norm(self.p_s - self.p_e)
        self.r = r_body
        self.dims = np.r_[self.p_s, r_body]
        self.cm = trimesh.collision.CollisionManager()
        self.cm.add_object("dobs", trimesh.creation.icosphere(radius=r_body))

    def set_collision_geometries(self, t):
        dim, = self.get_dims_at_time(t)
        self.cm.set_transform("dobs", trimesh.transformations.translation_matrix(dim[:-1]))

    def get_dims_at_time(self, t):
        s = (self.v * t) % (2 * self.r_direction)
        if s < self.r_direction:
            a = self.p_s
            b = self.p_e
            s = s / self.r_direction
        else:
            b = self.p_s
            a = self.p_e
            s = (s - self.r_direction) / self.r_direction
        dims_t = np.r_[(1-s) * a + s * b, self.r]
        return dims_t[None]


def load_dynamic_scenario(dyn_o_speed=0.2):
    s_obs = StaticObstacles()
    d_obs = DynamicObstacle(v=dyn_o_speed)
    return s_obs, d_obs

