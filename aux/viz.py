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
from trimesh.path import Path3D
from trimesh.path.entities import Line


def render_sphere(scene, c=None, r=1, color=None, node_name=None, geom_name=None, transform=None):
    if color is None:
        color = [255, 0, 0]
    sph = trimesh.creation.icosphere(radius=r)
    if c is not None:
        sph.apply_translation(c[:3])
    sph.visual.face_colors = color
    scene.add_geometry(sph, node_name=node_name, geom_name=geom_name, transform=transform)


def render_path(scene, path, color=[0, 0, 255], node_name=None, geom_name=None):
    scene.add_geometry(Path3D(
        entities=[Line(np.r_[0:path.shape[0]], color=color)],
        vertices=path[:, :3]
    ), node_name=node_name, geom_name=geom_name
    )

