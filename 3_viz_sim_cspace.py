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


import pathlib
import pickle
import time

import trimesh
import numpy as np


from aux.viz import render_path, render_sphere

fname = f"sim_data"
p_c = pathlib.Path(__file__).parent

with (p_c / "results" / f"{fname}.pkl").open("rb") as fp:
    data = pickle.load(fp)


data_simulation_setup = data["sim_setup"]
data_sim = data["sim"]
data_static = data["static_data"]
data_inpt = data["inputs"]

cnt = 0
cnt_old = 0

n_stop = len(data_sim) - 1

percp_fq = data_simulation_setup["percp_fq"]
dt = data_simulation_setup["dt"]



T_tool = trimesh.transformations.translation_matrix(np.r_[0.1, 0.0, 0.0])

def callback(s):
    t_s = time.time()
    global cnt, cnt_old
    data_i = data_sim[cnt]
    data_inpt_i = data_inpt[cnt]
    scene.delete_geometry(["X", "X_stop", "x_g_v", "x_g_em_v", "path_track", "c", "c_stop", "r_re"])

    if "X" in data_i:
        X = data_i["X"]
        P = X[:, :3]
        render_path(s, P, color=[0, 0, 255], geom_name="X")

    if "x_g_v" in data_i:
        p, _ = np.split(data_i["x_g_v"], 2)
        render_sphere(scene, p, r=0.01, color=[0, 0, 255], geom_name="x_g_v")

    if "x_g_em_v" in data_i:
        p, _ = np.split(data_i["x_g_em_v"], 2)
        render_sphere(scene, p, r=0.01, color=[0, 255, 0], geom_name="x_g_em_v")
    if "X_stop" in data_i:
        X = data_i["X_stop"]
        P = X[:, :3]
        render_path(s, P, color=[0, 255, 0], geom_name="X_stop")
    if "path_track" in data_i:
        path_track = data_i["path_track"]
        render_path(s, path_track, color=[0, 0, 255], geom_name="path_track")
    cnt = (cnt + 1) % len(data_sim)

scene = trimesh.Scene()
if "rs_min" in data_static:
    cs = data_static["path_centers"]
    rs_min = data_static["rs_min"]
    render_path(scene, cs, color=[255, 0, 0])
    for i, (c, r) in enumerate(zip(cs, rs_min)):
        render_sphere(scene, c, r=r, color=[0, 255, 0, 20], geom_name=f"ball_{i}")
        render_sphere(scene, c, r=0.01, color=[255, 0, 0], geom_name=f"cs_{i}")

if "path_track_static" in data_static:
    path_centers = data_static["path_track_static"]
    render_path(scene, path_centers, color=[0, 255, 0], geom_name="path_track_static")

scene.add_geometry(trimesh.creation.axis())
q = np.zeros(3,)
scene.add_geometry(trimesh.creation.axis())

T_camera = np.array(
    [
        [-0.98678497, -0.08070933, 0.14050418, 1.00460842],
        [0.02722376, -0.93737513, -0.347256, -1.60817625],
        [0.15973193, -0.33884195, 0.9271849, 7.69001634],
        [0., 0., 0., 1.]
    ]
)
scene.graph[scene.camera.name] = T_camera

scene.show(
    callback=callback
)
