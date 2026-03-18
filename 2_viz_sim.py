import pathlib
import pickle

import pyglet
import trimesh
import numpy as np

from aux.viz import render_path, render_sphere
from manipulators.demo.man import DemoManipulator
from worlds.demo import load_dynamic_scenario


s_obs, d_obs = load_dynamic_scenario()
man = DemoManipulator()

fname = f"sim_data"
p_c = pathlib.Path(__file__).parent
with (p_c / "results" / f"{fname}.pkl").open("rb") as fp:
    data = pickle.load(fp)

data_sim = data["sim"]
data_simulation_setup = data["sim_setup"]
data_static = data.get("static", {})
cnt = 0

n_stop = len(data_sim) - 1
dt = data_simulation_setup["dt"]
link_ee = "link_6"

def callback(s):
    global cnt
    data_i = data_sim[cnt]
    scene.delete_geometry(["X", "X_stop", "x_g_v", "path_track"])
    if "X" in data_i:
        X = data_i["X"]
        P = X[:, :3]
        ps = []
        for p in P:
            fk, = man.get_link_fk(p, links=[link_ee])
            ps.append(fk[:3, -1])
        ps = np.vstack(ps)
        render_path(s, ps, color=[0, 0, 255], geom_name="X")
    if "x_g_v" in data_i:
        q_g_v, _ = np.split(data_i["x_g_v"], 2)
        fk, = man.get_link_fk(q_g_v, links=[link_ee])
        p_g = fk[:3, -1]
        render_sphere(scene, p_g, r=0.01, color=[0, 255, 0], geom_name="x_g_v")
    if "X_stop" in data_i:
        X = data_i["X_stop"]
        P = X[:, :3]
        ps = []
        for p in P:
            fk, = man.get_link_fk(p, links=[link_ee])
            ps.append(fk[:3, -1])
        ps = np.vstack(ps)
        render_path(s, ps, color=[0, 0, 255], geom_name="X_stop")
    if "path_track" in data_i:
        path_track = data_i["path_track"]
        ps = []
        for p in path_track:
            fk, = man.get_link_fk(p, links=[link_ee])
            ps.append(fk[:3, -1])
        ps = np.vstack(ps)
        render_path(s, ps, color=[0, 255, 0], geom_name="path_track")
    t = cnt * dt
    q, v = np.split(data_i["x"], 2)
    man.update_scene(scene, q, s_data)
    dims_t, = d_obs.get_dims_at_time(t)
    p_t = dims_t[:3]
    s.graph.update(
        frame_to="dobs", frame_from="world", matrix=trimesh.transformations.translation_matrix(p_t)
    )
    if cnt == n_stop:
        pyglet.app.exit()
    cnt = (cnt + 1) % len(data_sim)


dim, = d_obs.get_dims_at_time(cnt * 1e-2)
*_, r = dim
d_obs_mesh = trimesh.creation.icosphere(radius=r, face_colors=[255, 0, 0])
scene = trimesh.Scene()

scene.add_geometry(trimesh.creation.axis())
s_data = man.add_to_scene(scene, geom_name_suffix="t")
for *c, r in s_obs.dims:
    scene.add_geometry(trimesh.creation.icosphere(radius=r), transform=trimesh.transformations.translation_matrix(c))
scene.add_geometry(d_obs_mesh, geom_name="dobs")
scene.add_geometry(trimesh.creation.axis())

T_camera = np.array([
    [0.07600603, -0.53513592, 0.84133978, 1.27296495],
    [0.99656, 0.06872358, -0.04631674, -0.04310753],
    [-0.03303413, 0.84196592, 0.53851846, 0.77269436],
    [0., 0., 0., 1.]]
)

scene.graph[scene.camera.name] = T_camera
scene.show(callback=callback)
