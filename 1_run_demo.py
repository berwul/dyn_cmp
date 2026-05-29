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

import cvxpy as cp
import numpy as np

from aux.nscdf import nSCDF
from aux.simulation_pipeline import run_planner
from manipulators.demo.man import DemoManipulator
from planners.dpcmp.planner import DynamicCorridorTracking
from worlds.demo import load_dynamic_scenario


man = DemoManipulator()
s_obs, d_obs = load_dynamic_scenario()

nscdf = nSCDF.from_saved()
nscdf.set_dims(s_obs.dims)

np.random.seed(0)

q_s = np.r_[-0.25, 0.4, 0.5] * np.pi
q_g = np.r_[0.25, 0.4, 0.5] * np.pi

u_lim = 20.0
v_lim = 0.3

dt = 5 * 1e-2
percp_fq = 1


def initialize_cmpc():
    planner = DynamicCorridorTracking(
        nscdf,
        u_lim=u_lim,
        v_lim=v_lim,
        dt=dt,
        H=20
    )
    path = np.vstack([
        q_s,
        np.r_[-0.5, -0.1, 0.3] * np.pi,
        np.r_[0.5, -0.1, 0.3] * np.pi,
        q_g
    ])
    planner.set_path(path, delta=0.2)
    # If you want to run compiled:
    #   1) Install cvxpygen, pip install cvxpygen
    #   2) Uncomment code below:
    # from cvxpygen import cpg
    # code_name = "cmpc"
    # cpg.generate_code(planner.cont.problem, code_dir=str(P_COMPILED / code_name), solver=cp.CLARABEL)
    # assert False
    # compiled = False
    # if compiled:
    #     from compiled.cmpc_soft.cpg_solver import cpg_solve as mpc_cpg_solver
    #     planner.cont.problem.register_solve('CPG', mpc_cpg_solver)
    #     planner.cont.compiled = True
    planner.cont.set_solver(cp.CLARABEL)
    return planner

# Shared envelope
dims = np.r_[0.2, 0.2, 0.4] / 2
center = np.r_[0.4, 0.0, 0.2]
rr = 0.075
nr_samples = 1000
position_samples = np.random.uniform(-dims + rr, dims - rr, size=(nr_samples, 3)) + center[None]
dims_samples = np.c_[position_samples, np.ones(nr_samples) * rr]

planner = initialize_cmpc()
# Tighten the nominal corridor, gives safety corridor
rs = nscdf.sdf_batch(planner.path_centers, dims_samples)
rs_min = np.minimum(rs, planner.path_radii)

overlap_distance = np.linalg.norm(planner.path_centers[:-1] - planner.path_centers[1:], axis=-1) - (rs_min[:-1] + rs_min[1:])
is_connected = (overlap_distance < 0).all()
assert is_connected, "safety corridor is not overlapping"
planner.set_path_radii_min(rs_min)
path_track_static = planner.path_track.copy()

# Run planner
result, debug_data, ctimes = run_planner(
    man,
    planner,
    q_s, q_g,
    dt,
    percp_fq=percp_fq,
    human=d_obs,
    verbose=True,
    max_iters=1000,
    return_ctimes=True,
    return_debug=True
)

names = "MPC", "time-varying dyn corridor", "opt. path"
print("mean) " + " | ".join([f"{n} : {v * 1e3:0.2f} ms" for v, n in zip(np.nanmean(ctimes, axis=0), names)]))
print("max) " + " | ".join([f"{n} : {v * 1e3:0.2f} ms" for v, n in zip(np.nanmax(ctimes, axis=0), names)]))

p_c = pathlib.Path(__file__).parent
p_f = p_c / "results"
p_f.mkdir(exist_ok=True)

data_simulation_setup = {
    "percp_fq" : percp_fq,
    "dt": dt
}

with (p_f / "sim_data.pkl").open("wb") as fp:
    pickle.dump({
        **debug_data,
        "sim_setup": data_simulation_setup,
        "static_data": {"path_track_static": path_track_static, "path_centers": planner.path_centers, "rs_min": rs_min}
    },
        fp
    )

