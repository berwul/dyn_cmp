import pathlib
import pickle

import cvxpy as cp
import numpy as np

from aux.nscdf import nSCDF
from aux.simulation_pipeline import run_planner
from manipulators.demo.man import DemoManipulator
from planners.dcmp.planner import DynamicCorridorTracking
from worlds.demo import load_dynamic_scenario


man = DemoManipulator()
s_obs, d_obs = load_dynamic_scenario()

nscdf = nSCDF.from_saved()
nscdf.set_dims(s_obs.dims)

np.random.seed(0)

q_s = np.r_[-0.25, 0.4, 0.5] * np.pi
q_g = np.r_[0.25, 0.4, 0.5] * np.pi

u_lim = 10.0
v_lim = 1.0
dt = 1 * 1e-2
percp_fq = 1


def initialize_cmpc():
    planner = DynamicCorridorTracking(
        nscdf,
        u_lim=u_lim,
        v_lim=v_lim,
        dt=dt,
        add_slacks=True
    )
    path = np.vstack([
        q_s,
        np.r_[-0.5, -0.1, 0.5] * np.pi,
        np.r_[0.5, -0.1, 0.5] * np.pi,
        q_g
    ])
    planner.set_path(path, delta=0.05)
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


planner = initialize_cmpc()

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
        "sim_setup": data_simulation_setup
    },
        fp
    )
