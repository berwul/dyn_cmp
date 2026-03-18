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


from collections import Counter, defaultdict

import numpy as np


def run_planner(man, planner, q_s, q_g, dt, percp_fq, human=None, max_iters=2000, return_debug=False, return_ctimes=False, verbose=False):
    planner.initialize(q_s, q_g)
    x = np.r_[q_s, np.zeros_like(q_s)]
    cnt = 0
    goal_reached = False
    counter = Counter({"collision": 0, "standing_still": 0, "collision_in_motion": 0})
    ctimes = []
    debug_data = defaultdict(list)
    A, B = planner.get_dynamics()
    error = False
    while cnt < max_iters:
        planner.clear()
        p, v = np.split(x, 2)
        speed = np.linalg.norm(v)
        if verbose:
            print(f"--- {cnt}")
            print(f"speed : {speed:0.2f} q: {np.array2string(p, precision=2)}")
        if speed > 2:
            error = True
            break
        t = cnt * dt
        if human is not None:
            human.set_collision_geometries(t)
            new_measurement = (cnt % percp_fq) == 0
            if new_measurement:
                planner.observe(human.get_dims_at_time(t))
        u = planner.plan(x)
        # --------------
        if human is not None:
            coll = not man.is_collision_free(p, cm_obst=human.cm)
        else:
            coll = False
        is_moving = speed > 1e-3
        coll_in_motion = coll and is_moving
        if coll and not coll_in_motion:
            counter["collision"] += 1
            counter["standing_still"] += speed < 1e-3
        elif coll_in_motion:
            counter["collision_in_motion"] += 1
            break
        if return_debug:
            debug_data["sim"].append(planner.get_simulation_data())
            debug_data["inputs"].append(planner.get_input_data())
            debug_data["perf"].append(planner.get_performance_data())
            status_data = planner.get_status_data()
            debug_data["stats"].append(
                {**status_data, "coll_free": not coll, "speed": speed}
            )
        if return_ctimes:
            ctimes.append(planner.get_performance_data()["times"])
        # NEXT TIME STEP
        x = A @ x + B @ u
        cnt += 1
        goal_reached = np.linalg.norm(x - np.r_[q_g, np.zeros_like(q_g)]) < 0.1
        if goal_reached:
            p, v = np.split(x, 2)
            speed = np.linalg.norm(v)
            if verbose:
                print(f"[GOAL REACHED] speed : {speed:0.2f} q: {np.array2string(p, precision=2)}")
            break
    eval_data = {
        "error": error,
        "goal_reached": goal_reached,
        "cnt": cnt,
        **counter
    }
    multi_output = return_debug or return_ctimes
    if multi_output:
        outs = (eval_data, )
        if return_debug:
            outs += (debug_data, )
        if return_ctimes:
            outs += (ctimes,)
        return outs
    else:
        return eval_data

