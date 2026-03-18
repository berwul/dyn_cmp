# Safe corridor motion planning for dynamic pick and place applications

<p align="center">
<img src="demo.gif"/>
</p>

This is the project repository for the paper entitled:

"Safe corridor motion planning for dynamic pick and place applications"

For proprietary reasons, the repository does not include the robot used in the experiments of the paper. The training of a neural signed configuration distance function (nSCDF) is not included in this repo. We instead provide a pre-learned network. To understand the learning process of the nSCDF we refer to the paper:

https://arxiv.org/abs/2502.16205


# Run demo example
The following example runs our motion planner on a generic manipulator. The example includes a pre-trained nSCDF with spheres as obstacle representation.

## Requirements:
- Python 3

##  Instructions
1) Install requirements
```bash
pip install -r requirements.txt
```
2) Run demo
```bash
python 1_run_demo.py
```
3) Visualize the simulation
```bash
python 2_viz_sim.py
```
The visualization illustrates the following:
- Static obstacles (gray spheres)
- Moving obstacle (red sphere)
- Manipulator collision geometries (gray tubes)
- Optimized performance path (green curve)
- Setpoint along performance path (green sphere)
- MPC trajectories (blue curves)

