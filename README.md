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
3) Run animation of the simulation in world space
```bash
python 2_viz_sim_wspace.py
```
The animation shows the following:
- Static obstacles (gray spheres)
- Moving obstacle (red sphere)
- Manipulator collision geometries (gray tubes)
- Optimized performance path and MPC performance trajectory  (blue curves)
- Setpoint along performance path (blue sphere)
- Optimized safety path and MPC safety trajectory  (green curves)
- Setpoint along safety path (blue sphere)

4) Run animation of the simulation in configuration space
```bash
python 3_viz_sim_cspace.py
```
The animation shows the following:
- Centerline of nominal corridor  (red curve)
- Safety corridor (green spheres)
- Optimized performance path and MPC performance trajectory  (blue curves)
- Optimized safety path and MPC safety trajectory  (green curves)
