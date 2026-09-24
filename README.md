# Advanced-Robotics-Hopping-Robot
Advanced Robotics (ENGG5402@CUHK) course project, which includes:
- 2D Hybrid Dynamics (with soft contacts) of hopping robot;
- Implementation of Raibert controller that controls forward velocity and hopping height;
- Visualization of simulation results and animations.

## Project structure
```
matlab_sim/   MATLAB 2D hopping robot: hybrid dynamics, Raibert controller, plots/animation
              (entry point: Main.m; Falling Ball/ is the contact-model warm-up)
mujoco_sim/   Python + MuJoCo bipedal simulation with MPC controller (see mujoco_sim/README.md)
```
