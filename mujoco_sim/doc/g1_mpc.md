# Convex MPC on the Unitree G1

The same convex MPC that walks and runs the 1.3 kg biped in this package,
driving a Unitree G1: a 33 kg, 29-joint humanoid whose legs are 43% of its
mass. It stands, marches, walks up to 0.7 m/s, sidesteps, turns, and takes
120 N shoves while walking, all under the same SRBD model and QP (`mpc.py`,
unchanged). The acceptance tests are in `g1_validation.md`, 9/9 passing.

```bash
uv run g1 --view                              # walk at 0.3 m/s, with the viewer
uv run g1 --vs 0.6 --view                     # faster (above 0.6 it ramps up, see --ramp)
uv run g1 --vs 0.3 --yaw-rate 0.3 --view      # walk a circle
uv run g1 --vs 0 --vy 0.1 --view              # sidestep left
uv run g1 --stand --push 4 100 0 --view       # stand, shoved forward at t = 4 s
uv run g1 --vs 0.3 --push 6 120 0 --video g1_push.mp4
uv run g1 --no-wbc --view                     # -J^T w instead of the whole-body QP
uv run python -m mujoco_sim.validate_g1 --jobs 8 --write
```

It simulates at about real time headless (1 ms step, full controller every
step), and at about half real time while recording video.

## The robot

`mujoco_sim/assets/unitree_g1/` is MuJoCo Menagerie's `unitree_g1` (29 DOF,
rev 1.0, Menagerie commit c96a32d), copied verbatim with only the meshes
`g1.xml` references, and with Unitree's BSD-3 licence. `g1.load_model()`
changes it on an `MjSpec` at load time and leaves the files alone:

- Every **position servo becomes a torque motor**, its control range the
  joint's own torque limit (88 / 139 N m at hips and knees, 50 at the
  ankles, 25 at the shoulders).
- A **sole site** per foot, the ankle-roll axis projected onto the sole
  plane, 3.5 cm down: the reference point the MPC's moment arms and contact
  moments are taken about. G1's foot touches the floor through four 5 mm
  spheres spanning 0.12 m ahead of the ankle to 0.05 m behind it and
  ±0.025–0.03 m across, which the MPC's CoP limits copy.
- **Waist damping** (80 N m s/rad), for the plain `-J^T w` mode only; see
  fault 3 below.
- Floor, cameras, a 1 ms step, the elliptic friction cone, and the
  `mpc_stand` keyframe: knees bent 0.9 rad, CoM at 0.645 m.

## The controller

`g1_controller.G1WalkController` subclasses the biped's
`MPCWalkController`. The gait clock, the Raibert footstep plan, the heading
set-point, the integral trims and the MPC are the biped's code; to share it,
`controller.py` gained class attributes for the MJCF names it binds to and
two small hooks (`_stance_feedforward`, `leg_lengths`), with the biped's
behaviour unchanged (all 12 of its validation stages give the same numbers).
What is new:

```
 (v_fwd, v_left, yaw rate) -> gait clock + footstep plan -> contact schedule, moment arms
                                                                  |
 CoM + pelvis attitude ----------------------------------> convex MPC (QP, 33 Hz)
                                                                  |  contact wrenches
 swing legs: numerical IK -> joint targets --+                    v
 waist, arms: pose + arm swing -------------+--> whole-body QP (1 kHz) -> joint torques
```

- **Whole-body QP** (`wbc.py`). A weighted QP over the full rigid-body
  dynamics: joint accelerations and contact wrenches, subject to the
  floating base's equations of motion and the MPC's own friction and CoP
  rows. It keeps the wrenches within 0.2 N of the MPC's, so it acts as
  inverse dynamics: the torques that make the ground push back with the
  MPC's wrench while swing legs, waist and arms follow their own targets.
  DAQP solves it every millisecond.
- **Numerical swing IK.** G1's hip is pitch-roll-yaw with offsets, and its
  thigh and shin differ, so the biped's closed form does not apply. Two
  damped-least-squares iterations per tick on the sole site's 6×6 Jacobian,
  warm-started, solved against a pelvis at the *heading set-point's*
  attitude rather than the measured one (the biped's faults 3 and 6).
- **Upper body.** The waist is held straight; the arms hold a relaxed pose
  and swing against the legs as the biped's do.

The SRBD's parameters are in `g1_srbd_params()`: mass 33.34 kg, inertia
diag(1.2, 0.6, 0.33), horizon 21 × 30 ms, friction 0.6, CoP rectangle from
the sole geometry. The gait (`G1GaitParams`) is a 2.5 Hz clock at duty 0.6
(0.16 s swings), 5 cm swing clearance, CoM at 0.63 m, feet 0.09 m either side
of the CoM.

## Results

| | |
|---|---|
| stand, 100 N × 0.1 s shove | 0.2 mm drift, peak pitch 0.040 rad, without a step |
| march in place, 12 s | 1.6 mm/s drift, roll / pitch p95 0.007 / 0.006 rad |
| walk 0.3 m/s, 20 s | 0.3004 m/s, roll / pitch p95 0.007 / 0.010 rad, heading drift 0.04 rad |
| ramp to 0.6 m/s | 0.596 m/s, pitch p95 0.017 rad |
| sidestep 0.1 m/s | 0.0998 m/s |
| turn 0.3 rad/s at 0.3 m/s | 0.303 rad/s at 0.300 m/s |
| 120 N × 0.1 s forward / 80 N sideways shove, walking | back to within 0.003 m/s of the command 3 s later |

(`g1_validation.md`; roll and pitch are the pelvis's.) Beyond those stages,
checked once: a step from rest to 0.6 m/s, a 4.2 s ramp to 0.7 m/s (0.69
m/s, pitch p95 0.02 rad), a 0.6 rad/s turn, a 0.25 m/s sidestep, and walking
shoves of 180 N forward, 150 N backward and 120 N sideways.

**Limits.** A 6 s ramp to 0.8 m/s falls 1.5 s after reaching it, as does a step
from rest to 0.7 m/s or a 2 s ramp to it. Standing with both feet down, a
150 N shove (0.45 m/s) is too much without a step. There is no running
gait: G1 is not built for it and it is not tried.

## What it took

Each of these was found with a number, in the spirit of the biped's list.
The first three are the biped's faults again at a different scale.

1. **The legs are not massless.** `-J^T w` gives the torque that delivers
   `w` at the foot only if the joints are not also holding the leg up. For
   the biped that is a rounding error; for G1's 7.2 kg legs it is not:
   standing, a wrench that should have held the robot still raised it 1.6 cm
   in 0.3 s. The plain mapping adds the leg's own gravity and Coriolis
   torques (`qfrc_bias`); the whole-body QP accounts for them by
   construction.
2. **The effective inertia is not the rigid one** (the biped's fault 1).
   Measured by driving the standing robot with a known extra foot force:
   roll 1.1–1.8, pitch 0.60, yaw 0.33 kg m², against 3.55 / 3.17 / 0.54 for
   the rigid robot and 1.61 / 1.38 / 0.29 for pelvis and upper body alone.
   Before the whole-body QP, the composite value made the standing robot's
   pitch rate double in sign-alternating steps at the MPC's rate. It is a
   compromise between standing and walking, as the table below shows:
   larger values walk better and cannot stand still.
3. **A soft waist makes the pelvis wobble.** The MPC measures the 3.8 kg
   pelvis; the 19 kg upper body hangs on it through a three-joint waist. At
   400 N m/rad the pelvis rolled ±0.02 rad against the torso at the MPC's
   own rate, and the QP chased that with a sideways force that flipped sign
   every solve. The plain mapping holds the waist at 3000 N m/rad with the
   damping in the model, where the integrator treats it implicitly; the
   whole-body QP needs neither.
4. **Heavy legs cost trunk steadiness at speed.** With the plain mapping G1
   passes as many tasks, but at 0.7 m/s its roll and pitch p95 are 0.14 and
   0.18 rad against 0.01 and 0.04 through the whole-body QP.
5. **Walk low enough to keep the knee bent** (the biped's fault 10). At a
   0.66 m CoM the robot loses the 0.7 m/s ramps and the two largest shoves;
   an earlier version was seen vaulting off a knee locked at its stop.
6. **A 2.5 Hz clock.** G1's pendulum time constant is sqrt(z/g) = 0.25 s,
   and the longer the swing against it, the further the sideways fall over
   the stance foot grows before the next foot catches it. At 1.6 Hz a ramp
   to 0.6 m/s makes only 0.42 and pitches 0.29 rad.

And one that cost more than all of those together:

7. **A plausible fix that was the bug.** The swing legs lagged their IK
   targets - the joint PD damped towards zero velocity while the target
   moved fast - so the target's joint velocity, differenced tick to tick
   from the IK and low-passed, was fed to the damping term. Tracking error
   dropped from 0.2 to 0.02 rad, and it stayed in for most of the port. It
   was also what kept the robot falling: the differenced velocity is noisy,
   and removing it took pitch p95 at 0.3 m/s from 0.04 to 0.01 rad and made
   settings walk that had fallen within two seconds. Every conclusion drawn
   while it was in had to be measured again, and several did not survive:
   that the plain mapping cannot walk G1 at all, that a smoother swing
   profile was needed, that the foot had to be planted behind the CoM, and
   that the whole-body QP needed a guard against feet bouncing loose. What
   is listed above is what survived.

Also tried and dropped:

- **A pelvis-attitude task in the whole-body QP**, asking the pelvis to turn
  the way the MPC's model says its wrench would, so the QP shades the wrench
  to cancel the swing leg's reaction. The QP's only way to find more moment
  inside the CoP limits is to raise `f_z`, since the limits scale with it:
  it drove the stance foot to its heel limit and `f_z` to its 700 N cap, and
  launched the robot.
- **Exact zero-order-hold discretisation** in place of the MPC's forward
  Euler: no measurable difference, so `mpc.py` is unchanged.

## Ablations

Each row changes one thing from the defaults and runs 18 tasks, 12 s each:
march; walk 0.3 m/s; ramps to 0.5 (3 s), 0.6 (4 s), 0.7 (5 s), 0.7 (2 s)
and 0.8 m/s (6 s); a step to 0.5 m/s; sidesteps at 0.1 and 0.25 m/s; turns
of 0.3 and 0.6 rad/s at 0.3 m/s; walking shoves of 60, 180 and −150 N forward
and 40 and 120 N sideways; and a 150 N shove standing. A task passes if the
robot is still up at the end.

| variant | passed | notes |
|---|---|---|
| **defaults** | **15/18** | fails the 0.8 ramp, the 2 s ramp to 0.7 and the standing 150 N shove |
| `-J^T w` instead of the whole-body QP | 15/18 | at 0.7 m/s, roll / pitch p95 0.14 / 0.18 rad against 0.01 / 0.04 |
| inertia (1.6, 1.4, 0.3), pelvis + upper body | 14/18 | also loses the 5 s ramp to 0.7 |
| inertia (2.4, 2.0, 0.45) | 16/18 | holds the 2 s ramp to 0.7, but falls standing still, before the shove |
| inertia (3.55, 3.17, 0.54), rigid | 16/18 | the same; falls standing at 1.6 s |
| CoM height 0.66 m | 12/18 | loses the 0.7 ramps and the 180 N and 120 N shoves |
| 2.0 Hz clock | 15/18 | pitch p95 0.08 rad at 0.7 m/s against 0.04 |
| roll / pitch weight 400 | 15/18 | |
| forward Euler vs. exact ZOH | 15/18 | no difference |

## Files

- `mujoco_sim/g1.py` — loading and modifying the Menagerie model.
- `mujoco_sim/g1_controller.py` — `G1WalkController`, `G1GaitParams`, `g1_srbd_params`.
- `mujoco_sim/wbc.py` — the whole-body QP.
- `mujoco_sim/g1_sim.py` — the `g1` command.
- `mujoco_sim/validate_g1.py` — the acceptance tests; `--write` regenerates
  `doc/g1_validation.md`, `--video DIR` records every stage (the recordings
  are in `result/g1/`).
