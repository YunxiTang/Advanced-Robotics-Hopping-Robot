# mujoco_sim

A two-legged robot walking **in 3-D** in MuJoCo under a **convex
model-predictive controller**, in the style of MIT Cheetah 3 (Di Carlo et al.,
IROS 2018): forward, backward, sideways, turning, or any mix of the three.

```bash
uv sync && ./fix_mjpython.sh        # once (the shell script is macOS-only)
uv run biped --view                 # walk, with the viewer
```

That is the whole thing: `biped` is the only entry point, MPC is the only
controller. `--view` opens the interactive viewer (Tab cycles through the
chase cameras); on macOS the command re-execs itself under `mjpython`
automatically, because Cocoa requires GUI calls on the main thread.

```bash
uv run biped                                   # headless, 30 s, prints a summary
uv run biped --vs 0.7 --duration 60            # faster, for longer
uv run biped --vs 0 --vy 0.2 --view            # sidestep to the left
uv run biped --vs 0.3 --yaw-rate 0.5 --view    # walk a circle
uv run biped --vs 0 --view                     # march in place
uv run biped --plot --save out.npz             # plots, and the raw log
uv run biped --vs 0.3 --yaw-rate 0.5 --video circle.mp4   # record to MP4
uv run python -m mujoco_sim.validate_mpc --jobs 8 --write   # the acceptance tests
uv run python -m mujoco_sim.validate_mpc --jobs 8 --video result   # ... with a video per stage
```

| flag | meaning | default |
|---|---|---|
| `--view` | interactive viewer | off |
| `--vs` | forward speed target, m/s (tracks 0 to 0.7) | 0.5 |
| `--vy` | sideways speed target, m/s, positive = left | 0 |
| `--yaw-rate` | turn rate, rad/s, positive = left | 0 |
| `--duration` | seconds of simulated time | 30 |
| `--height` | CoM height target, m | 0.36 |
| `--plot` / `--save` | figures (including a top-view path) / `.npz` log | off |
| `--video` | record the run from the chase camera to an MP4, with contact forces drawn as red arrows (needs `ffmpeg`) | off |

## The robot

- `mujoco_sim/biped.xml` — MJCF model: a rigid **box torso on a free joint**
  (all six floating DOFs, none of them actuated) and **two identical
  three-joint legs**. Each leg has a 2-DOF hip — **hip roll** (abduction,
  about the torso's x axis) and **hip pitch** — whose axes intersect at the
  hip pivot, then a **knee**, and a spherical point foot at the end of the
  shin. There are **no passive springs anywhere**: like the MIT Cheetah /
  ANYmal / Unitree quadrupeds, all six joints are plain torque motors, and
  every bit of compliance the gait shows is produced actively by the
  controller. Contact is handled by MuJoCo's solver — no manual phase
  switching, and stance vs. swing is read off the contact list.
- **There is no hip-yaw joint, and it is not needed to turn.** A point foot
  cannot resist spinning, so the stance leg simply pivots on its foot while
  the MPC makes the yaw moment from `r × f` — the same way it makes the roll
  and pitch moments.
- Roll, pitch and yaw are all **unactuated**: nothing drives the trunk's
  attitude directly, which is what makes this an underactuated problem.
- Total mass 1.30 kg, walking CoM height 0.36 m, legs 0.17 + 0.17 m, hips
  ±0.068 m off the centreline, friction 0.8, timestep 0.5 ms.

## The controller

```
 (v_fwd, v_left, yaw rate) -> gait clock + Raibert footstep plan -> contact schedule, moment arms
                                                                          |
 CoM + attitude state -------------------------------------------> convex MPC (QP, 50 Hz)
                                                                          |  3-D ground forces f_i
                     stance leg: tau = -J_i^T f_i  <----------------------+
                     swing  leg: 3-joint IK + joint PD
```

- `mujoco_sim/mpc.py` — `SRBDParams` + `ConvexMPC`. The Cheetah-3 13-state
  single-rigid-body model — roll/pitch/yaw, CoM position, world angular
  velocity, CoM velocity, and a constant 1 that turns gravity into a linear
  term and keeps the problem a QP — linearised about the reference heading at
  each predicted step, so turning makes the dynamics time-varying but still
  linear. Condensed over 20 steps of 20 ms into 120 decision variables and
  200 inequality rows: a friction pyramid (inscribed in MuJoCo's elliptic
  cone) per foot per step, plus unilateral force bounds gated by the gait's
  contact schedule. Only the first step is applied; the whole problem is
  re-solved on the next tick. OSQP is set up once and updated in place so its
  factorisation stays warm — **about 1.7 ms per solve**, against the 20 ms
  budget.
- `mujoco_sim/controller.py` — `MPCWalkController`: the gait clock, the 2-D
  Raibert footstep plan (forward and lateral, placed around the arc when
  turning), the heading set-point, `τ = −Jᵀf` on stance legs and closed-form
  3-joint IK with a joint PD on swing legs.
- `mujoco_sim/sim.py` — simulation loop and CLI.
- `mujoco_sim/video.py` — offscreen MP4 recording, piped to `ffmpeg`.
- `mujoco_sim/validate_mpc.py` — the staged acceptance tests; `--write`
  regenerates `doc/mpc_validation.md`, `--video DIR` records every simulated
  stage (2-9) to `DIR/stageN_*.mp4`. The recorded videos are in `result/`.
- Design: `doc/convec_mpc.md` (§16 is the as-built record of the planar
  version, §17 of the move to 3-D). Measured results: `doc/mpc_validation.md`.

**There is no attitude PD anywhere.** That is the point of the exercise. A PD
law on the hip — what this package used before — has its gain capped by the
*leg's* mass-matrix diagonal rather than the torso's, so its authority is
bounded by a numerical limit rather than a physical one. The MPC asks for
ground forces directly, so the bound becomes the friction cone, and during
double support it can also trade load between the two feet — a moment a
single hip PD cannot produce at all.

### Measured

All nine validation stages pass (`doc/mpc_validation.md`):

| | result |
|---|---|
| steady walk, 0.5 m/s, 30 s | 0.501 m/s (0.2% error), roll / pitch p95 0.049 / 0.074 rad, heading drift 0.17 rad |
| speed steps 0.3 → 0.7 → 0.3 m/s | 0.299 / 0.707 / 0.303 m/s |
| sidestep, 0.2 m/s left | 0.199 m/s, no forward creep |
| turn, 0.5 rad/s at 0.3 m/s | 0.495 rad/s at 0.302 m/s |
| 5 N × 100 ms push, forward / sideways | recovers to within 0.011 / 0.012 m/s |
| march in place, 10 s | 8 cm total drift, CoM height range 17 mm |
| `−Jᵀf` with the torso pinned | contact force within 0.03 N of the command |

**Speed ceiling: 0.7 m/s.** At 0.8 m/s the robot walks for 5–15 s and then
trips (yaw jumps ~1 rad in 0.1 s and the stance leg straightens). The planar
version of this robot reached 0.8 m/s, because it could not roll, yaw or
sway, and its feet could not scuff sideways.

## Leg kinematics

Both closed-form relations below rely on thigh and shin being the **same**
length (`L1 == L2`), which makes the hip-knee-foot triangle isosceles:

    theta = hip + knee/2                                  (leg angle inside the roll plane)
    l     = sqrt(L1^2 + L2^2 + 2*L1*L2*cos(knee))         (leg length)

The first says the hip-to-foot line exactly bisects the thigh/shin angle; the
second says leg length depends on the **knee alone**. Because the hip-roll
and hip-pitch axes intersect, the foot relative to the hip (torso frame) is

    d = ( l sin(theta),  l cos(theta) sin(roll),  -l cos(theta) cos(roll) )

so the swing-leg IK is one step: `|d|` fixes the knee, `atan2(d_y, -d_z)`
fixes the roll, and `asin(d_x / l)` then fixes the hip. Stance legs do not use
this at all — they are driven by `−Jᵀf` with MuJoCo's own Jacobian, taken at
the contact point.

## What this cost, and what it teaches

Each of the faults below was fatal or crippling on its own, and each is
written up at the point in the code where it bites. They share a theme worth
stating once: *the model this controller optimises and the robot it drives
disagree in specific, measurable ways, and every one of those disagreements
has to be found with a number rather than a plot.*

From the planar version, all still in force:

1. **The rigid-body inertia is wrong by 2x** for this purpose. Only the
   torso's share responds when a ground force rotates the trunk, because the
   legs are torque-controlled and do not ride along. The measured effective
   pitch inertia was 0.0145 against a rigid-body 0.027; at the rigid-body
   value the robot could not stand for one second.
2. **`kd*dt/I < 2` is the wrong stability test for a multi-DOF leg.** The
   leg's inverse mass matrix is strongly off-diagonal, so hip and knee
   damping destabilise each other. The right condition is on the eigenvalues
   of `dt M^-1 diag(kd)`; the planar version once shipped gains where one was
   3.52 — a 2.5x amplification every timestep.
3. **Holding a swing foot at a world-frame angle is positive feedback.** Hip
   torque couples straight into trunk rotation through the mass matrix, so a
   `- torso_angle` term in the swing IK pushes the trunk the way it is already
   falling, at ~100 rad/s.
4. **A QP with an `f_z >= 0` bound will go ballistic to dodge an attitude
   cost.** A 3 N minimum load on scheduled stance feet forbids it.
5. **A stance leg is held in task space, not joint space.** A planted foot is
   stationary while its joints sweep, so joint-space damping on a stance leg
   brakes exactly the motion that carries the robot forward.

New in 3-D:

6. **Fault 3 has a yaw version, and it is sneakier.** No joint rotates about
   the vertical, so it looks as if the swing IK can safely use the measured
   yaw. It cannot: a trunk yawed by `ψ` carries each hip pivot `0.068·ψ`
   forwards or backwards, a world-fixed swing target then moves relative to
   the hip, and the hip's response yaws the trunk further. Measured, a
   0.5 m/s walk drifted 1 rad off heading in 4 s and fell. The swing leg is
   now resolved against the heading *set-point* and a nominal hip position, so
   no measured attitude enters its joint targets at all.
7. **A leashed set-point must be leashed where the MPC sees it, not where it
   is stored.** The heading set-point was clamped to ±0.3 rad of the measured
   yaw; whenever the error exceeded that, the set-point was dragged along, and
   a heading lost that way was never recovered (1.4 rad off course before a
   fall). The clamp now applies only to the reference handed to the MPC.
8. **The joints' own damping steals ground force.** `biped.xml` gives each leg
   joint a little viscous damping (0.2 at the hips). On a stance leg the
   joints sweep at 1–3 rad/s, so up to ~0.6 N m of every hip torque went into
   the damper — several newtons at the foot, against a 13 N body weight. The
   robot sagged, landed late (16% of scheduled stance time with the foot
   still in the air at 0.5 m/s) and bounced (13% of the time with both feet
   off the ground). Feeding the damping torque forward took those to 4% and
   2%, halved the roll excursion, and removed a 0.08 rad steady pitch offset.
   (Fault 9's fix then took them to 3% and 0%.)
9. **`−Jᵀf` must be taken at the contact point, not at the foot's centre.**
   The ground pushes one foot radius below the sphere's centre, on a shin that
   rotates. With the torso pinned, the centre Jacobian delivered 11% too
   little horizontal force; at the contact point the error is 0.03 N. (Pinning
   the torso properly took two tries: a weld constraint let 14 N of thrust
   lift the pelvis 8 cm, and resetting the base state each step still let the
   legs accelerate it inside the step, which loaded them like gravity.)
10. **Top speed is set by knee reach.** At the planar version's 0.408 m CoM
    height the stance leg hit its knee limit at the ends of the sweep once the
    3-D gait got going, the load went through the joint limit instead of the
    controller, and the robot pole-vaulted off a straight leg. Walking at
    0.36 m took the ceiling from 0.5 m/s to 0.7 m/s.
11. **Yaw must be weighted lightly.** Each swing leg carries angular momentum
    about the vertical, and a trunk with 0.002 kg m² of yaw inertia
    counter-rotates ±0.1 rad every step, the motion humans cancel by swinging
    their arms. It is internal and zero-mean, so fighting it wastes friction:
    raising the yaw weight from 150 to 600 makes the robot fall.

### Where it deviates from `doc/convec_mpc.md`

| item | doc | here | why |
|---|---|---|---|
| model | planar, 7 states | 3-D, 13 states | this is the 3-D version; §17 of the doc |
| inertia | 0.02686 (pitch) | diag(0.0145, 0.0145, 0.003) | measured effective values; see fault 1 and `SRBDParams.inertia` |
| `step_freq` | 2.5 Hz | 4.0 Hz | stride length is `v*duty/step_freq`, so a faster clock plants each foot closer to the CoM and shrinks the very moment the MPC spends its friction cone rejecting |
| `Q` pitch weight | 250 | 500 (roll too); yaw 150 | measured optimum; see fault 11 for yaw |
| friction | cone, `mu` | pyramid, `mu/sqrt(2)` | a linear pyramid inscribed in MuJoCo's elliptic cone, so its corners cannot slip |
| `f_z` lower bound | 0 | 3 N when in contact | see fault 4 |
| CoM height | 0.408 m | 0.36 m | see fault 10 |
| predicted CoM in `r_i(k)` | `v_des` | measured `v` | with a tracking error the doc's form misplaces every moment arm in the horizon by more than the moment arms themselves |
| set-point integral | not in the doc | on speed (forward and lateral) and height | the SRBD's force deficit is *steady*, and a proportional controller answers a steady disturbance with a steady error |
| stance torque | `−Jᵀf` | `−Jᵀf` at the contact point, plus joint-damping feedforward and a task-space foot hold | faults 5, 8 and 9 |
| Raibert baselines | kept as controls (§11) | **deleted** | they were written for the single-leg SLIP model this package started from and were broken for two legs |

The set-point integral is the interesting one. Everything the SRBD leaves out
— 18% of the mass and more than half the inertia in the legs, a compliant
contact, a `−Jᵀf` mapping that is only exact for a massless leg on a rigid
contact — shows up as forces the robot asks for and does not get. Trimming
the set-points the MPC aims at recovers the tracking without touching the
QP's structure, and the same trim drives both channels that regulate speed:
the MPC's velocity reference and the Raibert footstep offset.

## Appearance

The robot is styled after current commercial humanoids (Unitree H1/G1 and
similar). It has pale glossy shells over a graphite frame and machined grey
motor housings drawn on each hinge axis: crossed roll and pitch motors at the
hip, and one at the knee. Cyan LED accents run around the hips and the waist,
across the chest, and on the head's black visor. The visor faces forward, so
pitch and heading can be read at a glance. The arms are for looks only. The
model has no arm joints, so they are carried rigidly, slightly bent.

The floor is a checkerboard, and the model carries two chase cameras (Tab in
the viewer): `track`, side-on as before, and `chase`, a three-quarter view
from behind that shows roll and lateral sway. Both follow the CoM without
inheriting the torso's attitude, so the horizon stays level. The lights
follow the CoM too, so the shadow map stays tight around the robot and the
shadows stay sharp.

Appearance and dynamics are fully separated:

- **Mass without looks:** the geoms that carry mass (`torso_geom`, `head`,
  `thigh_*_visual`, `shin_*_visual`) are the original plain box, sphere and
  capsules, with their exact sizes and masses. They are in the `inertial`
  class or in geom group 3, which is hidden by default; press `3` in the
  viewer to show them.
- **Looks without mass:** every drawn shell belongs to the `visual` class,
  which forces `mass="0"` and turns off collision. This matters more than it
  looks: the gait is sensitive to leg inertia, and a decorative geom left at
  MuJoCo's default density would quietly retune the robot.
- **Contact:** the feet are the only colliding geoms on the robot.

## Setup

```bash
cd mujoco_sim
uv sync
./fix_mjpython.sh   # macOS only, see below; re-run after any `uv sync` that recreates .venv
```

## Notes

- **macOS + `mjpython`**: the interactive viewer requires Cocoa's GUI calls to
  happen on the main thread, so on macOS it must run under `mjpython` (shipped
  by the `mujoco` package) rather than plain `python`. `uv run biped --view`
  handles this for you by re-execing. With a `uv`-managed venv, `mjpython`'s
  native trampoline can still fail to `dlopen` `libpython3.11.dylib`, because
  it resolves `@executable_path` against the symlinked `.venv/bin/python`
  rather than the real interpreter path; `fix_mjpython.sh` symlinks the dylib
  into `.venv/lib/` to fix that — run it once after `uv sync`.
- **Sign conventions**: the attitude is standard right-handed Z-Y-X Euler
  angles (x forward, y left, z up): positive roll drops the right side,
  positive pitch is **nose-down**, positive yaw turns left. (The planar
  version measured pitch about −y, nose-up positive — the opposite sign.) The
  hip-pitch and knee axes are `0 -1 0`, so positive hip swings the leg forward.
  The free joint's angular velocity is in the body frame; the controller
  rotates it into the world frame the SRBD is written in.
- **Initial pose**: the gait starts from the `mpc_stand` keyframe, both feet
  directly under the hips at the walking height. There is no "stand still"
  pose: with two point feet in 3-D the support polygon is a line segment,
  and the robot is an inverted pendulum about that line however the feet are
  placed — it cannot be balanced at all without stepping. The `−Jᵀf` check in
  the validation therefore pins the torso rather than standing on the feet.
- **Keep the knee's joint limit off the load path**: that is the whole story
  of fault 10. You want the controller absorbing loads, not the joint
  constraint.
- **A diverging controller used to hang the process.** MuJoCo detects a bad
  `qacc`, prints `Nan, Inf or huge value in QACC` and *resets the simulation
  state*, sending `data.time` back to zero — so a `while data.time < duration`
  loop never terminates. The main loop is bounded by step count as well, and
  reports the reset rather than swallowing it.
