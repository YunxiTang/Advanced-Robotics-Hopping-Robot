# mujoco_sim

A two-legged robot walking and **running in 3-D** in MuJoCo under a
**convex model-predictive controller**, in the style of MIT Cheetah 3 (Di
Carlo et al., IROS 2018): forward, backward, sideways, turning, or any mix of
the three. It walks up to 1.3 m/s and runs, with both feet off the ground 30%
of the time, up to 1.7 m/s, swinging its arms against its legs as it goes.
By default it runs above 0.7 m/s and walks below.

The same MPC also drives a **Unitree G1** humanoid (33 kg, 29 joints), up to
0.7 m/s; see [Unitree G1](#unitree-g1).

```bash
uv sync && ./fix_mjpython.sh        # once (the shell script is macOS-only)
uv run biped --view                 # walk, with the viewer
uv run g1 --view                    # the Unitree G1, walking
```

That is the whole thing: `biped` and `g1` are the only entry points, MPC is
the only controller. `--view` opens the interactive viewer (Tab cycles through the
chase cameras); on macOS the command re-execs itself under `mjpython`
automatically, because Cocoa requires GUI calls on the main thread.

```bash
uv run biped                                   # headless, 30 s, prints a summary
uv run biped --vs 0.7 --duration 60            # faster, for longer
uv run biped --vs 1.0 --view                   # run (above 0.7 m/s it runs by itself)
uv run biped --gait run --vs 0 --view          # run in place
uv run biped --vs 0 --vy 0.2 --view            # sidestep to the left
uv run biped --vs 0.3 --yaw-rate 0.5 --view    # walk a circle
uv run biped --vs 0 --view                     # march in place
uv run biped --stand --view                    # stand still on both flat feet
uv run biped --plot --save out.npz             # plots, and the raw log
uv run biped --vs 0.3 --yaw-rate 0.5 --video circle.mp4   # record to MP4
uv run biped --vs 1.0 --view --head-cam        # live head-camera RGB-D in the viewer
uv run biped --vs 1.0 --duration 15 --rgbd result/head_rgbd_run_1.0mps   # record it
uv run biped-avoid --course slalom --view      # steer around obstacles it sees
uv run biped-avoid --course forest --video result/avoid_forest.mp4   # record from above, plus avoid_forest_headcam.mp4
uv run python -m mujoco_sim.validate_mpc --jobs 8 --write   # the acceptance tests
uv run python -m mujoco_sim.validate_mpc --jobs 8 --video result   # ... with a video per stage
```

| flag | meaning | default |
|---|---|---|
| `--view` | interactive viewer | off |
| `--vs` | forward speed target, m/s (walks up to 1.3, runs up to 1.7) | 0.5 |
| `--vy` | sideways speed target, m/s, positive = left | 0 |
| `--yaw-rate` | turn rate, rad/s, positive = left | 0 |
| `--gait` | `walk`, `run`, or `auto` (run above 0.7 m/s, walk again below 0.6) | `auto` |
| `--stand` | stand still on both feet instead of stepping | off |
| `--duration` | seconds of simulated time | 30 |
| `--height` | CoM height target, m | 0.36 |
| `--plot` / `--save` | figures (including a top-view path) / `.npz` log | off |
| `--video` | record the run from the chase camera to an MP4, with contact forces drawn as red arrows (needs `ffmpeg`) | off |
| `--rgbd STEM` | record the head camera to `STEM.mp4` (colour and depth side by side) and `STEM.npz` (raw frames, see [Head RGB-D camera](#head-rgb-d-camera)) | off |
| `--head-cam` | with `--view`, show the head camera's live colour and depth images in the viewer's bottom-right corner | off |

## The robot

- `mujoco_sim/biped.xml` — MJCF model: a rigid **box torso on a free joint**
  (all six floating DOFs, none of them actuated) and **two identical
  six-joint legs**. Each leg has a 3-DOF hip — **hip yaw**, **hip roll**
  (abduction) and **hip pitch** — whose axes all meet at the hip pivot, then
  a **knee**, then a 2-DOF **ankle** (pitch, then roll, axes meeting at the
  ankle point) carrying a **flat foot**. Two **three-joint arms** hang from
  the shoulders: **shoulder pitch**, **shoulder roll** and **elbow**. There
  are **no passive springs anywhere**: all eighteen joints are plain torque motors, and every bit of
  compliance the gait shows is produced actively by the controller. Contact
  is handled by MuJoCo's solver — no manual phase switching, and stance vs.
  swing is read off the contact list.
- **The foot is a 10 × 5 cm sole contacting the floor at four points**:
  small spheres at its corners (spanning 9.2 × 4.2 cm, 3.1 cm behind the
  ankle to 6.1 cm ahead), each a point-on-plane contact with friction.
  Together they carry a full contact wrench: the centre of pressure (CoP)
  can move anywhere inside the rectangle, and their friction resists
  spinning. This is used instead of MuJoCo's box-on-plane collision, whose
  contact count jumps between 1 and 4 as the sole rocks.
- **The hip yaw came with the flat foot.** A point foot cannot resist
  spinning, so the point-foot robot turned by pivoting on its stance foot. A
  flat foot does resist, so without a hip-yaw joint the trunk's heading would
  be locked to the stance foot and the robot could only turn by slipping.
  With it, the stance feet stay put while the trunk turns on the hip yaws,
  and each new foot is planted at the heading set-point.
- Roll, pitch and yaw are all **unactuated**: nothing drives the trunk's
  attitude directly. The feet now give the MPC a bounded moment to work
  with, limited by the CoP staying inside the sole, which is enough to
  **stand still**. The point-foot robot could not: two point feet span a
  line segment, and it had to keep stepping.
- **The arms swing against the legs** (see "Arm swing" below). They are
  0.05 kg each, taken out of the trunk's mass, so the robot still weighs
  the same.
- Total mass 1.31 kg, walking CoM height 0.36 m, legs 0.17 + 0.17 m plus
  0.028 m from ankle to sole, hips ±0.068 m off the centreline, friction
  0.8, timestep 0.5 ms.

## The controller

```
 (v_fwd, v_left, yaw rate) -> gait clock + Raibert footstep plan -> contact schedule, moment arms
                                                                          |
 CoM + attitude state -------------------------------------------> convex MPC (QP, 50 Hz)
                                                                          |  contact wrenches w_i = (f_i, m_i)
                     stance leg: tau = -J_i^T w_i  <----------------------+
                     swing  leg: 6-joint IK + joint PD
```

- `mujoco_sim/mpc.py` — `SRBDParams` + `ConvexMPC`. The Cheetah-3 13-state
  single-rigid-body model — roll/pitch/yaw, CoM position, world angular
  velocity, CoM velocity, and a constant 1 that turns gravity into a linear
  term and keeps the problem a QP — linearised about the reference heading at
  each predicted step, so turning makes the dynamics time-varying but still
  linear. The input is a **contact wrench per foot**: a world-frame force
  and a moment about the foot's sole point, expressed in the foot's yaw
  frame so the constraints stay constant. Condensed over 20 steps of 20 ms
  into 240 decision variables and 440 inequality rows. Per foot per step,
  those rows are a friction pyramid (inscribed in MuJoCo's elliptic cone),
  unilateral force bounds gated by the gait's contact schedule, **CoP limits**
  that keep the centre of pressure inside the sole with a 20% margin, and a
  torsional-friction bound. Only the first step is applied; the whole
  problem is re-solved on the next tick. The solver is **DAQP** (dense
  active set), which solves it exactly in **about 7 ms**, against the 20 ms
  budget; see fault 12 for why not OSQP.
- `mujoco_sim/controller.py` — `MPCWalkController`: the gait clock (walk or
  run, see below), the 2-D
  Raibert footstep plan (forward and lateral, placed around the arc when
  turning) and each foot's planned heading, the heading set-point, `τ = −Jᵀw`
  with the leg's full 6×6 Jacobian on stance legs, and closed-form 6-joint IK
  with a joint PD on swing legs.
- `mujoco_sim/sim.py` — simulation loop and CLI.
- `mujoco_sim/video.py` — offscreen MP4 recording, piped to `ffmpeg`.
- `mujoco_sim/camera.py` — the head RGB-D camera: rendering, recording and
  back-projection to point clouds.
- `mujoco_sim/local_planner.py` — obstacle avoidance from the head camera's
  depth image, the four obstacle courses, and the `biped-avoid` CLI.
- `mujoco_sim/validate_mpc.py` — the staged acceptance tests; `--write`
  regenerates `doc/mpc_validation.md`, `--video DIR` records every simulated
  stage (2-12) to `DIR/stageN_*.mp4`. The recorded videos are in `result/`
  (the point-foot robot's are kept in `result/point_foot/`).
- Design: `doc/convec_mpc.md` (§16 is the as-built record of the planar
  version, §17 of the move to 3-D, §18 of the ankle and flat feet, §19 of
  the running gait). Measured
  results: `doc/mpc_validation.md`.

**There is no attitude PD anywhere.** That is the point of the exercise. A PD
law on the hip — what this package used before — has its gain capped by the
*leg's* mass-matrix diagonal rather than the torso's, so its authority is
bounded by a numerical limit rather than a physical one. The MPC asks for
ground forces directly, so the bound becomes the friction cone, and during
double support it can also trade load between the two feet — a moment a
single hip PD cannot produce at all. With flat feet it also places each
foot's centre of pressure, which in single support is the only moment
authority left once the friction cone is spent.

### Walking and running

Both gaits are the same controller. The gait clock gives each foot a stance
of `duty` of every 0.25 s cycle: **0.65 walking**, so there is always a foot
down and twice a cycle both are, and **0.35 running**, so twice a cycle
*neither* is. Nothing else changes. A flight phase is just a stretch of the
MPC's contact schedule with no foot in it; the QP sees it coming up to
0.4 s ahead and plans the push-off that carries the body over it.

With `--gait auto`, the default, the controller runs whenever the commanded
speed is above 0.7 m/s and walks again below 0.6 m/s. The duty is slewed
between the two at 0.5 per second rather than switched, so a change of gait
takes about two strides. An instant switch would reassign a leg that is
halfway through its stance to halfway through its swing.

Running needed one fix, and it improved walking too: **the foot must
actually land before it is given its stance wrench** (fault 16).

### Arm swing

Each arm swings with the *opposite* leg, the way a person's does. The
shoulder-pitch target is 1.5 × half the difference between the two legs'
measured angles (hip + knee/2, the hip-to-ankle line relative to the
trunk), low-passed over 20 ms and capped at ±0.6 rad. Following the legs'
real angles means the swing grows with stride length and disappears when
marching in place, with no speed schedule of its own: about ±10° walking at
0.5 m/s and ±12° running at 1.0 m/s. The elbows bend from 0.5 rad walking to
1.3 rad running, following the duty as the gait changes. The shoulder roll
holds the hands clear of the thighs. Each arm joint is a PD loop on its own
torque motor.

The MPC does not model the arms. To it they are two more unmodelled bodies
on the trunk, like the legs. Antiphase arms are the helpful kind of
disturbance, though: their pitch reactions cancel, and their yaw reaction
opposes the one the swinging legs put on the trunk (fault 11). They raised
both top speeds (fault 18):

| | walking top speed | running top speed | walk 0.5 m/s: yaw wobble p95 | run 1.0 m/s: roll / pitch p95 |
|---|---|---|---|---|
| no arm joints | 0.7 m/s | 1.25 m/s | — | 0.012 / 0.037 rad |
| arms held still | 0.9 m/s | 1.6 m/s (pitch p95 0.16 rad) | 0.038 rad | 0.013 / 0.039 rad |
| **arms swinging** | **1.3 m/s** | **1.7 m/s (0.09 rad)** | **0.028 rad** | **0.009 / 0.031 rad** |

A larger gain is not better: at 3.0 the arms swing ±18° walking and the yaw
wobble at 0.5 m/s rises to 0.047 rad, because the arms now over-cancel the
legs.

### Measured

All twelve validation stages pass (`doc/mpc_validation.md`). The right-hand
column is the point-foot robot, for comparison:

| | flat feet | point feet |
|---|---|---|
| stand still, 10 s, 2 N side push | 7 mm drift, attitude within 0.02 rad | impossible |
| steady walk, 0.5 m/s, 30 s | 0.500 m/s; roll / pitch p95 0.008 / 0.021 rad; heading drift 0.04 rad | 0.501 m/s; 0.049 / 0.074 rad; 0.17 rad |
| **run, 1.0 m/s, 20 s** | **0.999 m/s; both feet off the ground 29% of the time; roll / pitch p95 0.009 / 0.027 rad** | impossible (0.7 m/s ceiling) |
| **walk → run → walk, 0.5 → 1.0 → 0.5 m/s** | **0.51 / 0.99 / 0.51 m/s; airborne 0% / 29% / 0%; worst pitch 0.11 rad, at the changes** | — |
| speed steps 0.3 → 0.7 → 0.3 m/s | 0.301 / 0.699 / 0.301 m/s | 0.299 / 0.707 / 0.303 m/s |
| sidestep, 0.2 m/s left | 0.200 m/s, pitch p95 0.009 rad | 0.199 m/s, 0.062 rad |
| turn, 0.5 rad/s at 0.3 m/s | 0.500 rad/s at 0.300 m/s | 0.495 rad/s at 0.302 m/s |
| 5 N × 100 ms push, forward / sideways | recovers to within 0.002 / 0.002 m/s | 0.011 / 0.012 m/s |
| march in place, 10 s | 1.5 cm total drift, pitch p95 0.014 rad | 8 cm, 0.058 rad |
| `−Jᵀw` with the torso pinned | force within 0.004 N, moment within 6e-5 N m | force within 0.03 N |

Running also holds a turn (0.500 rad/s at 1.0 m/s), a sidestep (0.20 m/s),
running in place, and 5 N pushes forward and sideways at 1.0 m/s. Those are
not validation stages, just checks run once.

**Speed ceilings: walking 1.3 m/s, running 1.7 m/s.** Measured by ramping
to the speed over 4 s and holding it to 20 s, with the arms swinging:

| target | walk | run |
|---|---|---|
| 0.5 m/s | 0.50 m/s, pitch p95 0.02 rad | 0.50 m/s, 0.01 rad |
| 0.8 m/s | 0.80 m/s, 0.03 rad | — |
| 1.0 m/s | 1.00 m/s, 0.04 rad | 1.00 m/s, 0.03 rad |
| 1.2 m/s | 1.20 m/s, 0.05 rad | 1.20 m/s, 0.05 rad |
| 1.3 m/s | 1.31 m/s, 0.06 rad | 1.30 m/s, 0.05 rad |
| 1.4 m/s | falls | 1.40 m/s, 0.07 rad |
| 1.6 m/s | — | 1.60 m/s, 0.08 rad |
| 1.7 m/s | — | 1.70 m/s, 0.09 rad |
| 1.8 / 1.9 m/s | — | stays up, but makes only 1.73 / 1.79 m/s |
| 2.0 m/s | — | falls |

Before the arms had joints, the ceilings were 0.7 m/s walking (knee reach,
fault 10) and 1.25 m/s running (the trunk rode up into its legs until the
stance leg straightened before push-off ended, and the robot vaulted).
Walking was also the gait that tracked best at low speed, so the auto gait
still switches at 0.7 m/s: running is kept as a gait of its own rather than
something only needed at speed. At 1.0 m/s the two are about equally steady.

## Leg kinematics

The swing-leg IK stays closed form because the joints split cleanly into
three groups. The **hip yaw** is the foot's desired heading relative to the
heading set-point. The hip roll, hip pitch and knee then place the **ankle
point**, solved in the hip-yawed frame exactly as the point foot's IK was.
The two **ankle** joints level the sole: for a level trunk
`ankle_pitch = −(hip + knee)` and `ankle_roll = −hip_roll`. The ankle point
is 0.028 m above the sole, where the old foot sphere's centre was, so all
the leg geometry below is unchanged.

Both closed-form relations below rely on thigh and shin being the **same**
length (`L1 == L2`), which makes the hip-knee-ankle triangle isosceles:

    theta = hip + knee/2                                  (leg angle inside the roll plane)
    l     = sqrt(L1^2 + L2^2 + 2*L1*L2*cos(knee))         (leg length)

The first says the hip-to-foot line exactly bisects the thigh/shin angle; the
second says leg length depends on the **knee alone**. Because the hip-roll
and hip-pitch axes intersect, the foot relative to the hip (torso frame) is

    d = ( l sin(theta),  l cos(theta) sin(roll),  -l cos(theta) cos(roll) )

so that part of the IK is one step: `|d|` fixes the knee, `atan2(d_y, -d_z)`
fixes the roll, and `asin(d_x / l)` then fixes the hip. Stance legs do not use
this at all. They are driven by `−Jᵀw` with MuJoCo's own 6×6 Jacobian, taken
at the sole point below the ankle.

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

New with the ankle and flat feet:

12. **Contact moments make the QP too ill-conditioned for ADMM.** Two feet
    can trade moment against load in many nearly equivalent ways, and only
    the light regularisation tells them apart, so the Hessian's condition
    number is ~2×10⁶. OSQP needed a median 1275 iterations (~70 ms, 3.5×
    the 20 ms budget), and rescaling the moments to force units only halved
    that. A dense active-set solver does not mind: DAQP solves the same QPs
    exactly, in ~7 ms.
13. **Torsion transmits exactly, but only if the joint damping is fed
    forward.** With the torso pinned, `−Jᵀw` delivered every force and
    tilting moment to within 1%, but the twist about the vertical came out
    25–33% short. MuJoCo's soft friction lets a foot creep at 0.05 rad/s
    under a steady twist, and the hip yaw's own 0.2 damping took 0.01 N m of
    it. This is fault 8 again, for a new joint. The walking controller
    already feeds the damping forward, and the test now runs the same code.
14. **Attitude needs the same integral trim as speed and height.** The robot
    leaned back by a steady 0.057 rad marching in place (the point-foot robot
    too, by 0.026), 0.013 rad at 0.5 m/s and forward by 0.012 rad at
    0.7 m/s. That is a moment the SRBD does not model, and it depends on
    speed, so no fixed offset removes it. A clamped integral on the MPC's
    roll/pitch reference takes all three means below 0.001 rad.
15. **The ankle needs armature to be controllable.** The foot weighs 20 g,
    and its inertia about the ankle is ~10⁻⁵ kg m². Explicit integration of
    the swing PD would then cap the ankle's damping gain near 0.03. A
    2×10⁻⁴ kg m² armature (the reflected rotor inertia any geared ankle
    has) raises that tenfold without changing the foot's weight.

New with running:

16. **A scheduled stance is not a landed foot.** The gait clock is open
    loop, so stance starts on time whether or not the foot is down. After a
    flight phase it can still be centimetres up, and the MPC's wrench then
    has nothing to push on: `−Jᵀf` on an unloaded leg just flings it to full
    extension (the knee was measured past its stop), and the robot lands on
    a strut and pole-vaults off it. Now a scheduled-stance foot that has
    not touched keeps its swing controller, reaching down through its
    planned footstep at 0.5 m/s. The MPC does not count on it for the
    current step, and the QP is re-solved the moment it lands. Running at
    1.0 m/s from a standing start, that cut the height swing from 8.4 to
    2.0 cm and the pitch p95 from 0.080 to 0.036 rad. Ramped to 1.2 m/s, the
    robot went from making 1.12 m/s with 15 cm of bounce to 1.20 m/s with
    2 cm. Walking had the
    same fault on a smaller scale: its pitch p95 at 0.5 m/s halved, from
    0.040 to 0.021 rad. As a side effect, a stance foot that bounced loose
    used to reach for its *next* footstep; it now reaches for the one it was
    aimed at.
17. **Change gait by slewing the duty, not switching it.** Dropping the duty
    from 0.65 to 0.35 in one tick turns every leg between phase 0.35 and
    0.65 from stance into swing, part-way through its swing trajectory, so
    its target jumps. Slewed at 0.5 per second, the change takes about two
    strides and the swing target moves continuously.

New with the arms:

18. **Swinging arms are worth more than their mass.** Two 0.05 kg arms hung
    on joints, with the trunk 0.1 kg lighter to match, already raised the
    top speeds from 0.7 to 0.9 m/s walking and from 1.25 to 1.6 m/s running
    while held still. I have not isolated why; the arm mass now sits higher
    and wider than it did inside the trunk box, on PD-held joints. Swinging them against the legs took the ceilings to 1.3 and
    1.7 m/s. At 0.8 m/s walking it cut the pitch p95 from 0.078 to
    0.031 rad; at 1.6 m/s running, from 0.158 to 0.084 rad. The MPC still
    knows nothing about the arms. The swing works because it cancels the
    legs' angular momentum about the vertical, which fault 11 says the MPC
    should not fight with its friction cone, and because two arms in
    antiphase put no net pitch moment on the trunk.

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
| set-point integral | not in the doc | on speed (forward and lateral), height, and trunk roll/pitch | the SRBD's force deficit is *steady*, and a proportional controller answers a steady disturbance with a steady error (fault 14) |
| stance torque | `−Jᵀf` | `−Jᵀw` (force and moment) at the sole point, plus joint-damping feedforward and a 6-D task-space foot hold that also levels the sole | faults 5, 8, 9 and 13 |
| contact | point foot, force only | flat foot, force + moment, with CoP and torsion limits | §18 of the doc |
| gait | walking trot, fixed duty | walk (duty 0.65) or run (0.35), chosen from the commanded speed | §19 of the doc |
| arms | none | 3 DOF each, swung against the legs by a PD loop outside the MPC | fault 18 |
| touchdown | on schedule | on measured contact, after the schedule opens | fault 16 |
| QP solver | osqp | DAQP (dense active set) | fault 12 |
| Raibert baselines | kept as controls (§11) | **deleted** | they were written for the single-leg SLIP model this package started from and were broken for two legs |

The set-point integral is the interesting one. Everything the SRBD leaves out
— 18% of the mass and more than half the inertia in the legs, a compliant
contact, a `−Jᵀf` mapping that is only exact for a massless leg on a rigid
contact — shows up as forces the robot asks for and does not get. Trimming
the set-points the MPC aims at recovers the tracking without touching the
QP's structure, and the same trim drives both channels that regulate speed:
the MPC's velocity reference and the Raibert footstep offset.

## Head RGB-D camera

`head_cam` is fixed to the trunk just in front of the visor, 0.54 m above the
ground, looking forward and 20° down. It is modelled on a RealSense D435 in
its 424×240 mode: `fovy` 58° gives an 89° horizontal field. The image centre
meets the ground 1.5 m ahead, the bottom edge 0.47 m ahead, and the top edge
sits just above the horizon. Because the camera is rigid on the trunk, the
image rocks with the torso's pitch and roll, as a real head camera's does.

`camera.HeadCamera.capture(data)` returns two pixel-aligned images:

- **Colour:** `uint8`, H×W×3.
- **Depth:** `float32` metres, measured along the optical axis (z-depth, as
  depth cameras report it). It reads 0 where there is no data: nearer than
  the 0.12 m clip plane, or farther than 10 m.

Both come from the same render, so there is no registration step. The
intrinsics are `HeadCamera.K` (OpenCV convention, fx = fy = 216.5 px).
`depth_to_points(depth, K, cam_pos, cam_mat)` back-projects a depth image to
world points. Checked against the floor plane, the depth is exact to 0.3 mm
out to 7 m. Beyond 7 m, near the horizon, a single pixel covers metres of
floor, so a pixel's depth and the exact depth at its centre can differ by
that much.

`--rgbd STEM` writes two files:

- **`STEM.mp4`:** colour and colourised depth side by side at 30 fps,
  upsampled 2×. Depth runs red (0.3 m) to blue (5 m); black means no data.
- **`STEM.npz`:** raw frames at 10 Hz. The arrays are `rgb` (N,H,W,3
  uint8), `depth` (N,H,W uint16 millimetres, 0 = no data), `t`, `cam_pos`,
  `cam_mat` (the MuJoCo world pose, whose camera looks along −z with y up)
  and `K`.

`result/head_rgbd_run_1.0mps.*` is 15 s of running at 1.0 m/s.

A floor alone gives a depth image that is one gradient. To give the camera
something to see, props line both sides of the path along +x: boxes,
cylinders and spheres from x = 2.5 m to 20 m, at |y| ≥ 0.7 m. They are
`visual`-class geoms with no mass and no collisions, so the dynamics and
the validation results are unchanged. They are also placed clear of every
validation path.

The camera is a sensor only. The MPC still walks blind, and nothing reads
the images back into the controller. The local planner below uses them to
steer, through the same speed and turn-rate set-points as the command line.

On macOS, MuJoCo prints `ARB_clip_control unavailable ... depth accuracy
will be limited` when it first renders depth. The accuracy figures above
were measured with that warning present.

## Local planner: obstacle avoidance

`uv run biped-avoid --course NAME` walks the robot to a goal through an
obstacle course and steers around what the head camera sees
([local_planner.py](mujoco_sim/local_planner.py)). It runs at 10 Hz:

1. **Perceive.** Back-project the depth image (every 2nd pixel) to world
   points. Keep the ones 0.04–1.0 m above the floor and within 3 m. Drop
   anything within 0.3 m of the CoM, which is the robot's own hands.
2. **Remember.** Mark those points in a 5 cm world-frame grid. The camera
   loses sight of an obstacle before the robot has passed it, so the grid
   keeps a cell until it is 5 m behind. The world is static.
3. **Route.** A wavefront from the goal over a 10 cm copy of the grid, with
   obstacles grown by 0.25 m, gives each cell its walking distance to the
   goal around the obstacles seen so far. Unseen space counts as free.
4. **Choose a command.** Try 78 (speed, turn-rate) pairs (0–0.5 m/s,
   ±0.6 rad/s), each rolled out 2.5 s as a unicycle under the command's
   acceleration limits. Drop any that bring a 0.2 m-radius body into an
   occupied cell. Score the rest on that walking distance, taken 0.3 m
   ahead along the heading so facing the right way counts, plus clearance
   and speed. If nothing that moves is free, turn on the spot.
5. **Command.** Slew `Vs` and `yaw_rate` toward the winner. The MPC and
   gait are unchanged: the planner steers like a joystick would.

Scoring by straight-line distance to the goal, rather than the wavefront's
walking distance, fails in a known way. With a box dead ahead, creeping at
it ever slower always scored better than turning away from the goal, and
the robot stopped 0.7 m short of it.

Pose (CoM position and heading) comes from the simulator, so odometry is
perfect. Only the obstacles come from the camera.

The courses replace the scenery props with obstacles and add a
top-down orthographic camera, `overhead`, which `--video` records from. The
viewer and the video draw the seen cells in red, the chosen rollout in
green (orange when turning on the spot), and the goal in blue. Like the
props, the obstacles don't collide: only the feet have collision geometry,
so a solid obstacle would trip a foot while the trunk passed through it.
So a clearance monitor measures the smallest 3-D gap between any drawn part
of the robot and any obstacle, using `mj_geomDistance`. The planner never
sees it.

| course | obstacles | goal reached | closest approach |
|---|---|---|---|
| `single` | one 0.5 m box on the line | 16.3 s | 0.267 m |
| `slalom` | four posts, alternating sides | 20.7 s | 0.255 m |
| `wall` | a wall with a 0.9 m gap off the line | 16.1 s | 0.257 m |
| `forest` | 14 random posts and boxes | 22.5 s | 0.084 m |

No run fell, and max pitch stayed at 0.045 rad. In the forest the closest
approach is the left hand, which swings slightly outside the 0.2 m disc
the planner models the body as. The videos are `result/avoid_*.mp4` (from
overhead) and `result/avoid_*_headcam.mp4` (what the planner sees: colour and
depth side by side).

Limits. The planner only knows what the camera has shown it, so a dead end
it cannot yet see is still a place it will walk into and have to turn out
of. It commands only walking speeds (the auto gait runs from 0.7 m/s).

## Appearance

The robot is styled after current commercial humanoids (Unitree H1/G1 and
similar). It has pale glossy shells over a graphite frame and machined grey
motor housings drawn on each hinge axis: crossed roll and pitch motors at the
hip, one at the knee, and crossed pitch and roll motors at the ankle over a
dark rubber sole. Cyan LED accents run around the hips and the waist,
across the chest, and on the head's black visor. The visor faces forward, so
pitch and heading can be read at a glance. The head camera's lens sits
just below the visor's LED bar. The arms have a motor housing
at the shoulder and at the elbow, and a shell over the upper arm.

The floor is a checkerboard, and the model carries two chase cameras (Tab in
the viewer): `track`, side-on as before, and `chase`, a three-quarter view
from behind that shows roll and lateral sway. Both follow the CoM without
inheriting the torso's attitude, so the horizon stays level. The lights
follow the CoM too, so the shadow map stays tight around the robot and the
shadows stay sharp.

Appearance and dynamics are fully separated:

- **Mass without looks:** the geoms that carry mass (`torso_geom`, `head`,
  `thigh_*_visual`, `shin_*_visual`, `foot_*_mass`, `upper_arm_*_mass`,
  `forearm_*_mass`) are plain boxes, a
  sphere and capsules, with fixed sizes and masses. They are in the `inertial`
  class or in geom group 3, which is hidden by default; press `3` in the
  viewer to show them.
- **Looks without mass:** every drawn shell belongs to the `visual` class,
  which forces `mass="0"` and turns off collision. This matters more than it
  looks: the gait is sensitive to leg inertia, and a decorative geom left at
  MuJoCo's default density would quietly retune the robot.
- **Contact:** the four sole-corner spheres of each foot are the only
  colliding geoms on the robot. They are hidden with the mass geoms (group
  3); the drawn sole covers them.

## Unitree G1

`uv run g1` walks MuJoCo Menagerie's Unitree G1 (`mujoco_sim/assets/unitree_g1/`,
29 DOF, 33 kg, BSD-3) under the same MPC. The gait clock, footstep plan,
SRBD model and QP are the biped's code (`G1WalkController` subclasses
`MPCWalkController`). Three things are new, because G1's legs are 43% of its
mass where the biped's are nearly weightless:

- a **whole-body QP** (`wbc.py`) that turns the MPC's wrenches into torques
  through the full rigid-body dynamics instead of `-J^T w`;
- a **numerical swing-leg IK**, since G1's legs have no closed form;
- a held **waist** and swinging **arms**.

```bash
uv run g1 --view                              # walk at 0.3 m/s
uv run g1 --vs 0.6 --view                     # faster (ramped above 0.6 m/s)
uv run g1 --vs 0.3 --yaw-rate 0.3 --view      # walk a circle
uv run g1 --vs 0 --vy 0.1 --view              # sidestep
uv run g1 --stand --push 4 100 0 --view       # stand, shoved at t = 4 s
uv run python -m mujoco_sim.validate_g1 --jobs 8 --write   # its acceptance tests
```

All nine validation stages pass (`doc/g1_validation.md`, videos in
`result/g1/`): standing through a 100 N shove, marching, walking at 0.3 and
0.6 m/s, sidestepping, turning, and recovering from 120 N and 80 N shoves
while walking, with roll and pitch p95 at or below 0.02 rad. It tops out
near 0.7 m/s. `doc/g1_mpc.md` has the design, the measurements behind each
choice, an ablation table, and the one "fix" that turned out to be the bug.

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
  hip-pitch, knee and ankle-pitch axes are `0 -1 0`, so positive hip swings
  the leg forward and positive ankle pitch lifts the toe.
  The free joint's angular velocity is in the body frame; the controller
  rotates it into the world frame the SRBD is written in.
- **Initial pose**: the gait starts from the `mpc_stand` keyframe, both soles
  flat and directly under the hips at the walking height. With flat feet it
  is also a pose the robot can simply stand in (`--stand`, validation stage
  3). The `−Jᵀw` check in the validation still pins the torso, because that
  isolates the mapping from balance altogether.
- **Keep the knee's joint limit off the load path**: that is the whole story
  of fault 10. You want the controller absorbing loads, not the joint
  constraint.
- **A diverging controller used to hang the process.** MuJoCo detects a bad
  `qacc`, prints `Nan, Inf or huge value in QACC` and *resets the simulation
  state*, sending `data.time` back to zero — so a `while data.time < duration`
  loop never terminates. The main loop is bounded by step count as well, and
  reports the reset rather than swallowing it.
