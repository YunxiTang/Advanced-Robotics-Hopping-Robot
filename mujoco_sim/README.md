# mujoco_sim

MuJoCo reimplementation of the planar 1-legged (SLIP-style) Raibert hopping
robot from the MATLAB code in the parent directory (`set_model.m`, `Main.m`,
`Flight_*`, `Stance_*`, `Cart2Planar.m`/`Planar2Cart.m`).

The MATLAB version hand-derives the equations of motion for the flight phase
(free-falling point mass) and stance phase (polar leg-spring dynamics), then
manually switches between the two with `ode45` + zero-crossing events
(`FlightEvent.m`/`StanceEvent.m`) and re-expresses the state at each switch
(`Cart2Planar.m`/`Planar2Cart.m`).

Here there is no hand-derived dynamics at all: MuJoCo is the simulator.

- `mujoco_sim/hopper.xml` — MJCF model: a real rigid **box torso** (mass
  `model.m`, sized for enough rotational inertia that foot-contact impulses
  don't fling it into full tumbles) free to translate in x/z and pitch, and a
  genuine **two-link leg** — an actuated revolute **hip** (torso→thigh) and
  an actuated revolute **knee** (thigh→shin), with the foot at the end of the
  shin. There are **no passive springs anywhere in the model**: like the MIT
  Cheetah / ANYmal / Unitree quadrupeds, both joints are plain torque motors
  and all leg compliance is produced actively by the controller. Contact
  between the foot and the floor is handled by MuJoCo's contact solver — no
  manual phase-switching/event detection is needed; stance vs. flight is
  simply read off from whether the foot is in contact.
- `mujoco_sim/controller.py` — the same Raibert control laws as
  `Flight_Controller.m`/`Stance_Controller.m`/`Main.m` (energy-based leg
  thrust during stance, and the touchdown-angle law
  `theta = asin(v*T_stance/2/L0) + gain*(v - Vs)` during flight), but
  expressed as **leg impedance** rather than as commands on a prismatic DOF,
  because the two-link leg has no leg-length joint to command. See
  "Two-link leg kinematics and control" below.
- `mujoco_sim/sim.py` — simulation loop + CLI.
- `mujoco_sim/plotting.py` — result plots (height/velocity/leg length/energy
  vs. time), analogous to `Data_draw.m`.

## Two-link leg kinematics and control

With a prismatic leg, length and angle were independent DOFs and each had its
own actuator. A knee-based leg has neither: both leg length *and* leg
direction fall out of `(torso_pitch, hip, knee)`. Two closed-form relations
make this tractable, and both rely on thigh and shin being the **same
length** (`L1 == L2`), which makes the hip-knee-foot triangle isosceles:

    theta_leg = torso_pitch + hip + knee/2        (world-frame leg direction)
    l(knee)   = sqrt(L1^2 + L2^2 + 2*L1*L2*cos(knee))          (leg length)

The first says the hip-to-foot line exactly bisects the thigh/shin angle; the
second says leg length depends on the **knee alone**, independent of how the
thigh is oriented. So:

- **Leg length / thrust → knee.** The virtual leg spring and the
  energy-regulation thrust produce a radial force `F_r`, converted to knee
  torque through the length Jacobian: `tau_knee = (dl/dknee) * F_r`.
- **Leg angle (flight) and torso attitude (stance) → hip.** `tau_hip` is the
  PD effort directly, exactly as in the single-link version.

Note what is deliberately *not* done here. A textbook Jacobian transpose
would also put `0.5 * F_t` on the knee, since the knee swings the leg
direction too. That is correct only if `F_t` were a force applied at the
foot. It isn't — it's a torque between torso and thigh, which the hip motor
delivers by itself through its reaction on the torso. Routing half of that
high-gain signal into the 0.04 kg shin instead makes the knee torque
sign-flip every timestep and blow up (see Notes).

## Appearance

Amber cylinders mark the two actuated joints (hip and knee), drawn on their
hinge axes so each reads as a real revolute joint rather than a kink in the
leg; an amber block on the nose of the torso shows which way the robot faces,
making pitch readable at a glance. The floor is a checkerboard and the model
carries a `track` chase camera (select it in the viewer with Tab) — without
both of those the robot appears to hop in place, since it covers 20+ m and
there is otherwise nothing to reference the motion against.

Every cosmetic geom belongs to the `visual` default class, which forces
`mass="0"` and disables collision. This matters more than it looks: the gait
is sensitive to leg inertia, and a decorative geom left at MuJoCo's default
density would quietly retune the robot. The geoms that *do* carry mass
(`torso_geom`, `thigh_visual`, `shin_visual`, `foot_geom`) keep their exact
sizes and masses — restyling them changed nothing but their material. The
foot remains the only colliding geom on the robot.

## Setup

```bash
cd mujoco_sim
uv sync
./fix_mjpython.sh   # macOS only, see note below; re-run after any `uv sync` that recreates .venv
```

## Run

```bash
# headless, then plot the results
uv run python -m mujoco_sim.sim --duration 30 --plot

# interactive MuJoCo viewer (macOS requires mjpython, see note below)
uv run mjpython -m mujoco_sim.sim --duration 60 --view

# custom target height/velocity, save the log
uv run python -m mujoco_sim.sim --hs 0.4 --vs 1.0 --duration 30 --save out.npz
```

## Notes

- **macOS + `mjpython`**: the interactive viewer requires Cocoa's GUI calls to
  happen on the main thread, so on macOS you must launch it with `mjpython`
  (shipped by the `mujoco` package) instead of plain `python`. With a `uv`-managed
  venv, `mjpython`'s native trampoline can fail to `dlopen` `libpython3.11.dylib`
  because it resolves `@executable_path` against the symlinked
  `.venv/bin/python` rather than the real interpreter path. `fix_mjpython.sh`
  symlinks the dylib into `.venv/lib/` to fix this — run it once after `uv sync`
  (and again any time `uv sync` recreates `.venv`).
- **Sign convention for hip torque during stance**: the hip motor's torque
  acts directly on the `hip` DOF, but by Newton's third law the *reaction* on
  the torso (a separate body, not the same coordinate) has the **opposite**
  sign — commanding positive hip torque pushes `torso_pitch` more negative,
  not less. The attitude-balancing law in `stance_control` therefore has the
  same sign as `(torso_pitch, dtorso_pitch)`, not the negated
  target-minus-current form you'd write for a directly-actuated DOF. Getting
  this backwards (an easy mistake — it looks exactly like a plain PD "regulate
  to zero" law otherwise) turns the balancing loop into positive feedback and
  tips the robot over within the first stance phase.
- **Don't route leg-angle effort through the knee** (the bug that cost the
  most time on the two-link conversion): mapping the tangential effort with a
  full Jacobian transpose puts `0.5*F_t` on the knee. Because the shin is
  only 0.04 kg, feeding a high-gain attitude PD into it is the classic
  stiff-controller-on-tiny-inertia explicit-integration instability — the
  knee torque alternates sign every timestep (`+85, -307, +217, -243, ...`),
  spikes the ground reaction force to ~800 N, and throws the foot off the
  ground after 1-2 timesteps. The symptom is deceptive: the robot doesn't
  visibly explode, it just never gets airborne (stance ~3% of the time, body
  height pinned near the leg length) and skates along the floor, which looks
  like a tuning problem rather than a numerical one. Dropping the cross-term
  took stance from 3% to ~19% and made the gait work at the *original*
  attitude gains.
- **Attitude gains stayed at 40/4.** An intermediate version needed ~400/25
  because the tangential effort was being applied as a *foot force* (moment
  about the contact point is only `F_t * l ≈ 0.2*F_t`, so ~5x the gain for
  the same restoring torque). Once the hip applies its torque directly again,
  the original gains are correct — and the high ones are now actively
  unstable.
- **Initial pose must account for knee bend**: with a bent knee the leg only
  points straight down when `hip = -knee/2`. The keyframe originally kept
  `hip = 0` (correct for a straight single-link leg), which placed the foot
  ~25° off to one side and tipped the robot on its first touchdown.
- **Why the touchdown-angle law uses a fixed nominal stance duration, not a
  measured one**: `Main.m`'s law needs an estimate of how long the upcoming
  stance phase will last (`Tp(end)-Tp(1)`, measured from the *previous*
  stance). Re-measuring it live is fragile here: exactly when the gait
  degrades into fast, low, grazing bounces (the failure mode the law is
  supposed to correct), the measured duration collapses toward zero, which
  disables the law's velocity term right when it's needed most and freezes
  `theta_des` at a stale value — a vicious cycle. `ControllerParams.
  nominal_stance_duration` instead uses a fixed estimate from the leg
  spring's own natural period (`pi*sqrt(m_eff/k)`), so the law stays
  responsive to the *current* liftoff velocity on every bounce.
- **Target height/velocity and leg length**: `Vs`/`Hs` default to `1.0`
  m/s / `0.4` m (rather than the MATLAB source's `2.5`/`0.5`). With a short
  `L0 = 0.2` m leg, targeting a high forward speed geometrically forces a
  fast, flat, low-clearance gait (short legs *must* take fast, short strides
  to go fast — the same reason a person switches from bounding hops to a
  fast trot as they speed up). Pass `--vs`/`--hs` to push it faster, but
  expect the gait to flatten out as it approaches the leg's natural limits.
- **Known limitation — forward velocity undershoots `Vs`**: the steady gait
  settles around 0.76 m/s against a 1.0 m/s target (apex tracks better: ~0.36
  m vs. `Hs = 0.4`). Within each stride `dx` swings roughly 0.3–1.8 m/s, which
  is normal for hopping — it's the *mean* that sits low. The obvious knobs
  don't fix it: raising `landing_angle_gain` above ~0.4, `thrust_gain` above
  ~35, or `nominal_stance_duration` above ~0.08 all destabilise the gait
  rather than closing the gap. Removing the offset properly needs a richer
  controller than the MATLAB laws this port is reproducing.
- **Joint damping**: `hip` and `torso_pitch` carry a small `damping="0.2"`
  and the `knee` `damping="0.05"` for numerical robustness. Both leg joints
  are plain `<motor>` torque actuators — no position servos and no passive
  springs — so every bit of leg compliance is whatever the controller
  synthesises.
- **Keep the knee's joint limit off the load path**: the leg compresses to
  `l ≈ 0.162 m` (knee `≈ -1.5 rad`) on hard landings. An earlier `-1.3`
  lower limit meant the joint constraint, not the virtual spring, was
  absorbing those impacts; widening the range to `-1.9` left the limit
  unused in normal operation and noticeably improved pitch behaviour
  (|pitch|max 0.81 → 0.38 over 60 s).
