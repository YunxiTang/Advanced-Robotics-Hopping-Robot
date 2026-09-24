"""Convex-MPC controller for the 3-D two-legged (humanoid) robot.

MuJoCo owns the dynamics. Each leg is a real six-joint chain - a 3-DOF hip
(yaw + roll + pitch), a knee, and a 2-DOF ankle (pitch + roll) carrying a flat
foot - and two three-joint arms (shoulder pitch + roll, elbow) that swing
against the legs, all driven by plain torque motors, with no passive joint springs anywhere,
the same way the MIT Cheetah / ANYmal / Unitree robots are built. Every bit
of leg compliance the gait shows is produced actively by this file.

The controller itself is `MPCWalkController`: a convex model-predictive
controller on a single-rigid-body model (`mpc.py`) that optimises each foot's
3-D ground reaction *wrench* (force and moment), mapped to joint torques
through `-J^T w` with the leg's full 6x6 Jacobian. The same controller walks
and runs: the only difference is the gait clock's duty factor, and running's
flight phases are simply contact-schedule entries with no foot down. Design
and as-built notes: `doc/convec_mpc.md`.

Kinematics
----------
Joint angles are measured relative to the parent link: `hip_yaw` turns the
leg about the torso's z axis, `hip_roll` then tilts it about the yawed x axis,
`hip` is the thigh's pitch inside that rolled plane, and `knee` the shin angle
relative to the thigh (knee=0 is a fully straight leg). All three hip axes
intersect at the hip pivot. The ankle's pitch and roll axes intersect at the
ankle point, and the sole is `ANKLE_H` below it.

The ankle is what makes the IK below still closed form: the 4-joint chain
from hip to *ankle point* is solved for position (the old point-foot IK, in
the yawed frame), and the two ankle joints then only orient the sole. For a
level sole on a level trunk that is simply

    ankle_pitch = -(hip + knee),    ankle_roll = -hip_roll

because the pitch joints are parallel and so are the two roll axes once the
pitch chain has cancelled.

Because thigh and shin are the same length (L1 == L2), the hip-knee-foot
triangle is isosceles, so the hip-to-foot line exactly bisects the
thigh/shin angle, and the hip-to-foot distance depends on the knee alone:

    theta = hip + knee/2                         (leg angle inside the roll plane)
    l(q)  = sqrt(L1^2 + L2^2 + 2*L1*L2*cos(q))   (leg length, q = knee)

and the foot, relative to the hip, in the torso frame is

    d = ( l sin(theta),  l cos(theta) sin(roll),  -l cos(theta) cos(roll) )

where `d` is the hip-to-ankle vector in the hip-yawed frame. Those relations
make the inverse kinematics closed form: the foot's desired heading fixes the
hip yaw, leg length fixes the knee, `atan2(d_y, -d_z)` fixes the roll, the
in-plane angle `asin(d_x / l)` then fixes the hip, and the ankle levels the
sole.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mpc import (NU_FOOT, NX, ONE, PX, PZ, ROLL, YAW, VX, VZ, WX, WZ,
                  ConvexMPC, SRBDParams, moment_to_world, reference_trajectory, rot_z)

LEG_NAMES = ("l", "r")
SIDE_WORD = {"l": "left", "r": "right"}   # for MJCF names spelled out in full
# per-leg joint order: the column order of the 6x6 Jacobian, the torque
# vector, and the leg's block of data.ctrl
LEG_JOINTS = ("hip_yaw", "hip_roll", "hip", "knee", "ankle_pitch", "ankle_roll")
# per-arm joint order, the arm's block of data.ctrl after both legs'
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "elbow")
HIP_Y = 0.068                    # lateral hip offset in the torso frame (biped.xml)
ANKLE_H = 0.028                  # sole below the ankle point (biped.xml)


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def leg_length(q: float, L1: float, L2: float) -> float:
    """Hip-to-foot distance for knee angle q (q=0 is fully straight)."""
    return float(np.sqrt(L1**2 + L2**2 + 2 * L1 * L2 * np.cos(q)))


def knee_angle_for_length(l: float, L1: float, L2: float) -> float:
    """Inverse of leg_length: the (bent, i.e. negative) knee angle giving
    hip-to-foot distance l. Clips to the reachable range [|L1-L2|, L1+L2]."""
    l = float(np.clip(l, abs(L1 - L2) + 1e-6, L1 + L2 - 1e-6))
    c = (l**2 - L1**2 - L2**2) / (2 * L1 * L2)
    c = float(np.clip(c, -1.0, 1.0))
    return -float(np.arccos(c))


def leg_ik(d: np.ndarray, L1: float, L2: float) -> tuple[float, float, float]:
    """(hip_roll, hip, knee) placing the ankle at `d` from the hip, in the
    hip-yawed frame."""
    l = float(np.linalg.norm(d))
    knee = knee_angle_for_length(l, L1, L2)
    roll = float(np.arctan2(d[1], -d[2]))
    l_eff = leg_length(knee, L1, L2)          # the length actually reachable
    theta = float(np.arcsin(np.clip(d[0] / max(l_eff, 1e-6), -1.0, 1.0)))
    return roll, theta - knee / 2.0, knee


def level_sole_ankle(roll: float, hip: float, knee: float) -> tuple[float, float]:
    """(ankle_pitch, ankle_roll) that keep the sole parallel to the trunk's
    (yawed) x-y plane - i.e. level, for a level trunk."""
    return -(hip + knee), -roll


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    """MuJoCo (w, x, y, z) quaternion -> Z-Y-X Euler angles (roll, pitch, yaw)."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


@dataclass
class MPCGaitParams:
    """Gait scheduling, footstep planning and swing gains for `MPCWalkController`.

    Everything that shapes the *optimisation* lives in `SRBDParams` (weights,
    horizon, friction); what lives here is the part of the problem the MPC
    treats as exogenous - when each foot is down, and where it lands.
    """
    m: float = 1.31
    g: float = 10.0
    n_legs: int = 2
    L1: float = 0.17
    L2: float = 0.17
    # Gait clock. duty > 0.5 keeps a double-support window in every cycle,
    # which is where the attitude authority is (two feet can trade vertical
    # load; one foot can only lean on friction).
    #
    # step_freq is 4.0 Hz, not the design doc's 2.5 - measured in the planar
    # version over 25 s, mean speed vs. target and the 95th-percentile pitch:
    #
    #     v_des    2.5 Hz          3.5 Hz          4.0 Hz
    #     0.3      0.29 / 0.09     0.29 / 0.04     0.29 / 0.05
    #     0.5      0.49 / 0.22     0.50 / 0.08     0.50 / 0.06
    #     0.7      fell            0.71 / 0.16     0.69 / 0.12
    #     0.8      fell            0.80 / 0.26     0.79 / 0.14
    #
    # The reason is geometric: stride length is v*duty/step_freq, so at a
    # given speed a faster clock plants each foot closer to the CoM. The
    # moment `r x f` that the stance foot makes - the disturbance the MPC
    # spends its friction cone rejecting - shrinks in proportion. In 3-D it
    # matters twice over: in single support the stance foot is also HIP_Y off
    # to the side, and the lateral fall that starts over it has less time to
    # grow before the other foot catches it.
    step_freq: float = 4.0
    duty: float = 0.65
    # Running is the same clock with duty < 0.5: each foot's stance is
    # shorter than half a cycle, so between one foot's liftoff and the
    # other's touchdown both are in the air. Nothing else in the controller
    # changes - the MPC sees the flight phases in its contact schedule, and
    # plans the push-off that carries the body over them. At 0.35 the robot
    # is airborne 30% of the time (two 26 ms flights per 0.25 s cycle).
    #
    # Measured with the arms swinging (`arm_swing_gain`), ramping to each
    # speed over 4 s and holding to 20 s, mean speed / p95 pitch:
    #
    #     v_des    walk (0.65)     run (0.35)
    #     0.5      0.50 / 0.02     0.50 / 0.01
    #     0.8      0.80 / 0.03     -
    #     1.0      1.00 / 0.04     1.00 / 0.03
    #     1.3      1.31 / 0.06     1.30 / 0.05
    #     1.4      fell            1.40 / 0.07
    #     1.7      -               1.70 / 0.09
    #     2.0      -               fell
    #
    # Before the arms had joints the ceilings were 0.7 (walk: knee reach -
    # the stance foot is swept v*duty/f under the hip) and 1.25 (run: the
    # trunk rose into its legs until the stance leg straightened before
    # push-off ended). The auto gait below still runs from 0.7 up.
    run_duty: float = 0.35
    # Which duty the clock uses: "walk", "run", or "auto" - run whenever the
    # commanded speed is above `run_above`, and walk again once it is back
    # below `run_above - run_hysteresis`. The switch is not instantaneous:
    # the duty is slewed at `duty_rate` per second, so the gait passes
    # through ~2 cycles of intermediate duties. An instant change reassigns
    # a leg that is mid-stance to mid-swing, and its swing target jumps.
    gait: str = "auto"
    run_above: float = 0.7
    run_hysteresis: float = 0.1
    duty_rate: float = 0.5
    # Commanded motion, in the heading frame: forward and leftward speed, and
    # a turn rate (rad/s, positive = left).
    Vs: float = 0.5
    Vy: float = 0.0
    yaw_rate: float = 0.0
    # Desired CoM height: 0.36 m, i.e. 0.248 m of a 0.34 m leg - lower than
    # the planar version's 0.408 (0.30 m of leg), and that is what the 3-D
    # gait's top speed hangs on. The height trim below lifts the *aim* point
    # by 2-3 cm at speed, the CoM bobs another +/-2 cm in stride, and the
    # stance foot is swept further from the hip in 3-D than in the plane.
    # Stack those on 0.408 and the stance leg reaches full extension (0.337 m,
    # the knee's joint limit) at the ends of the sweep: the ground force then
    # goes through the joint limit rather than the controller, the leg turns
    # into a strut, and the robot pole-vaults off it. Measured, ramping to
    # each speed over 5 s and holding it to t = 20 s (time of the fall):
    #
    #     z_des    0.6 m/s   0.7 m/s   0.8 m/s
    #     0.408    7.4 s     6.2 s     6.1 s
    #     0.39     ok        8.9 s     6.9 s
    #     0.38     ok        17.6 s    5.3 s
    #     0.36     ok        ok        11.0 s
    #     0.35     ok        ok        8.8 s
    #
    # 0.36 and 0.35 are equally good up to 0.7 m/s; 0.36 keeps more knee
    # extension in hand. Nothing tried holds 0.8 m/s for long - see the
    # README. Low speeds are indifferent across the whole range.
    z_des: float = 0.36
    swing_clearance: float = 0.05
    # Raibert / capture-point footstep correction, applied in x *and* y. The
    # MPC has no say over where the feet go - it only optimises the forces
    # through them - so this is still what regulates speed, and laterally it
    # is also what keeps the side-to-side sway bounded.
    k_v: float = 0.12
    k_vy: float = 0.12
    speed_error_clip: float = 0.5
    # Nominal step width: each foot's reference point lands this far from
    # the CoM's line of travel. The hips are HIP_Y = 0.068 apart from it, but
    # a full hip-width step is a liability, not a margin: in single support
    # the CoM is an inverted pendulum over the stance foot, and every
    # centimetre of lateral offset is a centimetre it falls sideways before
    # the next foot comes down. (With point feet that was the whole story;
    # the flat foot's +/-2 cm of lateral CoP now absorbs some of it, but
    # not most.) Stepping slightly inside the hips trades a little hip roll
    # for a much smaller sway, and still leaves 5 cm between the two soles.
    step_width: float = 0.05
    # Integral trim on the regulated set-points.
    #
    # The MPC is a proportional controller on the SRBD, and the SRBD is wrong
    # in ways that do not average out: the legs carry 18% of the mass and
    # more than half the rotational inertia but are modelled as weightless,
    # the foot contact is compliant, and `tau = -J^T f` only realises the
    # requested force exactly for a massless leg on a rigid contact. The
    # residue is a *steady* force deficit, which a proportional law can only
    # answer with a steady error - in the planar version, 2 cm of height sag
    # and a forward speed stuck near half the target.
    #
    # Trimming the set-points the controller aims at (rather than adding a
    # force directly) keeps the QP's structure untouched and lets the same
    # integral act on *both* channels that regulate speed: the MPC's velocity
    # reference and the Raibert footstep offset. All trims are clamped, so a
    # target the robot simply cannot reach winds up to the clamp and stops.
    # The forward and lateral trims act in the heading frame.
    ki_v: float = 1.0
    ki_z: float = 0.6
    v_trim_max: float = 0.6
    # The height trim needs a wide range because it is compensating a *force*
    # deficit, not a kinematic one - the stance leg is nowhere near fully
    # extended when the robot sags, so nothing is stopping it rising except
    # the GRF the model fails to deliver.
    z_trim_max: float = 0.12
    # The same trim on the trunk's roll and pitch: the MPC's attitude
    # reference is offset against the measured attitude's integral. Its
    # steady error is the same kind of thing - a moment the SRBD does not
    # model - and it depends on speed, so no fixed offset can remove it.
    # Measured, the robot leaned back by a mean 0.057 rad marching in place,
    # 0.013 at 0.5 m/s, and forward by 0.012 at 0.7 m/s. (The point-foot
    # robot had it too, 0.026 marching.) At ki_att = 0.5 all three means are
    # below 0.001 rad, and the p95 pitch marching in place drops from 0.088
    # to 0.022; 0.5 to 2.0 all behave the same.
    ki_att: float = 0.5
    att_trim_max: float = 0.1
    # How far the heading the MPC is shown may lead the measured yaw. The
    # set-point integrates the commanded turn rate, so without a leash a
    # robot that falls behind in a turn - or is spun by a push - would be
    # asked to catch up the whole accumulated angle at once (`yaw_ref`).
    yaw_lead_max: float = 0.3
    # Swing leg: joint-space PD onto the IK targets, gains in LEG_JOINTS
    # order (hip_yaw, hip_roll, hip, knee, ankle_pitch, ankle_roll).
    #
    # The damping gains are set by an explicit-integration limit, but *not*
    # the `kd*dt/I < 2` diagonal rule - that rule is what made an earlier
    # version of this controller explode. The correct condition for a
    # multi-DOF leg is on the eigenvalues of `dt * M^-1 @ diag(kd)`, which
    # must all lie in (0, 2), and the leg's inverse mass matrix is strongly
    # *off*-diagonal:
    #
    #     M^-1[leg] = [[  685,    36,  -103],       (roll, hip, knee)
    #                  [   36,   790,  -562],       at the mpc_stand pose
    #                  [ -103,  -562,  1890]]
    #
    # The hip/knee cross term is comparable to the diagonals, so hip and knee
    # damping destabilise each other: the planar version shipped (3.0, 2.0)
    # on hip/knee, one eigenvalue reached 3.52 - a per-timestep amplification
    # of 2.5 - and the knee saturated within 8 ms of the first liftoff.
    #
    # With the hip yaw and the ankle the leg is six joints, and at the
    # mpc_stand pose its inverse mass matrix is
    #
    #     M^-1[leg] = [[1953, -344,  -56,  -26,   14,  186],   (yaw, roll, hip,
    #                  [-344,  674,   44,  -95,    5, -247],    knee, ankle
    #                  [ -56,   44,  765, -574, -178,    3],    pitch, ankle
    #                  [ -26,  -95, -574, 1683,  -39,    3],    roll)
    #                  [  14,    5, -178,  -39, 4457,   -1],
    #                  [ 186, -247,    3,    3,   -1, 4816]]
    #
    # The two ankle diagonals are the largest: the foot is light, and even
    # with the ankles' armature (biped.xml) a 1 kg-scale kd there would be
    # explosive. The hip yaw's 1953 is the leg's small inertia about its own
    # long axis. At the gains below the eigenvalues are (0.2, 0.2, 0.2, 0.3,
    # 0.5, 0.6): all modes decay. Ankle stiffness only has to hold a 20 g
    # foot level against the swing's accelerations; 5 N m/rad over the
    # ankle's ~2e-4 kg m^2 is a 150 rad/s loop, damping ratio 1.5.
    #
    # kp is nowhere near its own limit (`dt^2 * eig(M^-1 kp) < 4` allows
    # ~3000), so stiffness is free; only damping is scarce.
    swing_kp: tuple = (30.0, 60.0, 60.0, 60.0, 5.0, 5.0)
    swing_kd: tuple = (0.5, 1.0, 1.0, 0.5, 0.1, 0.1)
    # Stance-leg foot hold, layered *under* the MPC's -J^T f, and expressed in
    # **task space** (a virtual spring-damper on the foot), not joint space.
    #
    # The distinction is the whole point. A planted foot is stationary in the
    # world while the leg joints sweep at ~1 rad/s as the body travels over
    # it, so a joint-space `-kd*qdot` term is a brake on exactly the motion
    # that carries the robot forward - it fights the gait instead of holding
    # the foot. Damping the *foot's* world velocity instead costs nothing
    # while the foot is planted (v_foot = 0) and still catches a leg that is
    # swinging free.
    stance_hold_kp: float = 150.0
    stance_hold_kd: float = 3.0
    # ... and its rotational half, holding the sole level at its heading.
    # A foot that lands on its heel or toe edge - the swing leg is resolved
    # against the heading set-point, not the tilted trunk (`_ik_pd`) - is
    # rolled flat by this; a flat foot sees zero error and zero angular
    # velocity, so again it costs the MPC nothing. The damping is bounded by
    # the same explicit-integration limit as the swing ankle's.
    # Late touchdown. The gait clock is open loop: stance begins on schedule
    # whether or not the foot has reached the ground. Walking, the foot is
    # never more than a few millimetres short, but after a running flight
    # phase it can be centimetres up, and a foot that is handed its MPC
    # wrench there has nothing to push on - `-J^T f` just accelerates the
    # unloaded leg to full extension, the knee runs into its stop, and the
    # robot lands on a strut and pole-vaults off it. So a scheduled-stance
    # foot that has not yet touched keeps its swing controller, reaching on
    # down through the planned footstep at this speed (m/s), to at most this
    # depth below the floor.
    late_touchdown_speed: float = 0.5
    late_touchdown_depth: float = 0.05
    stance_hold_kr: float = 2.0
    stance_hold_kdr: float = 0.05
    # Arm swing. Each arm swings against its own leg, i.e. with the
    # *opposite* leg, the way a person's does: the shoulder-pitch target is
    # `arm_swing_gain` times half the difference of the two legs' angles
    # (theta = hip + knee/2, the hip-to-ankle line relative to the trunk),
    # measured, and low-passed over `arm_filter_tau` so stance-leg jitter
    # does not reach the arms. Because it follows the legs' real angles, the
    # swing grows with stride length and vanishes marching in place, with no
    # speed schedule of its own.
    #
    # The MPC does not model the arms - they are two small disturbances on
    # the trunk, like the legs - but antiphase arms are the *helpful* kind:
    # their pitch reactions cancel each other, and their yaw reaction
    # opposes the one the swinging legs put on the trunk.
    #
    # The elbow bends from `arm_elbow_walk` to `arm_elbow_run` as the duty
    # moves from walking to running, as a runner's does; the shoulder roll
    # holds the hands clear of the thighs.
    #
    # Measured at 1.5 the arms swing +/-10 deg walking at 0.5 m/s and
    # +/-12 deg running at 1.0. Against arms held still (gain 0), the yaw
    # wobble walking at 0.5 m/s drops from 0.038 to 0.028 rad (p95), and the
    # top speeds rise from 0.9 to 1.3 m/s walking and 1.6 to 1.7 running.
    # At 3.0 the arms over-cancel the legs: yaw wobble 0.047 rad.
    arm_swing_gain: float = 1.5
    arm_swing_max: float = 0.6
    arm_filter_tau: float = 0.02
    arm_roll: float = 0.12
    arm_elbow_walk: float = 0.5
    arm_elbow_run: float = 1.3
    # PD gains in ARM_JOINTS order. An arm is 0.05 kg with ~5e-4 kg m^2
    # about the shoulder, plus the joints' 2e-4 armature (biped.xml): the
    # shoulder loop is ~80 rad/s at damping ratio ~0.5, and dt*kd/I stays
    # below 0.1, far inside the explicit-integration limit that shapes
    # `swing_kd`.
    arm_kp: tuple = (4.0, 4.0, 2.0)
    arm_kd: tuple = (0.05, 0.05, 0.02)
    mpc: SRBDParams = field(default_factory=SRBDParams)



class MPCWalkController:
    """Convex-MPC walker: contact wrenches from a QP, joint torques from `-J^T w`.

    Structure (doc section 4)::

        (v_des, yaw_rate) -> gait clock (walk/run duty) + footstep plan -> contact schedule c_i(k), r_i(k), psi_i(k)
                                                              |
        CoM + attitude state ------------------------> convex MPC (QP)
                                                              |  w_i = (f_i, m_i) (first step)
                     stance leg: tau = -J_i^T w_i   <---------+---> swing leg: IK + PD

    There is deliberately **no attitude PD anywhere**. Roll, pitch and yaw are
    regulated by the MPC, by choosing how to split the load between the two
    feet, how far to lean each GRF inside the friction cone, and where to put
    each foot's centre of pressure - authority a hip PD cannot express (see
    `mpc.py`).

    The controller owns `model`/`data` because it needs MuJoCo's own foot
    Jacobians and body frames; the sign convention it relies on is derived in
    `mpc.py` and checked end-to-end by validation stage 2.
    """

    # MJCF names the controller binds to. Another robot (`g1_controller.py`)
    # overrides these; everything else here only needs a floating base, two
    # six-joint legs and a sole site on each foot.
    ROOT_JOINT = "root"
    BASE_BODY = "torso"          # the free joint's body
    FOOT_BODY = "foot_{s}"
    SOLE_SITE = "sole_{s}"       # the foot's reference point on the sole
    LEG_JOINT = "{j}_{s}"
    LEG_JOINTS = LEG_JOINTS
    ARM_JOINTS = ARM_JOINTS

    def __init__(self, model, data, params: MPCGaitParams | None = None):
        import mujoco  # local: only the MPC gait needs MuJoCo inside the controller

        self._mj = mujoco
        self.model = model
        self.data = data
        self.p = params or MPCGaitParams()
        p = self.p
        # Keep the QP's model constants consistent with the gait's.
        p.mpc.mass, p.mpc.g, p.mpc.n_feet = p.m, p.g, p.n_legs
        self.mpc = ConvexMPC(p.mpc)

        names = LEG_NAMES[: p.n_legs]
        self._side = np.array([1.0, -1.0])[: p.n_legs]   # +y for left, -y for right
        self._foot_bid = [model.body(self.FOOT_BODY.format(s=s, side=SIDE_WORD[s])).id for s in names]
        # the foot's reference point: the ankle projected onto the sole
        self._sole_sid = [model.site(self.SOLE_SITE.format(s=s, side=SIDE_WORD[s])).id for s in names]
        self._torso_bid = model.body(self.BASE_BODY).id
        self._floor_gid = model.geom("floor").id
        root = model.joint(self.ROOT_JOINT).id
        self._root_qadr = model.jnt_qposadr[root]
        self._root_vadr = model.jnt_dofadr[root]
        # the six DOFs per leg, in the LEG_JOINTS column order the 6x6
        # Jacobian and the torque vector both use
        jids = [[model.joint(self.LEG_JOINT.format(j=j, s=s, side=SIDE_WORD[s])).id for j in self.LEG_JOINTS]
                for s in names]
        self._leg_dofs = [model.jnt_dofadr[j] for j in jids]
        self._leg_qadr = [model.jnt_qposadr[j] for j in jids]
        # joint limits for clipping IK targets; +/-inf on unlimited joints
        self._q_range = []
        for j in jids:
            rng = model.jnt_range[j].copy()
            rng[~model.jnt_limited[j].astype(bool)] = (-np.inf, np.inf)
            self._q_range.append(rng)
        self._ctrl_range = model.actuator_ctrlrange.copy()
        ajids = [[model.joint(self.LEG_JOINT.format(j=j, s=s, side=SIDE_WORD[s])).id for j in self.ARM_JOINTS]
                 for s in names]
        self._arm_dofs = [model.jnt_dofadr[j] for j in ajids]
        self._arm_qadr = [model.jnt_qposadr[j] for j in ajids]
        self._arm_swing = np.zeros(p.n_legs)   # filtered shoulder-pitch targets
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        self._f = np.zeros((p.n_legs, NU_FOOT))  # held wrenches between QP solves
        self._next_solve = -np.inf
        self._was_stance = [True] * p.n_legs
        self._lift = [np.zeros(2)] * p.n_legs  # foot xy at the last liftoff
        self._lift_yaw = [0.0] * p.n_legs      # ... and its heading
        # where the current / last swing was aimed, and whether the foot has
        # actually come down since its stance was scheduled (late touchdown)
        self._td_plan = [data.site_xpos[s][:2].copy() for s in self._sole_sid]
        self._td_yaw = [0.0] * p.n_legs
        self._landed = [True] * p.n_legs
        self._stance_t0 = [0.0] * p.n_legs
        self._run_mode = False                 # auto gait's hysteresis state
        self.duty = self._duty_target()        # the clock's current duty
        self._v_trim = np.zeros(2)             # integral trims (see MPCGaitParams)
        self._z_trim = 0.0
        self._att_trim = np.zeros(2)           # (roll, pitch) reference offset
        self._last_t = 0.0
        self._yaw_prev = None                  # for unwrapping yaw
        self._yaw_turns = 0.0
        self._yaw_des = None                   # heading set-point
        self._yaw_meas = 0.0
        self.last_status = "not-solved"

    # --- trimmed set-points -------------------------------------------------

    @property
    def v_cmd(self) -> np.ndarray:
        """Commanded (forward, left) velocity in the heading frame."""
        return np.array([self.p.Vs, self.p.Vy])

    @property
    def v_des(self) -> np.ndarray:
        """Heading-frame velocity target the controller actually aims at."""
        return self.v_cmd + self._v_trim

    @property
    def z_des(self) -> float:
        return self.p.z_des + self._z_trim

    def _update_setpoints(self, dt: float, v_head: np.ndarray, com_z: float,
                          rpy: np.ndarray) -> None:
        p = self.p
        yaw = float(rpy[2])
        self._att_trim = np.clip(self._att_trim - p.ki_att * rpy[:2] * dt,
                                 -p.att_trim_max, p.att_trim_max)
        self._v_trim = np.clip(self._v_trim + p.ki_v * (self.v_cmd - v_head) * dt,
                               -p.v_trim_max, p.v_trim_max)
        self._z_trim = float(np.clip(
            self._z_trim + p.ki_z * (p.z_des - com_z) * dt, -p.z_trim_max, p.z_trim_max))
        if self._yaw_des is None:
            self._yaw_des = yaw
        self._yaw_des += p.yaw_rate * dt

    @property
    def yaw_ref(self) -> float:
        """The heading the MPC is asked to hold: the set-point, leashed to
        within `yaw_lead_max` of the measured yaw.

        The leash goes on what the MPC *sees*, not on the set-point itself.
        Clamping the set-point instead makes it follow the robot whenever the
        error exceeds the leash, and a heading lost that way is never
        recovered - measured, a 0.6 m/s walk wandered 1.4 rad off course
        before it fell.
        """
        yaw = self._yaw_meas
        return float(np.clip(self._yaw_des, yaw - self.p.yaw_lead_max,
                             yaw + self.p.yaw_lead_max))

    # --- gait clock ---------------------------------------------------------

    def _duty_target(self) -> float:
        """The duty the gait is heading for (see `MPCGaitParams.gait`)."""
        p = self.p
        if p.duty >= 1.0 or p.gait == "walk":          # standing, or forced
            return p.duty
        if p.gait == "run":
            return p.run_duty
        speed = float(np.hypot(p.Vs, p.Vy))
        thresh = p.run_above - (p.run_hysteresis if self._run_mode else 0.0)
        self._run_mode = speed > thresh
        return p.run_duty if self._run_mode else p.duty

    @property
    def running(self) -> bool:
        """Is the clock's duty below 1/2, i.e. does the gait have flight?"""
        return self.duty < 0.5

    @property
    def stance_duration(self) -> float:
        return self.duty / self.p.step_freq

    def _phase(self, t: float, leg: int) -> float:
        """Where leg `leg` is in its own cycle at time `t` (0 = touchdown)."""
        return (t * self.p.step_freq + leg / self.p.n_legs) % 1.0

    def _scheduled(self, phase: float) -> bool:
        return phase < self.duty

    def _plan_yaw(self, touchdown_t: float, t_now: float) -> float:
        """Heading for a foot touching down at `touchdown_t`.

        The heading *set-point* carried forward, not the measured yaw: the
        trunk counter-rotates by ~0.1 rad every step (see `SRBDParams.Q`),
        and a foot planted at whatever angle the trunk happened to have would
        walk a zigzag of splayed feet. The hip-yaw joint takes up the
        difference.
        """
        return self.yaw_ref + self.p.yaw_rate * (touchdown_t - t_now)

    def _plan_step(self, leg: int, touchdown_t: float, t_now: float,
                   com: np.ndarray, v: np.ndarray, yaw: float) -> np.ndarray:
        """Raibert / capture-point footstep (world xy) for a touchdown at `touchdown_t`.

            p_foot = p_hip(at touchdown) + v*T_stance/2 + K (v - v_des)

        The first term centres the stance sweep about the hip (so a steady
        stride is force-neutral); the last is the feedback that actually
        regulates velocity - land further forward than neutral and the stance
        brakes, further back and it accelerates; the same sideways. Both the
        hip and the feedback are resolved in the heading the robot will have
        at touchdown, so a turning robot places its feet around the arc.
        """
        p = self.p
        dt = touchdown_t - t_now
        yaw_td = yaw + p.yaw_rate * dt
        R = rot_z(yaw_td)[:2, :2]
        v_des_w = R @ self.v_des
        err = np.clip(v - v_des_w, -p.speed_error_clip, p.speed_error_clip)
        err_h = R.T @ err                               # heading frame
        hip = com[:2] + v * dt + R @ np.array([0.0, self._side[leg] * p.step_width])
        fb = R @ np.array([p.k_v * err_h[0], p.k_vy * err_h[1]])
        return hip + v * self.stance_duration / 2.0 + fb

    def _horizon_plan(self, t: float, com: np.ndarray, v: np.ndarray, yaw: float):
        """Contact schedule `c_i(k)`, moment arms `r_i(k)` and foot headings
        `psi_i(k)` over the horizon.

        `r_i(k) = p_foot,i - p_com(k)`, from the CoM to the foot's reference
        point on the sole, where a foot already planted keeps its *measured*
        position and heading and a foot that touches down inside the horizon
        uses its planned footstep. The predicted CoM uses the **measured**
        velocity rather than `v_des`: the body cannot change speed much within
        0.4 s, so this keeps the moment arms consistent with the same
        prediction that places the feet. (The doc writes `v_des` here; with a
        0.5 m/s tracking error that would misplace every moment arm in the
        horizon by up to 0.2 m, which is larger than the moment arms
        themselves.)
        """
        p = self.p
        N, dt = p.mpc.horizon, p.mpc.dt
        contact = np.zeros((N, p.n_legs))
        r = np.zeros((N, p.n_legs, 3))
        psi = np.zeros((N, p.n_legs))

        for i in range(p.n_legs):
            plant = plant_yaw = None
            if self._scheduled(self._phase(t, i)):
                if self._landed[i]:
                    plant = self.data.site_xpos[self._sole_sid[i]][:2].copy()
                    plant_yaw = self._foot_yaw(i)
                else:
                    plant, plant_yaw = self._td_plan[i], self._td_yaw[i]
            for k in range(N):
                tk = t + k * dt
                ph = self._phase(tk, i)
                com_k = com[:2] + v[:2] * (k * dt)
                if self._scheduled(ph):
                    if plant is None:
                        # this stance interval starts inside the horizon:
                        # its touchdown was when the phase last crossed 0
                        td = max(tk - ph / p.step_freq, t)
                        plant = self._plan_step(i, td, t, com, v[:2], yaw)
                        plant_yaw = self._plan_yaw(td, t)
                    # a foot that is late for its touchdown carries nothing
                    # *now*; the QP is re-solved the moment it lands
                    contact[k, i] = 0.0 if (k == 0 and not self._landed[i]) else 1.0
                    r[k, i] = (*(plant - com_k), -com[2])   # the sole is on the floor
                    psi[k, i] = plant_yaw
                else:
                    # the foot lifts; the next stance gets a freshly planned
                    # footstep rather than inheriting this one
                    plant = plant_yaw = None
                    r[k, i] = (0.0, 0.0, -com[2])   # unused: the wrench is clamped to 0
        return contact, r, psi

    # --- MuJoCo state helpers ----------------------------------------------

    def _in_contact(self, leg: int) -> bool:
        """Any of the foot's four sole contacts touching the floor."""
        bid, gbody = self._foot_bid[leg], self.model.geom_bodyid
        for c in self.data.contact[: self.data.ncon]:
            if c.geom1 == self._floor_gid and gbody[c.geom2] == bid:
                return True
            if c.geom2 == self._floor_gid and gbody[c.geom1] == bid:
                return True
        return False

    def _foot_yaw(self, leg: int) -> float:
        """Heading of the sole (its x axis projected on the floor), unwrapped
        to lie within pi of the trunk's unwrapped yaw."""
        R = self.data.site_xmat[self._sole_sid[leg]].reshape(3, 3)
        a = float(np.arctan2(R[1, 0], R[0, 0]))
        return self._yaw_meas + wrap_angle(a - self._yaw_meas)

    def base_state(self) -> tuple[np.ndarray, np.ndarray]:
        """(rpy, world angular velocity) of the torso, with yaw unwrapped.

        The free joint's angular velocity is in the *body* frame; the SRBD is
        written in world axes, so it is rotated out here. Yaw is unwrapped so
        that a robot turning in circles does not see a 2 pi jump in its
        heading error.
        """
        d = self.data
        q = d.qpos[self._root_qadr + 3:self._root_qadr + 7]
        rpy = quat_to_rpy(q)
        if self._yaw_prev is not None:
            jump = rpy[2] - self._yaw_prev
            self._yaw_turns -= 2 * np.pi * np.round(jump / (2 * np.pi))
        self._yaw_prev = rpy[2]
        rpy[2] += self._yaw_turns
        R = d.xmat[self._torso_bid].reshape(3, 3)
        w_body = d.qvel[self._root_vadr + 3:self._root_vadr + 6]
        return rpy, R @ w_body

    def _foot_jacobian(self, leg: int) -> np.ndarray:
        """6x6 d(sole point, foot rotation)/d(leg joints), base held fixed.

        Rows are the translational then the rotational Jacobian of the foot
        at its reference point; columns the six leg DOFs in LEG_JOINTS order.
        `mj_jac` gives the full 6 x nv Jacobian; taking only the leg columns
        is exactly the "base fixed" Jacobian of doc section 8 (the base DOFs
        move the foot too, but no motor drives them).

        With six joints the matrix is square, so `-J^T w` can realise any
        contact wrench - force *and* moment - at the foot. The point matters
        for the moment: a wrench is (f, m) *about a point*, and this must be
        the same point the MPC's moment arms `r_i` and contact moments
        `m_i` are taken about. (With the old point foot, the same lesson
        cost 11% of the horizontal force when the Jacobian was taken at the
        foot sphere's centre instead of its contact point.)
        """
        point = self.data.site_xpos[self._sole_sid[leg]]
        self._mj.mj_jac(self.model, self.data, self._jacp, self._jacr, point,
                        self._foot_bid[leg])
        dofs = self._leg_dofs[leg]
        return np.vstack((self._jacp[:, dofs], self._jacr[:, dofs]))

    def _foot_hold(self, leg: int, plan: np.ndarray, plan_yaw: float) -> np.ndarray:
        """Task-space spring-damper holding a stance foot flat on its plant point.

        Returns joint torques for `J^T [kp (p_plant - p_foot) - kd v_foot;
        kr e_R - kdr w_foot]`, where `e_R` is the rotation taking the sole to
        level at its target heading. While the foot is genuinely planted
        every term vanishes - the target is the foot's own position and
        heading, it is flat, and its world velocity is zero - so the MPC
        keeps full authority over the leg. It only bites when the foot is off
        its plant point or off the floor, i.e. when a stance foot has landed
        on an edge or been knocked loose, where it reaches for the footstep
        it was aimed at and flattens the sole. (A foot that has not landed
        yet is not held here at all; it keeps swinging - see
        `late_touchdown_speed`.)
        """
        p, d = self.p, self.data
        J = self._foot_jacobian(leg)                  # 6x6, leg columns only
        v_foot = self._jacp @ d.qvel                  # full Jacobian: world velocity
        w_foot = self._jacr @ d.qvel
        pos = d.site_xpos[self._sole_sid[leg]]
        planted = self._in_contact(leg)
        xy = pos[:2] if planted else plan
        err = np.array([xy[0], xy[1], 0.0]) - pos
        R = d.site_xmat[self._sole_sid[leg]].reshape(3, 3)
        E = rot_z(self._foot_yaw(leg) if planted else plan_yaw) @ R.T
        e_rot = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
        wrench = np.concatenate((p.stance_hold_kp * err - p.stance_hold_kd * v_foot,
                                 p.stance_hold_kr * e_rot - p.stance_hold_kdr * w_foot))
        return J.T @ wrench

    def _ik_pd(self, leg: int, target: np.ndarray, foot_yaw: float) -> np.ndarray:
        """Joint-space PD onto the 6-joint IK solution for a world sole target:
        the reference point at `target`, the sole level at heading `foot_yaw`.

        The target is a world-frame point, but it is resolved into joint
        angles in the **reference heading** frame - the heading set-point, not
        the torso's measured attitude - from a hip position built the same
        way. The joint targets therefore do not depend on the trunk's measured
        roll, pitch *or* yaw at all: the leg is commanded as if the trunk were
        level and pointing where it should.

        That is not sloppiness, it is the only thing keeping this loop
        stable. The mass matrix couples each hip joint to a trunk rotation -
        `M^-1[pitch, hip] = 156`, `M^-1[roll, hip_roll] = -156`,
        `M^-1[yaw, hip] = 109` - so a hip torque that grows with a trunk
        angle is **positive** feedback on that angle whenever the sign works
        out, and for a world-frame target it always does:

        - roll/pitch: holding a world-frame leg direction subtracts the trunk
          angle from the joint target. Growth rate `sqrt(156*kp)` ~ 100 rad/s
          at kp = 60; in the planar version, the moment the first foot left
          the ground the hip torque alternated -192, +141, -111, +102 N m and
          the torso was on its back within half a second.
        - yaw: a trunk yawed by `psi` carries the hip pivot `HIP_Y*psi`
          backwards on one side and forwards on the other, so a world-fixed
          target moves *forward* relative to the hip on the side that swung
          back. The hip swings that leg forward, and the reaction yaws the
          trunk further the same way - measured, a 0.5 m/s walk drifted 1 rad
          off heading in 4 s and then fell.

        (The stance leg's `-J^T f` has no such path: it is a real ground
        reaction, and its moment is exactly the one the MPC asked for.)

        The price is that a tilted or mis-headed torso lands the foot off the
        plan by `l*sin(angle)` - about 4 cm at 0.15 rad, and the MPC's own job
        is to keep the trunk far below that. The same holds for the sole: it
        is levelled relative to the *reference* trunk, so a trunk pitched by
        0.07 rad lands the foot 0.07 rad onto its heel or toe, and the
        stance-side foot hold (`_foot_hold`) rolls it flat after touchdown.
        The ankle is light enough that doing it here instead, from the
        measured shin, would be harmless - but it would also be the one joint
        in the swing leg whose target moved with the trunk.

        The hip yaw is the foot's heading relative to the heading set-point,
        so it too is free of measured attitude.
        """
        p, d = self.p, self.data
        lo, hi = self._q_range[leg].T
        R = rot_z(self.yaw_ref)
        pelvis = d.xpos[self._torso_bid]
        hip_pos = pelvis + R @ np.array([0.0, self._side[leg] * HIP_Y, 0.0])
        ankle = np.asarray(target, dtype=float) + np.array([0.0, 0.0, ANKLE_H])
        yaw_des = float(np.clip(wrap_angle(foot_yaw - self.yaw_ref), lo[0], hi[0]))
        dvec = rot_z(yaw_des).T @ (R.T @ (ankle - hip_pos))
        roll_des, hip_des, knee_des = leg_ik(dvec, p.L1, p.L2)
        knee_des = float(np.clip(knee_des, lo[3], hi[3]))
        roll_des = float(np.clip(roll_des, lo[1], hi[1]))
        q_des = np.array([yaw_des, roll_des, hip_des, knee_des,
                          *level_sole_ankle(roll_des, hip_des, knee_des)])
        q_des = np.clip(q_des, lo, hi)
        q = d.qpos[self._leg_qadr[leg]]
        dq = d.qvel[self._leg_dofs[leg]]
        return np.asarray(p.swing_kp) * (q_des - q) - np.asarray(p.swing_kd) * dq

    def _stance_feedforward(self, leg: int) -> np.ndarray:
        """Joint torques a stance leg adds on top of `-J^T w`: here the
        joints' own viscous damping, cancelled (see `control`)."""
        dofs = self._leg_dofs[leg]
        return self.model.dof_damping[dofs] * self.data.qvel[dofs]

    def leg_lengths(self) -> list:
        """Hip-to-ankle distance of each leg, for logging."""
        return [leg_length(self.data.qpos[a[3]], self.p.L1, self.p.L2)
                for a in self._leg_qadr]

    def _arm_torques(self, dt: float) -> np.ndarray:
        """PD torques for both arms, in data.ctrl order (ARM_JOINTS, l then r).

        Each arm's swing target is set by the legs' measured angles
        (`arm_swing_gain`); the elbow bend by the gait's duty.
        """
        p, d = self.p, self.data
        theta = np.array([d.qpos[a[2]] + 0.5 * d.qpos[a[3]] for a in self._leg_qadr])
        # left arm forward when the right leg is: +(theta_r - theta_l)/2
        target = np.clip(p.arm_swing_gain * 0.5 * (theta[::-1] - theta),
                         -p.arm_swing_max, p.arm_swing_max)
        a = dt / (p.arm_filter_tau + dt) if dt > 0 else 0.0
        self._arm_swing += a * (target - self._arm_swing)
        run = float(np.clip((p.duty - self.duty) / (p.duty - p.run_duty), 0.0, 1.0))
        elbow = p.arm_elbow_walk + run * (p.arm_elbow_run - p.arm_elbow_walk)
        out = []
        for i in range(p.n_legs):
            q_des = np.array([self._arm_swing[i], p.arm_roll, elbow])
            q = d.qpos[self._arm_qadr[i]]
            dq = d.qvel[self._arm_dofs[i]]
            out += list(np.asarray(p.arm_kp) * (q_des - q) - np.asarray(p.arm_kd) * dq)
        return np.asarray(out)

    # --- main entry point ---------------------------------------------------

    def control(self, t: float, com: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Torques in `data.ctrl` order: LEG_JOINTS for the left leg, then
        for the right, then ARM_JOINTS for the left and right arms.

        `com` and `vel` are the whole robot's CoM position and velocity, 3-D.
        """
        p, d = self.p, self.data
        com, vel = np.asarray(com, dtype=float), np.asarray(vel, dtype=float)
        rpy, omega = self.base_state()
        yaw = self._yaw_meas = float(rpy[2])

        dt_ctrl = max(t - self._last_t, 0.0)
        target = self._duty_target()
        step = p.duty_rate * dt_ctrl
        self.duty = float(np.clip(target, self.duty - step, self.duty + step))

        phases = [self._phase(t, i) for i in range(p.n_legs)]
        sched = [self._scheduled(ph) for ph in phases]

        # Remember where each foot left the ground: the swing trajectory is
        # drawn from the *actual* liftoff point to the planned touchdown, so a
        # stumble doesn't make the foot teleport back to a nominal stride.
        for i in range(p.n_legs):
            if self._was_stance[i] and not sched[i]:
                self._lift[i] = d.site_xpos[self._sole_sid[i]][:2].copy()
                self._lift_yaw[i] = self._foot_yaw(i)
            if sched[i] and not self._was_stance[i]:
                self._landed[i] = False
                self._stance_t0[i] = t
            # Touchdown is the *measured* contact, latched for the rest of
            # the stance (see `late_touchdown_speed`). The QP is re-solved
            # at once: the wrench it was holding assumed this foot carried
            # nothing.
            if sched[i] and not self._landed[i] and self._in_contact(i):
                self._landed[i] = True
                self._next_solve = t
        self._was_stance = list(sched)

        # Rolling horizon: re-solve on the MPC clock (every mpc.dt = 40 sim
        # steps) and zero-order-hold the GRFs in between.
        v_head = rot_z(yaw)[:2, :2].T @ vel[:2]
        self._update_setpoints(max(t - self._last_t, 0.0), v_head, float(com[2]), rpy)
        self._last_t = t
        if t >= self._next_solve:
            contact, r, psi = self._horizon_plan(t, com, vel, yaw)
            x0 = np.zeros(NX)
            x0[ROLL:YAW + 1] = rpy
            x0[PX:PZ + 1] = com
            x0[WX:WZ + 1] = omega
            x0[VX:VZ + 1] = vel
            x0[ONE] = 1.0
            x_ref = reference_trajectory(x0, self.v_des, self.z_des, p.mpc,
                                         yaw_des=self.yaw_ref, yaw_rate=p.yaw_rate)
            x_ref[:, ROLL:ROLL + 2] = self._att_trim
            self._f = self.mpc.solve(x0, contact, r, x_ref, psi)
            self.last_status = self.mpc.status
            self._next_solve = t + p.mpc.dt

        out = []
        for i in range(p.n_legs):
            td = t + (1.0 - phases[i]) / p.step_freq      # next touchdown time
            plan = self._plan_step(i, td, t, com, vel[:2], yaw)
            plan_yaw = self._plan_yaw(td, t)

            if sched[i] and not self._landed[i]:
                # Scheduled stance, but the foot is still in the air: keep
                # reaching down through the planned footstep until it lands.
                z = -min(p.late_touchdown_speed * (t - self._stance_t0[i]),
                         p.late_touchdown_depth)
                tau = self._ik_pd(i, np.array([*self._td_plan[i], z]), self._td_yaw[i])
            elif sched[i]:
                # Virtual work: the leg pushes on the *ground* with -w, so
                # tau = J^T (-w) = -J^T w. Sign derived in doc section 8 and
                # checked end-to-end by validation stage 2. The QP's moment
                # is in the foot's yaw frame; J's rows are world axes.
                f, m = self._f[i, :3], self._f[i, 3:]
                w = np.concatenate((f, moment_to_world(m, self._foot_yaw(i))))
                tau = -self._foot_jacobian(i).T @ w
                # ... plus a soft hold on the foot's plant point.
                #
                # Gating this branch on *measured* contact instead, and
                # reaching stiffly for the ground when airborne, chatters at
                # the timestep: a foot the MPC has assigned no load goes
                # limp (tau = -J^T 0 = 0), drops out of contact, gets yanked
                # back by the stiff reach, touches, goes limp again. The
                # joint velocities alternate sign every step (+65, -66, +65
                # rad/s), the torques saturate, and the robot is thrown. The
                # MPC's own force does not depend on measured contact, so
                # keeping the whole branch on the schedule alone - plus a
                # soft posture term that is continuous across touchdown -
                # removes the discontinuity the chatter was feeding on.
                tau = tau + self._foot_hold(i, self._td_plan[i], self._td_yaw[i])
                # ... plus the joints' own viscous damping, cancelled.
                #
                # biped.xml gives every leg joint a little damping (0.2 at
                # the hips, 0.05 at the knee) - friction in a real gearbox.
                # It is invisible to the SRBD, and on a stance leg it is not
                # small: the joints sweep at 1-3 rad/s as the body vaults
                # over the foot, so up to ~0.6 N m of every hip torque goes
                # into the damper instead of the ground - several newtons at
                # the foot, against a 13 N body weight. Uncancelled, the
                # robot sagged (the height trim sat at +4.6 cm), landed late
                # (16% of scheduled stance time with the foot still in the
                # air at 0.5 m/s) and bounced (13% of the time with *both*
                # feet off the ground). Feeding the damping torque forward
                # took those to 4% and 2%, halved the roll excursion, and
                # removed a 0.08 rad steady pitch offset outright.
                tau = tau + self._stance_feedforward(i)
            else:
                s = (phases[i] - self.duty) / (1.0 - self.duty)
                xy = self._lift[i] + s * (plan - self._lift[i])
                z = p.swing_clearance * np.sin(np.pi * s)
                psi = self._lift_yaw[i] + s * wrap_angle(plan_yaw - self._lift_yaw[i])
                self._td_plan[i], self._td_yaw[i] = plan, plan_yaw
                tau = self._ik_pd(i, np.array([xy[0], xy[1], z]), psi)
            out += list(tau)

        tau = np.concatenate((out, self._arm_torques(dt_ctrl)))
        return np.clip(tau, self._ctrl_range[:, 0], self._ctrl_range[:, 1])
