"""Convex-MPC controller for the 3-D two-legged (humanoid) robot.

MuJoCo owns the dynamics. Each leg is a real three-joint chain - a 2-DOF hip
(roll + pitch) and a knee - driven by plain torque motors, with no passive
joint springs anywhere, the same way the MIT Cheetah / ANYmal / Unitree
quadrupeds are built. Every bit of leg compliance the gait shows is produced
actively by this file.

The controller itself is `MPCWalkController`: a convex model-predictive
controller on a single-rigid-body model (`mpc.py`) that optimises the 3-D
ground reaction forces, mapped to joint torques through `-J^T f`. Design and
as-built notes: `doc/convec_mpc.md`.

Kinematics
----------
Joint angles are measured relative to the parent link: `hip_roll` rotates the
leg about the torso's x axis, `hip` is then the thigh's pitch inside that
rolled plane, and `knee` the shin angle relative to the thigh (knee=0 is a
fully straight leg). The hip-roll and hip-pitch axes intersect at the hip
pivot.

Because thigh and shin are the same length (L1 == L2), the hip-knee-foot
triangle is isosceles, so the hip-to-foot line exactly bisects the
thigh/shin angle, and the hip-to-foot distance depends on the knee alone:

    theta = hip + knee/2                         (leg angle inside the roll plane)
    l(q)  = sqrt(L1^2 + L2^2 + 2*L1*L2*cos(q))   (leg length, q = knee)

and the foot, relative to the hip, in the torso frame is

    d = ( l sin(theta),  l cos(theta) sin(roll),  -l cos(theta) cos(roll) )

Those relations make the 3-joint inverse kinematics closed form: leg length
fixes the knee, `atan2(d_y, -d_z)` fixes the roll, and the in-plane angle
`asin(d_x / l)` then fixes the hip.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mpc import (NX, ONE, PX, PZ, ROLL, YAW, VX, VZ, WX, WZ,
                  ConvexMPC, SRBDParams, reference_trajectory, rot_z)

LEG_NAMES = ("l", "r")
HIP_Y = 0.068                    # lateral hip offset in the torso frame (biped.xml)


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
    """(hip_roll, hip, knee) placing the foot at `d` from the hip, torso frame."""
    l = float(np.linalg.norm(d))
    knee = knee_angle_for_length(l, L1, L2)
    roll = float(np.arctan2(d[1], -d[2]))
    l_eff = leg_length(knee, L1, L2)          # the length actually reachable
    theta = float(np.arcsin(np.clip(d[0] / max(l_eff, 1e-6), -1.0, 1.0)))
    return roll, theta - knee / 2.0, knee


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
    m: float = 1.30
    g: float = 10.0
    n_legs: int = 2
    L1: float = 0.17
    L2: float = 0.17
    foot_radius: float = 0.028   # foot sphere radius: the foot centre's height in stance
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
    # Nominal step width: each foot lands this far from the CoM's line of
    # travel. The hips are HIP_Y = 0.068 apart from it, but with point feet a
    # full hip-width step is a liability, not a margin: in single support the
    # CoM is an inverted pendulum over the stance foot, and every centimetre
    # of lateral offset is a centimetre it falls sideways before the next
    # foot comes down. Stepping slightly inside the hips trades a little hip
    # roll for a much smaller sway.
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
    # How far the heading the MPC is shown may lead the measured yaw. The
    # set-point integrates the commanded turn rate, so without a leash a
    # robot that falls behind in a turn - or is spun by a push - would be
    # asked to catch up the whole accumulated angle at once (`yaw_ref`).
    yaw_lead_max: float = 0.3
    # Swing leg: joint-space PD onto the IK targets, gains in
    # (hip_roll, hip, knee) order.
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
    # of 2.5 - and the knee saturated within 8 ms of the first liftoff. At
    # (1.0, 1.0, 0.5) the eigenvalues here are (0.64, 0.34, 0.23): all modes
    # decay. (At (3, 3, 2) the largest is 2.33 - unstable.) Roll is nearly
    # decoupled from the other two, so it can take the same kd as the hip.
    #
    # kp is nowhere near its own limit (`dt^2 * eig(M^-1 kp) < 4` allows
    # ~7500), so stiffness is free; only damping is scarce.
    swing_kp: tuple = (60.0, 60.0, 60.0)
    swing_kd: tuple = (1.0, 1.0, 0.5)
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
    mpc: SRBDParams = field(default_factory=SRBDParams)

    @property
    def stance_duration(self) -> float:
        return self.duty / self.step_freq


class MPCWalkController:
    """Convex-MPC walker: GRFs from a QP, joint torques from `-J^T f`.

    Structure (doc section 4)::

        (v_des, yaw_rate) -> gait clock + footstep plan -> contact schedule c_i(k), r_i(k)
                                                              |
        CoM + attitude state ------------------------> convex MPC (QP)
                                                              |  f_i (first step)
                     stance leg: tau = -J_i^T f_i   <---------+---> swing leg: IK + PD

    There is deliberately **no attitude PD anywhere**. Roll, pitch and yaw are
    regulated by the MPC, by choosing how to split the load between the two
    feet and how far to lean each GRF inside the friction cone - authority a
    hip PD cannot express (see `mpc.py`).

    The controller owns `model`/`data` because it needs MuJoCo's own foot
    Jacobians and body frames; the sign convention it relies on is derived in
    `mpc.py` and checked end-to-end by validation stage 2.
    """

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
        self._foot_gid = [model.geom(f"foot_{s}").id for s in names]
        self._foot_bid = [model.geom_bodyid[g] for g in self._foot_gid]
        self._torso_bid = model.body("torso").id
        self._floor_gid = model.geom("floor").id
        root = model.joint("root").id
        self._root_qadr = model.jnt_qposadr[root]
        self._root_vadr = model.jnt_dofadr[root]
        # the three DOFs per leg, in the (hip_roll, hip, knee) column order
        # the 3x3 Jacobian and the torque vector both use
        joints = ("hip_roll_{}", "hip_{}", "knee_{}")
        self._leg_dofs = [np.array([model.jnt_dofadr[model.joint(j.format(s)).id]
                                    for j in joints]) for s in names]
        self._leg_qadr = [np.array([model.jnt_qposadr[model.joint(j.format(s)).id]
                                    for j in joints]) for s in names]
        self._knee_range = [model.joint(f"knee_{s}").range.copy() for s in names]
        self._roll_range = [model.joint(f"hip_roll_{s}").range.copy() for s in names]
        self._ctrl_range = model.actuator_ctrlrange.copy()
        self._jacp = np.zeros((3, model.nv))

        self._f = np.zeros((p.n_legs, 3))     # held GRFs between QP solves
        self._next_solve = -np.inf
        self._was_stance = [True] * p.n_legs
        self._lift = [np.zeros(2)] * p.n_legs  # foot xy at the last liftoff
        self._v_trim = np.zeros(2)             # integral trims (see MPCGaitParams)
        self._z_trim = 0.0
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
                          yaw: float) -> None:
        p = self.p
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

    def _phase(self, t: float, leg: int) -> float:
        """Where leg `leg` is in its own cycle at time `t` (0 = touchdown)."""
        return (t * self.p.step_freq + leg / self.p.n_legs) % 1.0

    def _scheduled(self, phase: float) -> bool:
        return phase < self.p.duty

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
        return hip + v * p.stance_duration / 2.0 + fb

    def _horizon_plan(self, t: float, com: np.ndarray, v: np.ndarray, yaw: float):
        """Contact schedule `c_i(k)` and moment arms `r_i(k)` over the horizon.

        `r_i(k) = p_foot,i - p_com(k)`, where a foot already planted keeps its
        *measured* position and a foot that touches down inside the horizon
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
        foot_z = p.foot_radius

        for i in range(p.n_legs):
            plant = None
            if self._scheduled(self._phase(t, i)) and self._in_contact(i):
                plant = self.data.geom_xpos[self._foot_gid[i]][:2].copy()
            for k in range(N):
                tk = t + k * dt
                ph = self._phase(tk, i)
                com_k = com[:2] + v[:2] * (k * dt)
                if self._scheduled(ph):
                    if plant is None:
                        # this stance interval starts inside the horizon:
                        # its touchdown was when the phase last crossed 0
                        td = tk - ph / p.step_freq
                        plant = self._plan_step(i, max(td, t), t, com, v[:2], yaw)
                    contact[k, i] = 1.0
                    r[k, i] = (*(plant - com_k), foot_z - com[2])
                else:
                    # the foot lifts; the next stance gets a freshly planned
                    # footstep rather than inheriting this one
                    plant = None
                    r[k, i] = (0.0, 0.0, foot_z - com[2])   # unused: f is clamped to 0
        return contact, r

    # --- MuJoCo state helpers ----------------------------------------------

    def _in_contact(self, leg: int) -> bool:
        gid = self._foot_gid[leg]
        for c in self.data.contact[: self.data.ncon]:
            pair = {c.geom1, c.geom2}
            if gid in pair and self._floor_gid in pair:
                return True
        return False

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
        """3x3 d(contact point)/d(hip_roll, hip, knee), with the floating base held fixed.

        `mj_jac` gives the full 3 x nv translational Jacobian; taking only the
        three leg columns is exactly the "base fixed" Jacobian of doc section
        8 (the base DOFs move the foot too, but no motor drives them).

        The point is the bottom of the foot sphere - where the ground actually
        pushes - not the sphere's centre. The difference is one foot radius
        (2.8 cm) on a shin that rotates, and `-J^T f` taken at the centre
        realises a different force at the contact: measured with the torso
        pinned (validation stage 2), the centre Jacobian delivers 11% too
        little horizontal force on both feet; at the contact point the worst
        component is off by 0.03 N, under 1%. (It also makes `J qdot` the
        contact point's velocity, which is zero for a planted foot that rolls
        without slipping, so the stance hold's damping stays out of the way
        even while the shin rotates.)
        """
        point = self.data.geom_xpos[self._foot_gid[leg]].copy()
        point[2] -= self.p.foot_radius
        self._mj.mj_jac(self.model, self.data, self._jacp, None, point,
                        self._foot_bid[leg])
        return self._jacp[:, self._leg_dofs[leg]]

    def _foot_hold(self, leg: int, plan: np.ndarray) -> np.ndarray:
        """Task-space spring-damper holding a stance foot on its plant point.

        Returns joint torques for `J^T (kp (p_plant - p_foot) - kd v_foot)`.
        While the foot is genuinely planted both terms vanish - the target is
        the foot's own position and its world velocity is zero - so the MPC
        keeps full authority over the leg. It only bites when the foot is off
        its plant point, i.e. when a scheduled-stance leg is still in the air
        or has been knocked loose, where it reaches for the planned footstep.
        """
        p = self.p
        J = self._foot_jacobian(leg)                  # 3x3, leg columns only
        v_foot = self._jacp @ self.data.qvel          # full Jacobian: world velocity
        pos = self.data.geom_xpos[self._foot_gid[leg]]
        xy = pos[:2] if self._in_contact(leg) else plan
        err = np.array([xy[0], xy[1], p.foot_radius]) - pos
        return J.T @ (p.stance_hold_kp * err - p.stance_hold_kd * v_foot)

    def _ik_pd(self, leg: int, target: np.ndarray) -> np.ndarray:
        """Joint-space PD onto the 3-joint IK solution for a world foot target.

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
        is to keep the trunk far below that.
        """
        p, d = self.p, self.data
        R = rot_z(self.yaw_ref)
        pelvis = d.xpos[self._torso_bid]
        hip_pos = pelvis + R @ np.array([0.0, self._side[leg] * HIP_Y, 0.0])
        dvec = R.T @ (target - hip_pos)
        roll_des, hip_des, knee_des = leg_ik(dvec, p.L1, p.L2)
        knee_des = float(np.clip(knee_des, *self._knee_range[leg]))
        roll_des = float(np.clip(roll_des, *self._roll_range[leg]))
        q_des = np.array([roll_des, hip_des, knee_des])
        q = d.qpos[self._leg_qadr[leg]]
        dq = d.qvel[self._leg_dofs[leg]]
        return np.asarray(p.swing_kp) * (q_des - q) - np.asarray(p.swing_kd) * dq

    # --- main entry point ---------------------------------------------------

    def control(self, t: float, com: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Torques in `data.ctrl` order:
        (hip_roll_l, hip_l, knee_l, hip_roll_r, hip_r, knee_r).

        `com` and `vel` are the whole robot's CoM position and velocity, 3-D.
        """
        p, d = self.p, self.data
        com, vel = np.asarray(com, dtype=float), np.asarray(vel, dtype=float)
        rpy, omega = self.base_state()
        yaw = self._yaw_meas = float(rpy[2])

        phases = [self._phase(t, i) for i in range(p.n_legs)]
        sched = [self._scheduled(ph) for ph in phases]

        # Remember where each foot left the ground: the swing trajectory is
        # drawn from the *actual* liftoff point to the planned touchdown, so a
        # stumble doesn't make the foot teleport back to a nominal stride.
        for i in range(p.n_legs):
            if self._was_stance[i] and not sched[i]:
                self._lift[i] = d.geom_xpos[self._foot_gid[i]][:2].copy()
        self._was_stance = list(sched)

        # Rolling horizon: re-solve on the MPC clock (every mpc.dt = 40 sim
        # steps) and zero-order-hold the GRFs in between.
        v_head = rot_z(yaw)[:2, :2].T @ vel[:2]
        self._update_setpoints(max(t - self._last_t, 0.0), v_head, float(com[2]), yaw)
        self._last_t = t
        if t >= self._next_solve:
            contact, r = self._horizon_plan(t, com, vel, yaw)
            x0 = np.zeros(NX)
            x0[ROLL:YAW + 1] = rpy
            x0[PX:PZ + 1] = com
            x0[WX:WZ + 1] = omega
            x0[VX:VZ + 1] = vel
            x0[ONE] = 1.0
            x_ref = reference_trajectory(x0, self.v_des, self.z_des, p.mpc,
                                         yaw_des=self.yaw_ref, yaw_rate=p.yaw_rate)
            self._f = self.mpc.solve(x0, contact, r, x_ref)
            self.last_status = self.mpc.status
            self._next_solve = t + p.mpc.dt

        out = []
        for i in range(p.n_legs):
            td = t + (1.0 - phases[i]) / p.step_freq      # next touchdown time
            plan = self._plan_step(i, td, t, com, vel[:2], yaw)

            if sched[i]:
                # Virtual work: the leg pushes on the *ground* with -f, so
                # tau = J^T (-f) = -J^T f. Sign derived in doc section 8 and
                # checked end-to-end by validation stage 2.
                tau = -self._foot_jacobian(i).T @ self._f[i]
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
                tau = tau + self._foot_hold(i, plan)
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
                dofs = self._leg_dofs[i]
                tau = tau + self.model.dof_damping[dofs] * d.qvel[dofs]
            else:
                s = (phases[i] - p.duty) / (1.0 - p.duty)
                xy = self._lift[i] + s * (plan - self._lift[i])
                z = p.foot_radius + p.swing_clearance * np.sin(np.pi * s)
                tau = self._ik_pd(i, np.array([xy[0], xy[1], z]))
            out += list(tau)

        tau = np.asarray(out)
        return np.clip(tau, self._ctrl_range[:, 0], self._ctrl_range[:, 1])
