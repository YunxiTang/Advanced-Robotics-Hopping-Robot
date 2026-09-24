"""The convex-MPC walking controller, driving the Unitree G1 (`g1.py`).

`G1WalkController` is `MPCWalkController` - the same gait clock, footstep
plan, SRBD model and QP (`mpc.py`) - bound to G1's joints. G1 is a 33 kg
robot whose legs are 43% of its mass, where the small biped is a 1.3 kg
robot whose legs are nearly weightless, and three parts are replaced or
added for that. The measurements behind every choice are in `doc/g1_mpc.md`.

- **A whole-body QP between the MPC and the motors** (`wbc.py`). The biped
  maps each stance foot's wrench to torques with `-J^T w`, exact only for a
  massless leg. The QP solves the full rigid-body dynamics instead, for
  torques that realise the MPC's wrenches while the swing legs, waist and
  arms follow their own targets. The plain mapping (`use_wbc=False`, with
  the stance leg's own gravity added) walks G1 too, but at 0.7 m/s the
  trunk pitches and rolls four to ten times as much.
- **Swing-leg IK is numerical.** The biped's closed-form IK needs equal thigh
  and shin and intersecting hip axes. G1 has neither: its hip is pitch, then
  roll, then yaw, with offsets between them. Each swing target is solved by
  damped least squares on the sole site's 6x6 Jacobian (position, and a
  level sole at the planned heading), warm-started from the previous tick,
  so two iterations per tick keep it converged. It is solved on a scratch
  `MjData` whose pelvis has the *heading set-point's* orientation, not the
  measured one, for the reason the biped's closed form is (faults 3 and 6 in
  the README): a swing target that moves with the measured trunk attitude is
  positive feedback on that attitude through the mass matrix.
- **Upper body.** The waist (yaw, roll, pitch) is held straight, so pelvis
  and torso move as the single rigid body the MPC assumes. The seven-joint
  arms hold a relaxed pose and swing against the legs as the biped's do
  (`arm_swing_gain`).

What the MPC is told about the robot is in `g1_srbd_params` and
`G1GaitParams`; the reasoning behind each weight and gain that is not
explained here is the biped's, in `controller.py` and `mpc.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import g1
from .controller import SIDE_WORD, MPCGaitParams, MPCWalkController
from .mpc import N_CON_FOOT, NU_FOOT, SRBDParams, rot_z
from .wbc import WBCParams, WholeBodyQP

G1_LEG_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
G1_ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")
G1_WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


def g1_srbd_params() -> SRBDParams:
    return SRBDParams(
        mass=33.34,             # model.body_subtreemass[0]
        # Effective inertia, measured as the biped's was (fault 1): drive the
        # standing robot through -J^T w with a known extra foot force and
        # read off the pelvis's angular acceleration. Roll 1.1-1.8, pitch
        # 0.60, yaw 0.33, against 3.55 / 3.17 / 0.54 for the rigid robot and
        # 1.61 / 1.38 / 0.29 for pelvis plus upper body alone.
        #
        # It is a compromise between standing and walking. Standing, the legs
        # are planted and only the trunk turns: at the rigid value the robot
        # cannot stand still (with both feet down it falls at 1.6 s), and
        # without the whole-body QP even (1.6, 1.4, 0.3) made the pelvis
        # pitch rate double in sign-alternating steps at the MPC's rate.
        # Walking, a swinging leg turns with the trunk, and the rigid value
        # walks best - it alone holds a 2 s ramp from rest to 0.7 m/s. The
        # measured value does both, and passes 15 of the 18 tasks in the
        # battery the doc describes; (1.6, 1.4, 0.3) passes 14.
        inertia=(1.2, 0.6, 0.33),
        g=9.81,
        # 0.03 s x 21 = 0.63 s, about 1.6 gait cycles at 2.5 Hz, as the
        # biped's 0.4 s is at 4 Hz.
        dt=0.03,
        horizon=21,
        # Menagerie's feet are friction 0.6 (they have contact priority over
        # the floor); pyramid inscribed in the cone.
        mu=0.6 / np.sqrt(2.0),
        f_z_max=700.0,          # ~2.1 Mg
        f_z_min=25.0,           # ~0.08 Mg: keeps a scheduled stance leg loaded
        foot_front=g1.FOOT_FRONT,
        foot_back=g1.FOOT_BACK,
        foot_half_width=g1.FOOT_HALF_WIDTH,
        # four corners 0.06-0.13 m from the reference point, mu = 0.6, a
        # third of it promised (see SRBDParams.mu_torsion)
        mu_torsion=0.02,
        # Roll and pitch are weighted a fifth of the biped's 500; 400 does as
        # well here (15 of 18 either way).
        #    roll   pitch  yaw    px   py   pz     wx   wy   wz   vx    vy    vz   1
        Q=(100.0, 100.0, 100.0, 0.0, 0.0, 200.0, 5.0, 5.0, 2.0, 30.0, 30.0, 5.0, 0.0),
        # The forces are ~25x the biped's, so the regularisation is scaled
        # down by ~25^2 to buy the same trade-off against the state cost.
        R=2e-6,
        R_moment=2e-6 / 0.08**2,
    )


@dataclass
class G1GaitParams(MPCGaitParams):
    m: float = 33.34
    g: float = 9.81
    L1: float = 0.31            # thigh and shin, for the logged leg length only
    L2: float = 0.32
    # A 0.4 s cycle, 0.16 s of swing. 2.0 Hz passes as many tasks but
    # pitches twice as much at 0.7 m/s; at 1.6 Hz a ramp to 0.6 m/s makes
    # only 0.42 and pitches 0.29 rad. G1's pendulum time constant is
    # sqrt(z/g) = 0.25 s, and the longer the swing is against it, the
    # further the sideways fall over the stance foot grows before the next
    # foot catches it.
    step_freq: float = 2.5
    duty: float = 0.6
    # G1's knees and 7 kg legs are not built to run, and it is not tried.
    gait: str = "walk"
    Vs: float = 0.3
    # 0.07 m below the straight-legged CoM height (0.704 m). At 0.66 the
    # robot loses the 0.7 m/s ramps and the two largest pushes: the higher it
    # walks, the less knee travel is left before the stance leg locks
    # straight at the end of its sweep (fault 10 in the README, again - an
    # earlier version was seen vaulting off a knee at its -0.09 rad stop).
    z_des: float = 0.63
    swing_clearance: float = 0.05
    # Raibert gains scale with sqrt(z/g): 0.25 s here against the biped's
    # 0.19, so its 0.12 becomes ~0.17.
    k_v: float = 0.17
    k_vy: float = 0.17
    # The hips are 0.116 m off the centreline.
    step_width: float = 0.09
    z_trim_max: float = 0.08
    # swing PD, in G1_LEG_JOINTS order (hip pitch, roll, yaw, knee, ankle
    # pitch, ankle roll), on top of gravity compensation - used as torques
    # only with use_wbc=False. Each joint has 0.01 kg m^2 of armature
    # (g1.xml); the eigenvalues of dt M^-1 diag(kd) for the leg are at most
    # 0.6, far inside the explicit-integration limit of 2.
    swing_kp: tuple = (500.0, 500.0, 200.0, 500.0, 80.0, 50.0)
    swing_kd: tuple = (20.0, 20.0, 8.0, 20.0, 4.0, 2.0)
    # stance foot hold (use_wbc=False only), scaled up from the biped's
    stance_hold_kp: float = 2000.0
    stance_hold_kd: float = 40.0
    stance_hold_kr: float = 40.0
    stance_hold_kdr: float = 1.0
    late_touchdown_speed: float = 0.4
    # Arms: swing as the biped's do; elbows held at the pose's bend.
    arm_swing_gain: float = 1.0
    arm_swing_max: float = 0.5
    # waist PD (yaw, roll, pitch) holding the torso on the pelvis, as torques
    # (use_wbc=False only; its damping is the model's, see g1.WAIST_DAMPING)
    waist_kp: tuple = (3000.0, 3000.0, 3000.0)
    waist_kd: tuple = (0.0, 0.0, 0.0)
    # arm PD, in G1_ARM_JOINTS order (use_wbc=False only)
    arm_kp: tuple = (60.0, 60.0, 40.0, 40.0, 10.0, 10.0, 10.0)
    arm_kd: tuple = (2.0, 2.0, 1.5, 1.5, 0.3, 0.3, 0.3)
    # swing IK: damped-least-squares iterations per tick, and the damping
    ik_iters: int = 2
    ik_damping: float = 1e-3
    # Whole-body QP (`wbc.py`) between the MPC and the motors. Off, the
    # torques are the biped's: -J^T w plus the leg's own gravity on stance
    # legs, joint PD elsewhere.
    use_wbc: bool = True
    wbc: WBCParams = field(default_factory=WBCParams)
    # the QP's joint-space tasks - swing legs, then waist and arms - as
    # acceleration gains (1/s^2, 1/s) and weights
    wbc_leg_kp: float = 900.0
    wbc_leg_kd: float = 60.0
    wbc_leg_w: float = 1.0
    wbc_upper_kp: float = 400.0
    wbc_upper_kd: float = 40.0
    wbc_upper_w: float = 1.0
    mpc: SRBDParams = field(default_factory=g1_srbd_params)


class G1WalkController(MPCWalkController):
    """`MPCWalkController` on the Unitree G1 (see the module docstring).

    `data.ctrl` order is Menagerie's: left leg, right leg (G1_LEG_JOINTS),
    waist, left arm, right arm (G1_ARM_JOINTS).
    """

    ROOT_JOINT = "floating_base_joint"
    BASE_BODY = "pelvis"
    FOOT_BODY = "{side}_ankle_roll_link"
    SOLE_SITE = "sole_{s}"
    LEG_JOINT = "{side}_{j}_joint"
    LEG_JOINTS = G1_LEG_JOINTS
    ARM_JOINTS = G1_ARM_JOINTS

    def __init__(self, model, data, params: G1GaitParams | None = None):
        super().__init__(model, data, params or G1GaitParams())
        p = self.p
        wj = [model.joint(n).id for n in G1_WAIST_JOINTS]
        self._waist_dofs = model.jnt_dofadr[wj]
        self._waist_qadr = model.jnt_qposadr[wj]
        self._arm_pose = [np.array([g1._arm_q(j, s) for j in G1_ARM_JOINTS])
                          for s in SIDE_WORD]
        # the swing IK's scratch copy of the robot, and its warm starts
        self._ik = self._mj.MjData(model)
        self._q_ik = [None] * p.n_legs
        self._ik_t = [-np.inf] * p.n_legs
        # per tick: each swinging leg's joint targets, None for a stance leg
        self._swing_des = [None] * p.n_legs
        self._upper_dofs = np.concatenate([self._waist_dofs, *self._arm_dofs])
        self._upper_qadr = np.concatenate([self._waist_qadr, *self._arm_qadr])
        self._upper_des = np.zeros(len(self._upper_dofs))
        self.wbc = WholeBodyQP(model, data, self._sole_sid,
                               self.mpc._A[:N_CON_FOOT, :NU_FOOT],
                               p.mpc.f_z_min, p.mpc.f_z_max, p.wbc)
        self.wbc_wrench = np.zeros((p.n_legs, NU_FOOT))

    # --- swing leg ------------------------------------------------------------

    def _swing_ik(self, leg: int, target: np.ndarray, foot_yaw: float) -> np.ndarray:
        """Leg joint angles putting the sole site at world `target`, level and
        at heading `foot_yaw`, with the pelvis at its measured position but
        the heading set-point's orientation (level, yawed to `yaw_ref`)."""
        mj, m, p = self._mj, self.model, self.p
        ik, d = self._ik, self.data
        qadr, dofs = self._leg_qadr[leg], self._leg_dofs[leg]
        lo, hi = self._q_range[leg].T

        # warm start from the last solution, unless this leg has not been
        # swinging (it has just lifted off, or the run has just begun)
        t = d.time
        q = self._q_ik[leg]
        if q is None or t - self._ik_t[leg] > 5 * m.opt.timestep:
            q = d.qpos[qadr].copy()
        self._ik_t[leg] = t

        ik.qpos[:] = d.qpos
        ra = self._root_qadr
        half = 0.5 * self.yaw_ref
        ik.qpos[ra + 3:ra + 7] = (np.cos(half), 0.0, 0.0, np.sin(half))
        R_des = rot_z(foot_yaw)
        sid = self._sole_sid[leg]
        jacp, jacr = self._jacp, self._jacr
        for _ in range(p.ik_iters):
            ik.qpos[qadr] = q
            mj.mj_kinematics(m, ik)
            mj.mj_comPos(m, ik)
            e_p = target - ik.site_xpos[sid]
            E = R_des @ ik.site_xmat[sid].reshape(3, 3).T
            e_r = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
            mj.mj_jacSite(m, ik, jacp, jacr, sid)
            J = np.vstack((jacp[:, dofs], jacr[:, dofs]))
            e = np.concatenate((e_p, e_r))
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + p.ik_damping * np.eye(6), e),
                        lo, hi)
        self._q_ik[leg] = q
        return q

    def _ik_pd(self, leg: int, target: np.ndarray, foot_yaw: float) -> np.ndarray:
        """Joint PD onto the swing IK, plus the leg's own gravity. Also
        records the targets for the whole-body QP."""
        p, d = self.p, self.data
        q_des = self._swing_ik(leg, np.asarray(target, dtype=float), foot_yaw)
        qadr, dofs = self._leg_qadr[leg], self._leg_dofs[leg]
        self._swing_des[leg] = q_des
        return (np.asarray(p.swing_kp) * (q_des - d.qpos[qadr])
                - np.asarray(p.swing_kd) * d.qvel[dofs] + d.qfrc_bias[dofs])

    # --- stance leg -----------------------------------------------------------

    def _stance_feedforward(self, leg: int) -> np.ndarray:
        """`-J^T w` assumes a massless leg; G1's joints must also hold up
        their own 7 kg of leg, which is what `qfrc_bias` is on these DOFs.
        Without it the standing robot rose 1.6 cm in 0.3 s under a wrench
        that should have held it still."""
        dofs = self._leg_dofs[leg]
        return super()._stance_feedforward(leg) + self.data.qfrc_bias[dofs]

    def leg_lengths(self) -> list:
        """Hip-pitch pivot to sole site, per leg."""
        d = self.data
        hips = [self.model.body(f"{SIDE_WORD[s]}_hip_pitch_link").id for s in SIDE_WORD]
        return [float(np.linalg.norm(d.xpos[h] - d.site_xpos[s]))
                for h, s in zip(hips, self._sole_sid)]

    # --- upper body -----------------------------------------------------------

    def _arm_torques(self, dt: float) -> np.ndarray:
        """Waist, then left and right arm, in data.ctrl order: PD onto the
        pose plus gravity compensation, the shoulders swinging against the
        legs. Also records the targets for the whole-body QP."""
        p, d = self.p, self.data
        # leg angle: hip pitch + knee/2 is the hip-to-ankle line (thigh and
        # shin are nearly equal); positive = foot behind the hip
        theta = np.array([d.qpos[a[0]] + 0.5 * d.qpos[a[3]] for a in self._leg_qadr])
        target = np.clip(p.arm_swing_gain * 0.5 * (theta[::-1] - theta),
                         -p.arm_swing_max, p.arm_swing_max)
        a = dt / (p.arm_filter_tau + dt) if dt > 0 else 0.0
        self._arm_swing += a * (target - self._arm_swing)

        wq, wv = self._waist_qadr, self._waist_dofs
        out = [np.asarray(p.waist_kp) * -d.qpos[wq] - np.asarray(p.waist_kd) * d.qvel[wv]
               + d.qfrc_bias[wv]]
        upper = [np.zeros(len(wq))]
        for i in range(p.n_legs):
            q_des = self._arm_pose[i].copy()
            q_des[0] += self._arm_swing[i]
            upper.append(q_des)
            dofs = self._arm_dofs[i]
            out.append(np.asarray(p.arm_kp) * (q_des - d.qpos[self._arm_qadr[i]])
                       - np.asarray(p.arm_kd) * d.qvel[dofs] + d.qfrc_bias[dofs])
        self._upper_des = np.concatenate(upper)
        return np.concatenate(out)

    # --- whole-body QP ----------------------------------------------------------

    def control(self, t: float, com: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Torques in data.ctrl order.

        The base controller plans - gait clock, footsteps, the MPC - and, on
        the way, records each swinging leg's and the upper body's targets.
        The whole-body QP then turns the MPC's wrenches and those targets
        into torques.
        """
        self._swing_des = [None] * self.p.n_legs
        tau = super().control(t, com, vel)
        if not self.p.use_wbc:
            return tau
        p, d, m = self.p, self.data, self.model
        # every leg the base controller did not swing is a loaded stance leg
        # (a scheduled stance foot that has not landed yet is swung down)
        loaded = [self._swing_des[i] is None for i in range(p.n_legs)]

        qdd_des = np.zeros(m.nv - 6)
        w_joint = np.zeros(m.nv - 6)
        for i in range(p.n_legs):
            if loaded[i]:
                continue
            dofs, qadr = self._leg_dofs[i], self._leg_qadr[i]
            qdd_des[dofs - 6] = (p.wbc_leg_kp * (self._swing_des[i] - d.qpos[qadr])
                                 - p.wbc_leg_kd * d.qvel[dofs])
            w_joint[dofs - 6] = p.wbc_leg_w
        ud = self._upper_dofs
        qdd_des[ud - 6] = (p.wbc_upper_kp * (self._upper_des - d.qpos[self._upper_qadr])
                           - p.wbc_upper_kd * d.qvel[ud])
        w_joint[ud - 6] = p.wbc_upper_w

        foot_yaw = [self._foot_yaw(i) for i in range(p.n_legs)]
        u_mpc = np.where(np.asarray(loaded)[:, None], self._f, 0.0)
        tau_wbc, u = self.wbc.solve(loaded, u_mpc, foot_yaw, qdd_des, w_joint)
        if tau_wbc is None:
            return tau
        self.wbc_wrench = u
        return np.clip(tau_wbc, self._ctrl_range[:, 0], self._ctrl_range[:, 1])
