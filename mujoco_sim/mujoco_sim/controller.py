"""Raibert-style controller for the planar two-legged (humanoid) hopper.

Reimplements the control logic from Flight_Controller.m / Stance_Controller.m /
Main.m, but MuJoCo owns the dynamics and each leg is now a real two-link
(thigh/shin) chain driven by two plain torque motors - no passive joint
springs anywhere, the same way the MIT Cheetah / ANYmal / Unitree quadrupeds
are built. All compliance is *active*: the SLIP leg spring k*(l-L0) of
Stance_EoM.m is emulated as a virtual spring-damper along the leg.

Two legs
--------
The robot has two identical legs, and they are driven as a **symmetric pair**
(a two-legged pronk), not in an alternating walking gait. Each leg runs its
own copy of the impedance law below on its own joint state, which is what
makes this a decentralised controller in the quadruped sense rather than a
single virtual leg: if the legs are disturbed apart, each corrects itself.

What *is* shared is the load. During stance the two legs together have to
produce the body's radial force and the torso's attitude torque, so each one
commands `leg_share = 1/n_legs` of it; otherwise the robot would see double
the intended leg stiffness and double the attitude gain. Flight is the
opposite: holding a leg at L0 and swinging it to the touchdown angle acts on
that leg's own inertia, so both flight terms use the full gain per leg.

Kinematics
----------
Joint angles are measured relative to the parent link: `hip` is the thigh
angle relative to the torso, `knee` the shin angle relative to the thigh
(knee=0 is a fully straight leg). The world-frame thigh angle is therefore
theta_thigh = torso_pitch + hip.

Because thigh and shin are the same length (L1 == L2), the hip-knee-foot
triangle is isosceles, so the hip-to-foot line exactly bisects the
thigh/shin angle, and the hip-to-foot distance depends on the knee alone:

    theta_leg = theta_thigh + knee/2          (world-frame leg direction)
    l(q)      = sqrt(L1^2 + L2^2 + 2*L1*L2*cos(q))     (leg length, q = knee)

Control ("MIT-style" leg impedance)
-----------------------------------
The leg is commanded in polar coordinates (l, theta_leg) - a radial force
F_r along the leg and a tangential effort F_t rotating it - which are mapped
to joint torques by the Jacobian transpose. Differentiating the relations
above w.r.t. the joint coordinates (theta_thigh, knee):

    dl        = (dl/dq) * dknee                 -> l does not depend on the thigh
    dtheta_leg = dtheta_thigh + 0.5 * dknee

    J = [[0,      dl/dq],
         [1,      0.5  ]]        tau = J^T @ [F_r, F_t]

which would give tau_hip = F_t and tau_knee = (dl/dq)*F_r + 0.5*F_t.

The 0.5*F_t cross-term is deliberately dropped - see `_joint_torques`. J^T is
the right mapping for a force applied at the *foot*, but F_t here is a torque
between torso and thigh, which the hip motor already delivers by itself
through its reaction on the torso. Feeding half of it into the light shin
instead makes the knee torque sign-flip every timestep and blow up. So the
mapping actually used is:

    tau_hip  = F_t
    tau_knee = (dl/dq) * F_r
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def leg_length(q: float, L1: float, L2: float) -> float:
    """Hip-to-foot distance for knee angle q (q=0 is fully straight)."""
    return float(np.sqrt(L1**2 + L2**2 + 2 * L1 * L2 * np.cos(q)))


def dlength_dq(q: float, L1: float, L2: float) -> float:
    """d(leg length)/d(knee angle): the radial force <-> knee torque lever arm."""
    l = leg_length(q, L1, L2)
    return -L1 * L2 * np.sin(q) / l


def knee_angle_for_length(l: float, L1: float, L2: float) -> float:
    """Inverse of leg_length: the (bent, i.e. negative) knee angle giving
    hip-to-foot distance l. Clips to the reachable range [|L1-L2|, L1+L2]."""
    l = float(np.clip(l, abs(L1 - L2) + 1e-6, L1 + L2 - 1e-6))
    c = (l**2 - L1**2 - L2**2) / (2 * L1 * L2)
    c = float(np.clip(c, -1.0, 1.0))
    return -float(np.arccos(c))


@dataclass
class ControllerParams:
    m: float = 1.298         # total robot mass (torso + both legs)
    g: float = 10.0
    n_legs: int = 2          # legs sharing the stance load
    L1: float = 0.17         # thigh length  (0.245 H)
    L2: float = 0.17         # shin length   (0.246 H)
    L0: float = 0.30         # nominal (rest) leg length
    Hs: float = 0.4          # desired apex height
    Vs: float = 1.0          # desired forward velocity
    settle_bounces: int = 10  # thrust disabled for the first N stance phases
    # Raibert thrust, applied by extending the leg spring's rest length during
    # the extension half of stance. thrust_gain converts an energy deficit (J)
    # into metres of extra rest length; max_thrust_extension caps it, which is
    # what bounds the energy injected per hop.
    thrust_gain: float = 0.02
    max_thrust_extension: float = 0.05
    # Virtual leg spring-damper (replaces the passive joint spring the old
    # prismatic model had in the XML). Stance is stiff - this *is* the SLIP
    # spring that makes the robot bounce - while flight only needs enough
    # authority to hold the leg near L0 while it swings.
    leg_kp_stance: float = 2200.0
    leg_kd_stance: float = 8.0
    leg_kp_flight: float = 300.0
    leg_kd_flight: float = 8.0
    # Nominal stance duration used by the touchdown-angle law, taken as half
    # the leg spring's natural period (pi*sqrt(m_eff/k)) rather than measured
    # per-bounce. Measuring it live is fragile: exactly when the gait
    # degrades into fast, low, grazing bounces (the failure mode this law is
    # supposed to prevent) the measured duration collapses toward zero,
    # which disables the law's velocity term right when it's needed most and
    # freezes theta_des at a stale value. A fixed physically-motivated
    # constant keeps the law responsive to the *current* liftoff velocity on
    # every single bounce.
    nominal_stance_duration: float = 0.06
    landing_angle_gain: float = 0.4
    velocity_error_clip: float = 1.0
    # flight: PD gains swinging the leg to the desired touchdown angle
    hip_kp: float = 40.0
    hip_kd: float = 4.0
    # stance: PD gains holding the torso upright (torso_pitch -> 0).
    #
    # attitude_kp is an order of magnitude above the flat-torso version: the
    # humanoid trunk puts the upper body's mass ~0.13 m above the hip axis,
    # which inflates the pitch inertia about that axis by ~6x, almost all of
    # it the m*h^2 parallel-axis term, and kp has to grow with the inertia to
    # keep the same attitude bandwidth.
    #
    # attitude_kd, however, has a hard ceiling that has nothing to do with the
    # torso. The damping torque is applied at the *hip*, whose own mass-matrix
    # diagonal is the leg's (~0.006 kg m^2), and explicit damping is unstable
    # once kd*dt/I > 2. With leg_share halving it per leg that caps
    # attitude_kd near 50; the first humanoid attempt used 40 on a 0.003
    # kg m^2 hip and sign-flipped the torque every timestep (+488, -276, +538,
    # ...), which looks exactly like a tuning problem but is purely numerical.
    # Raising the trunk therefore does NOT let you scale kp and kd together -
    # kd stays put, which is why the trunk mass has to sit as low as the
    # humanoid proportions allow.
    attitude_kp: float = 550.0
    attitude_kd: float = 10.0


@dataclass
class ControllerState:
    """Mutable state carried between control steps (mirrors the MATLAB globals)."""
    theta_des: float = 0.0
    bounce_count: int = 0
    # apex tracking during the current flight phase
    apex_h: float = 0.0
    apex_vx: float = 0.0


class RaibertController:
    def __init__(self, params: ControllerParams | None = None):
        self.p = params or ControllerParams()
        self.s = ControllerState()
        self.Ed = 0.5 * self.p.m * self.p.Vs**2 + self.p.m * self.p.g * self.p.Hs
        # nominal knee angle whose leg length is exactly L0
        self.q0 = knee_angle_for_length(self.p.L0, self.p.L1, self.p.L2)
        # fraction of the stance load (body weight, attitude torque) each leg
        # is responsible for - see "Two legs" in the module docstring
        self.leg_share = 1.0 / self.p.n_legs

    # --- leg state helpers (polar coordinates of the two-link leg) ----------

    def leg_state(self, torso_pitch: float, dtorso_pitch: float, hip: float,
                   dhip: float, knee: float, dknee: float) -> tuple:
        """(l, dl, theta_leg, dtheta_leg, dl/dq) for the current joint state."""
        dl_dq = dlength_dq(knee, self.p.L1, self.p.L2)
        l = leg_length(knee, self.p.L1, self.p.L2)
        dl = dl_dq * dknee
        theta_leg = torso_pitch + hip + knee / 2.0
        dtheta_leg = dtorso_pitch + dhip + dknee / 2.0
        return l, dl, theta_leg, dtheta_leg, dl_dq

    def _joint_torques(self, f_radial: float, f_tangential: float,
                        dl_dq: float) -> np.ndarray:
        """Map the polar leg command to (hip, knee) motor torques.

        The radial (leg-length) force goes to the knee through the length
        Jacobian dl/dq, since leg length depends on the knee alone.

        The tangential term is deliberately *not* run through the Jacobian
        transpose. J^T would also put 0.5*f_tangential on the knee, which is
        the right thing only if f_tangential were a force applied at the foot.
        It isn't: it is a torque between torso and thigh (attitude control in
        stance, leg swing in flight), which the hip motor delivers on its own
        via its reaction on the torso. Feeding half of that high-gain,
        high-bandwidth signal into the 0.04 kg shin instead drives the knee
        unstable - the torque sign-flips every timestep and grows without
        bound, throwing the foot off the ground. Any radial leakage from the
        hip torque is just a disturbance for the stiff leg-length loop to
        reject.
        """
        return np.array([f_tangential, dl_dq * f_radial])

    # --- event bookkeeping --------------------------------------------------

    def update_apex_tracking(self, z: float, vx: float, in_stance: bool) -> None:
        """Track the peak height / velocity-at-peak during flight (apex)."""
        if in_stance:
            return
        if z > self.s.apex_h:
            self.s.apex_h = z
            self.s.apex_vx = vx

    def on_liftoff(self, t: float, vx_liftoff: float, z_liftoff: float) -> None:
        """Called the instant the foot leaves the ground: update touchdown-angle
        law and reset apex trackers for the next flight phase (Main.m loop body)."""
        self.s.bounce_count += 1

        err = vx_liftoff - self.p.Vs
        err = float(np.clip(err, -self.p.velocity_error_clip, self.p.velocity_error_clip))

        arg = vx_liftoff * self.p.nominal_stance_duration / 2.0 / self.p.L0
        arg = float(np.clip(arg, -1.0, 1.0))
        self.s.theta_des = np.arcsin(arg) + self.p.landing_angle_gain * err

        # Seed the apex trackers with the state at liftoff rather than with
        # zero. `update_apex_tracking` only ever raises apex_h, so seeding it
        # at 0 means a grazing bounce - liftoff and touchdown within a few
        # timesteps, before the peak detector sees anything - reports an apex
        # of 0 m and an apex speed of 0 m/s. The energy law then reads that as
        # "the robot has no energy at all", and the error becomes the entire
        # target Ed, firing a thrust spike of several hundred newtons on the
        # next stance. That is what turned an otherwise steady gait into the
        # occasional 1 m launch, and it gets worse the higher thrust_gain is.
        # The liftoff state is always a valid lower bound on the flight apex.
        self.s.apex_h = z_liftoff
        self.s.apex_vx = vx_liftoff

    # --- phase controllers --------------------------------------------------

    def control(self, in_stance: bool, torso_pitch: float, dtorso_pitch: float,
                 legs) -> np.ndarray:
        """Torques for every leg, flattened into MuJoCo's `data.ctrl` layout.

        `legs` is a sequence of (hip, dhip, knee, dknee), one per leg, in the
        same order the actuators are declared in hopper.xml - i.e. the result
        is (hip_l, knee_l, hip_r, knee_r), interleaved per leg rather than
        grouped per joint type.
        """
        phase = self.stance_control if in_stance else self.flight_control
        return np.concatenate([
            phase(torso_pitch=torso_pitch, dtorso_pitch=dtorso_pitch,
                  hip=hip, dhip=dhip, knee=knee, dknee=dknee)
            for hip, dhip, knee, dknee in legs
        ])

    def flight_control(self, torso_pitch: float, dtorso_pitch: float,
                        hip: float, dhip: float,
                        knee: float, dknee: float) -> np.ndarray:
        """Swing the leg to the Raibert touchdown angle while holding it at its
        nominal length L0, both as virtual impedances in polar leg coordinates.

        Neither term is scaled by `leg_share`: in flight each leg is moving
        only its own thigh/shin inertia towards its own target, so sharing the
        effort between legs would just make both of them track sluggishly.
        """
        l, dl, theta_leg, dtheta_leg, dl_dq = self.leg_state(
            torso_pitch, dtorso_pitch, hip, dhip, knee, dknee)

        f_radial = self.p.leg_kp_flight * (self.p.L0 - l) - self.p.leg_kd_flight * dl
        f_tangential = (self.p.hip_kp * (self.s.theta_des - theta_leg)
                        - self.p.hip_kd * dtheta_leg)
        return self._joint_torques(f_radial, f_tangential, dl_dq)

    def stance_control(self, torso_pitch: float, dtorso_pitch: float,
                        hip: float, dhip: float,
                        knee: float, dknee: float) -> np.ndarray:
        """Virtual SLIP spring plus energy-regulating thrust along the leg, and
        a real attitude controller rotating the leg to hold the torso upright.

        Both outputs are scaled by `leg_share`, because the legs are in stance
        together: the virtual spring and the attitude torque are properties of
        the *body*, and each leg supplies its fraction. Without this the robot
        would ride on `n_legs` times the intended leg stiffness.

        Sign note: the hip motor's torque acts directly on the "hip" DOF, but
        by Newton's third law the *reaction* on the torso (a separate body,
        not the same coordinate) has the opposite sign - i.e. positive hip
        torque pushes torso_pitch *more* negative, not less. So the restoring
        law here needs the same sign as (torso_pitch, dtorso_pitch), not the
        negated "target-minus-current" form you'd write for a direct DOF.
        """
        l, dl, theta_leg, dtheta_leg, dl_dq = self.leg_state(
            torso_pitch, dtorso_pitch, hip, dhip, knee, dknee)

        # Virtual leg spring - this is what makes the robot bounce at all -
        # with Raibert's thrust applied as an *extension of the spring's rest
        # length* during the second half of stance, rather than as a raw extra
        # force.
        #
        # Stance_Controller.m adds +/- thrust_gain*energy_error directly to the
        # radial force, which is a bang-bang energy pump: it swamps the spring
        # (measured ~10x larger, so leg_kp_stance stopped mattering at all),
        # and it is unbounded, so a single bad apex estimate asks for hundreds
        # of newtons. Extending the rest length instead injects at most
        # leg_kp_stance * max_thrust_extension, keeps the force a genuine
        # spring force that stores and returns energy over the stroke, and
        # leaves the leg stiffness in charge of the bounce.
        rest = self.p.L0
        if self.s.bounce_count > self.p.settle_bounces and dl > 0.0:
            en_last_flight = (self.p.m * self.p.g * self.s.apex_h
                              + 0.5 * self.p.m * self.s.apex_vx**2)
            deficit = self.Ed - en_last_flight
            rest += float(np.clip(self.p.thrust_gain * deficit,
                                  0.0, self.p.max_thrust_extension))

        f_radial = self.p.leg_kp_stance * (rest - l) - self.p.leg_kd_stance * dl

        f_tangential = self.p.attitude_kp * torso_pitch + self.p.attitude_kd * dtorso_pitch

        return self._joint_torques(self.leg_share * f_radial,
                                    self.leg_share * f_tangential, dl_dq)


@dataclass
class GaitParams:
    """Parameters of the alternating (walking) gait.

    Unlike the hopper, this is a *scheduled* gait: a clock decides which leg
    is on the ground, instead of the phase being read from contact. The two
    legs run the same schedule half a cycle apart, so their swing reactions on
    the torso largely cancel - which is the main reason a biped can walk with
    a heavy trunk where the symmetric two-legged pronk could not.
    """
    m: float = 1.30
    g: float = 10.0
    n_legs: int = 2
    L1: float = 0.17
    L2: float = 0.17
    step_freq: float = 2.5       # full gait cycles per second
    duty: float = 0.65           # fraction of the cycle each leg spends in stance
    Vs: float = 0.8              # desired forward velocity
    stand_height: float = 0.30   # hip height above the foot centre in stance
    swing_clearance: float = 0.05
    # stance leg: virtual spring along the leg holding the body up, on top of
    # a gravity feed-forward so the spring only has to carry the error
    stance_kp: float = 3000.0
    stance_kd: float = 30.0
    # swing leg: joint-space PD onto the inverse-kinematics targets. knee_kd
    # is small on purpose - the shin's mass-matrix diagonal is ~8e-4 kg m^2,
    # and explicit damping needs kd*dt/I < 2.
    swing_hip_kp: float = 60.0
    swing_hip_kd: float = 3.0
    swing_knee_kp: float = 60.0
    swing_knee_kd: float = 2.0
    # torso attitude, carried by whichever hips are in stance
    attitude_kp: float = 300.0
    attitude_kd: float = 12.0
    # Raibert-style foot-placement correction: shifts the touchdown point
    # forward when the robot is running fast, which is what actually regulates
    # speed (the schedule alone only sets a nominal stride).
    speed_gain: float = 0.12
    speed_error_clip: float = 0.5

    @property
    def step_length(self) -> float:
        """Foot travel relative to the hip during one stance phase.

        The body advances at Vs while the foot is planted, so over a stance of
        duty/step_freq seconds the foot sweeps back by exactly this much.
        """
        return self.Vs * self.duty / self.step_freq


class WalkingController:
    """Scheduled alternating-gait walker for the planar biped.

    Each leg follows a foot trajectory expressed in the hip frame and converted
    to joint targets by the 2-link inverse kinematics above:

    - **Stance** (phase < duty): the foot sweeps backwards from +S/2 to -S/2,
      which is what carries the body forward. The knee runs a virtual spring
      along the leg (plus a gravity feed-forward) to hold body height, and the
      hip is given over to torso attitude - the same split Raibert uses, and
      the same one that worked for the hopper.
    - **Swing** (phase >= duty): the foot lifts by swing_clearance and returns
      to +S/2 ready for touchdown. Here both joints track the IK solution with
      a joint-space PD, since the leg is free and has only its own inertia to
      move.
    """

    def __init__(self, params: GaitParams | None = None):
        self.p = params or GaitParams()

    def _leg_polar(self, torso_pitch, dtorso_pitch, hip, dhip, knee, dknee):
        dl_dq = dlength_dq(knee, self.p.L1, self.p.L2)
        l = leg_length(knee, self.p.L1, self.p.L2)
        theta_leg = torso_pitch + hip + knee / 2.0
        return l, dl_dq * dknee, theta_leg, dl_dq

    def _foot_target(self, phase: float, vx: float) -> tuple:
        """Desired foot position (forward, up) relative to the hip, in world
        axes, for a leg at this point in its own cycle."""
        p = self.p
        S = p.step_length
        # Raibert foot placement: land further forward than the nominal stride
        # when travelling too fast, which brakes; further back when too slow.
        err = float(np.clip(vx - p.Vs, -p.speed_error_clip, p.speed_error_clip))
        shift = p.speed_gain * err

        if phase < p.duty:                       # stance: sweep the foot back
            s = phase / p.duty
            return (0.5 - s) * S + shift, -p.stand_height
        s = (phase - p.duty) / (1.0 - p.duty)    # swing: lift and return
        x = (-0.5 + s) * S + shift
        return x, -p.stand_height + p.swing_clearance * np.sin(np.pi * s)

    def control(self, t: float, torso_pitch: float, dtorso_pitch: float,
                legs, vx: float = 0.0, contacts=None) -> np.ndarray:
        """`contacts` is a per-leg bool saying whether that foot is actually
        on the ground. The gait schedule alone is open-loop, and a leg the
        schedule calls "stance" may still be in the air - early on, or after a
        stumble. Pushing the weight feed-forward through an unloaded leg just
        throws the robot upwards (it launched itself on the first attempt,
        never touching the ground again), so the support term is gated on real
        contact and only the legs actually carrying the robot share its
        weight."""
        p = self.p
        base = (t * p.step_freq) % 1.0
        phases = [(base + i / p.n_legs) % 1.0 for i in range(p.n_legs)]
        if contacts is None:
            contacts = [True] * p.n_legs
        loaded = [ph < p.duty and ct for ph, ct in zip(phases, contacts)]
        n_stance = max(1, sum(loaded))

        out = []
        for i, (ph, (hip, dhip, knee, dknee)) in enumerate(zip(phases, legs)):
            l, dl, theta_leg, dl_dq = self._leg_polar(
                torso_pitch, dtorso_pitch, hip, dhip, knee, dknee)
            x_f, z_f = self._foot_target(ph, vx)
            l_des = float(np.hypot(x_f, z_f))

            if ph < p.duty:
                # Gravity feed-forward: share the body weight between the legs
                # actually on the ground and project it along the leg, so the
                # virtual spring only has to correct the residual.
                support = (p.m * p.g / n_stance / max(np.cos(theta_leg), 0.5)
                           if loaded[i] else 0.0)
                f_radial = (support + p.stance_kp * (l_des - l)
                            - p.stance_kd * dl)
                tau_hip = (p.attitude_kp * torso_pitch
                           + p.attitude_kd * dtorso_pitch) / n_stance
                tau_knee = dl_dq * f_radial
            else:
                theta_des = float(np.arctan2(x_f, -z_f))
                knee_des = knee_angle_for_length(l_des, p.L1, p.L2)
                hip_des = theta_des - torso_pitch - knee_des / 2.0
                tau_hip = p.swing_hip_kp * (hip_des - hip) - p.swing_hip_kd * dhip
                tau_knee = p.swing_knee_kp * (knee_des - knee) - p.swing_knee_kd * dknee
            out += [tau_hip, tau_knee]
        return np.asarray(out)
