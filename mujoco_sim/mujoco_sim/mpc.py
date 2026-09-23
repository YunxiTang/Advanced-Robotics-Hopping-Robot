"""Convex MPC over a 3-D single-rigid-body model (SRBD).

This is the optimiser half of the controller described in `doc/convec_mpc.md`,
and it follows Di Carlo et al., *Dynamic Locomotion in the MIT Cheetah 3
Through Convex Model-Predictive Control* (IROS 2018) - now in full 3-D rather
than reduced to the sagittal plane.

Why an MPC and not a PD
-----------------------
The trunk's attitude can only be driven by (a) the moment the ground
reaction wrenches make about the CoM - each foot's force through its moment
arm, plus the flat foot's own contact moment - and (b) where the feet are
put. The obvious alternative - a PD law on
the hips, which is what this package used before - regulates attitude with a
gain capped by the *leg's* tiny mass-matrix diagonal rather than the torso's:
damping much above 30 makes the explicit integrator sign-flip the torque every
timestep, so the attitude authority is bounded by a numerical limit that has
nothing to do with the physics. An MPC asks for the GRFs directly, so the
bound becomes the friction cone - the real limit - and during double support
it can also trade load between the two feet, a moment a hip PD cannot produce
at all.

The model
---------
State (13 - the trailing constant 1 turns gravity into a *linear* term so the
whole problem stays a QP):

    x = [phi, theta, psi,  p_x, p_y, p_z,  w_x, w_y, w_z,  v_x, v_y, v_z,  1]
         roll pitch yaw    CoM position    world ang. vel.  CoM velocity

Control (a ground reaction *wrench* per flat foot - force and moment):

    u_i = [f_x, f_y, f_z,  m_x', m_y', m_z']        u = [u_l, u_r]

    d(Theta) = Rz(psi)^T w                             (small roll/pitch)
    dp       = v
    dw       = I_w^-1 sum_i (r_i x f_i + Rz(psi_i) m_i')   (I_w = Rz I_body Rz^T)
    dv       = (1/M) sum_i f_i - g e_z

The force is in world axes. The moment is taken about the foot's reference
point (the ankle projected onto the sole) and expressed in the foot's **yaw
frame** (primed, rotated by the foot's heading `psi_i`). Writing it in the
foot frame is what keeps the constraint matrix below constant: the
centre-of-pressure limits are a fixed rectangle in that frame whichever way
the foot points, and the foot's heading moves into `B` instead, which is
rebuilt every solve anyway.

With point feet `m_i` was identically zero, and all the attitude authority in
single support came from `r x f` - leaning the force inside the friction
cone. A flat foot adds a moment of up to `f_z` times the half-length of the
sole about each horizontal axis, and a little torsional friction about the
vertical, and that is the whole difference between a robot that must keep
stepping to stay up and one that can stand still.

The two approximations are the Cheetah paper's: roll and pitch are assumed
small, so the Euler-rate map collapses to a yaw rotation, and the gyroscopic
term `w x I w` is dropped. Both hold here - the MPC keeps roll and pitch
within a few degrees, and the trunk never spins fast.

`r_i` is the vector from the CoM to foot `i`. Treating it as a *known,
externally scheduled* quantity (from the gait scheduler's footstep plan) is
exactly what keeps the problem convex: the true bilinear term `r x f` would
otherwise make it a nonconvex program, and the contact sequence itself would
make it a mixed-integer one. Likewise `psi` inside `Rz` is taken from the yaw
*reference* at each predicted step, so turning makes the dynamics
time-varying but still linear.

Sign convention: Euler angles are standard right-handed Z-Y-X (x forward,
y left, z up), so positive roll drops the right side, positive pitch is
**nose-down** and positive yaw turns left. Check it with `r x f`: a foot
planted ahead of the CoM (`r_x > 0`) pushing up (`f_z > 0`) gives
`(r x f)_y = r_z f_x - r_x f_z < 0`, i.e. nose-up - as it should. (The planar
version measured pitch about -y, nose-up positive; the 3-D one uses the
conventional axes, so its pitch has the opposite sign.)

Discretisation is forward Euler (`A_d = I + A_c dt`, `B_d = B_c dt`) as in the
Cheetah paper; over a 0.02 s step the error is negligible compared to the
SRBD approximation itself.

The QP
------
Condensed form: the states are eliminated, leaving only the input sequence
`U in R^(12N)` as decision variables (240 of them at N=20), with

    X = A_qp x0 + B_qp U
    min  ||X - X_ref||^2_Qbar + ||U||^2_Rbar
       = 1/2 U' [2(B_qp' Qbar B_qp + Rbar)] U + [2 B_qp' Qbar (A_qp x0 - X_ref)]' U

Constraints, per predicted step and foot (11 rows each):

    +/- f_ix - mu f_iz <= 0      (friction pyramid, x)
    +/- f_iy - mu f_iz <= 0      (friction pyramid, y)
    f_z_min c_i(k) <= f_iz <= f_z_max c_i(k)
                                 (unilateral; c=0 zeroes a swing leg entirely,
                                  and every other row then forces the rest of
                                  the wrench to 0 too)
    +/- m_ix' - Y f_iz <= 0      (CoP inside the sole, sideways)
    -X_front f_iz <= m_iy' <= X_back f_iz
                                 (CoP inside the sole, fore-aft: a CoP at x'
                                  gives m_y' = -x' f_z)
    +/- m_iz' - mu_t f_iz <= 0   (torsional friction)

The CoP rows are the ZMP condition, written per foot: a moment that would put
the centre of pressure outside the sole cannot be delivered, because the foot
would roll onto its edge instead. They are linear in the wrench because the
bounds scale with `f_z`, which is what lets a flat foot stay in a QP.

The pyramid is inscribed in MuJoCo's elliptic cone (`mu_mpc = mu/sqrt(2)`),
so a force the QP accepts can never slip at a corner.

The solver is DAQP, a dense active-set method, not the ADMM solver (OSQP)
the point-foot version used. The condensed QP is small and fully dense, and
with contact moments in it it is also badly conditioned: two feet can trade
moment against load in many near-equivalent ways, only the light
regularisation tells them apart, and the Hessian's condition number is ~2e6.
ADMM's convergence rate degrades with exactly that - OSQP took a median 1275
iterations (~70 ms, three times the 20 ms budget) on the same problems, and
rescaling the moments to force units only halved it - while an active-set
method does not care: DAQP solves them *exactly*, in ~7 ms, with no
tolerances to tune. It has no warm start through its Python API, so the
problem is simply rebuilt and solved from scratch every tick.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import daqp

NX = 13         # state dimension (12 + the constant-1 gravity channel)
NU_FOOT = 6     # (f_x, f_y, f_z, m_x', m_y', m_z') per foot
N_CON_FOOT = 11 # pyramid (4), f_z (1), CoP (4), torsion (2)
FZ_ROW = 4      # the two-sided f_z row within a foot's block

# state indices
ROLL, PITCH, YAW = 0, 1, 2
PX, PY, PZ = 3, 4, 5
WX, WY, WZ = 6, 7, 8
VX, VY, VZ = 9, 10, 11
ONE = 12


def rot_z(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(r: np.ndarray) -> np.ndarray:
    """[r]_x, so that skew(r) @ f == np.cross(r, f)."""
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def moment_to_world(m_foot: np.ndarray, foot_yaw: float) -> np.ndarray:
    """A foot-yaw-frame contact moment (the QP's m_i') in world axes."""
    return rot_z(foot_yaw) @ m_foot


@dataclass
class SRBDParams:
    """Model, horizon and weights of the convex MPC (doc section 12)."""
    mass: float = 1.31          # measured: model.body_subtreemass[torso]
    # Effective body-frame inertia (roll, pitch, yaw), NOT the rigid-body one.
    #
    # The whole robot's inertia about the CoM is diag(0.024, 0.024, 0.004)
    # kg m^2, but only diag(0.011, 0.011, 0.002) of that is the torso; the
    # rest is the two legs, and the legs are *torque*-controlled, so they do
    # not ride along with the trunk when a GRF rotates it. In the planar
    # version, driving the model with a constant GRF through -J^T f and
    # measuring d(pitch)/dt showed an effective pitch inertia of 0.012-0.016,
    # i.e. roughly half the rigid-body figure - and at the rigid-body value
    # the robot could not stand for one second, because the MPC believed each
    # newton bought half the rotation it actually got.
    #
    # Pitch keeps that measured 0.0145. Roll is the same torso-plus-a-quarter
    # rule applied to the torso's own 0.011 (the trunk is almost square in
    # x-z and y-z, so the two axes behave alike). Yaw sits between the
    # torso's 0.0020 and the rigid 0.0040. It was set when the legs had no
    # hip-yaw joint and followed the trunk in yaw; with one they no longer
    # do, which argues for a lower value, but the flat-foot gait passes
    # every validation stage at the old one and it has not been re-tuned.
    inertia: tuple = (0.0145, 0.0145, 0.0030)
    g: float = 10.0
    n_feet: int = 2
    # 0.02 s x 20 = a 0.4 s horizon, 1.6 gait cycles at 4 Hz. At least one
    # full cycle is needed to "see" the double-support windows where the
    # attitude authority actually lives.
    dt: float = 0.02
    horizon: int = 20
    # MuJoCo's cone is elliptic with mu = 0.8; the linear pyramid is
    # inscribed in it so that its corners (|f_x| = |f_y| = mu f_z) stay
    # inside the real cone.
    mu: float = 0.8 / np.sqrt(2.0)
    f_z_max: float = 50.0       # ~3.8 Mg, leaves room to push off
    # Minimum vertical load on a foot the schedule says is down.
    #
    # With a pure `0 <= f_z` bound the QP discovers that it can dodge the
    # attitude cost of a stance foot that is not exactly under the CoM by
    # unloading *every* foot and going ballistic for a few steps - cheaper, in
    # the cost function, than fighting the moment. On the real robot that
    # reads as a leg the controller has stopped holding up: tau = -J^T 0 = 0,
    # the leg goes limp, the foot drops out of contact, and the gait falls
    # apart within one cycle. A small positive floor (MIT Cheetah uses 10 N
    # for a 9 kg robot; 3 N here is the same ~0.25 Mg) forbids the trick and
    # keeps every scheduled stance leg genuinely loaded.
    f_z_min: float = 3.0
    # Flat-foot contact geometry, in the foot's frame about its reference
    # point (the ankle projected onto the sole). The rectangle is the one the
    # four sole contacts in biped.xml span - heel 0.031 m behind the ankle,
    # toe 0.061 m ahead, 0.021 m either side - shrunk by `cop_margin`.
    #
    # The margin keeps the commanded CoP off the very edge. Every modelling
    # error in the SRBD shows up as a wrench slightly different from the one
    # asked for, and at the edge any error in the unloading direction lifts
    # the opposite corners: the foot rolls onto its edge, the contact
    # becomes a line, and the moment the QP was counting on is gone.
    foot_front: float = 0.061
    foot_back: float = 0.031
    foot_half_width: float = 0.021
    cop_margin: float = 0.8
    # Torsional friction, as a moment arm: |m_z'| <= mu_t f_z. Four corner
    # contacts at ~0.04 m from the reference point with mu = 0.8 could resist
    # up to ~0.035 f_z if nothing else used the friction, but the same
    # friction also carries the shear force, so only a third of that is
    # promised to the QP.
    mu_torsion: float = 0.012
    # Weights, in state order (see the module docstring).
    #
    # Attitude is weighted hardest: roll and pitch are the underactuated
    # directions. Pitch's 500 is the planar version's measured optimum (twice
    # the design doc's 250: it took |pitch| p95 from 0.250 to 0.226 rad, and
    # 900 made the robot fall because the QP then spent the whole friction
    # cone on attitude and had nothing left to steer with). Roll gets the
    # same. p_x and p_y have weight 0 - we regulate *velocity*, not absolute
    # position, so the robot is never asked to chase a position it has
    # already drifted from - and lateral sway over each stance foot is the
    # gait working, not an error. The *rate* weights are the touchy ones:
    # omega is the noisiest state the SRBD has (the legs swing, the trunk
    # does not follow rigidly), and weighting it hard turns that noise into
    # GRF commands.
    #
    # Yaw is weighted well below roll and pitch, and has to be. Each swing
    # leg carries angular momentum about the vertical (it is HIP_Y off the
    # centreline, and the stance leg sweeping back on the other side adds to
    # it), and a trunk with only 0.002 kg m^2 of yaw inertia pays for that
    # with a +/-0.1 rad counter-rotation every step - the motion humans cancel
    # by swinging their arms. The swing is internal and zero-mean, so fighting
    # it is wasted friction: at 0.5 m/s, raising the yaw weight from 150 to
    # 600 or 2000 made the robot fall.
    #       roll   pitch  yaw   px   py   pz     wx   wy   wz   vx    vy    vz   1
    Q: tuple = (500.0, 500.0, 150.0, 0.0, 0.0, 120.0, 8.0, 8.0, 3.0, 20.0, 20.0, 5.0, 0.0)
    R: float = 1e-3             # light regularisation, mostly for conditioning
    # Regularisation on the contact moments, per (N m)^2: R / (0.05 m)^2, so
    # a contact moment costs what the force making it on a 5 cm lever would.
    # The moments are two orders of magnitude smaller than the forces (a CoP
    # 3 cm off centre under 6 N is 0.18 N m), and at R itself they would be
    # all but free.
    R_moment: float = 0.4

    @property
    def nu(self) -> int:
        return NU_FOOT * self.n_feet


class ConvexMPC:
    """Rolling-horizon QP that turns a footstep/contact plan into contact wrenches."""

    def __init__(self, params: SRBDParams | None = None):
        self.p = params or SRBDParams()
        p = self.p
        self.nvar = p.horizon * p.nu
        self.ncon = p.horizon * p.n_feet * N_CON_FOOT
        self._I_body = np.diag(np.asarray(p.inertia, dtype=float))

        self._Qbar = np.tile(np.asarray(p.Q, dtype=float), p.horizon)
        r_foot = [p.R] * 3 + [p.R_moment] * 3
        self._Rbar = np.diag(np.tile(r_foot, p.n_feet * p.horizon))

        self._A = self._constraint_matrix()
        self._sense = np.zeros(self.ncon, dtype=np.intc)   # all rows two-sided inequalities
        self.status = "not-solved"
        self.solves = 0
        self.failures = 0

    # --- static QP structure ------------------------------------------------

    def _constraint_matrix(self) -> np.ndarray:
        """Block-diagonal contact-wrench constraints - constant for all time.

        Only the bounds `l`/`u` depend on the contact schedule, which is why
        this matrix never has to be rebuilt. The CoP and torsion rows are
        constant too, because the moment is expressed in the foot's own
        frame.
        """
        p = self.p
        rows_step = p.n_feet * N_CON_FOOT
        blk = np.zeros((rows_step, p.nu))
        k = p.cop_margin
        for i in range(p.n_feet):
            cx, cy, cz, mx, my, mz = NU_FOOT * i + np.arange(NU_FOOT)
            r = N_CON_FOOT * i
            blk[r + 0, [cx, cz]] = (1.0, -p.mu)
            blk[r + 1, [cx, cz]] = (-1.0, -p.mu)
            blk[r + 2, [cy, cz]] = (1.0, -p.mu)
            blk[r + 3, [cy, cz]] = (-1.0, -p.mu)
            blk[r + FZ_ROW, cz] = 1.0
            blk[r + 5, [mx, cz]] = (1.0, -k * p.foot_half_width)
            blk[r + 6, [mx, cz]] = (-1.0, -k * p.foot_half_width)
            blk[r + 7, [my, cz]] = (1.0, -k * p.foot_back)      # CoP behind the ankle
            blk[r + 8, [my, cz]] = (-1.0, -k * p.foot_front)    # CoP ahead of it
            blk[r + 9, [mz, cz]] = (1.0, -p.mu_torsion)
            blk[r + 10, [mz, cz]] = (-1.0, -p.mu_torsion)
        A = np.zeros((p.horizon * rows_step, self.nvar))
        for k in range(p.horizon):
            A[k * rows_step:(k + 1) * rows_step, k * p.nu:(k + 1) * p.nu] = blk
        return A

    # --- per-solve assembly -------------------------------------------------

    def _state_matrix(self, psi: float) -> np.ndarray:
        """A_d for one predicted step, linearised about heading `psi`."""
        p = self.p
        Ac = np.zeros((NX, NX))
        Ac[ROLL:YAW + 1, WX:WZ + 1] = rot_z(psi).T
        Ac[PX:PZ + 1, VX:VZ + 1] = np.eye(3)
        Ac[VZ, ONE] = -p.g
        return np.eye(NX) + Ac * p.dt

    def _input_matrix(self, r: np.ndarray, foot_yaw: np.ndarray, psi: float) -> np.ndarray:
        """B_d for one predicted step. `r` is (n_feet, 3): CoM -> foot
        reference point; `foot_yaw` (n_feet,) is each foot's heading."""
        p = self.p
        Rz = rot_z(psi)
        I_inv = Rz @ np.linalg.inv(self._I_body) @ Rz.T
        B = np.zeros((NX, p.nu))
        for i in range(p.n_feet):
            f = slice(NU_FOOT * i, NU_FOOT * i + 3)
            m = slice(NU_FOOT * i + 3, NU_FOOT * (i + 1))
            B[WX:WZ + 1, f] = I_inv @ skew(r[i])
            B[WX:WZ + 1, m] = I_inv @ rot_z(foot_yaw[i])
            B[VX:VZ + 1, f] = np.eye(3) / p.mass
        return B * p.dt

    def condense(self, feet_r: np.ndarray, feet_yaw: np.ndarray,
                 yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(A_qp, B_qp) for the footstep plan `feet_r` of shape (N, n_feet, 3)
        and foot headings `feet_yaw` of shape (N, n_feet).

        `yaw` (N,) is the heading each predicted step is linearised about.
        B_qp must be rebuilt every solve: `r_i` changes both because the CoM
        travels under the planted foot and because the feet swap.

        With a time-varying A_k the block-Toeplitz shortcut of the planar
        version no longer applies, so this uses the recursion
        `row_k = A_k row_(k-1) + [0 ... B_k ... 0]` - N matrix products
        instead of N^2/2.
        """
        p = self.p
        N, nu = p.horizon, p.nu
        Aqp = np.zeros((NX * N, NX))
        Bqp = np.zeros((NX * N, nu * N))
        A_prev, B_prev = np.eye(NX), np.zeros((NX, nu * N))
        for k in range(N):
            Ak = self._state_matrix(yaw[k])
            rows = slice(k * NX, (k + 1) * NX)
            Aqp[rows] = A_prev = Ak @ A_prev
            Bqp[rows] = Ak @ B_prev
            Bqp[rows, k * nu:(k + 1) * nu] = self._input_matrix(feet_r[k], feet_yaw[k], yaw[k])
            B_prev = Bqp[rows]
        return Aqp, Bqp

    def build(self, x0: np.ndarray, contact: np.ndarray, feet_r: np.ndarray,
              x_ref: np.ndarray, feet_yaw: np.ndarray | None = None):
        """Assemble (H, g, A, l, u) for the current plan.

        `contact` is (N, n_feet) in {0,1} and `x_ref` is (N, 13), both covering
        the predicted states x_1 ... x_N. `feet_yaw` (N, n_feet) is each
        foot's heading, default 0. The dynamics are linearised about the
        reference's yaw at each step (x_0's for the first).
        """
        p = self.p
        x_ref = np.asarray(x_ref, dtype=float)
        yaw = np.concatenate(([x0[YAW]], x_ref[:-1, YAW]))
        if feet_yaw is None:
            feet_yaw = np.zeros((p.horizon, p.n_feet))
        Aqp, Bqp = self.condense(np.asarray(feet_r, dtype=float),
                                 np.asarray(feet_yaw, dtype=float), yaw)
        err = Aqp @ np.asarray(x0, dtype=float) - x_ref.ravel()
        WB = self._Qbar[:, None] * Bqp
        H = 2.0 * (Bqp.T @ WB + self._Rbar)
        g = 2.0 * (Bqp.T @ (self._Qbar * err))
        # Symmetrise: H is symmetric analytically; rounding would otherwise
        # leave it slightly off, which a Cholesky-based solver notices.
        H = 0.5 * (H + H.T)

        lo = np.full(self.ncon, -np.inf)
        up = np.zeros(self.ncon)
        c = np.asarray(contact, dtype=float).ravel()     # (k, foot) order
        fz_rows = np.arange(p.horizon * p.n_feet) * N_CON_FOOT + FZ_ROW
        lo[fz_rows] = p.f_z_min * c                       # unilateral + loaded
        up[fz_rows] = p.f_z_max * c
        return H, g, self._A, lo, up

    # --- solve --------------------------------------------------------------

    def solve(self, x0: np.ndarray, contact: np.ndarray, feet_r: np.ndarray,
              x_ref: np.ndarray, feet_yaw: np.ndarray | None = None) -> np.ndarray:
        """Return the *first* step's contact wrenches, shaped (n_feet, 6):
        world-frame force, then foot-yaw-frame moment (`moment_to_world`).

        Rolling horizon: everything past the first step is discarded and the
        whole problem is re-solved on the next control tick with fresh state.
        """
        p = self.p
        H, g, A, lo, up = self.build(x0, contact, feet_r, x_ref, feet_yaw)
        # DAQP treats any bound at or beyond 1e30 as absent.
        x, _, flag, _ = daqp.solve(H, g, A, np.minimum(up, 1e30), np.maximum(lo, -1e30),
                                   self._sense)
        self.solves += 1
        self.status = "solved" if flag >= 1 else f"daqp-exitflag-{flag}"
        if flag < 1 or not np.all(np.isfinite(x)):
            self.failures += 1
            return self.fallback(contact[0])
        return np.asarray(x[:p.nu], dtype=float).reshape(p.n_feet, NU_FOOT)

    def fallback(self, contact_now: np.ndarray) -> np.ndarray:
        """Gravity compensation, used if the QP fails.

        Dropping to zero torque on a failed solve would make the robot
        collapse on a single bad tick, so the feet that are down simply share
        the body weight vertically, with no contact moment - the trivial
        feasible point of the same constraint set.
        """
        p = self.p
        f = np.zeros((p.n_feet, NU_FOOT))
        down = np.flatnonzero(np.asarray(contact_now) > 0.5)
        if down.size:
            f[down, 2] = min(p.f_z_max, p.mass * p.g / down.size)
        return f


def reference_trajectory(x0: np.ndarray, v_body: np.ndarray, z_des: float,
                         params: SRBDParams, yaw_des: float | None = None,
                         yaw_rate: float = 0.0) -> np.ndarray:
    """Steady cruise: walk at `v_body` = (forward, left) in the *heading*
    frame, hold z_des, torso level, turn at `yaw_rate` starting from
    `yaw_des` (default: the current heading).

    The velocity command is body-relative, so while turning the reference
    velocity rotates with the reference heading step by step - the robot is
    asked to walk an arc, not a straight line at a slowly changing angle.

    Covers x_1 ... x_N (the states the cost actually penalises). The p_x/p_y
    entries are carried along for completeness only - their weight is 0.
    """
    N, dt = params.horizon, params.dt
    yaw0 = float(x0[YAW]) if yaw_des is None else float(yaw_des)
    yaw = yaw0 + yaw_rate * dt * np.arange(1, N + 1)
    c, s = np.cos(yaw), np.sin(yaw)
    vx = c * v_body[0] - s * v_body[1]
    vy = s * v_body[0] + c * v_body[1]
    ref = np.zeros((N, NX))
    ref[:, YAW] = yaw
    ref[:, PX] = x0[PX] + np.cumsum(vx) * dt
    ref[:, PY] = x0[PY] + np.cumsum(vy) * dt
    ref[:, PZ] = z_des
    ref[:, WZ] = yaw_rate
    ref[:, VX] = vx
    ref[:, VY] = vy
    ref[:, ONE] = 1.0
    return ref
