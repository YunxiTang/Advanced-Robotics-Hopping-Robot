"""Whole-body inverse-dynamics QP between the convex MPC and the motors.

The small biped maps the MPC's contact wrenches to torques with `tau = -J^T w`
per stance leg, which is exact only for a massless leg on a still base. Its
legs are 18% of its mass and it gets away with it. Unitree G1's legs are 43%
of its 33 kg: `-J^T w` plus the leg's own gravity (`g1_controller`,
`use_wbc=False`) still walks it, but at 0.7 m/s the trunk pitches and rolls
four to ten times as much (p95 0.18 / 0.14 rad, against 0.04 / 0.01 here),
because nothing in it accounts for the leg's inertia, the base's
acceleration or the swinging leg.

This is the standard fix - the whole-body controller MPC humanoids put under
their MPC (MIT's WBIC, for one) - in its plainest form: one weighted QP over
the full rigid-body dynamics, re-solved every control tick.

    variables   x = [qdd (nv),  u_1 ... u_n (6 each)]     u_i as the MPC's:
                                                           world force, foot-yaw-frame moment
    subject to  M_b qdd + h_b = sum_i J_i,b^T T_i u_i      floating base, 6 rows (exact)
                each u_i inside the MPC's own friction / CoP / torsion rows,
                u_i = 0 for a foot that is not a loaded stance foot
    minimising  w       |u - u_mpc|^2                      stay with the plan
              + w_c     |J_i qdd + k_c J_i qd|^2           stance feet stay put
              + w_q_j   |qdd_j - qdd_j,des|^2              swing legs, waist, arms
              + eps     |qdd|^2

and the joint torques follow from the actuated rows of the same dynamics,

    tau = (M qdd + h - qfrc_passive - sum_i J_i^T T_i u_i)[actuated].

With the wrench weights as they are, `u` stays with the MPC's - within 0.2 N,
walking at 0.3 or 0.6 m/s - so this is in effect inverse dynamics:
the torques that make the ground push back with the MPC's wrench *while* the
swing legs, waist and arms follow their own acceleration targets, with every
coupling between them in the mass matrix accounted for. The CoM's
translation needs no task of its own: the floating-base rows tie it to the
total contact force exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import daqp
import mujoco
import numpy as np

from .mpc import FZ_ROW, NU_FOOT, rot_z


@dataclass
class WBCParams:
    # Weights on the wrench's deviation from the MPC's, per N^2 and
    # (N m)^2. Contact forces are ~10^2 N and joint accelerations ~10 rad/s^2,
    # so these keep the wrench within a fraction of a newton of the plan. The
    # vertical force is held closest: the CoP and torsion limits scale with
    # it, so a QP that could move it freely could widen its own limits by
    # loading the foot up (an earlier version with a pelvis-attitude task did
    # exactly that, and launched the robot).
    w_force: float = 1e-3
    w_fz: float = 5e-2
    w_moment: float = 1e-1
    w_contact: float = 100.0
    # the contact task's damping: J qdd = -k_c J qd pulls a sliding or
    # rocking stance foot back to rest
    k_contact: float = 20.0
    eps: float = 1e-6


class WholeBodyQP:
    """One contact-wrench-plus-acceleration QP per control tick (module docstring)."""

    def __init__(self, model, data, sole_sids, foot_block: np.ndarray,
                 f_z_min: float, f_z_max: float, params: WBCParams | None = None):
        """`sole_sids`: each foot's reference-point site; `foot_block`: one
        foot's rows of the MPC's constraint matrix (`ConvexMPC._A`)."""
        self.m, self.d = model, data
        self.p = params or WBCParams()
        self.sole_sids = list(sole_sids)
        self.nf = len(self.sole_sids)
        self.nv = model.nv
        self.nx = self.nv + NU_FOOT * self.nf
        self.blk = np.asarray(foot_block, dtype=float)
        self.f_z_min, self.f_z_max = f_z_min, f_z_max
        # actuator i drives DOF act_dof[i]
        self.act_dof = np.array([model.jnt_dofadr[model.actuator_trnid[i, 0]]
                                 for i in range(model.nu)])
        self._M = np.zeros((self.nv, self.nv))
        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))
        self.status = "not-solved"
        self.failures = 0

    def _contact_jacobian(self, i: int) -> np.ndarray:
        mujoco.mj_jacSite(self.m, self.d, self._jacp, self._jacr, self.sole_sids[i])
        return np.vstack((self._jacp, self._jacr))

    def solve(self, loaded, u_mpc: np.ndarray, foot_yaw, qdd_des: np.ndarray,
              w_joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint torques in actuator order, and the contact wrenches used.

        `loaded[i]`: foot i is a stance foot carrying load. `u_mpc` (nf, 6)
        is the MPC's wrench per foot, `foot_yaw` each foot's heading.
        `qdd_des` / `w_joint` (nv - 6 each) are the joint-space
        acceleration targets and their weights; 0 leaves a joint free (the
        stance legs', which the contact task and the dynamics determine).
        """
        m, d, p = self.m, self.d, self.p
        nv, nf = self.nv, self.nf
        mujoco.mj_fullM(m, d, self._M)
        M = self._M
        h = d.qfrc_bias - d.qfrc_passive
        qd = d.qvel

        # generalized force of each foot's wrench variables: J^T T
        JT = np.zeros((nv, NU_FOOT * nf))
        Js = []
        for i in range(nf):
            J = self._contact_jacobian(i)
            Js.append(J)
            T = np.eye(6)
            T[3:, 3:] = rot_z(foot_yaw[i])
            JT[:, NU_FOOT * i:NU_FOOT * (i + 1)] = J.T @ T

        H = np.zeros((self.nx, self.nx))
        g = np.zeros(self.nx)

        def add(A_rows, b, w, cols):
            """cost += w |A x[cols] - b|^2 (w scalar or per row)"""
            Aw = A_rows * (np.asarray(w, dtype=float)[:, None] if np.ndim(w) else w)
            H[np.ix_(cols, cols)] += 2.0 * A_rows.T @ Aw
            g[cols] -= 2.0 * Aw.T @ b

        qcols = np.arange(nv)
        ucols = nv + np.arange(NU_FOOT * nf)
        # stay with the MPC's wrenches
        w_u = np.tile([p.w_force, p.w_force, p.w_fz] + [p.w_moment] * 3, nf)
        add(np.eye(NU_FOOT * nf), np.asarray(u_mpc, dtype=float).ravel(), w_u, ucols)
        # joint-space tasks
        add(np.eye(nv)[6:], np.asarray(qdd_des, dtype=float), np.asarray(w_joint, dtype=float), qcols)
        # stance feet at rest
        for i in range(nf):
            if loaded[i]:
                add(Js[i], -p.k_contact * (Js[i] @ qd), p.w_contact, qcols)
        H[qcols, qcols] += 2.0 * p.eps
        H[ucols, ucols] += 2.0 * p.eps

        # constraints: floating-base dynamics (equality), then the wrench rows
        nb = self.blk.shape[0]
        A = np.zeros((6 + nb * nf, self.nx))
        lo = np.full(A.shape[0], -np.inf)
        up = np.zeros(A.shape[0])
        A[:6, :nv] = M[:6]
        A[:6, nv:] = -JT[:6]
        lo[:6] = up[:6] = -h[:6]
        for i in range(nf):
            r = 6 + nb * i
            A[r:r + nb, nv + NU_FOOT * i:nv + NU_FOOT * (i + 1)] = self.blk
            fz = r + FZ_ROW
            lo[fz], up[fz] = (self.f_z_min, self.f_z_max) if loaded[i] else (0.0, 0.0)
        sense = np.zeros(A.shape[0], dtype=np.intc)
        sense[:6] = 5
        x, _, flag, _ = daqp.solve(H, g, A, np.minimum(up, 1e30), np.maximum(lo, -1e30), sense)
        self.status = "solved" if flag >= 1 else f"daqp-exitflag-{flag}"
        if flag < 1 or not np.all(np.isfinite(x)):
            self.failures += 1
            return None, None
        qdd, u = x[:nv], x[nv:]
        tau_dof = M @ qdd + h - JT @ u
        return tau_dof[self.act_dof], u.reshape(nf, NU_FOOT)
