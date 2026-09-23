"""Staged validation of the convex-MPC controller (doc/convec_mpc.md, section 13).

Run it with::

    uv run python -m mujoco_sim.validate_mpc            # all stages, print a table
    uv run python -m mujoco_sim.validate_mpc --write    # ... and update doc/mpc_validation.md
    uv run python -m mujoco_sim.validate_mpc --stage 8  # just one
    uv run python -m mujoco_sim.validate_mpc --video result  # ... recording each stage to MP4

The plan is deliberately staged, and each stage has a numeric acceptance
criterion rather than a visual one: every failure this controller had during
development (the pitch inertia, the swing-damping eigenvalue, the attitude
feedback through the swing IK, the joint damping on the stance leg) looked
like "needs tuning" from a plot and was actually a modelling or numerical
error that a single scalar would have caught immediately.

Stage 1 and 2 are unit tests of the two things that are easiest to get
silently backwards - the QP's own force and moment balance, and the sign of
the `tau = -J^T w` mapping. They run in about a second and do not need a gait.
Stage 3 is new with the flat feet: stand still, which a point-foot biped
cannot do at all. Stages 4-7 are the planar version's plan carried over to
3-D; 8-10 exercise what only a 3-D robot can do - get pushed sideways, walk
sideways, and turn. 11-12 are the running gait: run steadily, and change
between walking and running on the move.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .controller import MPCGaitParams, MPCWalkController
from .mpc import NX, ONE, PZ, ConvexMPC, SRBDParams, moment_to_world, reference_trajectory
from .sim import MODEL_PATH, run
from .video import VideoRecorder

DOC_PATH = Path(__file__).resolve().parent.parent / "doc" / "mpc_validation.md"
FALL_Z = 0.2                  # a CoM below this is a fall


@dataclass
class Stage:
    name: str
    passed: bool
    metrics: list          # (label, value, criterion or None)
    note: str = ""


def _fmt(v) -> str:
    return v if isinstance(v, str) else f"{v:.4g}"


def _fell(a: dict) -> bool:
    return bool(a["z"].min() < FALL_Z)


def _yes_no(b: bool) -> str:
    return "YES" if b else "no"


# --- stage 1: the QP on its own ---------------------------------------------

def stage1_qp_selfcheck(video=None) -> Stage:
    """Standing: the QP must find the weight, and no net force or moment.

    If this fails, the SRBD matrices or the cost are wrong and nothing
    downstream is worth debugging. Two cases:

    - both feet down, split fore-aft (left ahead, right behind) as well as
      sitting at hip width, and turned to different headings, so every
      column of `r x f` and of the foot-frame moment's rotation is
      exercised;
    - **one** foot down, 2 cm to the side of and 1 cm behind the CoM. With
      a point foot that is a fall the QP cannot prevent. With a flat foot
      the CoM is over the sole, and the QP must hold it there with the
      contact moment alone - a CoP right under the CoM, which is inside the
      sole, so the answer must again be the weight and zero net moment.

    The horizontal-force bound is 2% of the weight rather than ~0, because
    in single support the exact answer is not quite the optimum: the
    regularisation prices the contact moment, and leaning the force a
    fraction of a newton (on the 0.36 m CoM height) buys some of that moment
    more cheaply than the ankle can. The net moment stays at zero.
    """
    p = SRBDParams()
    mpc = ConvexMPC(p)
    z = MPCGaitParams.z_des
    x0 = np.zeros(NX)
    x0[PZ], x0[ONE] = z, 1.0
    ref = reference_trajectory(x0, np.zeros(2), z, p)
    weight = p.mass * p.g
    metrics, ok = [], True
    cases = (("double support", np.ones(2),
              [(0.06, 0.068, -z), (-0.06, -0.068, -z)], (0.2, -0.3)),
             ("single support", np.array([1.0, 0.0]),
              [(-0.01, 0.02, -z), (0.0, -0.1, -z)], (0.1, 0.0)))
    for label, c, feet, yaws in cases:
        contact = np.tile(c, (p.horizon, 1))
        r = np.tile(np.asarray(feet, float), (p.horizon, 1, 1))
        psi = np.tile(yaws, (p.horizon, 1))
        w = mpc.solve(x0, contact, r, ref, psi)
        total = w[:, :3].sum(axis=0)
        moment = sum(np.cross(r[0, i], w[i, :3]) + moment_to_world(w[i, 3:], yaws[i])
                     for i in range(2))
        ok &= (abs(total[2] - weight) < 0.1 * weight
               and np.hypot(*total[:2]) < 0.02 * weight
               and np.linalg.norm(moment) < 0.05 and mpc.status == "solved")
        metrics += [(f"{label}: sum f_z", total[2], f"= Mg = {weight:.2f} N +/-10%"),
                    (f"{label}: abs sum f_xy", float(np.hypot(*total[:2])),
                     f"~ 0 (below 2% Mg = {0.02 * weight:.2f} N)"),
                    (f"{label}: abs net moment", float(np.linalg.norm(moment)),
                     "~ 0 (below 0.05 N m)"),
                    (f"{label}: solver status", mpc.status, "solved")]
    return Stage("1. QP self-check (standing)", bool(ok), metrics)


# --- stage 2: the J^T mapping ------------------------------------------------

def stage2_jacobian_pinned(settle: float = 0.5, video=None) -> Stage:
    """Pin the torso, command `tau = -J^T w`, and read the wrench back off
    MuJoCo's contact solver.

    This isolates the sign convention of `-J^T w` - the thing a sign error
    would break - from everything else, balance included: pinning the torso
    leaves exactly one question, whether the ground pushes back on each
    foot with the wrench that was asked for.

    Gravity is switched off so the legs' own weight does not show up in the
    contact force, and the two feet are given deliberately different, fully
    3-D wrenches - force inside the friction cone, CoP inside the sole,
    torsion inside the torsional limit - so that a swapped axis, a swapped
    leg, a flipped sign or a moment taken about the wrong point would each
    show up. The measured moment is summed over the four sole contacts about
    the foot's reference point.
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.gravity[:] = 0.0
    # The torso is pinned by making it immovable - a million times its own
    # mass and inertia - and, for good measure, putting its free-joint state
    # back after every step. Both halves matter:
    #   - a weld constraint is soft enough that 14 N of leg thrust lifts the
    #     pelvis 8 cm and the feet leave the floor;
    #   - resetting the state alone is not a pin either: inside each step
    #     the legs still accelerate a 1 kg torso at ~13 m/s^2, and through
    #     the mass matrix that loads the legs as if gravity were on - the
    #     measured f_z came out 0.6-0.7 N high on both feet.
    # The model is otherwise the one the controller walks on, so this
    # checks the controller's own code path.
    torso = model.body("torso").id
    model.body_mass[torso] *= 1e6
    model.body_inertia[torso] *= 1e6
    data = mujoco.MjData(model)
    # Recompute the model constants derived from the masses - among them the
    # trackcom camera offsets, which otherwise point the video camera at the
    # sky.
    mujoco.mj_setConst(model, data)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("mpc_stand").id)
    root = model.joint("root")
    qa, va = model.jnt_qposadr[root.id], model.jnt_dofadr[root.id]
    base_q = data.qpos[qa:qa + 7].copy()
    ctrl = MPCWalkController(model, data, MPCGaitParams())
    # (f_x, f_y, f_z, m_x, m_y, m_z), world axes; the feet point straight
    # ahead, so the foot frame and the world agree. Left: CoP 1.2 cm to the
    # left and 1.7 cm ahead of the ankle; right: 1 cm right, 1 cm behind.
    w_cmd = np.array([[1.5, -1.0, 6.0, 0.07, -0.10, 0.03],
                      [-1.0, 2.0, 8.0, -0.08, 0.08, -0.04]])
    floor = model.geom("floor").id

    rec = VideoRecorder(model, video) if video is not None else None
    while data.time < settle:
        mujoco.mj_forward(model, data)
        tau = []
        for j in range(2):
            # the stance leg's own path: -J^T w plus the joints' damping fed
            # forward (MPCWalkController.control)
            dofs = ctrl._leg_dofs[j]
            tau += list(-ctrl._foot_jacobian(j).T @ w_cmd[j]
                        + model.dof_damping[dofs] * data.qvel[dofs])
        data.ctrl[:len(tau)] = tau
        mujoco.mj_step(model, data)
        data.qpos[qa:qa + 7] = base_q
        data.qvel[va:va + 6] = 0.0
        if rec is not None:
            rec.maybe_capture(data)
    if rec is not None:
        rec.close()

    mujoco.mj_forward(model, data)
    w_meas = np.zeros((2, 6))
    n_pts = [0, 0]
    c6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if floor not in (c.geom1, c.geom2):
            continue
        g = c.geom2 if c.geom1 == floor else c.geom1
        for j, bid in enumerate(ctrl._foot_bid):
            if model.geom_bodyid[g] != bid:
                continue
            mujoco.mj_contactForce(model, data, i, c6)
            frame = c.frame.reshape(3, 3)          # rows: normal, t1, t2
            f_world = frame.T @ c6[:3]             # force on geom2 from geom1
            f = f_world if c.geom2 == g else -f_world
            arm = c.pos - data.site_xpos[ctrl._sole_sid[j]]
            w_meas[j] += np.concatenate((f, np.cross(arm, f)))
            n_pts[j] += 1
    f_err = np.abs(w_meas[:, :3] - w_cmd[:, :3]).max()
    m_err = np.abs(w_meas[:, 3:] - w_cmd[:, 3:]).max()
    f_tol = 0.02 * np.abs(w_cmd[:, :3]).max()
    m_tol = 0.05 * np.abs(w_cmd[:, 3:]).max()
    ok = f_err < f_tol and m_err < m_tol and min(n_pts) == 4
    return Stage("2. J^T mapping (torso pinned)", ok, [
        ("commanded w_left", str(w_cmd[0]), "-"),
        ("measured w_left", str(w_meas[0].round(3)), "-"),
        ("commanded w_right", str(w_cmd[1]), "-"),
        ("measured w_right", str(w_meas[1].round(3)), "-"),
        ("sole contacts left / right", f"{n_pts[0]} / {n_pts[1]}", "4 / 4 (foot flat)"),
        ("max abs force error", f_err, f"< {f_tol:.2f} N (2% of the largest component)"),
        ("max abs moment error", m_err, f"< {m_tol:.4f} N m (5% of the largest component)"),
    ], "A sign error here shows up as a force of the wrong sign, or no contact at all; "
       "a moment about the wrong point shows up as a moment error proportional to the "
       "force. Without the joint-damping feedforward the torsion comes out 25-33% "
       "short: MuJoCo's soft friction lets the foot creep in yaw at 0.05 rad/s under "
       "a steady twist, and the hip yaw's 0.2 damping eats 0.01 N m of it. "
       "(With the old point foot, the same test caught a Jacobian taken at the "
       "foot sphere's centre instead of its contact point: 11% too little horizontal "
       "force.)")


# --- stages 3-12: closed-loop gaits ------------------------------------------

def _gait_metrics(log, settle: float) -> dict:
    a = log.as_arrays()
    k = a["t"] > settle
    roll, pitch = np.abs(a["roll"][k]), np.abs(a["pitch"][k])
    t = a["t"][k]
    return {
        "a": a,
        "fwd": float(a["v_fwd"][k].mean()),
        "lat": float(a["v_lat"][k].mean()),
        "yaw_rate": float((a["yaw"][k][-1] - a["yaw"][k][0]) / (t[-1] - t[0])),
        "yaw_drift": float(np.abs(a["yaw"][k] - a["yaw"][0]).max()),
        "z_mean": float(a["z"][k].mean()),
        "z_range": float(a["z"][k].max() - a["z"][k].min()),
        "roll_p95": float(np.percentile(roll, 95)),
        "pitch_p95": float(np.percentile(pitch, 95)),
        "roll_max": float(roll.max()),
        "pitch_max": float(pitch.max()),
        "fell": _fell(a),
    }


def _attitude_rows(m: dict) -> list:
    return [("p95 abs roll", m["roll_p95"], "< 0.15 rad"),
            ("p95 abs pitch", m["pitch_p95"], "< 0.15 rad")]


def _attitude_ok(m: dict) -> bool:
    return m["roll_p95"] < 0.15 and m["pitch_p95"] < 0.15


def stage3_stand(duration: float = 10.0, video=None) -> Stage:
    """Stand still on both feet - no stepping at all.

    The test the point-foot robot could not have: two point feet span only
    a line segment, so it was an inverted pendulum about that line and had
    to keep stepping. With flat feet the support polygon is the hull of the
    two soles, and the MPC holds the CoM over it with the feet's contact
    moments. `duty = 1` makes every foot a permanent stance foot, so this is
    the same controller with the gait clock switched off. A 2 N sideways
    push at t = 5 s checks that the stance is actually being regulated, not
    just resting on friction.
    """
    log = run(duration=duration, params=MPCGaitParams(Vs=0.0, duty=1.0),
              disturbance=(5.0, 5.1, (0.0, 2.0)), video=video)
    a = log.as_arrays()
    k = a["t"] > 1.0
    drift = float(np.hypot(a["x"][-1] - a["x"][0], a["y"][-1] - a["y"][0]))
    roll, pitch = np.abs(a["roll"][k]), np.abs(a["pitch"][k])
    fell = _fell(a)
    ok = (not fell) and drift < 0.02 and roll.max() < 0.05 and pitch.max() < 0.05
    return Stage(f"3. Stand still on flat feet ({duration:.0f} s, 2 N side push)", ok, [
        ("fell", _yes_no(fell), "no"),
        ("CoM drift, start to end", drift, "< 0.02 m"),
        ("max abs roll", float(roll.max()), "< 0.05 rad"),
        ("max abs pitch", float(pitch.max()), "< 0.05 rad"),
        ("CoM height range", float(a["z"][k].max() - a["z"][k].min()), "(reported)"),
    ], "Impossible with point feet (their support polygon is a line segment).")


def stage4_march(duration: float = 10.0, video=None) -> Stage:
    m = _gait_metrics(run(duration=duration, params=MPCGaitParams(Vs=0.0), video=video),
                      settle=2.0)
    drift = float(np.hypot(m["a"]["x"][-1] - m["a"]["x"][0],
                           m["a"]["y"][-1] - m["a"]["y"][0]))
    ok = (not m["fell"]) and _attitude_ok(m) and m["z_range"] < 0.03 and drift < 0.3
    return Stage("4. March in place (v_des = 0, 10 s)", ok, [
        ("fell", _yes_no(m["fell"]), "no"),
        *_attitude_rows(m),
        ("CoM height range", m["z_range"], "< 0.03 m"),
        ("horizontal drift", drift, "< 0.3 m"),
        ("heading drift", m["yaw_drift"], "(reported)"),
    ])


def stage5_walk(v_des: float = 0.5, duration: float = 30.0, video=None) -> Stage:
    m = _gait_metrics(run(duration=duration, params=MPCGaitParams(Vs=v_des), video=video),
                      settle=10.0)
    err = abs(m["fwd"] - v_des) / v_des
    ok = (not m["fell"]) and err < 0.15 and _attitude_ok(m) and m["yaw_drift"] < 0.3
    return Stage(f"5. Steady walk (v_des = {v_des} m/s, {duration:.0f} s)", ok, [
        ("fell", _yes_no(m["fell"]), "no"),
        ("mean forward speed", m["fwd"], f"= {v_des} m/s +/-15%"),
        ("speed error", f"{100 * err:.1f}%", "< 15%"),
        ("mean sideways speed", m["lat"], "(reported)"),
        *_attitude_rows(m),
        ("max abs roll / pitch", f"{m['roll_max']:.3f} / {m['pitch_max']:.3f}", "(reported)"),
        ("heading drift", m["yaw_drift"], "< 0.3 rad"),
        ("mean CoM height", m["z_mean"], f"~ {MPCGaitParams.z_des} m"),
    ], "The acceptance bar is on the 95th percentile of the attitude; the maximum is a "
       "per-touchdown transient and is reported for information.")


def stage6_tracking(duration: float = 30.0, video=None) -> Stage:
    """Step the speed target and check it is followed."""
    lo, hi = 0.3, 0.7
    steps = [(0.0, lo), (10.0, hi), (20.0, lo)]

    def schedule(t: float) -> float:
        v = steps[0][1]
        for t0, val in steps:
            if t >= t0:
                v = val
        return v

    log = run(duration=duration, params=MPCGaitParams(Vs=lo), v_schedule=schedule,
              video=video)
    a = log.as_arrays()
    seg = []
    for t0, val in steps:
        k = (a["t"] > t0 + 6.0) & (a["t"] <= t0 + 10.0)
        seg.append((val, float(a["v_fwd"][k].mean())))
    fell = _fell(a)
    errs = [abs(m - v) / v for v, m in seg]
    ok = (not fell) and max(errs) < 0.2
    return Stage(f"6. Velocity tracking ({lo} -> {hi} -> {lo} m/s)", ok, [
        ("fell", _yes_no(fell), "no"),
        *[(f"segment {i + 1}: target {v} m/s", m, "+/-20%")
          for i, (v, m) in enumerate(seg)],
    ], "The planar version stepped to 0.8 m/s here. The 3-D robot holds 0.8 m/s for "
       "10-15 s and then trips, so the high step is 0.7 m/s - see the README.")


def _push(force, label: str, duration: float = 20.0, video=None) -> Stage:
    t0, t1 = 8.0, 8.1
    log = run(duration=duration, params=MPCGaitParams(Vs=0.3),
              disturbance=(t0, t1, force), video=video)
    a = log.as_arrays()
    before = (a["t"] > 4.0) & (a["t"] <= t0)
    during = (a["t"] > t0) & (a["t"] < t0 + 2.0)
    after = a["t"] > t0 + 5.0
    fell = _fell(a)
    v = np.stack([a["dx"], a["dy"]], axis=1)
    recovered = float(np.linalg.norm(v[after].mean(0) - v[before].mean(0)))
    ok = (not fell) and recovered < 0.1
    peak = float(np.linalg.norm(v[during] - v[before].mean(0), axis=1).max())
    return Stage(f"{label} ({np.linalg.norm(force):.0f} N for {1000 * (t1 - t0):.0f} ms)", ok, [
        ("fell", _yes_no(fell), "no"),
        ("peak velocity change", peak, "-"),
        ("velocity error after recovery", recovered, "< 0.1 m/s"),
        ("max abs roll during", float(np.abs(a["roll"][during]).max()), "-"),
        ("max abs pitch during", float(np.abs(a["pitch"][during]).max()), "-"),
    ])


def stage7_push_forward(force: float = 5.0, video=None) -> Stage:
    return _push((force, 0.0), "7. Push recovery, forward", video=video)


def stage8_push_sideways(force: float = 5.0, video=None) -> Stage:
    """The push the planar robot could not receive: straight into the side."""
    return _push((0.0, force), "8. Push recovery, sideways", video=video)


def stage9_sidestep(v_lat: float = 0.2, duration: float = 20.0, video=None) -> Stage:
    m = _gait_metrics(run(duration=duration, params=MPCGaitParams(Vs=0.0, Vy=v_lat),
                          video=video),
                      settle=8.0)
    err = abs(m["lat"] - v_lat) / v_lat
    ok = (not m["fell"]) and err < 0.2 and _attitude_ok(m) and m["yaw_drift"] < 0.3
    return Stage(f"9. Sidestep (v_left = {v_lat} m/s, {duration:.0f} s)", ok, [
        ("fell", _yes_no(m["fell"]), "no"),
        ("mean sideways speed", m["lat"], f"= {v_lat} m/s +/-20%"),
        ("mean forward speed", m["fwd"], "(reported)"),
        *_attitude_rows(m),
        ("heading drift", m["yaw_drift"], "< 0.3 rad"),
    ])


def stage10_turn(v_des: float = 0.3, yaw_rate: float = 0.5,
                duration: float = 20.0, video=None) -> Stage:
    m = _gait_metrics(run(duration=duration,
                          params=MPCGaitParams(Vs=v_des, yaw_rate=yaw_rate),
                          video=video),
                      settle=5.0)
    err = abs(m["yaw_rate"] - yaw_rate) / yaw_rate
    v_err = abs(m["fwd"] - v_des) / v_des
    ok = (not m["fell"]) and err < 0.1 and v_err < 0.15 and _attitude_ok(m)
    return Stage(f"10. Turn while walking ({v_des} m/s, {yaw_rate} rad/s, {duration:.0f} s)", ok, [
        ("fell", _yes_no(m["fell"]), "no"),
        ("mean turn rate", m["yaw_rate"], f"= {yaw_rate} rad/s +/-10%"),
        ("mean forward speed", m["fwd"], f"= {v_des} m/s +/-15%"),
        *_attitude_rows(m),
    ], "A flat foot resists spinning, so the trunk turns on the stance legs' hip-yaw "
       "joints while the feet stay put; each new foot is planted at the heading "
       "set-point.")


def _flight(a: dict, k) -> float:
    """Fraction of the time in `k` with both feet off the ground."""
    return float(1.0 - a["stance"][k].mean())


def stage11_run(v_des: float = 1.0, duration: float = 20.0, video=None) -> Stage:
    """Run: the same controller with the gait clock's duty at 0.35.

    Each foot is down for 35% of the cycle, so twice a cycle both are in
    the air; the nominal airborne fraction is 1 - 2 x 0.35 = 30%. The speed
    is 1.0 m/s, which the walking gait cannot reach (it tops out at 0.7).
    """
    p = MPCGaitParams(Vs=v_des, gait="run")
    m = _gait_metrics(run(duration=duration, params=p, video=video), settle=8.0)
    k = m["a"]["t"] > 8.0
    flight = _flight(m["a"], k)
    nominal = 1.0 - 2.0 * p.run_duty
    err = abs(m["fwd"] - v_des) / v_des
    ok = ((not m["fell"]) and err < 0.1 and _attitude_ok(m) and m["yaw_drift"] < 0.3
          and flight > 0.8 * nominal)
    return Stage(f"11. Run (v_des = {v_des} m/s, duty {p.run_duty}, {duration:.0f} s)", ok, [
        ("fell", _yes_no(m["fell"]), "no"),
        ("time with both feet off the ground", f"{100 * flight:.1f}%",
         f"> {80 * nominal:.0f}% (nominal {100 * nominal:.0f}%)"),
        ("mean forward speed", m["fwd"], f"= {v_des} m/s +/-10%"),
        *_attitude_rows(m),
        ("heading drift", m["yaw_drift"], "< 0.3 rad"),
        ("CoM height range", m["z_range"], "(reported)"),
    ], "The flight fraction is measured off MuJoCo's contact list, not the schedule: "
       "it is the time neither foot has any sole contact touching the floor.")


def stage12_gait_change(duration: float = 25.0, video=None) -> Stage:
    """Walk -> run -> walk on the move, chosen by the controller.

    The command ramps 0.5 -> 1.0 m/s over 5-8 s and back over 15-18 s, with
    `gait = "auto"`: the controller starts running when the command passes
    0.7 m/s and walks again below 0.6, slewing the duty between the two
    gaits. Each hold segment must show the right gait (flight or no flight)
    at the right speed.
    """
    lo, hi = 0.5, 1.0

    def schedule(t: float) -> float:
        if t < 5.0:
            return lo
        if t < 8.0:
            return lo + (hi - lo) * (t - 5.0) / 3.0
        if t < 15.0:
            return hi
        if t < 18.0:
            return hi - (hi - lo) * (t - 15.0) / 3.0
        return lo

    log = run(duration=duration, params=MPCGaitParams(Vs=lo), v_schedule=schedule,
              video=video)
    a = log.as_arrays()
    fell = _fell(a)
    segs = [("walk", lo, 2.0, 5.0), ("run", hi, 10.0, 15.0), ("walk", lo, 21.0, 25.0)]
    metrics, ok = [("fell", _yes_no(fell), "no")], not fell
    for i, (gait, v, t0, t1) in enumerate(segs):
        k = (a["t"] > t0) & (a["t"] <= t1)
        speed, flight = float(a["v_fwd"][k].mean()), _flight(a, k)
        want_flight = gait == "run"
        seg_ok = (abs(speed - v) / v < 0.1
                  and (flight > 0.2 if want_flight else flight < 0.02))
        ok &= seg_ok
        metrics += [(f"{t0:.0f}-{t1:.0f} s ({gait}): speed", speed, f"= {v} m/s +/-10%"),
                    (f"{t0:.0f}-{t1:.0f} s ({gait}): airborne", f"{100 * flight:.1f}%",
                     "> 20%" if want_flight else "< 2%")]
    trans = (a["t"] > 5.0) & (a["t"] <= 21.0)
    metrics += [("max abs pitch, 5-21 s", float(np.abs(a["pitch"][trans]).max()), "(reported)"),
                ("max abs roll, 5-21 s", float(np.abs(a["roll"][trans]).max()), "(reported)")]
    return Stage("12. Walk -> run -> walk (0.5 -> 1.0 -> 0.5 m/s, auto gait)", bool(ok),
                 metrics,
                 "The gait changes are the worst moments of the run: the integral trims "
                 "were wound up for the old gait, so the trunk dips ~2 cm entering the "
                 "run and pitches ~0.1 rad leaving it.")


STAGES = (stage1_qp_selfcheck, stage2_jacobian_pinned, stage3_stand, stage4_march,
          stage5_walk, stage6_tracking, stage7_push_forward, stage8_push_sideways,
          stage9_sidestep, stage10_turn, stage11_run, stage12_gait_change)


def report(stages: list) -> str:
    """Render the results as the markdown table in `doc/mpc_validation.md`."""
    lines = ["# Convex-MPC validation results (3-D biped)", "",
             "Generated by `uv run python -m mujoco_sim.validate_mpc --write`.",
             "Stages 1-2 and 4-7 follow section 13 of `convec_mpc.md`, carried over to "
             "3-D; 8-10 are new with the 3-D robot (section 17), 3 with the flat "
             "feet (section 18), and 11-12 with the running gait (section 19).", ""]
    lines += ["| stage | result |", "|---|---|"]
    for s in stages:
        lines.append(f"| {s.name} | {'PASS' if s.passed else 'FAIL'} |")
    lines.append("")
    for s in stages:
        lines += [f"## {s.name} — {'PASS' if s.passed else 'FAIL'}", "",
                  "| metric | value | criterion |", "|---|---|---|"]
        for label, value, crit in s.metrics:
            lines.append(f"| {label} | {_fmt(value)} | {crit or '-'} |")
        if s.note:
            lines += ["", s.note]
        lines.append("")
    return "\n".join(lines)


# Video file stem per stage. Stage 1 is a standalone QP solve with no
# simulation, so it has nothing to film.
VIDEO_NAMES = {stage2_jacobian_pinned: "stage2_jacobian_pinned",
               stage3_stand: "stage3_stand_still",
               stage4_march: "stage4_march_in_place",
               stage5_walk: "stage5_walk_0.5mps",
               stage6_tracking: "stage6_speed_tracking",
               stage7_push_forward: "stage7_push_forward",
               stage8_push_sideways: "stage8_push_sideways",
               stage9_sidestep: "stage9_sidestep",
               stage10_turn: "stage10_turn",
               stage11_run: "stage11_run_1.0mps",
               stage12_gait_change: "stage12_walk_run_walk"}


def _run_stage(job) -> Stage:
    fn, video_dir = job
    if video_dir is None or fn not in VIDEO_NAMES:
        return fn()
    return fn(video=Path(video_dir) / f"{VIDEO_NAMES[fn]}.mp4")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help=f"write the results to {DOC_PATH}")
    ap.add_argument("--stage", type=int, default=None,
                    help=f"run a single stage (1-{len(STAGES)})")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run stages in parallel processes (they are independent)")
    ap.add_argument("--video", type=str, default=None, metavar="DIR",
                    help="record each simulated stage from the chase camera "
                         "to DIR/stageN_*.mp4")
    args = ap.parse_args()

    todo = STAGES if args.stage is None else (STAGES[args.stage - 1],)
    jobs = [(fn, args.video) for fn in todo]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.jobs) as ex:
            results = list(ex.map(_run_stage, jobs))
    else:
        results = [_run_stage(j) for j in jobs]
    for s in results:
        print(f"\n{'PASS' if s.passed else 'FAIL'}  {s.name}")
        for label, value, crit in s.metrics:
            print(f"      {label:<32} {_fmt(value):>14}   {crit or ''}")

    print(f"\n{sum(s.passed for s in results)}/{len(results)} stages passed")
    if args.write:
        DOC_PATH.write_text(report(results))
        print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
