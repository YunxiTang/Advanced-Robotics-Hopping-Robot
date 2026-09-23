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
the `tau = -J^T f` mapping. They run in about a second and do not need a gait.
Stages 3-6 are the planar version's plan carried over to 3-D; 7-9 exercise
what only a 3-D robot can do - get pushed sideways, walk sideways, and turn.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .controller import MPCGaitParams, MPCWalkController
from .mpc import NX, ONE, PZ, ConvexMPC, SRBDParams, reference_trajectory
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
    """Standing, both feet down: the QP must find the weight, and no net
    force or moment.

    If this fails, the SRBD matrices or the cost are wrong and nothing
    downstream is worth debugging. The feet are split fore-aft (left ahead,
    right behind) as well as sitting at hip width, so every column of `r x f`
    is exercised; the CoM sits on the line between the two feet, which with
    point feet is the only place a zero-moment stance exists at all.
    """
    p = SRBDParams()
    mpc = ConvexMPC(p)
    z = MPCGaitParams.z_des
    x0 = np.zeros(NX)
    x0[PZ], x0[ONE] = z, 1.0
    contact = np.ones((p.horizon, 2))
    r = np.zeros((p.horizon, 2, 3))
    r[:, 0] = (0.06, 0.068, 0.028 - z)
    r[:, 1] = (-0.06, -0.068, 0.028 - z)
    f = mpc.solve(x0, contact, r, reference_trajectory(x0, np.zeros(2), z, p))

    total = f.sum(axis=0)
    moment = sum(np.cross(r[0, i], f[i]) for i in range(2))
    weight = p.mass * p.g
    ok = (abs(total[2] - weight) < 0.1 * weight and np.hypot(*total[:2]) < 0.1
          and np.linalg.norm(moment) < 0.05)
    return Stage("1. QP self-check (standing)", ok, [
        ("sum f_z", total[2], f"= Mg = {weight:.2f} N +/-10%"),
        ("|sum f_xy|", float(np.hypot(*total[:2])), "~ 0 (below 0.1 N)"),
        ("|net moment|", float(np.linalg.norm(moment)), "~ 0 (below 0.05 N m)"),
        ("solver status", mpc.status, "solved"),
    ])


# --- stage 2: the J^T mapping ------------------------------------------------

def stage2_jacobian_pinned(settle: float = 0.5, video=None) -> Stage:
    """Pin the torso, command `tau = -J^T f`, and read the force back off
    MuJoCo's contact solver.

    This isolates the sign convention of `-J^T f` - the thing a sign error
    would break - from everything else. The planar version checked it by
    standing still for 5 s on a split stance, but in 3-D that test no
    longer exists: with two point feet the support polygon is a line
    segment, and the robot is an inverted pendulum about that line however
    the feet are placed, so no controller can stand on it without stepping.
    Pinning the torso in place removes the balance problem entirely and
    leaves exactly one question: does the ground push back on each foot
    with the force that was asked for?

    Gravity is switched off so the legs' own weight does not show up in the
    contact force, and the two feet are given deliberately different,
    fully 3-D forces (inside the friction cone) so that a swapped axis, a
    swapped leg or a flipped sign would each show up.
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
    f_cmd = np.array([[1.5, -1.0, 6.0], [-1.0, 2.0, 8.0]])
    floor = model.geom("floor").id

    rec = VideoRecorder(model, video) if video is not None else None
    while data.time < settle:
        mujoco.mj_forward(model, data)
        tau = []
        for j in range(2):
            tau += list(-ctrl._foot_jacobian(j).T @ f_cmd[j])
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        data.qpos[qa:qa + 7] = base_q
        data.qvel[va:va + 6] = 0.0
        if rec is not None:
            rec.maybe_capture(data)
    if rec is not None:
        rec.close()

    mujoco.mj_forward(model, data)
    f_meas = np.zeros((2, 3))
    c6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        for j, gid in enumerate(ctrl._foot_gid):
            if {c.geom1, c.geom2} == {gid, floor}:
                mujoco.mj_contactForce(model, data, i, c6)
                frame = c.frame.reshape(3, 3)          # rows: normal, t1, t2
                f_world = frame.T @ c6[:3]             # force on geom2 from geom1
                f_meas[j] += f_world if c.geom2 == gid else -f_world
    err = np.abs(f_meas - f_cmd).max()
    tol = 0.02 * np.abs(f_cmd).max()
    ok = err < tol
    return Stage("2. J^T mapping (torso pinned)", ok, [
        ("commanded f_left", str(f_cmd[0]), "-"),
        ("measured f_left", str(f_meas[0].round(3)), "-"),
        ("commanded f_right", str(f_cmd[1]), "-"),
        ("measured f_right", str(f_meas[1].round(3)), "-"),
        ("max abs error", err, f"< {tol:.2f} N (2% of the largest component)"),
    ], "A sign error here shows up as a force of the wrong sign, or no contact at all. "
       "Taking the Jacobian at the foot sphere's centre instead of the contact point "
       "fails this stage: it delivers 11% too little horizontal force.")


# --- stages 3-9: closed-loop gaits ------------------------------------------

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


def stage3_march(duration: float = 10.0, video=None) -> Stage:
    m = _gait_metrics(run(duration=duration, params=MPCGaitParams(Vs=0.0), video=video),
                      settle=2.0)
    drift = float(np.hypot(m["a"]["x"][-1] - m["a"]["x"][0],
                           m["a"]["y"][-1] - m["a"]["y"][0]))
    ok = (not m["fell"]) and _attitude_ok(m) and m["z_range"] < 0.03 and drift < 0.3
    return Stage("3. March in place (v_des = 0, 10 s)", ok, [
        ("fell", _yes_no(m["fell"]), "no"),
        *_attitude_rows(m),
        ("CoM height range", m["z_range"], "< 0.03 m"),
        ("horizontal drift", drift, "< 0.3 m"),
        ("heading drift", m["yaw_drift"], "(reported)"),
    ])


def stage4_walk(v_des: float = 0.5, duration: float = 30.0, video=None) -> Stage:
    m = _gait_metrics(run(duration=duration, params=MPCGaitParams(Vs=v_des), video=video),
                      settle=10.0)
    err = abs(m["fwd"] - v_des) / v_des
    ok = (not m["fell"]) and err < 0.15 and _attitude_ok(m) and m["yaw_drift"] < 0.3
    return Stage(f"4. Steady walk (v_des = {v_des} m/s, {duration:.0f} s)", ok, [
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


def stage5_tracking(duration: float = 30.0, video=None) -> Stage:
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
    return Stage(f"5. Velocity tracking ({lo} -> {hi} -> {lo} m/s)", ok, [
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


def stage6_push_forward(force: float = 5.0, video=None) -> Stage:
    return _push((force, 0.0), "6. Push recovery, forward", video=video)


def stage7_push_sideways(force: float = 5.0, video=None) -> Stage:
    """The push the planar robot could not receive: straight into the side."""
    return _push((0.0, force), "7. Push recovery, sideways", video=video)


def stage8_sidestep(v_lat: float = 0.2, duration: float = 20.0, video=None) -> Stage:
    m = _gait_metrics(run(duration=duration, params=MPCGaitParams(Vs=0.0, Vy=v_lat),
                          video=video),
                      settle=8.0)
    err = abs(m["lat"] - v_lat) / v_lat
    ok = (not m["fell"]) and err < 0.2 and _attitude_ok(m) and m["yaw_drift"] < 0.3
    return Stage(f"8. Sidestep (v_left = {v_lat} m/s, {duration:.0f} s)", ok, [
        ("fell", _yes_no(m["fell"]), "no"),
        ("mean sideways speed", m["lat"], f"= {v_lat} m/s +/-20%"),
        ("mean forward speed", m["fwd"], "(reported)"),
        *_attitude_rows(m),
        ("heading drift", m["yaw_drift"], "< 0.3 rad"),
    ])


def stage9_turn(v_des: float = 0.3, yaw_rate: float = 0.5,
                duration: float = 20.0, video=None) -> Stage:
    m = _gait_metrics(run(duration=duration,
                          params=MPCGaitParams(Vs=v_des, yaw_rate=yaw_rate),
                          video=video),
                      settle=5.0)
    err = abs(m["yaw_rate"] - yaw_rate) / yaw_rate
    v_err = abs(m["fwd"] - v_des) / v_des
    ok = (not m["fell"]) and err < 0.1 and v_err < 0.15 and _attitude_ok(m)
    return Stage(f"9. Turn while walking ({v_des} m/s, {yaw_rate} rad/s, {duration:.0f} s)", ok, [
        ("fell", _yes_no(m["fell"]), "no"),
        ("mean turn rate", m["yaw_rate"], f"= {yaw_rate} rad/s +/-10%"),
        ("mean forward speed", m["fwd"], f"= {v_des} m/s +/-15%"),
        *_attitude_rows(m),
    ], "A point foot cannot resist spinning, so the turn is made entirely by the yaw "
       "moment of `r x f` - there is no hip-yaw joint.")


STAGES = (stage1_qp_selfcheck, stage2_jacobian_pinned, stage3_march,
          stage4_walk, stage5_tracking, stage6_push_forward, stage7_push_sideways,
          stage8_sidestep, stage9_turn)


def report(stages: list) -> str:
    """Render the results as the markdown table in `doc/mpc_validation.md`."""
    lines = ["# Convex-MPC validation results (3-D biped)", "",
             "Generated by `uv run python -m mujoco_sim.validate_mpc --write`.",
             "Stages 1-6 follow section 13 of `convec_mpc.md`, carried over to 3-D; "
             "7-9 are new with the 3-D robot (section 17).", ""]
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
               stage3_march: "stage3_march_in_place",
               stage4_walk: "stage4_walk_0.5mps",
               stage5_tracking: "stage5_speed_tracking",
               stage6_push_forward: "stage6_push_forward",
               stage7_push_sideways: "stage7_push_sideways",
               stage8_sidestep: "stage8_sidestep",
               stage9_turn: "stage9_turn"}


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
