"""Staged validation of the convex MPC on the Unitree G1 (`g1_controller.py`).

Run it with::

    uv run python -m mujoco_sim.validate_g1 --jobs 8            # all stages, print a table
    uv run python -m mujoco_sim.validate_g1 --jobs 8 --write    # ... and update doc/g1_validation.md
    uv run python -m mujoco_sim.validate_g1 --stage 4           # just one
    uv run python -m mujoco_sim.validate_g1 --jobs 8 --video result/g1   # ... recording each stage

As for the biped (`validate_mpc.py`), every stage has a numeric acceptance
criterion. Stage 1 checks the one new piece of machinery on its own: that
the whole-body QP's torques make the ground push back with the wrench the MPC
asked for. Stages 2-9 are the biped's plan at G1's scale - the pushes are
sized to the robot's 33 kg, and the top speed is G1's.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from .g1 import load_model
from .g1_controller import G1GaitParams, G1WalkController
from .sim import run
from .validate_mpc import Stage, _fmt, _yes_no
from .video import VideoRecorder

DOC_PATH = Path(__file__).resolve().parent.parent / "doc" / "g1_validation.md"
FALL_Z = 0.4                  # a CoM below this is a fall (walking height 0.63 m)
ATT_P95 = 0.1                 # rad, roll and pitch


def _run(duration, video=None, **kw):
    """sim.run on G1; `ramp=(v, T)` ramps the forward speed to v over T s."""
    ramp = kw.pop("ramp", None)
    push = kw.pop("push", None)
    params = G1GaitParams(**kw)
    vs = None if ramp is None else (lambda t: ramp[0] * min(1.0, t / ramp[1]))
    dist = None if push is None else (push[0], push[0] + 0.1, np.asarray(push[1:], float))
    log = run(duration=duration, params=params, model=load_model(),
              controller_cls=G1WalkController, v_schedule=vs, disturbance=dist,
              video=video)
    return log.as_arrays()


def _fell(a: dict) -> bool:
    return bool(a["z"].min() < FALL_Z)


def _p95(x) -> float:
    return float(np.percentile(np.abs(x), 95))


def _walk_metrics(a: dict, window, v_fwd=None, v_lat=None, yaw_rate=None):
    """(metrics, ok) for a steady window: speed, attitude and heading."""
    k = (a["t"] > window[0]) & (a["t"] <= window[1])
    fell = _fell(a)
    metrics, ok = [("fell", _yes_no(fell), "no")], not fell
    if v_fwd is not None:
        v = float(a["v_fwd"][k].mean())
        tol = max(0.1 * abs(v_fwd), 0.02)
        ok &= abs(v - v_fwd) < tol
        metrics.append(("mean forward speed", v, f"= {v_fwd} m/s +/-{tol:g}"))
    if v_lat is not None:
        v = float(a["v_lat"][k].mean())
        tol = max(0.1 * abs(v_lat), 0.02)
        ok &= abs(v - v_lat) < tol
        metrics.append(("mean sideways speed", v, f"= {v_lat} m/s +/-{tol:g}"))
    span = a["t"][k][-1] - a["t"][k][0]
    rate = float((a["yaw"][k][-1] - a["yaw"][k][0]) / span)
    if yaw_rate:
        ok &= abs(rate - yaw_rate) < 0.1 * abs(yaw_rate)
        metrics.append(("mean turn rate", rate, f"= {yaw_rate} rad/s +/-10%"))
    else:
        drift = abs(rate * span)
        ok &= drift < 0.2
        metrics.append(("heading drift", drift, "< 0.2 rad"))
    r, p = _p95(a["roll"][k]), _p95(a["pitch"][k])
    ok &= r < ATT_P95 and p < ATT_P95
    metrics += [("p95 abs roll", r, f"< {ATT_P95} rad"), ("p95 abs pitch", p, f"< {ATT_P95} rad"),
                ("mean CoM height", float(a["z"][k].mean()), "(reported)")]
    return metrics, bool(ok)


# --- stage 1: the whole-body QP's wrench, measured ------------------------------

def stage1_wbc_wrench(video=None) -> Stage:
    """Standing, the contact force MuJoCo measures at each foot must be the
    one the QP computed its torques for, and that one the MPC's.

    This is the G1 counterpart of the biped's pinned `-J^T w` check. It runs
    the controller's own code path, free-standing, and compares per foot
    over the last second of 3 s: MPC force -> whole-body QP force (how far
    the QP shaded it) -> contact force (whether the torques delivered it).
    """
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("mpc_stand").id)
    mujoco.mj_forward(model, data)
    ctrl = G1WalkController(model, data, G1GaitParams(duty=1.0, Vs=0.0))
    rec = VideoRecorder(model, video) if video is not None else None
    f6 = np.zeros(6)
    err_qp, err_meas, weight = [], [], []
    while data.time < 3.0:
        mujoco.mj_subtreeVel(model, data)
        data.ctrl[:] = ctrl.control(data.time, data.subtree_com[0].copy(),
                                    data.subtree_linvel[0].copy())
        u_mpc, u_qp = ctrl._f[:, :3].copy(), ctrl.wbc_wrench[:, :3].copy()
        mujoco.mj_step(model, data)
        if rec is not None:
            rec.maybe_capture(data)
        if data.time < 2.0:
            continue
        # contact force on each foot, world axes, after the step it drove
        meas = np.zeros((2, 3))
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            for j, bid in enumerate(ctrl._foot_bid):
                if bid in (b1, b2):
                    mujoco.mj_contactForce(model, data, i, f6)
                    fw = c.frame.reshape(3, 3).T @ f6[:3]
                    meas[j] += fw if b2 == bid else -fw
        err_qp.append(np.abs(u_qp - u_mpc).max())
        err_meas.append(np.abs(meas - u_qp).max())
        weight.append(meas[:, 2].sum())
    if rec is not None:
        rec.close()
    mg = float(model.body_subtreemass[0] * -model.opt.gravity[2])
    e_qp, e_meas, w = float(np.max(err_qp)), float(np.max(err_meas)), float(np.mean(weight))
    ok = e_qp < 0.02 * mg and e_meas < 0.03 * mg and abs(w - mg) < 0.02 * mg
    metrics = [("max |QP force - MPC force|", e_qp, f"< 2% Mg = {0.02 * mg:.1f} N"),
               ("max |contact force - QP force|", e_meas, f"< 3% Mg = {0.03 * mg:.1f} N"),
               ("mean total contact f_z", w, f"= Mg = {mg:.1f} N +/-2%"),
               ("QP failures", ctrl.wbc.failures, "0")]
    ok &= ctrl.wbc.failures == 0
    return Stage("1. Whole-body QP delivers the MPC's wrench (standing, 3 s)", bool(ok), metrics,
                 "The QP stays with the MPC's force to within a hundredth of a newton "
                 "while standing, and the measured contact force matches it to within "
                 "about a newton against 327 N of weight.")


# --- stages 2-9 -----------------------------------------------------------------

def stage2_stand(video=None) -> Stage:
    a = _run(12.0, video, duty=1.0, Vs=0.0, push=(4.0, 100.0, 0.0))
    fell = _fell(a)
    k = a["t"] > 8.0
    drift = float(np.hypot(a["x"][-1] - a["x"][0], a["y"][-1] - a["y"][0]))
    peak = float(np.abs(a["pitch"]).max())
    ok = not fell and drift < 0.05 and peak < 0.1 and _p95(a["pitch"][k]) < 0.01
    return Stage("2. Stand still (12 s, 100 N x 0.1 s forward push at 4 s)", bool(ok),
                 [("fell", _yes_no(fell), "no"), ("CoM drift", drift, "< 0.05 m"),
                  ("max abs pitch", peak, "< 0.1 rad"),
                  ("p95 abs pitch, 8-12 s", _p95(a["pitch"][k]), "< 0.01 rad")],
                 "Both feet stay down (duty 1): the push is taken on the soles' "
                 "centres of pressure and friction alone, without a step.")


def stage3_march(video=None) -> Stage:
    a = _run(12.0, video, Vs=0.0)
    metrics, ok = _walk_metrics(a, (4.0, 12.0), v_fwd=0.0, v_lat=0.0)
    return Stage("3. March in place (12 s)", ok, metrics)


def stage4_walk(video=None) -> Stage:
    a = _run(20.0, video, Vs=0.3)
    metrics, ok = _walk_metrics(a, (5.0, 20.0), v_fwd=0.3, v_lat=0.0)
    return Stage("4. Steady walk (0.3 m/s, 20 s)", ok, metrics)


def stage5_fast(video=None) -> Stage:
    a = _run(14.0, video, Vs=0.0, ramp=(0.6, 4.0))
    metrics, ok = _walk_metrics(a, (9.0, 14.0), v_fwd=0.6, v_lat=0.0)
    return Stage("5. Speed ramp to 0.6 m/s (over 4 s, held to 14 s)", ok, metrics,
                 "Near G1's ceiling under this controller: a 5 s ramp to 0.7 m/s "
                 "holds, and a 6 s ramp to 0.8 m/s falls 1.5 s after reaching it.")


def stage6_sidestep(video=None) -> Stage:
    a = _run(15.0, video, Vs=0.0, Vy=0.1)
    metrics, ok = _walk_metrics(a, (5.0, 15.0), v_fwd=0.0, v_lat=0.1)
    return Stage("6. Sidestep (0.1 m/s left, 15 s)", ok, metrics)


def stage7_turn(video=None) -> Stage:
    a = _run(20.0, video, Vs=0.3, yaw_rate=0.3)
    metrics, ok = _walk_metrics(a, (5.0, 20.0), v_fwd=0.3, yaw_rate=0.3)
    return Stage("7. Turn while walking (0.3 m/s, 0.3 rad/s, 20 s)", ok, metrics)


def _push_stage(name, force, video) -> Stage:
    t0 = 6.0
    a = _run(12.0, video, Vs=0.3, push=(t0, *force))
    fell = _fell(a)
    pre = (a["t"] > t0 - 2.0) & (a["t"] <= t0)
    post = a["t"] > t0 + 3.0
    during = (a["t"] > t0) & (a["t"] <= t0 + 3.0)
    dv = np.hypot(a["dx"] - a["dx"][pre].mean(), a["dy"] - a["dy"][pre].mean())
    err = float(np.hypot(a["v_fwd"][post].mean() - 0.3, a["v_lat"][post].mean()))
    ok = not fell and err < 0.05
    return Stage(name, bool(ok),
                 [("fell", _yes_no(fell), "no"),
                  ("peak velocity change", float(dv[during].max()), "(reported)"),
                  ("velocity error 3 s after", err, "< 0.05 m/s"),
                  ("max abs roll during", float(np.abs(a["roll"][during]).max()), "(reported)"),
                  ("max abs pitch during", float(np.abs(a["pitch"][during]).max()), "(reported)")])


def stage8_push_forward(video=None) -> Stage:
    return _push_stage("8. Push recovery, forward (120 N x 0.1 s, walking 0.3 m/s)",
                       (120.0, 0.0), video)


def stage9_push_sideways(video=None) -> Stage:
    return _push_stage("9. Push recovery, sideways (80 N x 0.1 s, walking 0.3 m/s)",
                       (0.0, 80.0), video)


STAGES = (stage1_wbc_wrench, stage2_stand, stage3_march, stage4_walk, stage5_fast,
          stage6_sidestep, stage7_turn, stage8_push_forward, stage9_push_sideways)

VIDEO_NAMES = {stage1_wbc_wrench: "g1_stage1_wbc_wrench",
               stage2_stand: "g1_stage2_stand_push",
               stage3_march: "g1_stage3_march_in_place",
               stage4_walk: "g1_stage4_walk_0.3mps",
               stage5_fast: "g1_stage5_walk_0.6mps",
               stage6_sidestep: "g1_stage6_sidestep",
               stage7_turn: "g1_stage7_turn",
               stage8_push_forward: "g1_stage8_push_forward",
               stage9_push_sideways: "g1_stage9_push_sideways"}


def report(stages: list) -> str:
    """Render the results as the markdown in `doc/g1_validation.md`."""
    lines = ["# Convex-MPC validation results (Unitree G1)", "",
             "Generated by `uv run python -m mujoco_sim.validate_g1 --write`. "
             "Design and measurements: `g1_mpc.md`.", "",
             "| stage | result |", "|---|---|"]
    lines += [f"| {s.name} | {'PASS' if s.passed else 'FAIL'} |" for s in stages]
    lines.append("")
    for s in stages:
        lines += [f"## {s.name} — {'PASS' if s.passed else 'FAIL'}", "",
                  "| metric | value | criterion |", "|---|---|---|"]
        lines += [f"| {label} | {_fmt(v)} | {c or '-'} |" for label, v, c in s.metrics]
        if s.note:
            lines += ["", s.note]
        lines.append("")
    return "\n".join(lines)


def _run_stage(job) -> Stage:
    fn, video_dir = job
    if video_dir is None:
        return fn()
    return fn(video=Path(video_dir) / f"{VIDEO_NAMES[fn]}.mp4")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help=f"write the results to {DOC_PATH}")
    ap.add_argument("--stage", type=int, default=None,
                    help=f"run a single stage (1-{len(STAGES)})")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run stages in parallel processes (they are independent)")
    ap.add_argument("--video", type=str, default=None, metavar="DIR",
                    help="record each stage from the chase camera to DIR/g1_stageN_*.mp4")
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
