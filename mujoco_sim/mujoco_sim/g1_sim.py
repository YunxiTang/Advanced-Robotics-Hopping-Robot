"""`uv run g1`: the Unitree G1 walking under the convex MPC.

The same simulation loop as the biped (`sim.run`), with the G1 model
(`g1.load_model`) and controller (`g1_controller.G1WalkController`).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .g1 import load_model
from .g1_controller import G1GaitParams, G1WalkController
from .sim import _reexec_under_mjpython, run


def main():
    ap = argparse.ArgumentParser(
        description="Walk the Unitree G1 with the convex-MPC controller.")
    ap.add_argument("--view", action="store_true",
                    help="open the interactive MuJoCo viewer (select the chase "
                         "camera with Tab)")
    ap.add_argument("--vs", type=float, default=0.3,
                    help="forward speed target in m/s (0 marches in place; "
                         "reached by a ramp, see --ramp)")
    ap.add_argument("--vy", type=float, default=0.0,
                    help="sideways speed target in m/s, positive = left")
    ap.add_argument("--yaw-rate", type=float, default=0.0,
                    help="turn rate in rad/s, positive = left")
    ap.add_argument("--ramp", type=float, default=None,
                    help="seconds over which the forward speed ramps up from 0 "
                         "(default: 6 s per m/s above 0.6 m/s, else none)")
    ap.add_argument("--stand", action="store_true",
                    help="stand still on both feet instead of walking")
    ap.add_argument("--push", type=float, nargs=3, default=None,
                    metavar=("T", "FX", "FY"),
                    help="push the pelvis with (FX, FY) newtons for 0.1 s at time T")
    ap.add_argument("--no-wbc", action="store_true",
                    help="map the MPC's wrenches with -J^T w, as the biped "
                         "does, instead of the whole-body QP")
    ap.add_argument("--duration", type=float, default=20.0,
                    help="seconds of simulated time")
    ap.add_argument("--plot", action="store_true", help="plot results afterwards")
    ap.add_argument("--save", type=str, default=None, help="save log arrays to .npz")
    ap.add_argument("--video", type=str, default=None,
                    help="record the run from the chase camera to this .mp4 file")
    args = ap.parse_args()

    if args.view:
        _reexec_under_mjpython("mujoco_sim.g1_sim")

    if args.stand:
        args.vs = args.vy = args.yaw_rate = 0.0
    params = G1GaitParams(Vs=args.vs, Vy=args.vy, yaw_rate=args.yaw_rate,
                          duty=1.0 if args.stand else G1GaitParams.duty,
                          use_wbc=not args.no_wbc)
    # From rest straight to 0.6 m/s is fine; to 0.7 falls, and a ramp to it
    # does not.
    ramp = args.ramp if args.ramp is not None else (6.0 * args.vs if args.vs > 0.6 else 0.0)
    v_schedule = (lambda t: args.vs * min(1.0, t / ramp)) if ramp > 0 else None
    push = None if args.push is None else (args.push[0], args.push[0] + 0.1,
                                           np.array(args.push[1:]))

    wall = time.perf_counter()
    log = run(duration=args.duration, params=params, viewer=args.view,
              disturbance=push, v_schedule=v_schedule, video=args.video,
              model=load_model(), controller_cls=G1WalkController)
    wall = time.perf_counter() - wall
    if args.video:
        print(f"saved video to {args.video}")
    a = log.as_arrays()

    fell = np.flatnonzero(a["z"] < 0.4)
    if fell.size:
        print(f"FELL at t = {a['t'][fell[0]]:.2f} s")
    settled = a["t"] > max(min(10.0, 0.5 * args.duration), ramp + 2.0)
    if settled.any() and not fell.size:
        s = {k: v[settled] for k, v in a.items()}
        span = s["t"][-1] - s["t"][0]
        print(f"speed fwd {s['v_fwd'].mean():.3f} / left {s['v_lat'].mean():.3f} m/s "
              f"(target {args.vs} / {args.vy}), turn rate "
              f"{(s['yaw'][-1] - s['yaw'][0]) / span:.3f} rad/s (target {args.yaw_rate})")
        print(f"mean CoM height {s['z'].mean():.3f} m, p95 |roll| "
              f"{np.percentile(np.abs(s['roll']), 95):.3f} rad, p95 |pitch| "
              f"{np.percentile(np.abs(s['pitch']), 95):.3f} rad")
    if not args.view:
        print(f"simulated {a['t'][-1]:.1f} s in {wall:.1f} s wall time")

    if args.save:
        np.savez(args.save, **a)
        print(f"saved log to {args.save}")
    if args.plot:
        from .plotting import plot_results
        plot_results(a, params)


if __name__ == "__main__":
    main()
