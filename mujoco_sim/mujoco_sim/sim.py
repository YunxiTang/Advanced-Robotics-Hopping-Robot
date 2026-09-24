"""Drive the MuJoCo 3-D biped with the convex-MPC controller and log results.

MuJoCo simulates the whole hybrid system continuously - there is no
hand-derived dynamics and no manual phase switching anywhere; contact is read
off its contact list each step. The controller is `MPCWalkController`
(`controller.py` + `mpc.py`); the design is `doc/convec_mpc.md`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .controller import LEG_NAMES, MPCGaitParams, MPCWalkController
from .camera import HeadCamera, RGBDRecorder, colorize_depth, rgbd_stem
from .video import VideoRecorder

MODEL_PATH = Path(__file__).parent / "biped.xml"
# logged leg torque -> joint (the biped's name; G1 calls its hip "hip_pitch")
TAU_LOG = {"hip_yaw": "tau_yaw", "hip_roll": "tau_roll", "hip": "tau_hip", "knee": "tau_knee",
           "ankle_pitch": "tau_ankle_pitch", "ankle_roll": "tau_ankle_roll"}


@dataclass
class SimLog:
    t: list
    x: list
    y: list
    z: list
    dx: list
    dy: list
    dz: list
    # heading-frame velocity: forward / leftward relative to where the torso
    # faces, which is what the speed commands are expressed in
    v_fwd: list
    v_lat: list
    roll: list
    pitch: list
    yaw: list
    leg_len: list
    stance: list
    tau_yaw: list
    tau_roll: list
    tau_hip: list
    tau_knee: list
    tau_ankle_pitch: list
    tau_ankle_roll: list
    energy: list
    # the gait clock's duty factor: 0.65 walking, 0.35 running, in between
    # while the controller changes gait
    duty: list

    @classmethod
    def empty(cls) -> "SimLog":
        return cls(*([] for _ in range(len(cls.__dataclass_fields__))))

    def as_arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


def foot_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list:
    """Per-leg bool: is any of this foot's sole contacts touching the floor?"""
    floor_id = model.geom("floor").id
    ids = [model.body(f"foot_{s}").id for s in LEG_NAMES]
    touching = [False] * len(ids)
    for i in range(data.ncon):
        c = data.contact[i]
        if floor_id not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == floor_id else c.geom1
        bid = model.geom_bodyid[other]
        if bid in ids:
            touching[ids.index(bid)] = True
    return touching


def is_in_stance(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """True if *either* foot is on the ground; False is a flight phase.

    Walking, this is True all the time. Running, it is False for the two
    flights in each cycle, so `1 - mean(stance)` is the fraction of the time
    the robot is airborne.
    """
    return any(foot_contacts(model, data))


def run(duration: float = 30.0, params=None, viewer: bool = False,
        disturbance=None, v_schedule=None, video=None, rgbd=None,
        head_cam_view: bool = False, model=None, on_step=None, decorate=None,
        video_camera: str = "chase", stop=None,
        controller_cls=MPCWalkController) -> SimLog:
    """Simulate `duration` seconds of walking or running.

    `disturbance`, if given, is (t_start, t_end, force): a push applied to
    the torso's CoM over that window, used by the validation plan's
    push-recovery stages. `force` is a world-frame (f_x, f_y) pair, or a
    scalar for a purely forward push.

    `v_schedule`, if given, is a callable `t -> v_des` that retargets the
    commanded forward speed while the simulation runs (the velocity-tracking
    stage of the same plan). With `gait="auto"` that also switches the robot
    between walking and running as the command crosses `run_above`.

    `video`, if given, is a path: the run is also recorded offscreen from
    the chase camera to that MP4 file.

    `rgbd`, if given, is a path stem: the head camera's colour and depth are
    recorded to `<stem>.mp4` (side by side) and `<stem>.npz` (raw frames),
    see `camera.RGBDRecorder`.

    `head_cam_view` overlays the head camera's live colour and depth images
    in the corner of the interactive viewer.

    `model` replaces biped.xml (e.g. `local_planner.course_model`).
    `on_step(data, ctrl)` is called before every control step and may
    retarget `ctrl.p` (the local planner steers through it). `decorate(scene)`
    adds geoms to the viewer and video frames. `video_camera` names the
    camera `video` records from. `stop(data)`, if it returns True, ends the
    run early. `controller_cls` drives another robot through the same loop
    (`g1_controller.G1WalkController`, with `model` from `g1.load_model`).
    """
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("mpc_stand").id)
    # Populate xpos/subtree_com/contacts before the first control call:
    # mj_resetDataKeyframe only writes qpos/qvel, so without this the very
    # first controller invocation sees a CoM of (0,0,0) and a 0.4 m height
    # error, and answers with a full-saturation torque spike that kicks the
    # robot before the gait has started.
    mujoco.mj_forward(model, data)

    # The controller needs MuJoCo itself: foot Jacobians for the -J^T f
    # mapping, and body frames for the footstep plan.
    ctrl = controller_cls(model, data, params)
    torso_bid = ctrl._torso_bid
    log = SimLog.empty()

    def com_state() -> tuple[np.ndarray, np.ndarray]:
        """Whole-robot centre-of-mass position and velocity, both 3-D.

        This is the state the single-rigid-body model is written in, so it
        must be the CoM and not `qpos[0:3]` - the latter is the *pelvis*, the
        torso body's frame sitting on the hip axis to keep the leg kinematics
        simple. With the humanoid trunk the CoM rides ~0.07 m above it, which
        is a fifth of the leg length and would bias every moment arm.
        """
        mujoco.mj_subtreeVel(model, data)
        return data.subtree_com[0].copy(), data.subtree_linvel[0].copy()

    def step_and_log(in_stance: bool):
        com, vel = com_state()
        rpy, _ = ctrl.base_state()
        c, s = np.cos(rpy[2]), np.sin(rpy[2])
        # Hip to ankle. Logged as the mean over the two legs: in an alternating gait they
        # are half a cycle apart, so this is a stride-average rather than a
        # per-leg trace.
        l = float(np.mean(ctrl.leg_lengths()))
        # body mechanical energy only: there is no passive spring in the
        # model, so there is no elastic term to add here.
        ke = 0.5 * params_m * float(vel @ vel)
        pe = params_m * params_g * float(com[2])
        log.t.append(data.time)
        log.x.append(float(com[0])); log.y.append(float(com[1])); log.z.append(float(com[2]))
        log.dx.append(float(vel[0])); log.dy.append(float(vel[1])); log.dz.append(float(vel[2]))
        log.v_fwd.append(float(c * vel[0] + s * vel[1]))
        log.v_lat.append(float(-s * vel[0] + c * vel[1]))
        log.roll.append(float(rpy[0])); log.pitch.append(float(rpy[1]))
        log.yaw.append(float(rpy[2]))
        log.leg_len.append(l)
        log.stance.append(in_stance)
        # data.ctrl is interleaved per leg: LEG_JOINTS x (l, r). Logged as
        # totals across legs, which is the quantity acting on the body as a
        # whole.
        # Columns are looked up by joint role, so a robot whose legs order
        # their joints differently (G1: pitch first) logs the same way.
        n = len(ctrl.LEG_JOINTS)
        for joint, name in TAU_LOG.items():
            k = ctrl.LEG_JOINTS.index(joint if joint in ctrl.LEG_JOINTS else f"{joint}_pitch")
            getattr(log, name).append(float(data.ctrl[k:2 * n:n].sum()))
        log.energy.append(ke + pe)
        log.duty.append(ctrl.duty)

    p = ctrl.p
    params_m, params_g = p.m, p.g

    render_dt = 1.0 / 60.0  # target viewer frame rate for real-time pacing

    def show_head_cam(v, cam):
        """Overlay the head camera in the viewer's bottom-right corner:
        colour above depth, at the sensor's native 424x240."""
        vp = v.viewport
        if vp is None:                     # window not drawn yet
            return
        rgb, depth = cam.capture(data)
        h, w = rgb.shape[:2]
        left = max(0, vp.width - w - 10)
        v.set_images([(mujoco.MjrRect(left, 20 + h, w, h), rgb),
                      (mujoco.MjrRect(left, 10, w, h), colorize_depth(depth))])

    def render_loop(v=None, rec=None, cam=None):
        wall_start = time.perf_counter()
        sim_start = data.time
        next_render_t = sim_start + render_dt
        # Step-bounded, not just time-bounded. When a controller diverges,
        # MuJoCo detects the bad qacc, prints "Nan, Inf or huge value in
        # QACC" and *resets the simulation state*, which sends data.time back
        # to zero - so a `while data.time < duration` loop never terminates
        # and the run hangs forever instead of failing. Counting steps bounds
        # the run regardless, and a reset is reported rather than silently
        # swallowed.
        max_steps = int(np.ceil((duration - sim_start) / model.opt.timestep)) + 1
        prev_time = data.time
        resets = 0
        for _ in range(max_steps):
            if data.time >= duration:
                break
            in_stance = any(ctrl._in_contact(i) for i in range(ctrl.p.n_legs))

            if stop is not None and stop(data):
                break
            if v_schedule is not None:
                ctrl.p.Vs = float(v_schedule(data.time))
            if on_step is not None:
                on_step(data, ctrl)
            data.ctrl[:] = ctrl.control(data.time, *com_state())

            # External push (validation stages 6-7). xfrc_applied is in world
            # axes and is cleared again the moment the window closes, so the
            # recovery that follows is unforced.
            if disturbance is not None:
                t0, t1, force = disturbance
                fxy = np.broadcast_to(np.atleast_1d(np.asarray(force, float)), (2,)) \
                    if np.ndim(force) == 0 else np.asarray(force, float)
                data.xfrc_applied[torso_bid, 0:2] = fxy if t0 <= data.time < t1 else 0.0

            mujoco.mj_step(model, data)
            if data.time < prev_time:
                resets += 1
            prev_time = data.time
            step_and_log(in_stance)
            if rec is not None:
                rec.maybe_capture(data)
            if rgbd_rec is not None:
                rgbd_rec.maybe_capture(data)

            if v is not None and data.time >= next_render_t:
                sim_elapsed = data.time - sim_start
                wall_elapsed = time.perf_counter() - wall_start
                sleep_time = sim_elapsed - wall_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if cam is not None:
                    show_head_cam(v, cam)
                if decorate is not None:
                    v.user_scn.ngeom = 0
                    decorate(v.user_scn)
                v.sync()
                next_render_t += render_dt

        if resets:
            print(f"WARNING: the simulation went unstable and MuJoCo reset it "
                  f"{resets}x during this run - the controller diverged and "
                  f"the log is not physical.")

    rec = VideoRecorder(model, video, camera=video_camera, decorate=decorate) \
        if video is not None else None
    rgbd_rec = RGBDRecorder(model, rgbd) if rgbd is not None else None
    cam = HeadCamera(model) if viewer and head_cam_view else None
    try:
        if viewer:
            import mujoco.viewer as mj_viewer
            with mj_viewer.launch_passive(model, data) as v:
                v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
                render_loop(v, rec, cam)
        else:
            render_loop(None, rec)
    finally:
        for r in (rec, rgbd_rec, cam):
            if r is not None:
                r.close()

    return log


def _reexec_under_mjpython(module: str = "mujoco_sim.sim") -> None:
    """On macOS, hand the process over to `mjpython` for the viewer.

    Cocoa insists that GUI calls happen on the process's main thread, which
    is why MuJoCo ships `mjpython` and refuses `launch_passive` without it.
    Rather than make that the user's problem - the failure is an exception
    several frames deep, and the fix is a command they have to know about -
    re-exec ourselves under the `mjpython` sitting next to this interpreter,
    with the same arguments (`module` is the one whose CLI this is). Nothing has been simulated yet at this point,
    so replacing the process costs nothing.
    """
    import mujoco.viewer

    if sys.platform != "darwin" or getattr(mujoco.viewer, "_MJPYTHON", 1) is not None:
        return                                   # not macOS, or already there
    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.exists():                    # let MuJoCo raise its own error
        return
    os.execv(str(mjpython), [str(mjpython), "-m", module, *sys.argv[1:]])


def main():
    ap = argparse.ArgumentParser(
        description="Walk or run the 3-D biped with the convex-MPC controller.")
    ap.add_argument("--view", action="store_true",
                    help="open the interactive MuJoCo viewer (select a "
                         "chase camera with Tab)")
    ap.add_argument("--vs", type=float, default=0.5,
                    help="forward speed target in m/s (0 marches in place)")
    ap.add_argument("--vy", type=float, default=0.0,
                    help="sideways speed target in m/s, positive = left")
    ap.add_argument("--yaw-rate", type=float, default=0.0,
                    help="turn rate in rad/s, positive = left")
    ap.add_argument("--gait", choices=("auto", "walk", "run"), default="auto",
                    help="walk (duty 0.65), run (duty 0.35, with flight "
                         "phases), or auto: run above 0.7 m/s (default)")
    ap.add_argument("--stand", action="store_true",
                    help="stand still on both flat feet instead of walking "
                         "(ignores --vs/--vy/--yaw-rate)")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="seconds of simulated time")
    ap.add_argument("--height", type=float, default=MPCGaitParams.z_des,
                    help="CoM height target in m")
    ap.add_argument("--plot", action="store_true", help="plot results afterwards")
    ap.add_argument("--save", type=str, default=None, help="save log arrays to .npz")
    ap.add_argument("--video", type=str, default=None,
                    help="record the run from the chase camera to this .mp4 file")
    ap.add_argument("--rgbd", type=str, default=None, metavar="STEM",
                    help="record the head camera's colour and depth to "
                         "STEM.mp4 (side by side) and STEM.npz (raw frames)")
    ap.add_argument("--head-cam", action="store_true",
                    help="with --view, show the head camera's live colour "
                         "and depth images in the viewer's corner")
    args = ap.parse_args()

    if args.view:
        _reexec_under_mjpython()

    if args.stand:
        # duty = 1: every foot is a stance foot for the whole cycle, so the
        # gait clock never lifts one - the same controller, not stepping
        args.vs = args.vy = args.yaw_rate = 0.0
    params = MPCGaitParams(Vs=args.vs, Vy=args.vy, yaw_rate=args.yaw_rate,
                           z_des=args.height, gait=args.gait,
                           duty=1.0 if args.stand else MPCGaitParams.duty)
    log = run(duration=args.duration, params=params, viewer=args.view,
              video=args.video, rgbd=args.rgbd, head_cam_view=args.head_cam)
    if args.video:
        print(f"saved video to {args.video}")
    if args.rgbd:
        stem = rgbd_stem(args.rgbd)
        print(f"saved head-camera RGB-D to {stem}.mp4 and {stem}.npz")
    arrays = log.as_arrays()

    settled = arrays["t"] > min(10.0, 0.5 * args.duration)
    if settled.any():
        a = {k: v[settled] for k, v in arrays.items()}
        t_span = a["t"][-1] - a["t"][0]
        print(f"speed fwd {a['v_fwd'].mean():.3f} / left {a['v_lat'].mean():.3f} m/s "
              f"(target {args.vs} / {args.vy}), turn rate "
              f"{(a['yaw'][-1] - a['yaw'][0]) / t_span:.3f} rad/s "
              f"(target {args.yaw_rate})")
        flight = 1.0 - a["stance"].mean()
        print(f"gait {'running' if a['duty'][-1] < 0.5 else 'walking'}, both feet "
              f"off the ground {100 * flight:.0f}% of the time")
        print(f"mean CoM height {a['z'].mean():.3f} m, max |roll| "
              f"{np.abs(a['roll']).max():.3f} rad, max |pitch| "
              f"{np.abs(a['pitch']).max():.3f} rad")

    if args.save:
        np.savez(args.save, **arrays)
        print(f"saved log to {args.save}")

    if args.plot:
        from .plotting import plot_results
        plot_results(arrays, params)


if __name__ == "__main__":
    main()
