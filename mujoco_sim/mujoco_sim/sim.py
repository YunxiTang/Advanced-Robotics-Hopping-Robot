"""Drive the MuJoCo planar hopper with the Raibert controller and log results.

This replaces Main.m / run_Fight_simulation.m / run_Stance_simulation.m: instead
of hand-integrating Flight_EoM/Stance_EoM with ode45 and manually switching
phases at zero-crossing events (FlightEvent.m/StanceEvent.m), MuJoCo simulates
the whole hybrid system continuously and contact is detected from its contact
list each step.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .controller import (ControllerParams, GaitParams, RaibertController,
                         WalkingController, leg_length)

MODEL_PATH = Path(__file__).parent / "hopper.xml"


@dataclass
class SimLog:
    t: list
    x: list
    z: list
    dx: list
    dz: list
    theta: list
    torso_pitch: list
    leg_len: list
    stance: list
    tau_hip: list
    tau_knee: list
    energy: list

    @classmethod
    def empty(cls) -> "SimLog":
        return cls(*([] for _ in range(12)))

    def as_arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


LEG_NAMES = ("l", "r")


def foot_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list:
    """Per-leg bool: is this foot touching the floor right now?"""
    floor_id = model.geom("floor").id
    ids = [model.geom(f"foot_{s}").id for s in LEG_NAMES]
    touching = [False] * len(ids)
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if floor_id not in pair:
            continue
        for k, gid in enumerate(ids):
            if gid in pair:
                touching[k] = True
    return touching


def is_in_stance(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """True if *either* foot is on the ground.

    With both legs driven as a symmetric pair they touch down together, but
    treating stance as "any foot loaded" keeps the phase logic correct if they
    ever land a timestep apart - otherwise a leg could still be commanding the
    flight impedance while it is actually carrying the robot.
    """
    foot_ids = {model.geom(f"foot_{s}").id for s in LEG_NAMES}
    floor_id = model.geom("floor").id
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if floor_id in pair and pair & foot_ids:
            return True
    return False


def run(duration: float = 30.0, params=None, viewer: bool = False,
        gait: str = "walk") -> SimLog:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    walking = gait == "walk"
    if walking:
        ctrl = WalkingController(params)
    else:
        ctrl = RaibertController(params)
    log = SimLog.empty()

    def addr(name: str) -> tuple:
        jid = model.joint(name).id
        return model.jnt_qposadr[jid], model.jnt_dofadr[jid]

    torso_pitch_qpos_adr, torso_pitch_qvel_adr = addr("torso_pitch")
    # (hip_qpos, hip_qvel, knee_qpos, knee_qvel) per leg, in actuator order
    leg_adr = [addr(f"hip_{s}") + addr(f"knee_{s}") for s in LEG_NAMES]

    def read_legs() -> list:
        return [(data.qpos[hq], data.qvel[hv], data.qpos[kq], data.qvel[kv])
                for hq, hv, kq, kv in leg_adr]

    def com_state() -> tuple:
        """Whole-robot centre-of-mass (x, z, dx, dz).

        The Raibert laws are written for the point mass of set_model.m, so
        they have to be fed the CoM - not `qpos[0:2]`, which is the *pelvis*
        (the torso body's frame sits on the hip axis so the leg kinematics
        stay simple). With the humanoid trunk the CoM rides ~0.07 m above the
        pelvis, and feeding the pelvis height to the energy law instead would
        bias the apex target low by m*g*0.07 on every single bounce.
        """
        mujoco.mj_subtreeVel(model, data)
        com = data.subtree_com[0]
        vel = data.subtree_linvel[0]
        return float(com[0]), float(com[2]), float(vel[0]), float(vel[2])

    def step_and_log(in_stance: bool):
        x, z, dx, dz = com_state()
        torso_pitch = data.qpos[torso_pitch_qpos_adr]
        legs = read_legs()
        # The legs are driven as a symmetric pair, so the logged leg angle and
        # length are the mean over legs - for a healthy gait the two are
        # indistinguishable, and a visible divergence means they have split.
        theta_world = float(np.mean([torso_pitch + hip + knee / 2.0
                                     for hip, _, knee, _ in legs]))
        l = float(np.mean([leg_length(knee, ctrl.p.L1, ctrl.p.L2)
                           for _, _, knee, _ in legs]))
        # body mechanical energy only: with no passive spring left in the
        # model, the leg's compliance is actively supplied by the motors, so
        # there is no elastic term to add here.
        ke = 0.5 * params_m * (dx**2 + dz**2)
        pe = params_m * params_g * z
        log.t.append(data.time)
        log.x.append(x); log.z.append(z)
        log.dx.append(dx); log.dz.append(dz)
        log.theta.append(theta_world); log.torso_pitch.append(torso_pitch)
        log.leg_len.append(l)
        log.stance.append(in_stance)
        # data.ctrl is interleaved per leg: (hip_l, knee_l, hip_r, knee_r), so
        # the even entries are hips and the odd ones knees. Logged as totals
        # across legs, which is the quantity acting on the body as a whole.
        log.tau_hip.append(float(data.ctrl[0::2].sum()))
        log.tau_knee.append(float(data.ctrl[1::2].sum()))
        log.energy.append(ke + pe)

    p = ctrl.p
    params_m, params_g = p.m, p.g

    render_dt = 1.0 / 60.0  # target viewer frame rate for real-time pacing

    def render_loop(v=None):
        prev_stance = False
        wall_start = time.perf_counter()
        sim_start = data.time
        next_render_t = sim_start + render_dt
        while data.time < duration:
            in_stance = is_in_stance(model, data)
            _, com_z, com_dx, _ = com_state()

            if not walking:
                if (not in_stance) and prev_stance:
                    ctrl.on_liftoff(data.time, vx_liftoff=com_dx, z_liftoff=com_z)
                ctrl.update_apex_tracking(z=com_z, vx=com_dx, in_stance=in_stance)

            if walking:
                data.ctrl[:] = ctrl.control(
                    t=data.time,
                    torso_pitch=data.qpos[torso_pitch_qpos_adr],
                    dtorso_pitch=data.qvel[torso_pitch_qvel_adr],
                    legs=read_legs(), vx=com_dx,
                    contacts=foot_contacts(model, data))
            else:
                data.ctrl[:] = ctrl.control(
                    in_stance=in_stance,
                    torso_pitch=data.qpos[torso_pitch_qpos_adr],
                    dtorso_pitch=data.qvel[torso_pitch_qvel_adr],
                    legs=read_legs())

            mujoco.mj_step(model, data)
            step_and_log(in_stance)
            prev_stance = in_stance

            if v is not None and data.time >= next_render_t:
                sim_elapsed = data.time - sim_start
                wall_elapsed = time.perf_counter() - wall_start
                sleep_time = sim_elapsed - wall_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                v.sync()
                next_render_t += render_dt

    if viewer:
        import mujoco.viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as v:
            render_loop(v)
    else:
        render_loop(None)

    return log


def main():
    ap = argparse.ArgumentParser(description="Run the MuJoCo planar hopper simulation")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--gait", choices=("walk", "hop"), default="walk",
                    help="alternating walking gait, or the Raibert two-legged hop")
    ap.add_argument("--hs", type=float, default=0.4, help="desired apex height")
    ap.add_argument("--vs", type=float, default=1.0, help="desired forward velocity")
    ap.add_argument("--view", action="store_true", help="launch the interactive MuJoCo viewer")
    ap.add_argument("--plot", action="store_true", help="plot results after running")
    ap.add_argument("--save", type=str, default=None, help="save log arrays to .npz")
    args = ap.parse_args()

    if args.gait == "walk":
        params = GaitParams(Vs=args.vs)
    else:
        params = ControllerParams(Hs=args.hs, Vs=args.vs)
    log = run(duration=args.duration, params=params, viewer=args.view, gait=args.gait)
    arrays = log.as_arrays()

    if args.save:
        np.savez(args.save, **arrays)
        print(f"saved log to {args.save}")

    if args.plot:
        from .plotting import plot_results
        plot_results(arrays, params)


if __name__ == "__main__":
    main()
