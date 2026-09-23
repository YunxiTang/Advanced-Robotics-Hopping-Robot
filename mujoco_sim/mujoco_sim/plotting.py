"""Plot simulation results: path, speed, height, attitude, leg length and energy."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .controller import MPCGaitParams


def plot_results(arrays: dict, params: MPCGaitParams):
    t = arrays["t"]

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 2)
    ax_path = fig.add_subplot(gs[0:2, 0])
    axs = [fig.add_subplot(gs[i, 1]) for i in range(4)] + \
          [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[3, 0])]
    for ax in axs[1:]:
        ax.sharex(axs[0])

    # Top view: the one plot the planar version could not have.
    ax_path.plot(arrays["x"], arrays["y"], lw=1)
    ax_path.plot(arrays["x"][0], arrays["y"][0], "go", label="start")
    ax_path.plot(arrays["x"][-1], arrays["y"][-1], "rs", label="end")
    ax_path.set_aspect("equal", adjustable="datalim")
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.legend(loc="best")
    ax_path.set_title("CoM path (top view)")

    ax = axs[0]
    ax.plot(t, arrays["v_fwd"], label="forward")
    ax.plot(t, arrays["v_lat"], label="left")
    ax.axhline(params.Vs, color="C0", ls="--", lw=1)
    ax.axhline(params.Vy, color="C1", ls="--", lw=1)
    ax.set_ylabel("v (m/s)")
    ax.legend(loc="upper right")
    ax.set_title("Velocity in the heading frame (dashed: commanded)")

    ax = axs[1]
    ax.plot(t, np.rad2deg(arrays["roll"]), label="roll")
    ax.plot(t, np.rad2deg(arrays["pitch"]), label="pitch")
    ax.set_ylabel("angle (deg)")
    ax.legend(loc="upper right")
    ax.set_title("Torso attitude")

    ax = axs[2]
    ax.plot(t, np.rad2deg(arrays["yaw"]))
    if params.yaw_rate:
        ax.plot(t, np.rad2deg(arrays["yaw"][0] + params.yaw_rate * t), "r--", lw=1,
                label="commanded")
        ax.legend(loc="upper left")
    ax.set_ylabel("yaw (deg)")
    ax.set_title("Heading")

    ax = axs[3]
    ax.plot(t, arrays["z"], label="height z")
    ax.axhline(params.z_des, color="r", ls="--", label="desired")
    ax.set_ylabel("z (m)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right")
    ax.set_title("CoM height")

    ax = axs[4]
    ax.plot(t, arrays["leg_len"])
    ax.set_ylabel("l (m)")
    ax.set_title("Leg length (mean of both legs)")

    ax = axs[5]
    ax.plot(t, arrays["energy"])
    ax.set_ylabel("Energy (J)")
    ax.set_xlabel("time (s)")
    ax.set_title("Body mechanical energy")

    # No stance shading, unlike the planar plots: in a walk one foot or the
    # other is always down, so "in stance" is true almost everywhere and the
    # band would just grey out every panel.
    for ax in [ax_path, *axs]:
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"3-D biped (MuJoCo) — command: forward {params.Vs} m/s, "
                 f"left {params.Vy} m/s, turn {params.yaw_rate} rad/s, "
                 f"height {params.z_des} m")
    fig.tight_layout()
    plt.show()
