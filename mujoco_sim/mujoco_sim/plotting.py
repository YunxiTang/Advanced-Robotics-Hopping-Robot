"""Plot simulation results (mirrors the figures produced by Data_draw.m)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .controller import ControllerParams


def plot_results(arrays: dict, params: ControllerParams):
    t = arrays["t"]
    stance = arrays["stance"]

    fig, axs = plt.subplots(3, 2, figsize=(11, 9), sharex=True)

    axs[0, 0].plot(t, arrays["z"], label="height z")
    axs[0, 0].axhline(params.Hs, color="r", ls="--", label="Hs (desired)")
    axs[0, 0].set_ylabel("z (m)")
    axs[0, 0].legend(loc="upper right")
    axs[0, 0].set_title("Body height")

    axs[0, 1].plot(t, arrays["dx"], label="forward velocity dx")
    axs[0, 1].axhline(params.Vs, color="r", ls="--", label="Vs (desired)")
    axs[0, 1].set_ylabel("dx (m/s)")
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].set_title("Forward velocity")

    axs[1, 0].plot(t, arrays["x"])
    axs[1, 0].set_ylabel("x (m)")
    axs[1, 0].set_title("Horizontal position")

    axs[1, 1].plot(t, np.rad2deg(arrays["theta"]))
    axs[1, 1].set_ylabel("theta (deg)")
    axs[1, 1].set_title("Hip / leg angle")

    axs[2, 0].plot(t, arrays["leg_len"])
    axs[2, 0].set_ylabel("l (m)")
    axs[2, 0].set_xlabel("time (s)")
    axs[2, 0].set_title("Leg length")

    axs[2, 1].plot(t, arrays["energy"])
    axs[2, 1].set_ylabel("Energy (J)")
    axs[2, 1].set_xlabel("time (s)")
    axs[2, 1].set_title("Total mechanical energy")

    for ax in axs.flat:
        ax.fill_between(t, *ax.get_ylim(), where=stance, color="0.85", zorder=0,
                         step="post", label="_stance")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Planar hopper (MuJoCo) — Hs={params.Hs} m, Vs={params.Vs} m/s")
    fig.tight_layout()
    plt.show()
