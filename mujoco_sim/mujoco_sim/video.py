"""Offscreen MP4 recording of a simulation, for runs with no viewer attached.

Frames come from `mujoco.Renderer` and are piped straight into ffmpeg as raw
RGB, so nothing is buffered in memory and no extra Python package is needed -
only an `ffmpeg` binary on PATH.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import mujoco


class VideoRecorder:
    """Call `maybe_capture(data)` after every physics step; it renders a frame
    whenever simulated time has passed the next frame's timestamp, so the
    video plays back in real time whatever the physics step is."""

    def __init__(self, model: mujoco.MjModel, path, fps: int = 30,
                 camera: str = "chase", width: int = 1280, height: int = 720):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("video recording needs ffmpeg on PATH")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.camera = camera
        # Capped by <global offwidth/offheight> in the model.
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        # Draw the ground reaction force at each foot contact; arrow scale
        # and colour are set in the model's <visual> block.
        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        self.next_t = 0.0
        self.frames = 0
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
             "-movflags", "+faststart", str(self.path)],
            stdin=subprocess.PIPE)

    def maybe_capture(self, data: mujoco.MjData) -> None:
        if data.time + 1e-9 < self.next_t:
            return
        self.renderer.update_scene(data, camera=self.camera,
                                   scene_option=self.scene_option)
        self.proc.stdin.write(self.renderer.render().tobytes())
        self.frames += 1
        self.next_t += 1.0 / self.fps

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
