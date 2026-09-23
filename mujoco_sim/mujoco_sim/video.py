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


def ffmpeg_writer(path, width: int, height: int, fps: int,
                  scale: int = 1) -> subprocess.Popen:
    """Start ffmpeg encoding raw RGB frames written to its stdin as H.264.

    `scale` upsamples each frame by that integer factor with nearest-
    neighbour filtering, which keeps small sensor images (the head camera's
    424x240) legible in a player without blurring their pixels.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("video recording needs ffmpeg on PATH")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vf = ["-vf", f"scale=iw*{scale}:ih*{scale}:flags=neighbor"] if scale != 1 else []
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-", *vf,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
         "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE)


class VideoRecorder:
    """Call `maybe_capture(data)` after every physics step; it renders a frame
    whenever simulated time has passed the next frame's timestamp, so the
    video plays back in real time whatever the physics step is."""

    def __init__(self, model: mujoco.MjModel, path, fps: int = 30,
                 camera: str = "chase", width: int = 1280, height: int = 720,
                 decorate=None):
        self.path = Path(path)
        # `decorate(scene)`, if given, adds extra geoms (e.g. a planner's
        # path) to each frame after the model's own.
        self.decorate = decorate
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
        self.proc = ffmpeg_writer(self.path, width, height, fps)

    def maybe_capture(self, data: mujoco.MjData) -> None:
        if data.time + 1e-9 < self.next_t:
            return
        self.renderer.update_scene(data, camera=self.camera,
                                   scene_option=self.scene_option)
        if self.decorate is not None:
            self.decorate(self.renderer.scene)
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
