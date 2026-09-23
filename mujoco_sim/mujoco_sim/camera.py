"""The head-mounted RGB-D camera: rendering, recording and back-projection.

The camera itself is `head_cam` in biped.xml, fixed to the trunk just in
front of the visor and modelled on a RealSense D435 at 424x240. This module
turns it into a sensor:

- `HeadCamera.capture(data)` renders one colour image and one depth image
  (metres along the optical axis, 0 where there is no valid return).
- `RGBDRecorder` records a run: an MP4 with colour and colourised depth side
  by side, and an .npz of the raw frames with the intrinsics and camera
  poses needed to turn them into point clouds.
- `depth_to_points` back-projects a depth image into world coordinates.

Nothing here feeds the controller. The MPC walks blind, as before; the
camera only observes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import mujoco
import numpy as np

from .video import ffmpeg_writer


class HeadCamera:
    """Renders RGB and depth images from a camera in the model.

    Depth comes from the same OpenGL depth buffer the colour pass fills, so
    the two images are pixel-aligned by construction. A real RGB-D sensor
    has two lenses and has to register one image onto the other; this one
    does not.

    Depth is the z-distance along the optical axis (not the ray length), as
    real depth cameras report it. Pixels outside [min_depth, max_depth] are
    set to 0, the usual "no data" value: the lower bound is the model's near
    clipping plane (<map znear> x extent = 0.12 m), and the upper bound
    stands in for the D435's useful range, beyond which it gets noisy.
    """

    def __init__(self, model: mujoco.MjModel, camera: str = "head_cam",
                 width: int = 424, height: int = 240, max_depth: float = 10.0):
        self.model = model
        self.camera = camera
        self.cam_id = model.camera(camera).id
        self.width, self.height = width, height
        self.min_depth = float(model.vis.map.znear * model.stat.extent)
        self.max_depth = max_depth
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    @property
    def K(self) -> np.ndarray:
        """3x3 pinhole intrinsics in the OpenCV convention (pixel (0, 0) is
        the top-left pixel's centre; x right, y down, z forward).

        MuJoCo's `fovy` is the full vertical field of view, and pixels are
        square, so fx = fy.
        """
        fovy = np.deg2rad(self.model.cam_fovy[self.cam_id])
        f = 0.5 * self.height / np.tan(0.5 * fovy)
        return np.array([[f, 0.0, 0.5 * (self.width - 1)],
                         [0.0, f, 0.5 * (self.height - 1)],
                         [0.0, 0.0, 1.0]])

    def pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Camera position and orientation in the world frame.

        The rotation's columns are the camera's axes in MuJoCo's convention:
        x right, y up, looking along -z. `depth_to_points` converts from the
        OpenCV one.
        """
        return (data.cam_xpos[self.cam_id].copy(),
                data.cam_xmat[self.cam_id].reshape(3, 3).copy())

    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Render (rgb, depth): uint8 HxWx3 and float32 HxW in metres."""
        self.renderer.update_scene(data, camera=self.camera)
        rgb = self.renderer.render()
        self.renderer.enable_depth_rendering()
        try:
            depth = self.renderer.render()
        finally:
            self.renderer.disable_depth_rendering()
        depth = np.where((depth >= self.min_depth) & (depth <= self.max_depth),
                         depth, 0.0).astype(np.float32)
        return rgb, depth

    def close(self) -> None:
        self.renderer.close()


_TURBO = (matplotlib.colormaps["turbo"](np.linspace(0.0, 1.0, 256))[:, :3] * 255
          ).astype(np.uint8)


def colorize_depth(depth: np.ndarray, near: float = 0.3, far: float = 5.0) -> np.ndarray:
    """Depth image to a uint8 RGB picture: red near, blue far, black invalid.

    The range is fixed rather than taken from each frame, so a colour means
    the same distance in every frame of a video. 5 m covers what matters to
    a robot this size; everything beyond it is drawn the far colour.
    """
    idx = np.clip((far - depth) / (far - near) * 255.0, 0, 255).astype(np.uint8)
    out = _TURBO[idx]
    out[depth <= 0.0] = 0
    return out


def depth_to_points(depth: np.ndarray, K: np.ndarray, cam_pos: np.ndarray,
                    cam_mat: np.ndarray) -> np.ndarray:
    """Back-project the valid pixels of a depth image to world points (Nx3).

    `K` is OpenCV-convention (from `HeadCamera.K`); `cam_pos`/`cam_mat` are
    the MuJoCo camera pose (from `HeadCamera.pose`, or the `cam_pos` and
    `cam_mat` arrays in a recorded .npz).
    """
    v, u = np.nonzero(depth > 0.0)
    z = depth[v, u].astype(float)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    # OpenCV (x right, y down, z forward) -> MuJoCo (x right, y up, -z forward)
    p_cam = np.stack((x, -y, -z), axis=1)
    return cam_pos + p_cam @ cam_mat.T


def rgbd_stem(path) -> Path:
    """`path` without a trailing .mp4/.npz, so either output (or the bare
    stem) names the pair. Other dots are kept: `run_1.0mps` stays whole."""
    p = Path(path)
    return p.with_suffix("") if p.suffix in (".mp4", ".npz") else p


class RGBDRecorder:
    """Records the head camera over a run. Call `maybe_capture(data)` after
    every physics step, like `VideoRecorder`.

    Writes `<stem>.mp4`, colour and colourised depth side by side at `fps`
    (upsampled 2x, so the 424x240 sensor image is watchable), and
    `<stem>.npz`, the raw frames at `npz_hz`:

        rgb      (N, H, W, 3) uint8
        depth    (N, H, W)    uint16, millimetres, 0 = no data (the format
                              real depth cameras use)
        t        (N,)         simulated time, s
        cam_pos  (N, 3)       camera position, world frame
        cam_mat  (N, 3, 3)    camera orientation, MuJoCo convention
        K        (3, 3)       intrinsics, OpenCV convention

    `npz_hz` is lower than the video rate because raw frames are kept in
    memory until the end: 0.5 MB each, so 30 s at 10 Hz is 150 MB.
    """

    def __init__(self, model: mujoco.MjModel, stem, fps: int = 30, npz_hz: float = 10.0):
        self.stem = rgbd_stem(stem)
        self.cam = HeadCamera(model)
        self.fps = fps
        self.npz_every = max(1, round(fps / npz_hz)) if npz_hz > 0 else 0
        self.next_t = 0.0
        self.frames = 0
        self.proc = ffmpeg_writer(self.stem.with_name(self.stem.name + ".mp4"),
                                  2 * self.cam.width, self.cam.height, fps, scale=2)
        self.rgb, self.depth, self.t, self.cam_pos, self.cam_mat = [], [], [], [], []

    def maybe_capture(self, data: mujoco.MjData) -> None:
        if data.time + 1e-9 < self.next_t:
            return
        rgb, depth = self.cam.capture(data)
        self.proc.stdin.write(np.hstack((rgb, colorize_depth(depth))).tobytes())
        if self.npz_every and self.frames % self.npz_every == 0:
            pos, mat = self.cam.pose(data)
            self.rgb.append(rgb)
            self.depth.append(np.round(depth * 1000.0).astype(np.uint16))
            self.t.append(data.time)
            self.cam_pos.append(pos)
            self.cam_mat.append(mat)
        self.frames += 1
        self.next_t += 1.0 / self.fps

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.stem.name + '.mp4'}")
        if self.t:
            np.savez_compressed(self.stem.with_name(self.stem.name + ".npz"),
                                rgb=np.stack(self.rgb), depth=np.stack(self.depth),
                                t=np.array(self.t), cam_pos=np.stack(self.cam_pos),
                                cam_mat=np.stack(self.cam_mat), K=self.cam.K)
        self.cam.close()
