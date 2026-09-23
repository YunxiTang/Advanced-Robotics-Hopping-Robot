"""A camera-driven local planner: walk to a goal, around what the head sees.

The loop, run at `PlannerParams.rate` (10 Hz) on top of the 2 kHz physics:

1. **Perceive.** Render the head camera's depth image and back-project it to
   world points (`camera.depth_to_points`). Keep the points between
   `z_min` and `z_max` above the floor - obstacles the body would hit - and
   drop the floor, anything over the robot's head, and the robot's own
   swinging hands (anything within `self_radius` of its CoM).
2. **Remember.** Bin those points into a world-frame 2-D grid of occupied
   cells. The camera sees ~89 deg ahead, so an obstacle leaves its view
   well before the robot has walked past it; the grid remembers it (the
   world is static, so a cell is only forgotten once it is `forget_beyond`
   metres behind).
3. **Route.** A wavefront (grid Dijkstra) from the goal over a coarser copy
   of that grid, with each obstacle grown by the body's radius, gives every
   cell its walking distance to the goal *around* what has been seen.
   Unseen space counts as free.
4. **Choose a command.** A dynamic-window-style search over (forward speed,
   turn rate) pairs. Each candidate is rolled out for `horizon` seconds as
   a unicycle, ramping from the current command at the same acceleration
   limits the command itself is slewed with. Candidates that bring the body
   (a disc of `robot_radius`) into an occupied cell are discarded; the rest
   are scored on the cost-to-go a little ahead of their best point (which
   rewards both progress and facing the right way), clearance and speed,
   and the cheapest wins. Scoring against straight-line distance instead
   is the classic failure of this kind of planner: with an obstacle dead
   ahead, creeping straight at it slower and slower always looks better
   than turning away from the goal, and the robot stalls in front of it.
5. **Command.** Slew `ctrl.p.Vs` / `ctrl.p.yaw_rate` toward the winner. The
   MPC and the gait below it are untouched: the planner steers the robot
   the same way a joystick would, which is what those set-points are.

Pose (CoM position, trunk yaw) is read from the simulator, i.e. perfect
odometry; only obstacles come from the camera.

It only knows what the camera has shown it: a route through unseen space
is assumed open, and a detour is found once the obstacle blocking it is in
view. When every moving candidate would collide, it turns on the spot
toward the goal side until one is free.

Run a course with `uv run biped-avoid --course slalom` (see `main`).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .camera import HeadCamera, RGBDRecorder, depth_to_points


@dataclass
class PlannerParams:
    goal: tuple[float, float] = (10.0, 0.0)
    goal_tol: float = 0.3           # m: stop (march in place) within this
    rate: float = 10.0              # Hz, perception + planning
    # Command limits. 0.6 rad/s is a little past validation stage 10's
    # 0.5, and 0.5 m/s keeps the auto gait walking (it runs from 0.7).
    v_max: float = 0.5
    w_max: float = 0.6
    acc_v: float = 0.5              # m/s^2, slew on the forward command
    acc_w: float = 1.5              # rad/s^2, slew on the turn command
    # Body footprint: shoulders are 0.12 m off the centre line and the arms
    # swing outside them, so the trunk and arms fit a 0.2 m radius disc.
    robot_radius: float = 0.2
    comfort: float = 0.3            # m of clearance beyond which it is free
    # Perception
    z_min: float = 0.04             # m above the floor
    z_max: float = 1.0
    max_range: float = 3.0          # m, horizontal
    self_radius: float = 0.3        # m, ignore returns this close to the CoM
    cell: float = 0.05              # m, grid resolution
    pixel_stride: int = 2           # use every n-th row/column of the image
    forget_beyond: float = 5.0      # m from the robot
    # Rollout
    horizon: float = 2.5            # s
    dt: float = 0.1                 # s, rollout step
    n_v: int = 6
    n_w: int = 13
    # Route (cost-to-go) grid
    route_cell: float = 0.1         # m
    route_margin: float = 3.0       # m of grid around the robot and goal
    inflate: float = 0.25           # m: obstacles grown by this for routing
    lookahead: float = 0.3          # m ahead of a rollout's end, where it is scored
    # Cost weights
    w_dist: float = 1.0             # per metre of cost-to-go left
    w_clear: float = 1.0            # (comfort - clearance) / comfort, if > 0
    w_speed: float = 0.2            # (v_max - v) / v_max
    w_smooth: float = 0.05          # |w - w_now| / w_max


class LocalPlanner:
    """Call `update(data, ctrl)` before every control step; it replans at
    `p.rate` and between replans only slews the command toward the last plan.
    """

    def __init__(self, model: mujoco.MjModel, params: PlannerParams | None = None):
        self.p = params or PlannerParams()
        self.cam = HeadCamera(model)
        self.K = self.cam.K
        self.torso = model.body("torso").id
        self.goal = np.asarray(self.p.goal, float)
        self.cells = np.zeros((0, 2), np.int64)   # occupied grid cells
        self.next_t = 0.0
        self.last_t = None
        self.cmd = np.zeros(2)                   # (v, w) sent to the controller
        self.target = np.zeros(2)                # (v, w) chosen by the last plan
        self.plan = np.zeros((0, 3))             # chosen rollout (x, y, yaw)
        self.reached = False
        self.blocked = False                     # last plan found no free path

    # --- state ----------------------------------------------------------------

    def pose(self, data: mujoco.MjData) -> np.ndarray:
        """(x, y, yaw): CoM position and trunk heading."""
        xmat = data.xmat[self.torso].reshape(3, 3)
        com = data.subtree_com[0]
        return np.array([com[0], com[1], np.arctan2(xmat[1, 0], xmat[0, 0])])

    def obstacles(self) -> np.ndarray:
        """Centres of the occupied cells, world xy (Nx2)."""
        return (self.cells + 0.5) * self.p.cell

    # --- perception -------------------------------------------------------------

    def perceive(self, data: mujoco.MjData, pose: np.ndarray) -> None:
        p = self.p
        _, depth = self.cam.capture(data)
        s = p.pixel_stride
        K = self.K.copy()
        K[:2] /= s                    # pixel u' of depth[::s, ::s] is pixel s*u'
        pts = depth_to_points(depth[::s, ::s], K, *self.cam.pose(data))
        rel = np.hypot(pts[:, 0] - pose[0], pts[:, 1] - pose[1])
        keep = ((pts[:, 2] > p.z_min) & (pts[:, 2] < p.z_max)
                & (rel > p.self_radius) & (rel < p.max_range))
        new = np.floor(pts[keep, :2] / p.cell).astype(np.int64)
        cells = np.concatenate((self.cells, new))
        if len(cells):
            cells = np.unique(cells, axis=0)
            centre = (cells + 0.5) * p.cell
            near = np.hypot(centre[:, 0] - pose[0], centre[:, 1] - pose[1]) < p.forget_beyond
            cells = cells[near]
        self.cells = cells

    # --- planning ---------------------------------------------------------------

    def candidates(self) -> np.ndarray:
        p = self.p
        v, w = np.meshgrid(np.linspace(0.0, p.v_max, p.n_v),
                           np.linspace(-p.w_max, p.w_max, p.n_w), indexing="ij")
        return np.stack((v.ravel(), w.ravel()), axis=1)

    def rollout(self, pose: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Unicycle rollouts (C, S, 3) of each target (v, w), each ramping
        from the current command at the slew limits."""
        p = self.p
        steps = round(p.horizon / p.dt)
        C = len(targets)
        x = np.full(C, pose[0]); y = np.full(C, pose[1]); th = np.full(C, pose[2])
        v = np.full(C, self.cmd[0]); w = np.full(C, self.cmd[1])
        out = np.empty((C, steps, 3))
        for k in range(steps):
            v = v + np.clip(targets[:, 0] - v, -p.acc_v * p.dt, p.acc_v * p.dt)
            w = w + np.clip(targets[:, 1] - w, -p.acc_w * p.dt, p.acc_w * p.dt)
            th = th + w * p.dt
            x = x + v * np.cos(th) * p.dt
            y = y + v * np.sin(th) * p.dt
            out[:, k] = np.stack((x, y, th), axis=1)
        return out

    def clearance(self, xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Distance from each point of `xy` (..., 2) to the nearest occupied
        cell, less half a cell (the cell's own extent). inf if none nearby."""
        p = self.p
        obs = self.obstacles()
        if len(obs):
            reach = p.v_max * p.horizon + p.robot_radius + p.comfort
            obs = obs[np.hypot(obs[:, 0] - pose[0], obs[:, 1] - pose[1]) < reach]
        if not len(obs):
            return np.full(xy.shape[:-1], np.inf)
        d = np.linalg.norm(xy[..., None, :] - obs, axis=-1).min(axis=-1)
        return d - 0.5 * p.cell

    def route(self, pose: np.ndarray) -> None:
        """Wavefront cost-to-go to the goal on a grid spanning robot and
        goal: `self.dist[i, j]` is the 8-connected walking distance from cell
        (i, j) to the goal, inf inside (inflated) obstacles."""
        p = self.p
        res = p.route_cell
        lo = np.minimum(pose[:2], self.goal) - p.route_margin
        shape = tuple(np.ceil((np.maximum(pose[:2], self.goal) + p.route_margin - lo)
                              / res).astype(int))
        occ = np.zeros(shape, bool)
        idx = np.floor((self.obstacles() - lo) / res).astype(int)
        idx = idx[((idx >= 0) & (idx < shape)).all(axis=1)]
        occ[idx[:, 0], idx[:, 1]] = True
        # grow by the inflation radius (a disc of cell offsets)
        r = int(np.ceil(p.inflate / res))
        grown = occ.copy()
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di * di + dj * dj <= r * r and (di or dj):
                    grown[max(di, 0):shape[0] + min(di, 0), max(dj, 0):shape[1] + min(dj, 0)] |= \
                        occ[max(-di, 0):shape[0] + min(-di, 0), max(-dj, 0):shape[1] + min(-dj, 0)]
        dist = np.full(shape, np.inf)
        gi = tuple(np.clip(np.floor((self.goal - lo) / res).astype(int), 0, np.array(shape) - 1))
        dist[gi] = 0.0
        steps = [(di, dj, res * np.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                 if di or dj]
        # Relax all cells at once until nothing improves (Bellman-Ford on a
        # grid; each sweep extends the front by one cell).
        for _ in range(4 * sum(shape)):
            pad = np.pad(dist, 1, constant_values=np.inf)
            new = dist.copy()
            for di, dj, c in steps:
                np.minimum(new, pad[1 + di:1 + di + shape[0], 1 + dj:1 + dj + shape[1]] + c,
                           out=new)
            new[grown] = np.inf
            new[gi] = 0.0
            if np.array_equal(new, dist):
                break
            dist = new
        self.dist, self.dist_lo = dist, lo

    def cost_to_go(self, xy: np.ndarray) -> np.ndarray:
        """Look up `self.dist` at points (..., 2). Off the grid, or where
        the goal is walled off, fall back to straight-line distance."""
        euclid = np.linalg.norm(xy - self.goal, axis=-1)
        idx = np.floor((xy - self.dist_lo) / self.p.route_cell).astype(int)
        inside = ((idx >= 0) & (idx < self.dist.shape)).all(axis=-1)
        d = np.full(xy.shape[:-1], np.inf)
        d[inside] = self.dist[idx[inside, 0], idx[inside, 1]]
        d = np.where(inside, d, euclid + self.p.route_margin)
        if not np.isfinite(self.dist).sum() > 1:     # goal unreachable on the map
            return euclid
        return np.where(np.isfinite(d), d, euclid + 10.0)

    def choose(self, pose: np.ndarray) -> np.ndarray:
        p = self.p
        to_goal = self.goal - pose[:2]
        if np.hypot(*to_goal) < p.goal_tol:
            self.reached = True
            self.plan = pose[None]
            return np.zeros(2)

        self.route(pose)
        cand = self.candidates()
        traj = self.rollout(pose, cand)
        clear = self.clearance(traj[..., :2], pose).min(axis=1) - p.robot_radius
        ok = clear > 0.0
        moving = cand[:, 0] > 0.0
        self.blocked = not (ok & moving).any()

        # Score each rollout at its best point, each point `lookahead` ahead
        # along its heading (so a rollout facing the route beats one facing
        # a wall). Near the goal the lookahead shrinks to the distance left,
        # or it would overshoot and make stopping short look best.
        left = np.linalg.norm(traj[..., :2] - self.goal, axis=-1, keepdims=True)
        heading = np.stack((np.cos(traj[..., 2]), np.sin(traj[..., 2])), axis=-1)
        ahead = traj[..., :2] + np.minimum(p.lookahead, left) * heading
        cost = (p.w_dist * self.cost_to_go(ahead).min(axis=1)
                + p.w_clear * np.clip((p.comfort - clear) / p.comfort, 0.0, None)
                + p.w_speed * (p.v_max - cand[:, 0]) / p.v_max
                + p.w_smooth * np.abs(cand[:, 1] - self.cmd[1]) / p.w_max)
        cost[~ok] = np.inf
        if self.blocked:
            # Nothing that moves is free: turn on the spot, toward whichever
            # side the goal is on, until a way opens up.
            side = np.sign(np.angle(np.exp(1j * (np.arctan2(*to_goal[::-1]) - pose[2])))) or 1.0
            best = np.array([0.0, side * p.w_max])
            self.plan = self.rollout(pose, best[None])[0]
            return best
        i = int(np.argmin(cost))
        self.plan = traj[i]
        return cand[i]

    # --- loop -------------------------------------------------------------------

    def update(self, data: mujoco.MjData, ctrl) -> None:
        t = data.time
        if t + 1e-9 >= self.next_t:
            pose = self.pose(data)
            self.perceive(data, pose)
            self.target = self.choose(pose)
            self.next_t += 1.0 / self.p.rate
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        lim = np.array([self.p.acc_v, self.p.acc_w]) * dt
        self.cmd = self.cmd + np.clip(self.target - self.cmd, -lim, lim)
        ctrl.p.Vs, ctrl.p.Vy, ctrl.p.yaw_rate = float(self.cmd[0]), 0.0, float(self.cmd[1])

    def draw(self, scene: mujoco.MjvScene) -> None:
        """Add the obstacle map (red), chosen rollout (green, orange if
        blocked) and goal (blue) to a scene, for the viewer or a video."""
        def add(type_, size, pos, rgba):
            if scene.ngeom >= scene.maxgeom:
                return
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], type_, np.asarray(size, float),
                                np.asarray(pos, float), np.eye(3).ravel(),
                                np.asarray(rgba, np.float32))
            scene.ngeom += 1

        h = 0.5 * self.p.cell
        for c in self.obstacles():
            add(mujoco.mjtGeom.mjGEOM_BOX, (h, h, 0.005), (*c, 0.005), (0.9, 0.15, 0.1, 0.8))
        colour = (1.0, 0.6, 0.0, 0.9) if self.blocked else (0.1, 0.85, 0.2, 0.9)
        for x, y, _ in self.plan[::2]:
            add(mujoco.mjtGeom.mjGEOM_SPHERE, (0.03, 0, 0), (x, y, 0.03), colour)
        add(mujoco.mjtGeom.mjGEOM_CYLINDER, (self.p.goal_tol, 0.005, 0),
            (*self.goal, 0.005), (0.2, 0.4, 1.0, 0.5))

    def close(self) -> None:
        self.cam.close()


# --- obstacle courses ---------------------------------------------------------

@dataclass
class Course:
    goal: tuple[float, float]
    # (type, xy centre, size): box sizes are (half x, half y, half z),
    # cylinders (radius, half height). They stand on the floor.
    obstacles: list = field(default_factory=list)


def _forest(seed: int = 3, n: int = 14) -> list:
    """Random cylinders and boxes in x 2-9 m, |y| < 1.5 m, at least 0.75 m
    apart between surfaces - wide enough for the 0.4 m body plus margin."""
    rng = np.random.default_rng(seed)
    obs = []
    while len(obs) < n:
        xy = rng.uniform((2.0, -1.5), (9.0, 1.5))
        r = rng.uniform(0.1, 0.25)
        if all(np.hypot(*(xy - o[1])) - r - o[3] > 0.75 for o in obs):
            if rng.random() < 0.5:
                obs.append(("cylinder", xy, (r, rng.uniform(0.2, 0.6)), r))
            else:
                obs.append(("box", xy, (r, r, rng.uniform(0.2, 0.6)), r * np.sqrt(2)))
    return [o[:3] for o in obs]


COURSES = {
    # One box square on the straight line to the goal.
    "single": Course((8.0, 0.0), [("box", (3.0, 0.0), (0.25, 0.25, 0.4))]),
    # Alternating posts either side of the line: weave between them.
    "slalom": Course((10.0, 0.0), [
        ("cylinder", (2.5, 0.15), (0.2, 0.5)),
        ("cylinder", (4.5, -0.2), (0.2, 0.5)),
        ("cylinder", (6.5, 0.2), (0.2, 0.5)),
        ("cylinder", (8.5, -0.15), (0.2, 0.5)),
    ]),
    # A wall across the path with a 0.9 m gap off to the left.
    "wall": Course((8.0, 0.0), [
        ("box", (4.0, -1.35), (0.1, 1.15, 0.4)),  # y from -2.5 to -0.2
        ("box", (4.0, 1.6), (0.1, 0.9, 0.4)),     # y from 0.7 to 2.5
    ]),
    "forest": Course((11.0, 0.0), _forest()),
}


def course_model(name: str) -> mujoco.MjModel:
    """biped.xml with the course's obstacles in place of the scenery props,
    and a fixed overhead camera ("overhead") framing the whole course.

    The obstacles are drawn and seen by the camera but do not collide (like
    every other prop): only the feet have collision geometry, so a solid
    obstacle would trip a foot while the trunk passed through it. Whether
    the robot avoided them is measured instead, by `body_clearance`.
    """
    from .sim import MODEL_PATH

    course = COURSES[name]
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    for g in list(spec.worldbody.geoms):
        if g.name.startswith("prop_"):
            spec.delete(g)
    palette = [(0.85, 0.45, 0.2, 1), (0.25, 0.55, 0.85, 1), (0.3, 0.7, 0.35, 1),
               (0.8, 0.75, 0.25, 1), (0.55, 0.35, 0.8, 1)]
    for i, (kind, xy, size) in enumerate(course.obstacles):
        g = spec.worldbody.add_geom()
        g.name = f"obstacle_{i}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX if kind == "box" else mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = list(size) + [0.0] * (3 - len(size))
        g.pos = [xy[0], xy[1], size[-1]]
        g.rgba = palette[i % len(palette)]
        g.contype = g.conaffinity = 0
    goal = spec.worldbody.add_geom()
    goal.name = "goal_marker"
    goal.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    goal.size = [0.3, 0.002, 0]
    goal.pos = [course.goal[0], course.goal[1], 0.002]
    goal.rgba = [0.2, 0.4, 1.0, 0.6]
    goal.contype = goal.conaffinity = 0
    # Looking straight down (a camera looks along its -z), image x = world x.
    # Orthographic, so posts stand as circles instead of leaning outward; its
    # fovy is then the frame's height in metres, sized to fit the course.
    y_extent = max((abs(xy[1]) + max(size[:-1]) for _, xy, size in course.obstacles),
                   default=1.0)
    cam = spec.worldbody.add_camera()
    cam.name = "overhead"
    cam.pos = [0.5 * course.goal[0], 0.0, 5.0]
    cam.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
    cam.fovy = max((course.goal[0] + 3.0) * 9 / 16, 2 * y_extent + 0.5)
    return spec.compile()


class ClearanceMonitor:
    """Tracks the smallest 3-D gap between any drawn part of the robot and
    any obstacle, with `mj_geomDistance` on the true geometry. A diagnostic
    only - the planner never sees it."""

    def __init__(self, model: mujoco.MjModel, rate: float = 20.0):
        self.model = model
        self.obs = [i for i in range(model.ngeom)
                    if model.geom(i).name.startswith("obstacle_")]
        root = model.body("torso").id
        self.robot = [i for i in range(model.ngeom)
                      if model.body_rootid[model.geom_bodyid[i]] == root
                      and model.geom_group[i] == 0]
        self.rate = rate
        self.next_t = 0.0
        self.min_gap = np.inf
        self.fromto = np.zeros(6)

    def update(self, data: mujoco.MjData) -> None:
        if data.time + 1e-9 < self.next_t or not self.obs:
            return
        self.next_t += 1.0 / self.rate
        for o in self.obs:
            for r in self.robot:
                d = mujoco.mj_geomDistance(self.model, data, r, o, 1.0, self.fromto)
                self.min_gap = min(self.min_gap, d)


def main():
    from .controller import MPCGaitParams
    from .sim import run

    ap = argparse.ArgumentParser(
        description="Walk the biped through an obstacle course, steered by "
                    "the head camera's depth image.")
    ap.add_argument("--course", choices=sorted(COURSES), default="slalom")
    ap.add_argument("--v-max", type=float, default=PlannerParams.v_max,
                    help="top forward speed the planner may command, m/s")
    ap.add_argument("--duration", type=float, default=40.0,
                    help="seconds of simulated time (stops early at the goal)")
    ap.add_argument("--view", action="store_true", help="open the interactive viewer")
    ap.add_argument("--head-cam", action="store_true",
                    help="with --view, also show the head camera's images")
    ap.add_argument("--video", type=str, default=None,
                    help="record the run from overhead to this .mp4 file; the "
                         "head camera (colour | depth) goes to <name>_headcam.mp4")
    ap.add_argument("--rgbd", type=str, default=None, metavar="STEM",
                    help="record the head camera (see `biped --rgbd`)")
    args = ap.parse_args()

    if args.view:
        from .sim import _reexec_under_mjpython
        _reexec_under_mjpython("mujoco_sim.local_planner")

    course = COURSES[args.course]
    model = course_model(args.course)
    planner = LocalPlanner(model, PlannerParams(goal=course.goal, v_max=args.v_max))
    monitor = ClearanceMonitor(model)
    reached_at = []
    head_video = None
    if args.video:
        stem = Path(args.video).with_suffix("")
        head_video = stem.with_name(stem.name + "_headcam.mp4")
        # video only: npz_hz=0 skips the raw-frame dump that --rgbd writes
        head_rec = RGBDRecorder(model, head_video, npz_hz=0)

    def on_step(data, ctrl):
        planner.update(data, ctrl)
        if head_video is not None:
            head_rec.maybe_capture(data)
        monitor.update(data)
        if planner.reached and not reached_at:
            reached_at.append(data.time)

    def stop(data):
        # a couple of seconds of marching in place after arriving
        return bool(reached_at) and data.time > reached_at[0] + 2.0

    try:
        log = run(duration=args.duration, params=MPCGaitParams(Vs=0.0),
                  viewer=args.view, video=args.video, rgbd=args.rgbd,
                  head_cam_view=args.head_cam, model=model, on_step=on_step,
                  decorate=planner.draw, video_camera="overhead", stop=stop)
    finally:
        planner.close()
        if head_video is not None:
            head_rec.close()

    a = log.as_arrays()
    fell = a["z"].min() < 0.2
    print(f"course {args.course}: goal {course.goal}, "
          f"{len(course.obstacles)} obstacles")
    if reached_at:
        print(f"reached the goal at t = {reached_at[0]:.1f} s")
    else:
        end = np.array([a["x"][-1], a["y"][-1]])
        print(f"did NOT reach the goal: ended {np.hypot(*(end - course.goal)):.2f} m "
              f"short at ({end[0]:.2f}, {end[1]:.2f})")
    # mj_geomDistance searches only up to 1 m and returns that as the cap
    gap = "over 1 m" if monitor.min_gap >= 1.0 else f"{monitor.min_gap:.3f} m"
    print(f"closest the body came to an obstacle: {gap}"
          + ("  (COLLISION)" if monitor.min_gap <= 0 else ""))
    print(f"fell: {'YES' if fell else 'no'}; max |roll| {np.abs(a['roll']).max():.3f}, "
          f"max |pitch| {np.abs(a['pitch']).max():.3f} rad")
    if args.video:
        print(f"saved video to {args.video} and {head_video}")


if __name__ == "__main__":
    main()
