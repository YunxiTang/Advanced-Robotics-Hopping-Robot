"""The Unitree G1 humanoid, loaded for the convex-MPC controller.

`assets/unitree_g1/` is MuJoCo Menagerie's `unitree_g1` (29 DOF, rev 1.0),
copied verbatim with only the meshes `g1.xml` uses. It is left untouched;
everything this package needs is changed here, on an `MjSpec`, at load time:

- **Torque motors instead of position servos.** Menagerie drives every joint
  with a `kp = 500` position actuator. The MPC's output is a ground reaction
  wrench, turned into joint torques by `-J^T w`, so every actuator becomes a
  plain motor whose control range is the joint's own torque limit
  (`actuatorfrcrange`: 88 / 139 N m at the hips and knees, 50 at the ankles).
- **A sole site per foot**, `sole_l` / `sole_r`: the ankle-roll axis
  projected onto the sole, the reference point the MPC's moment arms and
  contact moments are taken about (see `mpc.py`). G1's foot touches the
  floor through four 5 mm spheres 3 cm below that axis, so the sole plane
  is 3.5 cm down.
- **Floor, light, cameras and the `mpc_stand` keyframe**, which `sim.run`
  starts from: knees bent, soles flat on the floor.

Gravity is 9.81 here (the small biped uses 10) and the timestep 1 ms.
Menagerie's `implicitfast` integrator is kept.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

G1_XML = Path(__file__).parent / "assets" / "unitree_g1" / "g1.xml"
SIDES = {"l": "left", "r": "right"}

# Sole geometry in the ankle-roll link's frame (g1.xml's four `foot` spheres:
# centres at x = -0.05 ... 0.12, y = +/-0.025 (heel) and +/-0.03 (toe),
# z = -0.03, radius 0.005).
SOLE_Z = -0.035
FOOT_FRONT = 0.12
FOOT_BACK = 0.05
FOOT_HALF_WIDTH = 0.025

# The crouch the robot starts in: hip pitch, knee, ankle pitch = -a, 2a, -a.
# The thigh (0.31 m) and shin (0.32 m) are nearly equal, so that keeps the
# ankle almost under the hip and the sole level. a = 0.45 puts the CoM at
# 0.645 m, just above the controller's 0.63 m walking height (straight legs:
# 0.704 m); see G1GaitParams.z_des for why it walks that low.
STAND_BEND = 0.45
# Upper-body posture the controller holds, by joint (without the side
# prefix): elbows bent, arms hanging just clear of the hips.
ARM_POSE = {"shoulder_pitch": 0.2, "shoulder_roll": 0.2, "shoulder_yaw": 0.0,
            "elbow": 1.0, "wrist_roll": 0.0, "wrist_pitch": 0.0, "wrist_yaw": 0.0}

# Damping on the three waist joints, applied by the integrator. It matters
# only for the controller's plain `-J^T w` mode (`use_wbc=False`), where the
# waist is held by a joint PD: the MPC measures the 3.8 kg pelvis, the 19 kg
# upper body hangs on it through the waist, and a soft waist let the pelvis
# roll +/-0.02 rad against the torso at the MPC's own rate - the QP chased
# that wobble with a sideways force that flipped sign every solve. The
# damping a stiff waist needs is out of reach of an explicit PD on so light a
# body at a 1 ms step, so it goes here, where `implicitfast` integrates it
# implicitly: a stand-in for the damping a real waist servo runs on board.
# The whole-body QP computes its torques through the full dynamics,
# `qfrc_passive` included, so under it this damping is cancelled exactly.
WAIST_DAMPING = 80.0


def _arm_q(joint: str, side: str) -> float:
    """ARM_POSE for one side: roll and yaw mirror across the sagittal plane."""
    q = ARM_POSE[joint]
    mirrored = joint.endswith("_roll") or joint.endswith("_yaw")
    return -q if (side == "r" and mirrored) else q


def _stand_qpos(model: mujoco.MjModel) -> np.ndarray:
    """`mpc_stand`: the crouch and arm pose, pelvis lowered onto the floor."""
    data = mujoco.MjData(model)
    q = model.qpos0.copy()
    for s, side in SIDES.items():
        for j, v in (("hip_pitch", -STAND_BEND), ("knee", 2 * STAND_BEND),
                     ("ankle_pitch", -STAND_BEND)):
            q[model.jnt_qposadr[model.joint(f"{side}_{j}_joint").id]] = v
        for j in ARM_POSE:
            q[model.jnt_qposadr[model.joint(f"{side}_{j}_joint").id]] = _arm_q(j, s)
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    sole_z = min(data.site_xpos[model.site(f"sole_{s}").id][2] for s in SIDES)
    q[2] -= sole_z
    return q


def load_model(scene: bool = True) -> mujoco.MjModel:
    """Compile G1 with torque motors, sole sites and the `mpc_stand` keyframe."""
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    spec.option.timestep = 0.001
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    for act in spec.actuators:
        joint = spec.joint(act.target)
        lim = np.asarray(joint.actfrcrange, dtype=float)
        act.set_to_motor()
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = lim
        act.inheritrange = 0.0

    # The waist servo's damping goes into the model, where `implicitfast`
    # integrates it implicitly; see WAIST_DAMPING.
    for j in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        spec.joint(j).damping = [WAIST_DAMPING, 0.0, 0.0]

    for s, side in SIDES.items():
        foot = spec.body(f"{side}_ankle_roll_link")
        foot.add_site(name=f"sole_{s}", pos=[0.0, 0.0, SOLE_Z], size=[0.01, 0, 0],
                      rgba=[1, 0, 0, 1], group=5)

    if scene:
        _add_scene(spec)

    # Menagerie's own `stand` keyframe is for its position servos (it carries
    # their ctrl); ours replaces it.
    for key in list(spec.keys):
        spec.delete(key)
    model = spec.compile()
    spec.add_key(name="mpc_stand", qpos=_stand_qpos(model))
    return spec.compile()


def _add_scene(spec: mujoco.MjSpec) -> None:
    """Floor, sky, lights and the chase cameras, as in biped.xml."""
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    # contact-force arrows, scaled for a 330 N robot rather than a 13 N one
    spec.visual.map.force = 0.002
    spec.stat.meansize = 0.1
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           mark=mujoco.mjtMark.mjMARK_EDGE,
                           rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3],
                           markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    mat = spec.add_material(name="grid", texrepeat=[5, 5], texuniform=True,
                            reflectance=0.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                   material="grid")
    pelvis = spec.body("pelvis")
    pelvis.add_camera(name="chase", mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
                      pos=[-1.7, -1.7, 0.5], xyaxes=[0.707, -0.707, 0, 0.1, 0.1, 0.99])
    pelvis.add_camera(name="track", mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
                      pos=[0, -3.5, 0.0], xyaxes=[1, 0, 0, 0, 0.1, 1])
