"""Unitree G1 humanoid cube-sorting scene for Isaac Lab 3.0 / Isaac Sim 6.2.

Built on the stock `IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs` task (fixed pelvis and legs,
Pink IK on both arms, Dex3 three-finger hands). Changes:

* the steering wheel is replaced by a red cube (`object`, left of centre) and a blue cube
  (`blue_cube`, right of centre), both randomised on every reset;
* two flat pads mark the goals: red pad on the robot's left, blue pad on its right;
* three 128x128 RGB cameras: chest (on `torso_link`), left wrist and right wrist;
* optionally a 960x640 third-person camera used only for screenshots.

Action (28): left wrist pose (pos 3, quat xyzw 4), right wrist pose (7), 14 hand joint targets.
Wrist poses are in the world frame; the action term converts them to the pelvis frame.

The robot stands at the env origin facing world +y, so its left is world -x.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

HUMANOID_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM_CONFIG = HUMANOID_ROOT / "configs" / "simulation.yaml"
TASK_ID = "IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs"
CAMERAS = ("chest_cam", "left_wrist_cam", "right_wrist_cam")


def load_sim_config(path: str | Path = DEFAULT_SIM_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def matrix_to_quat_xyzw(m: np.ndarray) -> tuple[float, float, float, float]:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return (float(x), float(y), float(z), float(w))


def look_at_quat_xyzw(pos, target, up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """Orientation (xyzw, "world" camera convention: +x forward, +z up) that points from pos to target."""
    f = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    f /= np.linalg.norm(f)
    left = np.cross(np.asarray(up, dtype=float), f)
    left /= np.linalg.norm(left)
    return matrix_to_quat_xyzw(np.stack([f, left, np.cross(f, left)], axis=1))


def _camera(sim_utils, CameraCfg, prim_path: str, c: dict, width: int, height: int):
    rot = tuple(c["rot_xyzw"]) if "rot_xyzw" in c else look_at_quat_xyzw(c["pos"], c["look_at"], c.get("up", (0, 0, 1)))
    convention = c.get("convention", "world")
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(c.get("focal_length", 18.0)),
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=tuple(c["pos"]), rot=rot, convention=convention),
    )


def make_env_cfg(cfg: dict, showcase: bool = False):
    """Isaac Lab env config. Requires Isaac Lab (imports are lazy)."""
    import isaaclab.envs.mdp as base_mdp
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import CameraCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    task = cfg["task"]
    env_cfg = parse_env_cfg(TASK_ID, device=task["device"], num_envs=1)
    env_cfg.seed = int(cfg.get("seed", 0))
    env_cfg.episode_length_s = float(task["episode_length_s"])
    ik_action = env_cfg.actions.upper_body_ik
    ik_action.controller.show_ik_warnings = False
    # The waist is shared by both arms. Letting IK roll it to reach with one hand drives it into
    # the joint limit and locks the other arm, so the waist stays at its default pose.
    ik_action.pink_controlled_joint_names = [n for n in ik_action.pink_controlled_joint_names if "waist" not in n]
    # The stock gain (0.075) leaves the wrists crawling. Track position; the palm only has to stay roughly down.
    for ik_task in ik_action.controller.variable_input_tasks:
        if hasattr(ik_task, "orientation_cost"):
            ik_task.orientation_cost = 0.05
            ik_task.gain = 0.2
            ik_task.lm_damping = 20.0
        if getattr(ik_task, "controlled_joints", None):
            ik_task.controlled_joints = [j for j in ik_task.controlled_joints if "waist" not in j]

    scene = env_cfg.scene
    del scene.robot_pov_cam
    # Stock Dex3 gains (stiffness 8, effort 5) stall the fingers on the cube face, so the curl
    # never hooks underneath. Stronger gains let the fingers close through that contact.
    waist = scene.robot.actuators["waist"]
    waist.joint_velocity_limit = {".*": 0.0}
    waist.joint_effort_limit = {".*": 200.0}
    waist.stiffness = {".*": 20000.0}
    waist.damping = {".*": 200.0}
    hands = scene.robot.actuators["hands"]
    hands.stiffness = 25.0
    hands.damping = 2.0
    hands.actuator_effort_limit = 15.0
    s = cfg["scene"]
    # The stock table carries a crate on the robot's right; shift the table so the crate is out of reach.
    scene.packing_table.init_state.pos = tuple(s["table_pos"])

    def cube(color, pos):
        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + ("RedCube" if color[0] > color[2] else "BlueCube"),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos), rot=(0.0, 0.0, 0.0, 1.0)),
            spawn=sim_utils.CuboidCfg(
                size=(s["cube_size"],) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=float(s["cube_mass"])),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.4, dynamic_friction=0.3),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
            ),
        )

    scene.object = cube((0.85, 0.05, 0.05), s["red_spawn"])
    scene.blue_cube = cube((0.05, 0.15, 0.85), s["blue_spawn"])

    def pad(name, color, pos):
        return AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(pos)),
            spawn=sim_utils.CuboidCfg(
                size=(s["pad_size"], s["pad_size"], 0.004),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
            ),
        )

    scene.red_pad = pad("RedPad", (0.95, 0.35, 0.35), s["red_goal"])
    scene.blue_pad = pad("BluePad", (0.35, 0.5, 0.95), s["blue_goal"])
    # The table top stays below the arms. This block is only under the cubes, so the wrists can
    # reach cube height without the thumb planting on the table at the home pose.
    scene.platform = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.40, 0.83)),
        spawn=sim_utils.CuboidCfg(
            size=(0.84, 0.16, 0.06),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.82, 0.82)),
        ),
    )

    cams = cfg["cameras"]
    res = int(cams["resolution"])
    scene.chest_cam = _camera(sim_utils, CameraCfg, "{ENV_REGEX_NS}/Robot/torso_link/ChestCam", cams["chest_cam"], res, res)
    scene.left_wrist_cam = _camera(
        sim_utils, CameraCfg, "{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/LeftWristCam", cams["left_wrist_cam"], res, res
    )
    scene.right_wrist_cam = _camera(
        sim_utils, CameraCfg, "{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/RightWristCam", cams["right_wrist_cam"], res, res
    )
    if showcase:
        sc = cams["showcase_cam"]
        scene.showcase_cam = _camera(sim_utils, CameraCfg, "{ENV_REGEX_NS}/ShowcaseCam", sc, sc["width"], sc["height"])

    # The stock success/drop checks refer to the steering wheel; success is judged by our own code.
    env_cfg.terminations.success = None
    env_cfg.terminations.object_dropping = None

    jitter = cfg["scene"]["spawn_jitter"]
    pose_range = {"x": [-jitter, jitter], "y": [-jitter, jitter], "z": [0.0, 0.0]}
    from isaaclab.utils import configclass

    @configclass
    class SortEventCfg:
        reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
        reset_red = EventTerm(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": pose_range, "velocity_range": {}, "asset_cfg": SceneEntityCfg("object")},
        )
        reset_blue = EventTerm(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": pose_range, "velocity_range": {}, "asset_cfg": SceneEntityCfg("blue_cube")},
        )

    env_cfg.events = SortEventCfg()

    # Default PhysX GPU buffers are sized for thousands of envs (~1 GB); one env needs a fraction.
    if cfg.get("physx_gpu"):
        from isaaclab_physx.physics import PhysxCfg

        if env_cfg.sim.physics is None:
            env_cfg.sim.physics = PhysxCfg()
        for k, v in cfg["physx_gpu"].items():
            setattr(env_cfg.sim.physics, k, int(v))
    return env_cfg


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def read_step_observation(env, cfg: dict) -> dict:
    """Everything recorded per control step for env 0, as numpy arrays (positions in the env frame)."""
    scene = env.unwrapped.scene
    origin = _t(scene.env_origins)[0]
    robot = scene["robot"]
    names = robot.data.body_names
    pose = _t(robot.data.body_link_pose_w)[0]
    li, ri = names.index("left_wrist_yaw_link"), names.index("right_wrist_yaw_link")
    term = env.unwrapped.action_manager.get_term("upper_body_ik")
    jn = robot.data.joint_names
    hand_idx = [jn.index(n) for n in term.cfg.hand_joint_names]
    arm_idx = [i for i, n in enumerate(jn) if any(k in n for k in ("shoulder", "elbow", "wrist", "waist"))]
    jp = _t(robot.data.joint_pos)[0]
    jv = _t(robot.data.joint_vel)[0]
    out = {
        "arm_joint_pos": jp[arm_idx],
        "hand_joint_pos": jp[hand_idx],
        "hand_joint_vel": jv[hand_idx],
        "left_eef_pos": pose[li, :3] - origin,
        "left_eef_quat": pose[li, 3:7],
        "right_eef_pos": pose[ri, :3] - origin,
        "right_eef_quat": pose[ri, 3:7],
        "red_cube_pos": _t(scene["object"].data.root_pos_w)[0] - origin,
        "blue_cube_pos": _t(scene["blue_cube"].data.root_pos_w)[0] - origin,
    }
    for cam in CAMERAS:
        out[cam] = _t(scene[cam].data.output["rgb"])[0, ..., :3]
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def hand_layout(env) -> tuple[dict, np.ndarray, np.ndarray]:
    """Indices of each hand's joints in the 14-D hand action, the open (pre-grasp) and closed targets.

    Open keeps the index and middle fingers straight and holds the thumb up (thumb_2 straight). A
    fully curled thumb hangs about 4 cm below the wrist and hits the table before the fingers reach
    the cube; this pose hangs about 2.5 cm, which is enough to get the fingertips beside the cube.
    """
    robot = env.unwrapped.scene["robot"]
    term = env.unwrapped.action_manager.get_term("upper_body_ik")
    names = list(term.cfg.hand_joint_names)
    jn = robot.data.joint_names
    lim = _t(robot.data.joint_pos_limits)[0].detach().cpu().numpy()
    closed = np.zeros(len(names), np.float32)
    opened = np.zeros(len(names), np.float32)
    for k, n in enumerate(names):
        lo, hi = lim[jn.index(n)]
        closed[k] = lo if abs(lo) > abs(hi) else hi
    # The right hand mirrors the left; the symmetric thumb_0 range would otherwise pick the same sign.
    for k, n in enumerate(names):
        if n.startswith("right_"):
            lo, hi = lim[jn.index(n)]
            closed[k] = np.clip(-closed[names.index("left_" + n[len("right_"):])], lo, hi)
    # Measured: this thumb pose is the one whose lowest link stays highest (see probe_fingers.py).
    thumb_open = {
        "thumb_0": -1.047,
        "thumb_1": 1.047,
        "thumb_2": 0.0,
    }
    for k, n in enumerate(names):
        sign = -1.0 if n.startswith("right_") else 1.0
        for joint, value in thumb_open.items():
            if joint in n:
                lo, hi = lim[jn.index(n)]
                opened[k] = float(np.clip(sign * value, lo, hi))
    index = {
        "left": [k for k, n in enumerate(names) if n.startswith("left_")],
        "right": [k for k, n in enumerate(names) if n.startswith("right_")],
    }
    return index, opened, closed


def cube_on_pad(cube_pos: np.ndarray, goal: np.ndarray, cfg: dict) -> bool:
    s = cfg["scene"]
    half = 0.5 * float(s["pad_size"])
    resting = cube_pos[2] < float(s["table_height"]) + 0.5 * float(s["cube_size"]) + 0.02
    return bool(abs(cube_pos[0] - goal[0]) < half and abs(cube_pos[1] - goal[1]) < half and resting)


def read_showcase(env):
    return _t(env.unwrapped.scene["showcase_cam"].data.output["rgb"])[0, ..., :3].detach().cpu().numpy()
