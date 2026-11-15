"""Print the G1 fixed-base scene layout: link prim paths, wrist poses, hand joint limits.

    python humanoid/simulation/probe_scene.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

TASK = "IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs"


def t(x):
    return x.torch if hasattr(x, "torch") else x


def main():
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env_cfg.scene.robot_pov_cam.height = 128
    env_cfg.scene.robot_pov_cam.width = 128
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        env.reset()
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if p.startswith("/World/envs/env_0/Robot") and any(
                k in p.split("/")[-1] for k in ("wrist_yaw_link", "torso_link", "head_link", "palm", "pelvis")
            ):
                print("[prim]", p, prim.GetTypeName())
        scene = env.unwrapped.scene
        robot = scene["robot"]
        names = robot.data.body_names
        pose = t(robot.data.body_link_pose_w)[0]
        for n in ("pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"):
            i = names.index(n)
            print("[body]", n, [round(float(v), 4) for v in pose[i]])
        print("[root]", [round(float(v), 4) for v in t(robot.data.root_pose_w)[0]])
        print("[object]", [round(float(v), 4) for v in t(scene["object"].data.root_pose_w)[0]])
        print("[origin]", [round(float(v), 4) for v in t(scene.env_origins)[0]])
        jn = robot.data.joint_names
        lim = t(robot.data.joint_pos_limits)[0]
        jp = t(robot.data.joint_pos)[0]
        for i, n in enumerate(jn):
            if "hand" in n:
                print("[hand]", n, round(float(lim[i, 0]), 3), round(float(lim[i, 1]), 3), "now", round(float(jp[i]), 3))
        print("[action_space]", env.action_space)
        term = env.unwrapped.action_manager.get_term("upper_body_ik")
        print("[hand_joint_order]", term.cfg.hand_joint_names)
        li, ri = names.index("left_wrist_yaw_link"), names.index("right_wrist_yaw_link")
        hold = torch.zeros(1, env.action_space.shape[-1], device=env.unwrapped.device)
        hold[0, 0:7] = pose[li]
        hold[0, 7:14] = pose[ri]
        for k in range(60):
            with torch.no_grad():
                env.step(hold)
        pose2 = t(robot.data.body_link_pose_w)[0]
        print("[after_hold] left", [round(float(v), 4) for v in pose2[li]], "right", [round(float(v), 4) for v in pose2[ri]])
        from PIL import Image

        rgb = t(scene["robot_pov_cam"].data.output["rgb"])[0, ..., :3].cpu().numpy()
        Image.fromarray(rgb).save("/tmp/g1_probe_head.png")
        print("[saved] /tmp/g1_probe_head.png")
        env.close()


if __name__ == "__main__":
    main()
