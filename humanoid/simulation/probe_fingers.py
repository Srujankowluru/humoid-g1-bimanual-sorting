"""Print finger-link positions relative to each cube, open and closed, palm down."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HUMANOID_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HUMANOID_ROOT.parent))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from humanoid.simulation.g1_sort_env import hand_layout, load_sim_config, make_env_cfg, read_step_observation  # noqa: E402
from humanoid.simulation.scripted_policy import LEFT_GRASP_QUAT, RIGHT_GRASP_QUAT, WRIST_FROM_CUBE  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True


def main():
    import gymnasium as gym
    import numpy as np
    import torch

    cfg = load_sim_config()
    env_cfg = make_env_cfg(cfg)
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(cfg["task"]["base_id"], cfg=env_cfg)
        env.reset()
        dev = env.unwrapped.device
        _, opened, closed = hand_layout(env)
        robot = env.unwrapped.scene["robot"]
        names = list(robot.data.body_names)

        def t(x):
            return x.torch if hasattr(x, "torch") else x

        def step(hand, n=50):
            obs = read_step_observation(env, cfg)
            left = obs["red_cube_pos"] + WRIST_FROM_CUBE
            right = obs["blue_cube_pos"] + WRIST_FROM_CUBE
            action = np.concatenate([left, LEFT_GRASP_QUAT, right, RIGHT_GRASP_QUAT, hand]).astype(np.float32)
            for _ in range(n):
                with torch.no_grad():
                    env.step(torch.as_tensor(action, device=dev).unsqueeze(0))
            return read_step_observation(env, cfg)

        names_j = list(env.unwrapped.action_manager.get_term("upper_body_ik").cfg.hand_joint_names)

        def k(n):
            return names_j.index(n)

        for t0 in (-1.047, 0.0, 1.047):
            for t1 in (-0.724, 0.0, 1.047):
                for t2 in (0.0, 0.872, 1.745):
                    hand = opened.copy()
                    hand[k("left_hand_thumb_0_joint")] = t0
                    hand[k("left_hand_thumb_1_joint")] = t1
                    hand[k("left_hand_thumb_2_joint")] = t2
                    hand[k("right_hand_thumb_0_joint")] = -t0
                    hand[k("right_hand_thumb_1_joint")] = -t1
                    hand[k("right_hand_thumb_2_joint")] = -t2
                    obs = step(hand, n=25)
                    pos = t(robot.data.body_pos_w)[0].detach().cpu().numpy()
                    origin = t(env.unwrapped.scene.env_origins)[0].detach().cpu().numpy()
                    eef = obs["left_eef_pos"]
                    lows = []
                    for n in names:
                        if "left_hand_" in n and n.endswith("_link"):
                            rel = pos[names.index(n)] - origin - eef
                            lows.append((float(rel[2]), n.split("left_hand_")[1], np.round(rel, 3)))
                    lows.sort()
                    print(f"[thumb] {t0:+.2f} {t1:+.2f} {t2:+.2f} wrist_z={obs['left_eef_pos'][2]:.3f} lowest {lows[0]}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
