"""Build the G1 sorting scene, hold the arms, open/close the hands, save every camera frame.

    python humanoid/simulation/smoke_test.py

Writes humanoid/outputs/metrics/sim_smoke_test.json and humanoid/outputs/figures/smoke_*.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HUMANOID_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HUMANOID_ROOT.parent))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from humanoid.simulation.g1_sort_env import CAMERAS, load_sim_config, make_env_cfg, read_showcase, read_step_observation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(HUMANOID_ROOT / "configs" / "simulation.yaml"))
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def t(x):
    return x.torch if hasattr(x, "torch") else x


def quat_rotate_inv(q_xyzw, v):
    x, y, z, w = q_xyzw
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return r.T @ v


def main():
    cfg = load_sim_config(args_cli.config)
    env_cfg = make_env_cfg(cfg, showcase=True)
    fig = HUMANOID_ROOT / "outputs" / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(cfg["task"]["base_id"], cfg=env_cfg)
        env.reset()
        dev = env.unwrapped.device
        obs = read_step_observation(env, cfg)
        robot = env.unwrapped.scene["robot"]
        term = env.unwrapped.action_manager.get_term("upper_body_ik")
        jn = robot.data.joint_names
        lim = t(robot.data.joint_pos_limits)[0].cpu().numpy()
        hand_idx = [jn.index(n) for n in term.cfg.hand_joint_names]
        hand_open = np.zeros(14, np.float32)
        hand_closed = np.array(
            [lim[i, 0] if abs(lim[i, 0]) > abs(lim[i, 1]) else lim[i, 1] for i in hand_idx], np.float32
        )
        hold = np.concatenate(
            [obs["left_eef_pos"], obs["left_eef_quat"], obs["right_eef_pos"], obs["right_eef_quat"], hand_open]
        ).astype(np.float32)

        def run(action, n):
            for _ in range(n):
                with torch.no_grad():
                    env.step(torch.as_tensor(action, device=dev).unsqueeze(0))

        run(hold, 50)
        obs = read_step_observation(env, cfg)
        names = robot.data.body_names
        pose = t(robot.data.body_link_pose_w)[0].cpu().numpy()
        report = {
            "action_space": str(env.action_space),
            "red_cube_settled": obs["red_cube_pos"].round(4).tolist(),
            "blue_cube_settled": obs["blue_cube_pos"].round(4).tolist(),
            "left_eef": obs["left_eef_pos"].round(4).tolist(),
            "right_eef": obs["right_eef_pos"].round(4).tolist(),
            "hand_joint_names": term.cfg.hand_joint_names,
            "hand_closed_targets": hand_closed.round(3).tolist(),
            "obs_shapes": {k: list(v.shape) for k, v in obs.items()},
        }
        for side in ("left", "right"):
            w = pose[names.index(f"{side}_wrist_yaw_link")]
            rel = {}
            for i, n in enumerate(names):
                if n.startswith(f"{side}_hand"):
                    rel[n] = quat_rotate_inv(w[3:7], pose[i, :3] - w[:3]).round(4).tolist()
            report[f"{side}_hand_links_in_wrist_frame_open"] = rel
        Image.fromarray(read_showcase(env)).save(fig / "smoke_showcase_open.png")
        for cam in CAMERAS:
            Image.fromarray(obs[cam]).save(fig / f"smoke_{cam}.png")

        closed = hold.copy()
        closed[14:] = hand_closed
        run(closed, 40)
        pose = t(robot.data.body_link_pose_w)[0].cpu().numpy()
        for side in ("left", "right"):
            w = pose[names.index(f"{side}_wrist_yaw_link")]
            rel = {}
            for i, n in enumerate(names):
                if n.startswith(f"{side}_hand"):
                    rel[n] = quat_rotate_inv(w[3:7], pose[i, :3] - w[:3]).round(4).tolist()
            report[f"{side}_hand_links_in_wrist_frame_closed"] = rel
        Image.fromarray(read_showcase(env)).save(fig / "smoke_showcase_closed.png")
        report["passed"] = True
        out = HUMANOID_ROOT / "outputs" / "metrics"
        out.mkdir(parents=True, exist_ok=True)
        (out / "sim_smoke_test.json").write_text(json.dumps(report, indent=2))
        print("[smoke]", json.dumps(report))
        env.close()


if __name__ == "__main__":
    main()
