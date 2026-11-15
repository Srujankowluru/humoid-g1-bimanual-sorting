"""Measure the Dex3 hand geometry with the palm down, for several thumb poses.

    python humanoid/simulation/probe_hand.py

For every pose, prints the axis-aligned world bounds of each left-hand link relative to the wrist.
Used to choose the pre-grasp thumb pose and the grasp offset in scripted_policy.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HUMANOID_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HUMANOID_ROOT.parent))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from humanoid.simulation.g1_sort_env import hand_layout, load_sim_config, make_env_cfg, read_step_observation  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from humanoid.simulation.scripted_policy import LEFT_GRASP_QUAT, RIGHT_GRASP_QUAT  # noqa: E402


def main():
    cfg = load_sim_config()
    env_cfg = make_env_cfg(cfg)
    with launch_simulation(env_cfg, args_cli):
        import omni.usd
        from pxr import Usd, UsdGeom

        env = gym.make(cfg["task"]["base_id"], cfg=env_cfg)
        env.reset()
        dev = env.unwrapped.device
        index, opened, closed = hand_layout(env)
        names = list(env.unwrapped.action_manager.get_term("upper_body_ik").cfg.hand_joint_names)
        stage = omni.usd.get_context().get_stage()
        left_wrist = np.array([-0.12, 0.30, 0.86], np.float32)
        right_wrist = np.array([0.12, 0.30, 0.86], np.float32)

        def act(hand):
            a = np.concatenate([left_wrist, LEFT_GRASP_QUAT, right_wrist, RIGHT_GRASP_QUAT, hand]).astype(np.float32)
            for _ in range(60):
                with torch.no_grad():
                    env.step(torch.as_tensor(a, device=dev).unsqueeze(0))

        def k(n):
            return names.index(n)

        poses = {}
        for t0 in (-1.047, 0.0, 1.047):
            for t1 in (-0.724, 0.0, 1.047):
                for t2 in (0.0, 1.745):
                    h = np.zeros(14, np.float32)
                    h[k("left_hand_thumb_0_joint")] = t0
                    h[k("left_hand_thumb_1_joint")] = t1
                    h[k("left_hand_thumb_2_joint")] = t2
                    poses[f"t0={t0:+.2f} t1={t1:+.2f} t2={t2:+.2f}"] = h
        poses["fingers_closed_thumb_closed"] = closed.copy()
        cache_purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        for label, h in poses.items():
            act(h)
            obs = read_step_observation(env, cfg)
            w = obs["left_eef_pos"]
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), cache_purposes)
            out = []
            for link in ("left_hand_palm_link", "left_hand_thumb_1_link", "left_hand_thumb_2_link", "left_hand_index_1_link", "left_hand_middle_1_link"):
                prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/left_hand/{link}")
                if not prim.IsValid():
                    prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/left_wrist_yaw_link/{link}")
                if not prim.IsValid():
                    continue
                r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                lo, hi = np.array(r.GetMin()) - w, np.array(r.GetMax()) - w
                out.append(f"{link.replace('left_hand_', '')}: x[{lo[0]:+.3f},{hi[0]:+.3f}] y[{lo[1]:+.3f},{hi[1]:+.3f}] z[{lo[2]:+.3f},{hi[2]:+.3f}]")
            print(f"[pose] {label} | wrist={np.round(w, 3)}")
            for line in out:
                print("    ", line)
        env.close()


if __name__ == "__main__":
    main()
