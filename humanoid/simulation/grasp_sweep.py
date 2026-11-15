"""Try grasp offsets until a hand lifts its cube. Prints the lift of each cube.

    python humanoid/simulation/grasp_sweep.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HUMANOID_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HUMANOID_ROOT.parent))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from humanoid.simulation.g1_sort_env import hand_layout, load_sim_config, make_env_cfg, matrix_to_quat_xyzw, read_step_observation  # noqa: E402
from humanoid.simulation.scripted_policy import LEFT_GRASP_QUAT, RIGHT_GRASP_QUAT  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def quat_from_axes(x_axis, y_axis):
    x = np.asarray(x_axis, np.float64)
    y = np.asarray(y_axis, np.float64)
    x /= np.linalg.norm(x)
    y = y - x * np.dot(x, y)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.array(matrix_to_quat_xyzw(np.column_stack([x, y, z])), np.float32)


# Palm down, fingers forward (+y): the orientation the policy already uses.
# Fingers down (-z), palm facing the cube from the robot side (+y).
PALM_DOWN = np.array(LEFT_GRASP_QUAT, np.float32)
PALM_DOWN_R = np.array(RIGHT_GRASP_QUAT, np.float32)
FINGERS_DOWN = quat_from_axes((0, 0, -1), (0, -1, 0))


def main():
    cfg = load_sim_config()
    env_cfg = make_env_cfg(cfg, showcase=False)
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(cfg["task"]["base_id"], cfg=env_cfg)
        env.reset()
        dev = env.unwrapped.device
        _, opened, closed = hand_layout(env)

        def read():
            return read_step_observation(env, cfg)

        def step(left, right, lquat, rquat, hand):
            # Right hand is the mirror image, so its wrist axes flip in x.
            action = np.concatenate([left, lquat, right, rquat, hand]).astype(np.float32)
            with torch.no_grad():
                env.step(torch.as_tensor(action, device=dev).unsqueeze(0))

        def mirror_quat(q):
            # Reflect the orientation through the robot's sagittal plane: x -> -x.
            m = np.diag([-1.0, 1.0, 1.0])
            x = np.array([1 - 2 * (q[1] ** 2 + q[2] ** 2), 2 * (q[0] * q[1] + q[2] * q[3]), 2 * (q[0] * q[2] - q[1] * q[3])])
            y = np.array([2 * (q[0] * q[1] - q[2] * q[3]), 1 - 2 * (q[0] ** 2 + q[2] ** 2), 2 * (q[1] * q[2] + q[0] * q[3])])
            x, y = m @ x, m @ y
            return quat_from_axes(x, y)

        r_down = mirror_quat(FINGERS_DOWN)

        trials = []
        for dy in (-0.07, -0.09, -0.11, -0.13):
            for dz in (0.035, 0.055, 0.075):
                for pre in (0.0, 0.5):
                    trials.append(("palm", PALM_DOWN, PALM_DOWN_R, np.array([0.0, dy, dz], np.float32), pre))
        for dz in (0.09, 0.12, 0.15):
            for pre in (0.0, 0.5):
                trials.append(("down", FINGERS_DOWN, r_down, np.array([0.0, -0.02, dz], np.float32), pre))

        print(f"[sweep] {len(trials)} trials", flush=True)
        for i, (kind, lquat, rquat, offset, pre) in enumerate(trials):
            env.reset()
            for _ in range(8):
                obs = read()
                step(obs["left_eef_pos"], obs["right_eef_pos"], lquat, rquat, opened)
            hand = ((1 - pre) * opened + pre * closed).astype(np.float32)
            cmd = {"left": None, "right": None}

            def go(z_extra, hand, steps, follow=True):
                for _ in range(steps):
                    obs = read()
                    for side, key in (("left", "red_cube_pos"), ("right", "blue_cube_pos")):
                        tgt = obs[key] + offset + np.array([0, 0, z_extra], np.float32)
                        prev = obs[f"{side}_eef_pos"] if cmd[side] is None else cmd[side]
                        d = tgt - prev
                        n = float(np.linalg.norm(d))
                        cmd[side] = (tgt if n <= 0.008 else prev + d * (0.008 / n)).astype(np.float32)
                    step(cmd["left"], cmd["right"], lquat, rquat, hand)

            go(0.09, hand, 45)
            go(0.0, hand, 30)
            go(0.0, closed, 30)
            z0 = {k: float(read()[k][2]) for k in ("red_cube_pos", "blue_cube_pos")}
            go(0.10, closed, 40)
            obs = read()
            lifts = {k: float(obs[k][2]) - z0[k] for k in z0}
            print(
                f"[sweep] {i:02d} {kind} off={np.round(offset, 3)} pre={pre:.1f} "
                f"red {lifts['red_cube_pos']:+.3f} blue {lifts['blue_cube_pos']:+.3f}",
                flush=True,
            )
        env.close()


if __name__ == "__main__":
    main()
