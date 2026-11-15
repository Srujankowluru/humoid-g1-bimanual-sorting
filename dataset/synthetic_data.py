"""SYNTHETIC test dataset generator (pipeline verification only).

These are hand-written kinematic trajectories: no physics, no robot, no simulator. Joint values
are smooth placeholder signals, not inverse kinematics. Images are 2D toy renderings. The output
is flagged `source=synthetic` / `synthetic=True` and every figure produced from it says so.

The layout mirrors what `simulation/collect_demonstrations.py` records for the Isaac Lab
Franka lift task (same keys, same 8-D IK-Abs action), so the whole pipeline can be exercised
without Isaac Sim.

Usage:
    python dataset/synthetic_data.py --output data/raw/synthetic_lift_demos.hdf5 --episodes 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.demo_io import DemoWriter, Episode  # noqa: E402

FPS = 50
DOWN_QUAT_XYZW = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # same target orientation as lift_cube_sm.py
HOME_EEF = np.array([0.45, 0.0, 0.40], dtype=np.float32)


def _render(eef: np.ndarray, obj: np.ndarray, gripper_open: float, size: int) -> np.ndarray:
    """Top-down toy image: table (grey), cube (red square), gripper (blue dot, radius ~ height)."""
    img = np.full((size, size, 3), 90, dtype=np.uint8)

    def to_px(p):  # x in [0.2, 0.8] -> rows, y in [-0.3, 0.3] -> cols
        r = int(np.clip((p[0] - 0.2) / 0.6 * (size - 1), 0, size - 1))
        c = int(np.clip((p[1] + 0.3) / 0.6 * (size - 1), 0, size - 1))
        return r, c

    r, c = to_px(obj)
    h = max(2, size // 32 + int(obj[2] * 20))
    img[max(0, r - h) : r + h, max(0, c - h) : c + h] = (200, 30, 30)
    r, c = to_px(eef)
    rad = max(2, int(2 + eef[2] * 15))
    yy, xx = np.ogrid[:size, :size]
    mask = (yy - r) ** 2 + (xx - c) ** 2 <= rad**2
    img[mask] = (30, 30, 220) if gripper_open > 0 else (30, 200, 30)
    return img


def make_episode(rng: np.random.Generator, image_size: int = 128) -> Episode:
    obj0 = np.array([rng.uniform(0.4, 0.6), rng.uniform(-0.2, 0.2), 0.055], dtype=np.float32)
    target = np.array([rng.uniform(0.4, 0.6), rng.uniform(-0.25, 0.25), rng.uniform(0.25, 0.5)], dtype=np.float32)
    above = obj0 + np.array([0, 0, 0.1], dtype=np.float32)

    # (waypoint, gripper command, number of steps)
    segments = [
        (HOME_EEF, 1.0, 10),
        (above, 1.0, int(rng.integers(45, 60))),
        (obj0, 1.0, int(rng.integers(30, 40))),
        (obj0, -1.0, 15),
        (target, -1.0, int(rng.integers(50, 70))),
        (target, -1.0, 20),
    ]
    eef = HOME_EEF.copy() + rng.normal(0, 0.01, 3).astype(np.float32)
    obj = obj0.copy()
    rows = {k: [] for k in ("joint_pos", "joint_vel", "eef_pos", "eef_quat", "object_pos", "target_pos", "table_cam")}
    actions = []
    grasped = False
    q_prev = None
    for goal, grip, n in segments:
        for _ in range(n):
            q = np.concatenate(
                [np.tanh(eef[[0, 1, 2, 0, 1, 2, 0]] * np.array([2, 3, 4, -2, -3, 1, 5])), [0.04, 0.04]]
            ).astype(np.float32)
            if grip < 0:
                q[7:] = 0.025 if grasped else 0.03
            qd = np.zeros(9, np.float32) if q_prev is None else (q - q_prev) * FPS
            q_prev = q
            rows["joint_pos"].append(q)
            rows["joint_vel"].append(qd)
            rows["eef_pos"].append(eef.copy())
            rows["eef_quat"].append(DOWN_QUAT_XYZW)
            rows["object_pos"].append(obj.copy())
            rows["target_pos"].append(target)
            rows["table_cam"].append(_render(eef, obj, grip, image_size))
            actions.append(np.concatenate([goal, DOWN_QUAT_XYZW, [grip]]).astype(np.float32))
            # first-order tracking of the commanded pose, like a stiff IK controller
            eef = eef + 0.12 * (goal - eef) + rng.normal(0, 0.0005, 3).astype(np.float32)
            if grip < 0 and np.linalg.norm(eef - obj) < 0.02:
                grasped = True
            if grasped:
                obj = eef.copy()
    success = bool(obj[2] - obj0[2] >= 0.10)
    return Episode(
        actions=np.stack(actions),
        obs={k: np.stack(v) for k, v in rows.items()},
        success=success,
        termination="success" if success else "time_out",
    )


def generate(output: str | Path, episodes: int, seed: int = 0, image_size: int = 128) -> Path:
    rng = np.random.default_rng(seed)
    with DemoWriter(
        output,
        source="synthetic",
        task="SYNTHETIC-Lift-Cube (kinematic placeholder, no physics)",
        fps=FPS,
        env_args={"generator": "dataset/synthetic_data.py", "seed": seed, "image_size": image_size},
    ) as w:
        for _ in range(episodes):
            w.add_episode(make_episode(rng, image_size))
    return Path(output)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="data/raw/synthetic_lift_demos.hdf5")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    a = p.parse_args()
    path = generate(a.output, a.episodes, a.seed, a.image_size)
    print(f"[SYNTHETIC] wrote {a.episodes} synthetic episodes to {path}")


if __name__ == "__main__":
    main()
