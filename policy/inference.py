"""Run a trained ACT checkpoint on raw observations.

`ACTInference.act(obs)` takes the same per-step dict that the simulator collector records
(numpy arrays keyed by joint_pos, eef_pos, ..., table_cam) and returns one action. Action
chunking is handled by LeRobot's `select_action` (a new chunk every `n_action_steps` steps).

Sanity check on a dataset frame (no simulator needed):
    python policy/inference.py --checkpoint outputs/checkpoints/act_nvidia_stack_human/best
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dataset.process_dataset import ACTION, ENV_STATE, IMAGE_PREFIX, STATE  # noqa: E402
from policy.act_model import load_checkpoint  # noqa: E402

# Orientations the Franka lift IK-Abs action actually uses: the pose the hand starts in, and
# gripper-down (xyzw). The recorded commands switch between these two discontinuously, so a chunk
# prediction blends them into a non-unit quaternion and the IK controller misses the cube.
_GRIPPER_DOWN_XYZW = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def sanitize_ik_abs_action(action: np.ndarray, eef_quat: np.ndarray | None) -> np.ndarray:
    """Renormalise the pose command of an 8-D IK-Abs action: pos(3), quat xyzw(4), gripper(1)."""
    if action.shape[-1] != 8:
        return action
    out = np.array(action, dtype=np.float32, copy=True)
    quat = out[3:7]
    norm = float(np.linalg.norm(quat))
    quat = quat / norm if norm > 1e-6 else _GRIPPER_DOWN_XYZW.copy()
    candidates = [_GRIPPER_DOWN_XYZW]
    if eef_quat is not None and np.linalg.norm(eef_quat) > 1e-6:
        candidates.append(np.asarray(eef_quat, dtype=np.float32) / np.linalg.norm(eef_quat))
    best, best_dot = quat, -1.0
    for cand in candidates:
        dot = float(np.dot(quat, cand))
        if abs(dot) > best_dot:
            best_dot = abs(dot)
            best = cand * (1.0 if dot >= 0 else -1.0)
    out[3:7] = best if best_dot > 0.5 else quat
    out[7] = 1.0 if out[7] >= 0.0 else -1.0
    return out


def frames_to_batch(frames: dict[str, np.ndarray], task: str) -> dict:
    """LeRobot-style batch from numpy arrays: state (B, D) float, images (B, H, W, 3) uint8 -> (B, 3, H, W) [0, 1]."""
    batch = {}
    for k, v in frames.items():
        if k.startswith(IMAGE_PREFIX):
            batch[k] = torch.from_numpy(np.ascontiguousarray(v)).permute(0, 3, 1, 2).float() / 255.0
        elif k in (STATE, ENV_STATE):
            batch[k] = torch.from_numpy(np.asarray(v, dtype=np.float32))
    n = next(iter(batch.values())).shape[0]
    batch["task"] = [task] * n
    return batch


class ACTInference:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        self.policy, self.pre, self.post, self.meta = load_checkpoint(checkpoint, device)
        # Replan every control step. Executing a whole chunk open-loop replays targets that were
        # computed for an earlier cube position, and the hand misses the grasp.
        self.policy.config.n_action_steps = 1
        self.policy.reset()
        keys = self.meta["provenance"]["keys"]
        self.state_keys, self.env_state_keys, self.image_keys = keys["state"], keys["env_state"], keys["images"]
        self.task = self.meta.get("task", "")

    def reset(self) -> None:
        self.policy.reset()

    def obs_to_frames(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        frames = {STATE: np.concatenate([obs[k] for k in self.state_keys])[None]}
        if self.env_state_keys:
            frames[ENV_STATE] = np.concatenate([obs[k] for k in self.env_state_keys])[None]
        for cam in self.image_keys:
            frames[IMAGE_PREFIX + cam] = obs[cam][None]
        return frames

    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        batch = self.pre(frames_to_batch(self.obs_to_frames(obs), self.task))
        action = self.policy.select_action(batch)
        action = self.post(action).squeeze(0).cpu().numpy()
        action = self._absolute_xy(action, obs)
        return sanitize_ik_abs_action(action, obs.get("eef_quat"))

    def _absolute_xy(self, action: np.ndarray, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Undo training-time xy offsets so the simulator receives an absolute IK target."""
        if self.meta["provenance"].get("action_xy_relative_to") != "object_pos":
            return action
        out = np.array(action, dtype=np.float32, copy=True)
        out[:2] += np.asarray(obs["object_pos"], dtype=np.float32)[:2]
        return out

    @torch.no_grad()
    def predict_chunks(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        """Full action chunks (B, chunk_size, action_dim) in raw action units for a batch of frames."""
        batch = self.pre(frames_to_batch(frames, self.task))
        chunk = self.policy.predict_action_chunk(batch)
        return self.post(chunk).cpu().numpy()


def main():
    from dataset.process_dataset import load_episodes

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--t", type=int, default=0)
    a = p.parse_args()
    inf = ACTInference(a.checkpoint)
    ep = load_episodes(inf.meta["dataset_root"])[a.episode]
    frames = {k: ep[k][a.t : a.t + 1] for k in ep if k == STATE or k == ENV_STATE or k.startswith(IMAGE_PREFIX)}
    chunk = inf.predict_chunks(frames)[0]
    gt = ep[ACTION][a.t : a.t + len(chunk)]
    np.set_printoptions(precision=4, suppress=True)
    print(f"checkpoint step {inf.meta.get('step')} | data source: {inf.meta['provenance']['source']}")
    print("predicted chunk (first 5 steps):\n", chunk[:5])
    print("ground truth    (first 5 steps):\n", gt[:5])
    print("MAE over chunk:", float(np.abs(chunk[: len(gt)] - gt).mean()))


if __name__ == "__main__":
    main()
