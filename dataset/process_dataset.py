"""Convert raw demonstration HDF5 files to LeRobot v3.0 datasets and create train/val splits.

    python dataset/process_dataset.py --profile g1_sort_vision --overwrite

Per frame the LeRobot dataset holds:
    observation.state              concat(state_keys)       float32
    observation.environment_state  concat(env_state_keys)   float32   (if any)
    observation.images.<cam>       (H, W, 3) uint8 image    (if any)
    action                         raw action               float32
Side files in the dataset root (ignored by LeRobot):
    provenance.json   where the demonstrations come from (synthetic / simulator / human)
    split.json        episode indices of the train / val split
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dataset.demo_io import SOURCES, read_episodes, read_file_attrs  # noqa: E402

DEFAULT_TRAIN_CONFIG = REPO_ROOT / "humanoid" / "configs" / "training.yaml"
STATE, ENV_STATE, ACTION = "observation.state", "observation.environment_state", "action"
IMAGE_PREFIX = "observation.images."


def load_train_config(path: str | Path = DEFAULT_TRAIN_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_profile(cfg: dict, name: str | None = None) -> dict:
    name = name or cfg["profile"]
    if name not in cfg["profiles"]:
        raise KeyError(f"unknown dataset profile {name!r}; available: {sorted(cfg['profiles'])}")
    prof = dict(cfg["profiles"][name])
    prof["name"] = name
    for k in ("raw_hdf5", "root"):
        prof[k] = str(REPO_ROOT / prof[k]) if not Path(prof[k]).is_absolute() else prof[k]
    return prof


def infer_source(raw_attrs: dict) -> str:
    """Isaac Lab's own recorder does not write a `source` attr; NVIDIA's public file is recognised by env name."""
    if "source" in raw_attrs:
        return str(raw_attrs["source"])
    env_args = json.loads(raw_attrs.get("env_args", "{}") or "{}")
    if "Stack-Cube-Franka" in env_args.get("env_name", ""):
        return "nvidia_isaaclab_human"
    return "isaaclab_teleop"


def build_features(prof: dict, first_episode) -> dict:
    def names(keys):
        return [f"{k}_{i}" for k in keys for i in range(first_episode.obs[k].shape[1])]

    features = {
        STATE: {"dtype": "float32", "shape": (len(names(prof["state_keys"])),), "names": names(prof["state_keys"])},
        ACTION: {
            "dtype": "float32",
            "shape": (first_episode.actions.shape[1],),
            "names": [f"action_{i}" for i in range(first_episode.actions.shape[1])],
        },
    }
    if prof["env_state_keys"]:
        n = names(prof["env_state_keys"])
        features[ENV_STATE] = {"dtype": "float32", "shape": (len(n),), "names": n}
    for cam in prof["image_keys"]:
        h, w, c = first_episode.obs[cam].shape[1:]
        features[IMAGE_PREFIX + cam] = {"dtype": "image", "shape": (h, w, c), "names": ["height", "width", "channels"]}
    return features


def episode_to_frames(prof: dict, ep) -> dict[str, np.ndarray]:
    actions = ep.actions.astype(np.float32, copy=True)
    # Close as soon as the hand is on the cube, and only then lift straight up. The recorded close
    # command is a single frame, so a reactive policy never emits it. A sideways lift sweeps the cube
    # out of the fingers; the goal height with the cube's own xy does not.
    if prof.get("vertical_lift_when_grasped"):
        obj = ep.obs["object_pos"]
        dist = np.linalg.norm(ep.obs["eef_pos"] - obj, axis=1)
        shut = ep.obs["finger_pos"].mean(axis=1) < 0.03
        hold = (dist < 0.025) & ~shut
        actions[hold, :3] = obj[hold]
        actions[hold, -1] = -1.0
        lift = shut & ((actions[:, -1] < 0) | (dist < 0.025))
        actions[lift, :2] = obj[lift, :2]
        actions[lift, 2] = ep.obs["target_pos"][lift, 2]
        actions[lift, -1] = -1.0
    elif prof.get("lift_on_close"):
        actions[actions[:, -1] < 0, :3] = ep.obs["target_pos"][actions[:, -1] < 0]
    # Predict xy as an offset from the measured cube. An absolute command drifts onto the wrong cube
    # once the hand leaves the demonstrated path; adding the cube position back each step cannot.
    if prof.get("action_xy_relative_to") == "object_pos":
        actions[:, :2] -= ep.obs["object_pos"][:, :2]
    out = {
        STATE: np.concatenate([ep.obs[k] for k in prof["state_keys"]], axis=1).astype(np.float32),
        ACTION: actions,
    }
    if prof["env_state_keys"]:
        out[ENV_STATE] = np.concatenate([ep.obs[k] for k in prof["env_state_keys"]], axis=1).astype(np.float32)
    for cam in prof["image_keys"]:
        out[IMAGE_PREFIX + cam] = ep.obs[cam].astype(np.uint8)
    return out


def make_split(num_episodes: int, val_fraction: float, seed: int) -> dict[str, list[int]]:
    if num_episodes < 2:
        raise ValueError("need at least 2 episodes for a train/val split")
    idx = np.random.default_rng(seed).permutation(num_episodes)
    n_val = min(max(1, round(num_episodes * val_fraction)), num_episodes - 1)
    return {"train": sorted(int(i) for i in idx[n_val:]), "val": sorted(int(i) for i in idx[:n_val])}


def convert(prof: dict, split_cfg: dict, overwrite: bool = False, successful_only: bool = True) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(prof["root"])
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} exists (use --overwrite)")
        shutil.rmtree(root)
    raw_attrs = read_file_attrs(prof["raw_hdf5"])
    source = infer_source(raw_attrs)
    keys = prof["state_keys"] + prof["env_state_keys"] + prof["image_keys"]
    load_keys = [k for k in keys if k != "finger_pos"]
    if "finger_pos" in keys and "joint_pos" not in load_keys:
        load_keys.append("joint_pos")
    episodes = list(read_episodes(prof["raw_hdf5"], obs_keys=load_keys, successful_only=successful_only))
    if not episodes:
        raise RuntimeError(f"no episodes in {prof['raw_hdf5']}")
    if "finger_pos" in keys:
        for ep in episodes:
            ep.obs["finger_pos"] = ep.obs["joint_pos"][:, -2:]

    ds = LeRobotDataset.create(
        repo_id=prof["repo_id"],
        fps=int(prof["fps"]),
        features=build_features(prof, episodes[0]),
        root=root,
        robot_type="franka_panda",
        use_videos=False,
    )
    for ep in episodes:
        frames = episode_to_frames(prof, ep)
        for t in range(len(ep)):
            ds.add_frame({k: v[t] for k, v in frames.items()} | {"task": prof["task"]})
        ds.save_episode()
    ds.finalize()

    split = make_split(len(episodes), float(split_cfg["val_fraction"]), int(split_cfg["seed"]))
    (root / "split.json").write_text(json.dumps(split, indent=2))
    provenance = {
        "source": source,
        "source_description": SOURCES.get(source, "unknown"),
        "synthetic": source == "synthetic",
        "raw_file": str(Path(prof["raw_hdf5"]).relative_to(REPO_ROOT))
        if Path(prof["raw_hdf5"]).is_relative_to(REPO_ROOT)
        else prof["raw_hdf5"],
        "raw_env_args": raw_attrs.get("env_args", ""),
        "raw_task": raw_attrs.get("task", ""),
        "profile": prof["name"],
        "num_episodes": len(episodes),
        "num_frames": int(sum(len(e) for e in episodes)),
        "successful_only": successful_only,
        "keys": {"state": prof["state_keys"], "env_state": prof["env_state_keys"], "images": prof["image_keys"]},
        "lift_on_close": bool(prof.get("lift_on_close", False)),
        "vertical_lift_when_grasped": bool(prof.get("vertical_lift_when_grasped", False)),
        "action_xy_relative_to": prof.get("action_xy_relative_to"),
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(
        f"[process] {provenance['num_episodes']} episodes / {provenance['num_frames']} frames "
        f"(source={source}) -> {root}; split train={len(split['train'])} val={len(split['val'])}"
    )
    return root


def load_dataset_info(root: str | Path) -> tuple[dict, dict]:
    root = Path(root)
    return json.loads((root / "provenance.json").read_text()), json.loads((root / "split.json").read_text())


def load_episodes(root: str | Path, repo_id: str = "local/dataset") -> list[dict[str, np.ndarray]]:
    """Load every episode of a local LeRobot dataset as dict[feature -> (T, ...) numpy array]."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    table = ds.hf_dataset.with_format(None)
    ep_index = np.asarray(table["episode_index"])
    keys = [k for k in (STATE, ENV_STATE, ACTION, "frame_index", "timestamp") if k in ds.meta.features]
    columns = {k: np.asarray(table[k]) for k in keys}
    image_keys = [k for k in ds.meta.features if k.startswith(IMAGE_PREFIX)]
    episodes = []
    for e in range(ds.meta.total_episodes):
        rows = np.flatnonzero(ep_index == e)
        ep = {k: v[rows] for k, v in columns.items()}
        for k in image_keys:
            ep[k] = np.stack([np.asarray(table[int(r)][k].convert("RGB")) for r in rows])
        ep["episode_index"] = e
        episodes.append(ep)
    return episodes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG))
    p.add_argument("--profile", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--include-failed", action="store_true", help="also convert episodes with success=False")
    a = p.parse_args()
    cfg = load_train_config(a.config)
    convert(get_profile(cfg, a.profile), cfg["split"], overwrite=a.overwrite, successful_only=not a.include_failed)


if __name__ == "__main__":
    main()
