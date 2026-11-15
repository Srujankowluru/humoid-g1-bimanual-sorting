"""Raw demonstration interface (HDF5), shared by every data source.

Layout follows the Isaac Lab recorder (`scripts/tools/record_demos.py`) so files written by
NVIDIA tools, by `simulation/collect_demonstrations.py` and by `dataset/synthetic_data.py`
can all be converted by `dataset/process_dataset.py`:

    /data                              attrs: source, task, fps, synthetic, env_args (json)
    /data/demo_<i>                     attrs: num_samples, success, termination
    /data/demo_<i>/actions             (T, action_dim) float32
    /data/demo_<i>/obs/<key>           (T, ...) float32, or uint8 (T, H, W, 3) for cameras

`source` must be one of SOURCES. Synthetic data is flagged in the file itself so it can never be
mistaken for a simulator or human demonstration downstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

SOURCES = {
    "isaaclab_scripted": "Scripted state-machine demonstrations recorded in Isaac Lab / Isaac Sim",
    "isaaclab_teleop": "Human teleoperation demonstrations recorded in Isaac Lab / Isaac Sim",
    "nvidia_isaaclab_human": "NVIDIA pre-recorded human teleop demos (Isaac Lab Mimic, Franka cube stack)",
    "synthetic": "SYNTHETIC kinematic trajectories for pipeline testing - NOT robot demonstrations",
}


@dataclass
class Episode:
    actions: np.ndarray
    obs: dict[str, np.ndarray]
    success: bool = False
    termination: str = "unknown"
    attrs: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.actions.shape[0])


class DemoWriter:
    """Append episodes to an Isaac Lab-style HDF5 file."""

    def __init__(self, path: str | Path, source: str, task: str, fps: int, env_args: dict | None = None):
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}; expected one of {sorted(SOURCES)}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(self.path, "w")
        data = self._f.create_group("data")
        data.attrs["source"] = source
        data.attrs["source_description"] = SOURCES[source]
        data.attrs["synthetic"] = source == "synthetic"
        data.attrs["task"] = task
        data.attrs["fps"] = int(fps)
        data.attrs["env_args"] = json.dumps(env_args or {})
        self.num_episodes = 0

    def add_episode(self, ep: Episode) -> None:
        if len(ep) == 0:
            raise ValueError("empty episode")
        for k, v in ep.obs.items():
            if v.shape[0] != len(ep):
                raise ValueError(f"obs/{k} has {v.shape[0]} steps, actions have {len(ep)}")
        g = self._f["data"].create_group(f"demo_{self.num_episodes}")
        g.attrs["num_samples"] = len(ep)
        g.attrs["success"] = bool(ep.success)
        g.attrs["termination"] = ep.termination
        for k, v in ep.attrs.items():
            g.attrs[k] = v
        g.create_dataset("actions", data=ep.actions.astype(np.float32))
        og = g.create_group("obs")
        for k, v in ep.obs.items():
            if v.dtype == np.uint8:
                og.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            else:
                og.create_dataset(k, data=v.astype(np.float32))
        self.num_episodes += 1
        self._f.flush()

    def close(self) -> None:
        self._f["data"].attrs["total"] = sum(
            int(g.attrs["num_samples"]) for g in self._f["data"].values() if isinstance(g, h5py.Group)
        )
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_file_attrs(path: str | Path) -> dict:
    with h5py.File(path, "r") as f:
        return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f["data"].attrs.items()}


def _demo_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


def read_episodes(path: str | Path, obs_keys: list[str] | None = None, successful_only: bool = False):
    """Yield `Episode`s from an Isaac Lab-style HDF5 file, ordered by demo index."""
    with h5py.File(path, "r") as f:
        for name in sorted(f["data"].keys(), key=_demo_sort_key):
            g = f["data"][name]
            success = bool(g.attrs.get("success", True))
            if successful_only and not success:
                continue
            keys = obs_keys if obs_keys is not None else list(g["obs"].keys())
            obs = {k: np.asarray(g["obs"][k]) for k in keys}
            yield Episode(
                actions=np.asarray(g["actions"], dtype=np.float32),
                obs=obs,
                success=success,
                termination=str(g.attrs.get("termination", "unknown")),
                attrs={"name": name},
            )
