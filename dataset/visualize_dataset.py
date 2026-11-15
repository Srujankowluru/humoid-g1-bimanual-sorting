"""Plot recorded trajectories and export example observation images.

    python dataset/visualize_dataset.py --profile nvidia_stack_human

Writes to outputs/figures/:
    demonstration_visualization[_<profile>].png   EEF paths, joint positions, actions
    example_observations[_<profile>].png          camera frames over one episode (if images exist)
    example_frames/<profile>/ep<k>_t<t>.png       individual exported frames
The suffix is omitted for the profile selected in configs/training.yaml.
Every figure title states the data source; synthetic data is labelled SYNTHETIC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dataset.process_dataset import (  # noqa: E402
    ACTION,
    IMAGE_PREFIX,
    STATE,
    get_profile,
    load_dataset_info,
    load_episodes,
    load_train_config,
)


def source_label(provenance: dict) -> str:
    if provenance["synthetic"]:
        return "SYNTHETIC DATA (kinematic placeholder - not robot demonstrations)"
    return f"source: {provenance['source_description']}"


def _cols(names: list[str], prefix: str) -> list[int]:
    return [i for i, n in enumerate(names) if n.startswith(prefix)]


def plot_demonstrations(episodes, state_names, provenance, fps, out: Path, max_eps: int = 6) -> Path:
    eef, joints = _cols(state_names, "eef_pos_"), _cols(state_names, "joint_pos_")
    fig = plt.figure(figsize=(15, 9))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    for ep in episodes[:max_eps]:
        p = ep[STATE][:, eef]
        ax3d.plot(p[:, 0], p[:, 1], p[:, 2], lw=1.2, label=f"ep {ep['episode_index']}")
        ax3d.scatter(*p[0], marker="o", s=15)
        ax3d.scatter(*p[-1], marker="x", s=25)
    ax3d.set(title="End-effector paths (o start, x end)", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    ax3d.legend(fontsize=7)

    ax = fig.add_subplot(2, 2, 2)
    for ep in episodes[:max_eps]:
        t = np.arange(len(ep[STATE])) / fps
        ax.plot(t, ep[STATE][:, eef[2]], lw=1)
    ax.set(title="End-effector height per episode", xlabel="time [s]", ylabel="z [m]")

    ep0 = episodes[0]
    t = np.arange(len(ep0[STATE])) / fps
    ax = fig.add_subplot(2, 2, 3)
    for j, c in enumerate(joints):
        ax.plot(t, ep0[STATE][:, c], lw=1, label=state_names[c])
    ax.set(title=f"Joint positions, episode {ep0['episode_index']}", xlabel="time [s]", ylabel="rad / m")
    ax.legend(fontsize=6, ncol=3)

    ax = fig.add_subplot(2, 2, 4)
    for d in range(ep0[ACTION].shape[1]):
        ax.plot(t, ep0[ACTION][:, d], lw=1, label=f"a{d}")
    ax.set(title=f"Actions, episode {ep0['episode_index']}", xlabel="time [s]")
    ax.legend(fontsize=6, ncol=4)

    lengths = [len(e[ACTION]) for e in episodes]
    fig.suptitle(
        f"Demonstrations: {len(episodes)} episodes, {sum(lengths)} frames @ {fps} Hz\n{source_label(provenance)}",
        color="red" if provenance["synthetic"] else "black",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def export_images(episodes, provenance, out: Path, frames_dir: Path, n: int = 6) -> Path | None:
    keys = [k for k in episodes[0] if k.startswith(IMAGE_PREFIX)]
    if not keys:
        return None
    ep = episodes[0]
    T = len(ep[ACTION])
    ts = np.linspace(0, T - 1, n).astype(int)
    fig, axes = plt.subplots(len(keys), n, figsize=(2.6 * n, 2.8 * len(keys)), squeeze=False)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for r, k in enumerate(keys):
        for c, t in enumerate(ts):
            img = ep[k][t]
            axes[r, c].imshow(img)
            axes[r, c].set_title(f"{k.removeprefix(IMAGE_PREFIX)} t={t}", fontsize=8)
            axes[r, c].axis("off")
            Image.fromarray(img).save(frames_dir / f"ep{ep['episode_index']}_{k.removeprefix(IMAGE_PREFIX)}_t{t}.png")
    fig.suptitle(f"Example observations, episode {ep['episode_index']}\n{source_label(provenance)}",
                 color="red" if provenance["synthetic"] else "black", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "humanoid" / "configs" / "training.yaml"))
    p.add_argument("--profile", default=None)
    a = p.parse_args()
    cfg = load_train_config(a.config)
    prof = get_profile(cfg, a.profile)
    suffix = "" if prof["name"] == cfg["profile"] else f"_{prof['name']}"
    figs = REPO_ROOT / cfg["train"]["figures_dir"]
    provenance, _ = load_dataset_info(prof["root"])
    episodes = load_episodes(prof["root"])
    names = json.loads((Path(prof["root"]) / "meta" / "info.json").read_text())["features"][STATE]["names"]
    print("[visualize]", plot_demonstrations(episodes, names, provenance, int(prof["fps"]), figs / f"demonstration_visualization{suffix}.png"))
    img = export_images(episodes, provenance, figs / f"example_observations{suffix}.png", figs / "example_frames" / prof["name"])
    print("[visualize]", img or "no image features in this dataset")


if __name__ == "__main__":
    main()
