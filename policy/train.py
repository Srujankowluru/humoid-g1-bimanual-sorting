"""Train an ACT policy on a converted LeRobot dataset.

    python policy/train.py --profile g1_sort_vision

Train/val episodes come from <dataset>/split.json. Validation loss is the L1 action-chunk error
in eval mode (no VAE posterior, latent = 0), i.e. exactly how the policy is used at inference.
Outputs:
    outputs/checkpoints/act_<profile>/{best,last,step_<n>}/   policy + processors + optimizer
    outputs/metrics/training_log_<profile>.csv
    outputs/figures/training_loss[_<profile>].png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dataset.process_dataset import get_profile, load_dataset_info, load_train_config  # noqa: E402
from dataset.visualize_dataset import source_label  # noqa: E402
from policy.act_model import build_policy, count_parameters, resolve_device, save_checkpoint  # noqa: E402


def make_datasets(prof: dict, split: dict, chunk_size: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    delta = {"action": [i / float(prof["fps"]) for i in range(chunk_size)]}
    train = LeRobotDataset(prof["repo_id"], root=prof["root"], episodes=split["train"], delta_timestamps=delta)
    val = LeRobotDataset(prof["repo_id"], root=prof["root"], episodes=split["val"], delta_timestamps=delta)
    return train, val


@torch.no_grad()
def evaluate_loss(policy, pre, loader, max_batches: int | None = None) -> float:
    """Mean (over valid, non-padded action steps) L1 error in normalised action space, eval mode."""
    policy.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = pre(batch)
        _, info = policy.forward(batch)
        bs = batch["action"].shape[0]
        total += info["l1_loss"] * bs
        n += bs
    policy.train()
    return total / max(n, 1)


def plot_losses(log: list[dict], provenance: dict, out: Path, title: str) -> None:
    steps = [r["step"] for r in log if r["train_loss"] is not None]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, [r["train_loss"] for r in log if r["train_loss"] is not None], lw=0.8, alpha=0.6, label="train total (L1 + kl_weight*KL)")
    ax.plot(steps, [r["train_l1"] for r in log if r["train_loss"] is not None], lw=0.8, label="train L1")
    vs = [(r["step"], r["val_l1"]) for r in log if r["val_l1"] is not None]
    if vs:
        ax.plot(*zip(*vs), "o-", color="black", label="val L1 (eval mode)")
    te = [(r["step"], r["train_eval_l1"]) for r in log if r.get("train_eval_l1") is not None]
    if te:
        ax.plot(*zip(*te), "s--", color="tab:green", label="train L1 (eval mode, 10 batches)")
    ax.set_yscale("log")
    ax.set(xlabel="optimizer step", ylabel="loss (normalised action units)", title=f"{title}\n{source_label(provenance)}")
    ax.title.set_color("red" if provenance["synthetic"] else "black")
    ax.title.set_fontsize(9)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def train(cfg: dict, profile: str | None = None, steps: int | None = None, device: str | None = None) -> dict:
    prof = get_profile(cfg, profile)
    tcfg, pcfg = dict(cfg["train"]), dict(cfg["policy"])
    if steps is not None:
        tcfg["steps"] = steps
    device = resolve_device(device or tcfg["device"])
    torch.manual_seed(int(tcfg["seed"]))
    np.random.seed(int(tcfg["seed"]))

    provenance, split = load_dataset_info(prof["root"])
    train_ds, val_ds = make_datasets(prof, split, int(pcfg["chunk_size"]))
    features = json.loads((Path(prof["root"]) / "meta" / "info.json").read_text())["features"]
    # Normalisation statistics come from LeRobot's per-dataset stats (computed over all episodes).
    policy, pre, post = build_policy(pcfg, features, train_ds.meta.stats, device)
    optim = torch.optim.AdamW(
        policy.get_optim_params(), lr=float(pcfg["optimizer_lr"]), weight_decay=float(pcfg["optimizer_weight_decay"])
    )
    loader_kw = dict(batch_size=int(tcfg["batch_size"]), num_workers=int(tcfg["num_workers"]), pin_memory=device == "cuda")
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > tcfg["batch_size"], **loader_kw)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kw)
    train_eval_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kw)

    run = f"act_{prof['name']}"
    ckpt_root = REPO_ROOT / tcfg["output_dir"] / run
    metrics_dir = REPO_ROOT / tcfg["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run": run,
        "profile": prof["name"],
        "dataset_root": prof["root"],
        "provenance": provenance,
        "split": split,
        "policy_cfg": pcfg,
        "fps": prof["fps"],
        "task": prof["task"],
        "device": device,
        "num_parameters": count_parameters(policy),
    }
    print(
        f"[train] {run}: {len(train_ds)} train frames ({len(split['train'])} eps), {len(val_ds)} val frames "
        f"({len(split['val'])} eps), {meta['num_parameters'] / 1e6:.2f}M params, device={device}, source={provenance['source']}"
    )

    log, best_val, step, t0 = [], math.inf, 0, time.time()
    policy.train()
    while step < int(tcfg["steps"]):
        for batch in train_loader:
            batch = pre(batch)
            loss, info = policy.forward(batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(tcfg["grad_clip_norm"]))
            optim.step()
            step += 1
            row = {"step": step, "train_loss": loss.item(), "train_l1": info["l1_loss"], "train_kld": info.get("kld_loss"), "train_eval_l1": None, "val_l1": None, "elapsed_s": round(time.time() - t0, 1)}
            if step % int(tcfg["eval_every"]) == 0 or step == int(tcfg["steps"]):
                # train L1 above uses the VAE posterior (sees the target actions); eval-mode L1 is comparable to val
                row["train_eval_l1"] = evaluate_loss(policy, pre, train_eval_loader, max_batches=10)
                row["val_l1"] = evaluate_loss(policy, pre, val_loader)
                print(f"[train] step {step:6d}  train_loss {row['train_loss']:.4f}  train_l1 {row['train_l1']:.4f}  train_eval_l1 {row['train_eval_l1']:.4f}  val_l1 {row['val_l1']:.4f}  ({row['elapsed_s']}s)")
                if row["val_l1"] < best_val:
                    best_val = row["val_l1"]
                    save_checkpoint(ckpt_root / "best", policy, pre, post, optim, step, meta | {"val_l1": best_val})
            if step % int(tcfg["save_every"]) == 0:
                save_checkpoint(ckpt_root / f"step_{step:06d}", policy, pre, post, optim, step, meta)
            log.append(row)
            if step >= int(tcfg["steps"]):
                break

    save_checkpoint(ckpt_root / "last", policy, pre, post, optim, step, meta | {"val_l1": log[-1]["val_l1"]})
    log_path = metrics_dir / f"training_log_{prof['name']}.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    suffix = "" if prof["name"] == cfg["profile"] else f"_{prof['name']}"
    fig = REPO_ROOT / tcfg["figures_dir"] / f"training_loss{suffix}.png"
    plot_losses(log, provenance, fig, f"ACT training - {prof['name']} ({step} steps)")
    summary = {
        "run": run,
        "steps": step,
        "final_train_loss": log[-1]["train_loss"],
        "final_train_l1": log[-1]["train_l1"],
        "final_train_eval_l1": log[-1]["train_eval_l1"],
        "final_val_l1": log[-1]["val_l1"],
        "best_val_l1": best_val,
        "train_time_s": round(time.time() - t0, 1),
        "checkpoint_dir": str(ckpt_root.relative_to(REPO_ROOT)),
        "log": str(log_path.relative_to(REPO_ROOT)),
        "figure": str(fig.relative_to(REPO_ROOT)),
        "device": device,
        "num_parameters": meta["num_parameters"],
        "source": provenance["source"],
        "synthetic": provenance["synthetic"],
    }
    (metrics_dir / f"training_summary_{prof['name']}.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done: {json.dumps(summary)}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "humanoid" / "configs" / "training.yaml"))
    p.add_argument("--profile", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    train(load_train_config(a.config), a.profile, a.steps, a.device)


if __name__ == "__main__":
    main()
