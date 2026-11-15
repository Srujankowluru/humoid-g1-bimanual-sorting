"""Offline evaluation of an ACT checkpoint on held-out (validation) episodes.

    python policy/evaluate.py --profile nvidia_stack_human [--checkpoint .../best]

Metrics (raw action units unless stated):
  * val_l1_normalised  - same L1 chunk loss as training, eval mode
  * chunk MAE / RMSE   - predicted vs recorded action chunk, per action dim and per horizon step
  * mean-action baseline MAE (predict the training-set mean action) for reference
  * open-loop trajectory comparison: at every step of a val episode the policy sees the recorded
    observation; we compare the first action of each predicted chunk with the recorded action
  * simulator rollout success rate - read from outputs/metrics/rollout_results_<profile>.json, which
    only `simulation/launch_simulation.py --mode rollout` writes. If absent, it is reported as NOT RUN.

Offline errors say nothing about closed-loop task success; that requires the simulator.
Outputs: outputs/metrics/evaluation_<profile>.json, outputs/figures/{trajectory_comparison,evaluation_results}[_<profile>].png
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
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dataset.process_dataset import (  # noqa: E402
    ACTION,
    ENV_STATE,
    IMAGE_PREFIX,
    STATE,
    get_profile,
    load_dataset_info,
    load_episodes,
    load_train_config,
)
from dataset.visualize_dataset import source_label  # noqa: E402
from policy.inference import ACTInference  # noqa: E402
from policy.train import evaluate_loss, make_datasets  # noqa: E402


def chunk_errors(inf: ACTInference, episodes: list[dict], chunk: int, batch: int = 64) -> dict:
    """Predict a chunk at every frame of every episode and compare with the recorded future actions."""
    abs_err, sq_err, counts, firsts = None, None, None, {}
    for ep in episodes:
        T = len(ep[ACTION])
        keys = [k for k in ep if k in (STATE, ENV_STATE) or k.startswith(IMAGE_PREFIX)]
        preds = np.concatenate(
            [inf.predict_chunks({k: ep[k][s : s + batch] for k in keys}) for s in range(0, T, batch)]
        )  # (T, chunk, A)
        A = preds.shape[-1]
        if abs_err is None:
            abs_err, sq_err, counts = np.zeros((chunk, A)), np.zeros((chunk, A)), np.zeros((chunk, 1))
        for h in range(chunk):
            valid = T - h
            if valid <= 0:
                break
            d = preds[:valid, h] - ep[ACTION][h:]
            abs_err[h] += np.abs(d).sum(0)
            sq_err[h] += (d**2).sum(0)
            counts[h] += valid
        firsts[ep["episode_index"]] = preds[:, 0]
    mae_hd = abs_err / np.maximum(counts, 1)
    w = counts / counts.sum()
    return {
        "mae_per_horizon": mae_hd.mean(1),
        "mae_per_dim": (mae_hd * w).sum(0),
        "rmse_per_dim": np.sqrt(((sq_err / np.maximum(counts, 1)) * w).sum(0)),
        "first_actions": firsts,
    }


def plot_trajectory(ep: dict, pred_first: np.ndarray, fps: int, provenance: dict, out: Path) -> None:
    A = ep[ACTION].shape[1]
    cols = 4
    rows = int(np.ceil(A / cols))
    t = np.arange(len(ep[ACTION])) / fps
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)
    for d in range(rows * cols):
        ax = axes[d // cols, d % cols]
        if d >= A:
            ax.axis("off")
            continue
        ax.plot(t, ep[ACTION][:, d], color="black", lw=1.4, label="recorded")
        ax.plot(t, pred_first[:, d], color="tab:orange", lw=1.0, ls="--", label="ACT (open-loop)")
        ax.set_title(f"action[{d}]", fontsize=9)
        ax.set_xlabel("time [s]", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        f"Validation episode {ep['episode_index']}: recorded vs predicted action (policy fed recorded observations)\n"
        f"{source_label(provenance)}",
        fontsize=10,
        color="red" if provenance["synthetic"] else "black",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_results(res: dict, provenance: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    A = len(res["policy_mae_per_dim"])
    x = np.arange(A)
    axes[0].bar(x - 0.2, res["baseline_mae_per_dim"], 0.4, label="mean-action baseline", color="grey")
    axes[0].bar(x + 0.2, res["policy_mae_per_dim"], 0.4, label="ACT", color="tab:orange")
    axes[0].set(title="Action-chunk MAE per action dim (val)", xlabel="action dim", xticks=x)
    axes[0].legend()
    axes[1].plot(np.arange(1, len(res["policy_mae_per_horizon"]) + 1), res["policy_mae_per_horizon"], "o-")
    axes[1].set(title="MAE vs prediction horizon (val)", xlabel="steps ahead", ylabel="MAE (mean over dims)")
    axes[1].grid(alpha=0.3)
    r = res["simulator_rollout"]
    sim_txt = (
        f"success rate {r['success_rate']:.0%} over {r['num_episodes']} episodes"
        if r["status"] == "ran"
        else "NOT RUN - simulator unavailable\n(no task-success claim is made)"
    )
    text = (
        f"profile: {res['profile']}\nsource: {provenance['source']}\nsynthetic: {provenance['synthetic']}\n"
        f"checkpoint: {res['checkpoint']}\nval episodes: {res['num_val_episodes']}  frames: {res['num_val_frames']}\n\n"
        f"val L1 (normalised): {res['val_l1_normalised']:.4f}\n"
        f"ACT chunk MAE: {np.mean(res['policy_mae_per_dim']):.4f}\n"
        f"baseline MAE:  {np.mean(res['baseline_mae_per_dim']):.4f}\n"
        f"first-action MAE: {res['first_action_mae']:.4f}\n\n"
        f"Simulator rollout:\n{sim_txt}"
    )
    axes[2].axis("off")
    axes[2].text(0.0, 1.0, text, va="top", family="monospace", fontsize=9)
    fig.suptitle(f"Offline policy evaluation - {source_label(provenance)}", color="red" if provenance["synthetic"] else "black")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def evaluate(cfg: dict, profile: str | None = None, checkpoint: str | None = None) -> dict:
    prof = get_profile(cfg, profile)
    run = f"act_{prof['name']}"
    ckpt = Path(checkpoint) if checkpoint else REPO_ROOT / cfg["train"]["output_dir"] / run / "best"
    inf = ACTInference(ckpt)
    chunk = inf.policy.config.chunk_size
    provenance, split = load_dataset_info(prof["root"])
    episodes = load_episodes(prof["root"])
    val_eps = [episodes[i] for i in split["val"]]
    train_eps = [episodes[i] for i in split["train"]]

    _, val_ds = make_datasets(prof, split, chunk)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
    val_l1 = evaluate_loss(inf.policy, inf.pre, loader)

    errs = chunk_errors(inf, val_eps, chunk)
    mean_action = np.concatenate([e[ACTION] for e in train_eps]).mean(0)
    all_val_actions = np.concatenate([e[ACTION] for e in val_eps])
    baseline_mae = np.abs(all_val_actions - mean_action).mean(0)
    first_mae = float(np.mean([np.abs(errs["first_actions"][e["episode_index"]] - e[ACTION]).mean() for e in val_eps]))

    rollout_path = REPO_ROOT / cfg["train"]["metrics_dir"] / f"rollout_results_{prof['name']}.json"
    rollout = json.loads(rollout_path.read_text()) if rollout_path.exists() else {"status": "not_run", "reason": "no simulator rollout results found"}
    if rollout_path.exists():
        rollout["status"] = "ran"

    res = {
        "profile": prof["name"],
        "checkpoint": str(ckpt.relative_to(REPO_ROOT)) if ckpt.is_relative_to(REPO_ROOT) else str(ckpt),
        "checkpoint_step": inf.meta.get("step"),
        "source": provenance["source"],
        "synthetic": provenance["synthetic"],
        "num_val_episodes": len(val_eps),
        "num_val_frames": int(len(all_val_actions)),
        "val_l1_normalised": float(val_l1),
        "policy_mae_per_dim": errs["mae_per_dim"].tolist(),
        "policy_rmse_per_dim": errs["rmse_per_dim"].tolist(),
        "policy_mae_per_horizon": errs["mae_per_horizon"].tolist(),
        "policy_chunk_mae": float(errs["mae_per_dim"].mean()),
        "first_action_mae": first_mae,
        "baseline_mae_per_dim": baseline_mae.tolist(),
        "baseline_mae": float(baseline_mae.mean()),
        "simulator_rollout": rollout,
        "note": "Offline, open-loop metrics on recorded observations. They do not measure closed-loop task success.",
    }
    suffix = "" if prof["name"] == cfg["profile"] else f"_{prof['name']}"
    figs = REPO_ROOT / cfg["train"]["figures_dir"]
    figs.mkdir(parents=True, exist_ok=True)
    plot_trajectory(val_eps[0], errs["first_actions"][val_eps[0]["episode_index"]], int(prof["fps"]), provenance, figs / f"trajectory_comparison{suffix}.png")
    plot_results(res, provenance, figs / f"evaluation_results{suffix}.png")
    out = REPO_ROOT / cfg["train"]["metrics_dir"] / f"evaluation_{prof['name']}.json"
    out.write_text(json.dumps(res, indent=2))
    print(
        f"[evaluate] {prof['name']} ({provenance['source']}): val_l1={val_l1:.4f} chunk_MAE={res['policy_chunk_mae']:.4f} "
        f"baseline_MAE={res['baseline_mae']:.4f} first_action_MAE={first_mae:.4f} sim_rollout={rollout['status']} -> {out}"
    )
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "humanoid" / "configs" / "training.yaml"))
    p.add_argument("--profile", default=None)
    p.add_argument("--checkpoint", default=None)
    a = p.parse_args()
    evaluate(load_train_config(a.config), a.profile, a.checkpoint)


if __name__ == "__main__":
    main()
