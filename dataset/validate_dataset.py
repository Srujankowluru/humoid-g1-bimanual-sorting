"""Consistency checks for a converted LeRobot dataset.

    python dataset/validate_dataset.py --profile nvidia_stack_human [--min-length 20]

Checks: required features and provenance present, feature dimensions, per-episode frame counts,
frame_index contiguity, timestamp spacing == 1/fps, NaN/Inf (missing values), constant-zero
actions, unit-norm quaternions (eef_quat), image dtype/shape, minimum episode length (>= ACT
chunk size by default) and a disjoint, complete train/val split.
Writes outputs/metrics/dataset_validation_<profile>.json and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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


def validate(root: str | Path, fps: int, min_length: int, expected: dict | None = None) -> dict:
    """Return {"passed": bool, "checks": [...], "summary": {...}}. `expected` maps feature -> dim."""
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})

    root = Path(root)
    check("provenance.json present", (root / "provenance.json").exists())
    check("split.json present", (root / "split.json").exists())
    provenance, split = load_dataset_info(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    features = info["features"]
    for key in (STATE, ACTION):
        check(f"feature {key} present", key in features)
    check("fps matches config", info["fps"] == fps, f"info.fps={info['fps']} config={fps}")
    for key, dim in (expected or {}).items():
        got = features.get(key, {}).get("shape", [None])[0]
        check(f"{key} dim == {dim}", got == dim, f"got {got}")

    episodes = load_episodes(root)
    check("episode count matches metadata", len(episodes) == info["total_episodes"], len(episodes))
    check("frame count matches metadata", sum(len(e[ACTION]) for e in episodes) == info["total_frames"])

    names = features[STATE].get("names") or []
    quat_cols = [i for i, n in enumerate(names) if n.startswith("eef_quat_")]
    lengths = []
    for ep in episodes:
        e, T = ep["episode_index"], len(ep[ACTION])
        lengths.append(T)
        for key in (STATE, ENV_STATE, ACTION):
            if key not in ep:
                continue
            arr = ep[key].astype(np.float64)
            check(f"ep{e} {key} same length as action", len(arr) == T)
            check(f"ep{e} {key} no NaN/Inf", np.isfinite(arr).all(), f"{(~np.isfinite(arr)).sum()} bad values")
        check(f"ep{e} frame_index contiguous", np.array_equal(ep["frame_index"], np.arange(T)))
        dt = np.diff(ep["timestamp"])
        check(f"ep{e} timestamps 1/fps", T < 2 or np.allclose(dt, 1.0 / fps, atol=1e-4), f"dt range {dt.min():.5f}..{dt.max():.5f}" if T > 1 else "")
        check(f"ep{e} length >= {min_length}", T >= min_length, T)
        check(f"ep{e} action not all zero", np.abs(ep[ACTION]).sum() > 0)
        if len(quat_cols) == 4:
            norms = np.linalg.norm(ep[STATE][:, quat_cols], axis=1)
            check(f"ep{e} eef_quat unit norm", np.allclose(norms, 1.0, atol=1e-2), f"norm range {norms.min():.4f}..{norms.max():.4f}")
        for key in (k for k in ep if k.startswith(IMAGE_PREFIX)):
            img = ep[key]
            check(f"ep{e} {key} uint8 HxWx3", img.dtype == np.uint8 and img.ndim == 4 and img.shape[-1] == 3, img.shape)
            check(f"ep{e} {key} not blank", img.std() > 1.0, f"std={img.std():.2f}")

    tr, va = set(split["train"]), set(split["val"])
    check("split train/val disjoint", not (tr & va))
    check("split covers all episodes", tr | va == set(range(len(episodes))))
    check("split has val episodes", len(va) > 0)

    failed = [c for c in checks if not c["passed"]]
    return {
        "passed": not failed,
        "num_checks": len(checks),
        "num_failed": len(failed),
        "failed": failed,
        "summary": {
            "root": str(root),
            "source": provenance["source"],
            "synthetic": provenance["synthetic"],
            "episodes": len(episodes),
            "frames": int(sum(lengths)),
            "episode_length_min_mean_max": [int(min(lengths)), float(np.mean(lengths)), int(max(lengths))],
            "features": {k: v["shape"] for k, v in features.items() if k.startswith("observation") or k == ACTION},
            "split": {"train": len(tr), "val": len(va)},
        },
        "checks": checks,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "humanoid" / "configs" / "training.yaml"))
    p.add_argument("--profile", default=None)
    p.add_argument("--min-length", type=int, default=None, help="default: policy.chunk_size")
    a = p.parse_args()
    cfg = load_train_config(a.config)
    prof = get_profile(cfg, a.profile)
    report = validate(prof["root"], int(prof["fps"]), a.min_length or int(cfg["policy"]["chunk_size"]))
    out = REPO_ROOT / cfg["train"]["metrics_dir"] / f"dataset_validation_{prof['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    s = report["summary"]
    print(f"[validate] {prof['name']}: source={s['source']} synthetic={s['synthetic']} episodes={s['episodes']} frames={s['frames']}")
    print(f"[validate] {report['num_checks'] - report['num_failed']}/{report['num_checks']} checks passed -> {out}")
    for c in report["failed"]:
        print(f"  FAIL {c['check']}: {c['detail']}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
