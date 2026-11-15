"""ACT policy construction and checkpoint I/O.

The model itself is LeRobot's maintained implementation (`lerobot.policies.act`, LeRobot 0.7.0):
ResNet image encoder + transformer encoder/decoder predicting `chunk_size` future actions,
trained with an L1 reconstruction loss plus a CVAE KL term. Normalisation (mean/std from the
dataset statistics) lives in LeRobot's pre/post-processor pipelines, saved with the checkpoint.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import torch


def _import_act():
    """Import ACT without executing ``lerobot.policies``'s registry.

    That registry imports every policy. Under Isaac Lab's pinned ``transformers`` one of them
    (G0.5) fails to import, which would block loading ACT inside the simulator. ACT's own modules
    do not need the registry, so the package namespace is registered first and the modules load
    directly. A normal ``lerobot`` install (no incompatible transformers) is unaffected: if the
    package was already imported, this is a no-op.
    """
    import lerobot

    name = "lerobot.policies"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(Path(lerobot.__file__).resolve().parent / "policies")]
        pkg.__package__ = name
        sys.modules[name] = pkg
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    return ACTConfig, ACTPolicy, make_pre_post_processors


ACTConfig, ACTPolicy, make_pre_post_processors = _import_act()
from lerobot.configs import FeatureType  # noqa: E402
from lerobot.utils.feature_utils import dataset_to_policy_features  # noqa: E402

PRETRAINED_DIR = "pretrained_model"


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_act_config(policy_cfg: dict, dataset_features: dict, device: str = "cpu") -> ACTConfig:
    """`dataset_features` is the `features` dict of a LeRobot dataset's meta/info.json."""
    feats = dataset_to_policy_features(dataset_features)
    output_features = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in feats.items() if f.type is not FeatureType.ACTION}
    return ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device=device,
        push_to_hub=False,
        **policy_cfg,
    )


def build_policy(policy_cfg: dict, dataset_features: dict, dataset_stats: dict, device: str = "cpu"):
    """Return (policy, preprocessor, postprocessor) ready for training."""
    config = build_act_config(policy_cfg, dataset_features, device)
    policy = ACTPolicy(config).to(device)
    pre, post = make_pre_post_processors(config, dataset_stats=dataset_stats)
    return policy, pre, post


def count_parameters(policy: torch.nn.Module) -> int:
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)


def save_checkpoint(
    ckpt_dir: str | Path, policy, pre, post, optimizer=None, step: int = 0, metadata: dict | None = None
) -> Path:
    ckpt_dir = Path(ckpt_dir)
    model_dir = ckpt_dir / PRETRAINED_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(model_dir)
    pre.save_pretrained(model_dir)
    post.save_pretrained(model_dir)
    state = {"step": step}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, ckpt_dir / "training_state.pt")
    (ckpt_dir / "metadata.json").write_text(json.dumps({"step": step, **(metadata or {})}, indent=2, default=str))
    return ckpt_dir


def load_checkpoint(ckpt_dir: str | Path, device: str = "auto"):
    """Return (policy, preprocessor, postprocessor, metadata) from a directory written by save_checkpoint."""
    ckpt_dir = Path(ckpt_dir)
    device = resolve_device(device)
    model_dir = ckpt_dir / PRETRAINED_DIR
    policy = ACTPolicy.from_pretrained(model_dir)
    policy.config.device = device
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=model_dir,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    meta_path = ckpt_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return policy, pre, post, metadata
