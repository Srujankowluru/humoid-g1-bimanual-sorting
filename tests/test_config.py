"""Checks that do not start Isaac Sim."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_simulation_config_matches_the_recorded_smoke_test():
    cfg = yaml.safe_load((ROOT / "humanoid" / "configs" / "simulation.yaml").read_text())
    assert cfg["task"]["base_id"] == "IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs"
    assert cfg["task"]["action_dim"] == 28
    assert cfg["cameras"]["resolution"] == 128
    assert cfg["collection"]["num_episodes"] == 60
    assert not (ROOT / "humanoid" / "data" / "raw" / "g1_sort_scripted_demos.hdf5").exists()


def test_vision_profile_has_three_cameras_and_no_cube_state():
    cfg = yaml.safe_load((ROOT / "humanoid" / "configs" / "training.yaml").read_text())
    vision = cfg["profiles"]["g1_sort_vision"]
    assert vision["image_keys"] == ["chest_cam", "left_wrist_cam", "right_wrist_cam"]
    assert vision["env_state_keys"] == []
    assert cfg["profiles"]["g1_sort_privileged_state"]["image_keys"] == []


def test_scripted_demonstrator_pushes_sideways():
    from humanoid.simulation.scripted_policy import PHASES

    assert PHASES == ("lift", "aside", "forward", "lower", "push", "retreat", "done")
    assert "grasp" not in PHASES
