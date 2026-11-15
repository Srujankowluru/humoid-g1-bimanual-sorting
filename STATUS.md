# STATUS

Last updated: 2026-11-15. The narrative record is also in `humanoid/STATUS.md`.

## COMPLETED

- Isaac Sim scene for a fixed-base Unitree G1 with Dex3 hands, a red cube, a blue cube, and a pad for each. Pink IK on both arms; waist joints removed from the controlled set.
- Smoke test passed. `humanoid/outputs/metrics/sim_smoke_test.json` records a 28-dimensional action and 128×128 images from the chest, left wrist, and right wrist cameras.
- Scripted sideways sorter. One successful episode is `humanoid/screenshots/sort_success.png`. With cubes starting off the pads, the first 9 attempts produced 4 successes.
- LeRobot v3 conversion scripts and `humanoid/configs/training.yaml` (vision profile and privileged-state profile). Default `--config` for those scripts points at that file.
- Unit tests that do not start Isaac Sim: `tests/test_config.py`. Re-run 2026-11-15: 3 passed. This folder has no virtualenv; the run used the Franka project's `.venv`, which already has PyYAML, NumPy, and pytest. A fresh install from `requirements.txt` was not performed here.

## BLOCKED_EXTERNAL_ACTION

- **GitHub upload.** This folder is prepared for the name `humoid-g1-bimanual-sorting`. No remote exists and nothing was pushed. Upload it yourself.
- Isaac Sim is required to collect demonstrations. That install lives with the Franka project environment on this machine. This cleanup did not reinstall it and did not launch the simulator again.

## NOT_COMPLETED

- Collection of 60 successful episodes. Configured in `humanoid/configs/simulation.yaml`. No HDF5 has been saved under `humanoid/data/`.
- LeRobot dataset conversion. There is nothing to convert.
- ACT training on G1 demonstrations. No checkpoint. No learned-policy success rate.
- Physical robot. None of these frames are from hardware, and the demonstrations are not human teleoperation.
