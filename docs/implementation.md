# Implementation

- `humanoid/simulation/g1_sort_env.py` builds the env config and reads observations, including the three cameras.
- `humanoid/simulation/scripted_policy.py` is a state machine. The wrists stay in a palm-down orientation the arms can reach. A palm-down grasp does not close on the cube: the thumb meets the table first. Each hand therefore slides in from the side, presses the cube toward its pad, and retreats. The left hand finishes before the right hand starts. The phase named `lift` moves the wrist clear of the table. It does not lift the cube.
- The demonstrator reads simulator cube positions. `humanoid/configs/training.yaml` defines `g1_sort_vision`, which would train on joints, wrist poses, and the three cameras, with no cube coordinates, and `g1_sort_privileged_state`, which would include cube positions and no cameras. Neither has been trained.
- `humanoid/simulation/collect_demonstrations.py` is set to keep successful episodes only, up to 60, with at most 120 attempts. The output HDF5 is not on disk.
- `dataset/` and `policy/` convert HDF5 to LeRobot v3.0 and train LeRobot 0.7.0 `ACTPolicy` (chunk 25, `dim_model` 256). Their default config is `humanoid/configs/training.yaml`.
- `dataset/synthetic_data.py` is a Franka-shaped kinematic placeholder copied with the dataset tools. It does not generate G1 demonstrations. Do not treat its output as G1 data.
- `probe_scene.py`, `probe_hand.py`, `probe_fingers.py`, and `grasp_sweep.py` are diagnostic scripts from the grasp investigation. They are not the sorting demonstrator.

Smoke-test record: `humanoid/outputs/metrics/sim_smoke_test.json` (`passed: true`, action box shape `(1, 28)`, three 128×128 cameras).
