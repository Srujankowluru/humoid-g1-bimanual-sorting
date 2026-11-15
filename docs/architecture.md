# Architecture

Unitree G1 with Dex3 hands in Isaac Sim 6.2 and Isaac Lab 3.0. The scene starts from `IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs`: fixed pelvis and legs, Pink inverse kinematics on both arms. `humanoid/simulation/g1_sort_env.py` removes waist joints from the Pink controlled set, replaces the stock object with a red cube and a blue cube, and adds a pad for each.

The action is 28-dimensional: left wrist pose (position 3, quaternion xyzw 4), right wrist pose (7), and 14 Dex3 hand joints. Wrist targets are written in the world frame; the Isaac Lab action term converts them to the pelvis frame.

Three observation cameras are 128×128 RGB: chest on `torso_link`, left wrist, right wrist. A 960×640 third-person camera is used only for screenshots.

```text
humanoid/simulation/scripted_policy.py     one hand at a time, sideways push
humanoid/simulation/collect_demonstrations.py
        -> humanoid/data/raw/g1_sort_scripted_demos.hdf5   (not collected)
dataset/process_dataset.py                 LeRobot v3.0, when an HDF5 exists
policy/train.py                            ACT config is ready; training has not run
```

This repository is separate from the Franka lift project.
