# Humanoid status

Repository status for upload is `STATUS.md` at the repository root. This file is the original scene note.



The Unitree G1 scene runs in Isaac Sim on this machine: fixed base, both Dex3 hands, a red cube, a blue cube, a pad for each, and three cameras (chest, left wrist, right wrist). Stills are in `humanoid/screenshots/`. `sort_success.png` is one episode where both cubes ended on their pads.

The scripted demonstrator can place both cubes. With the cubes starting off the pads, the first 9 attempts produced 4 successes. On the misses the hand never quite reached the cube, or the blue cube slid off the front of the platform. Only successful episodes are kept, and none have been saved into `humanoid/data` yet, so there is still no dataset and no trained G1 policy.

A palm-down grasp does not work on these Dex3 hands: the thumb hits the table before the fingers can close on the cube. The demonstrator slides each cube sideways instead, one hand at a time, with the waist held still so the arms do not lock each other.

`humanoid/configs/training.yaml` is ready for the shared dataset and ACT scripts once demonstrations are collected. Training is not what is missing: there has been nothing to train.
