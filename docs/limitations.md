# Limitations

- Simulation only. No physical Unitree G1 was used.
- The motion that put cubes on the pads is a sideways slide, one hand at a time. It is not a grasp and not a lift of the cube.
- Demonstrations are scripted. They are not human teleoperation.
- The first 9 scripted attempts, with cubes starting off the pads, produced 4 successes. Misses were a hand that did not reach the cube, or the blue cube sliding off the front of the platform. That is the recorded attempt count, not a learned-policy success rate.
- Collection is configured for 60 successful episodes and has not been run to completion. There is no saved G1 HDF5 and no LeRobot dataset.
- ACT training has not been run. There is no checkpoint and no learned-policy success rate.
- `dataset/synthetic_data.py` emits Franka lift placeholders, not G1 episodes.
