"""Scripted bimanual sorter for the G1 scene.

The hands stay palm down, which is the orientation the arms can actually reach. The thumb is held
in the pose that only drops ~2.5 cm, and the fingers stay straight, so they stick out ~16 cm
ahead of the wrist along world +y. The wrists themselves stall near y=0.32, so the cubes sit
further forward and the fingers are what touch them.

Each hand slides in x while its fingertips are still short of the platform, then moves forward
to a reachable y and presses sideways. The left hand finishes and lifts clear before the right
hand starts, because the two approach poses meet in the middle.

    lift -> aside -> forward -> lower -> push -> retreat -> done

The demonstrator reads the true cube positions to plan the contact. The training config in
configs/training.yaml would withhold those positions from a vision policy and give it the cameras
and the robot's own joints instead. No G1 policy has been trained.
"""

from __future__ import annotations

import numpy as np

from humanoid.simulation.g1_sort_env import matrix_to_quat_xyzw

PHASES = ("lift", "aside", "forward", "lower", "push", "retreat", "done")


def _side_quat(push_x: float) -> np.ndarray:
    """Fingers (wrist +x) point along the push. Finger spread (wrist +z) is world up, so the
    thumb (wrist -y) is horizontal and the middle finger hangs only ~3 cm below the wrist.
    """
    x = np.array([push_x, 0.0, 0.0], np.float64)
    z = np.array([0.0, 0.0, 1.0], np.float64)
    y = np.cross(z, x)
    return np.array(matrix_to_quat_xyzw(np.column_stack([x, y, z])), np.float32)


# Palm down, fingers pointing forward (world +y). Same orientation for the settle hold.
LEFT_GRASP_QUAT = (0.5, 0.5, 0.5, 0.5)
RIGHT_GRASP_QUAT = (-0.5, -0.5, 0.5, 0.5)
PALM_DOWN_LEFT = LEFT_GRASP_QUAT
PALM_DOWN_RIGHT = RIGHT_GRASP_QUAT


class HandFSM:
    def __init__(self, side: str, cfg: dict, hand_pose: np.ndarray):
        self.side = side
        self.quat = np.array(LEFT_GRASP_QUAT if side == "left" else RIGHT_GRASP_QUAT, np.float32)
        self.cube_key = "red_cube_pos" if side == "left" else "blue_cube_pos"
        self.goal = np.array(cfg["scene"]["red_goal" if side == "left" else "blue_goal"], np.float32)
        s = cfg["scene"]
        self.pad_half = 0.5 * float(s["pad_size"])
        self.table_z = float(s["table_height"])
        self.cube_half = 0.5 * float(s["cube_size"])
        p = cfg["scripted"]
        self.reach_tol = float(p["reach_tolerance"])
        self.max_step = float(p["max_step"])
        self.hand = hand_pose
        # Wrist sits this far from the cube center on the trailing side. The near finger is ~3 cm
        # closer, so 9 cm clears the cube and 4 cm puts that finger into the face.
        self.behind = 0.10
        self.press = 0.07
        self.push_step = 0.002
        self.z_off = -0.005
        # Travel above the cubes. Fingertips are ~16 cm ahead of the wrist, so the x slide stays
        # at y=0.12 and the contact pose is y=0.25, where the fingers cover a cube near y=0.38.
        self.travel_z = 0.96
        self.aside_y = 0.12
        self.contact_y = 0.25
        self.reset()

    def reset(self):
        self.phase = 0
        self.steps_in_phase = 0
        self.cmd = None
        self.anchor = None
        self.lift_xy = None
        self.home_x = None

    @property
    def name(self) -> str:
        return PHASES[self.phase]

    def _advance(self):
        self.phase = min(self.phase + 1, len(PHASES) - 1)
        self.steps_in_phase = 0

    def _on_pad(self, cube) -> bool:
        resting = cube[2] < self.table_z + self.cube_half + 0.02
        return abs(cube[0] - self.goal[0]) < self.pad_half and abs(cube[1] - self.goal[1]) < self.pad_half and resting

    def _direction(self, cube) -> np.ndarray:
        sign = 1.0 if float(self.goal[0]) >= float(cube[0]) else -1.0
        return np.array([sign, 0.0], np.float32)

    def _beside(self, cube, press: float) -> np.ndarray:
        direction = self._direction(cube)
        target = cube.copy()
        target[:2] = cube[:2] - direction * (self.behind - press)
        target[2] = cube[2] + self.z_off
        return target

    def __call__(self, obs: dict) -> tuple[np.ndarray, np.ndarray, bool]:
        eef = obs[f"{self.side}_eef_pos"]
        cube = obs[self.cube_key]
        name = self.name
        if self.anchor is None:
            self.anchor = cube.copy()
            self.lift_xy = eef[:2].copy()
            self.home_x = float(eef[0])
        beside = self._beside(self.anchor, 0.0)
        if name == "lift":
            # Straight up first. The resting fingertips sit in the platform, and the arm will not
            # translate until they are clear.
            # Pull back while rising so the fingertips, which stick out ~16 cm, clear the cubes.
            target = np.array([self.lift_xy[0], min(float(self.lift_xy[1]), 0.14), self.travel_z], np.float32)
        elif name == "aside":
            target = beside.copy()
            target[1] = self.aside_y
            target[2] = self.travel_z
        elif name == "forward":
            target = beside.copy()
            target[1] = self.contact_y
            target[2] = self.travel_z
        elif name == "lower":
            target = beside.copy()
            target[1] = self.contact_y
        elif name == "push":
            # Stay on the trailing face. Follow the cube in x only; y stays where the wrist can reach.
            target = self._beside(cube, self.press)
            target[1] = self.contact_y
        else:
            target = eef.copy() if self.cmd is None else self.cmd.copy()
            target[2] = 1.02
        # Never reach across the body. The waist is shared, and crossing it locks the other arm.
        if self.home_x is not None and name != "lift":
            if self.side == "left":
                target[0] = min(float(target[0]), self.home_x)
            else:
                target[0] = max(float(target[0]), self.home_x)
        target = np.asarray(target, np.float32)
        prev = eef if self.cmd is None else self.cmd
        delta = target - prev
        n = float(np.linalg.norm(delta))
        step = self.push_step if name == "push" else self.max_step
        cmd = target if n <= step else prev + delta * (step / n)
        self.cmd = cmd.astype(np.float32)
        reached = float(np.linalg.norm(eef - target)) < self.reach_tol and np.allclose(self.cmd, target)
        self.steps_in_phase += 1
        xy_there = float(np.linalg.norm(eef[:2] - target[:2])) < 0.02 and np.allclose(self.cmd[:2], target[:2])
        x_there = abs(float(eef[0] - target[0])) < 0.025 and abs(float(self.cmd[0] - target[0])) < 1e-3
        low_enough = float(eef[2]) < float(self.anchor[2]) + 0.04
        if name in ("aside", "forward", "lower", "push") and self._on_pad(cube):
            self.phase = PHASES.index("retreat")
            self.steps_in_phase = 0
        elif name == "lift" and self.steps_in_phase > 10 and float(eef[2]) > self.travel_z - 0.03:
            self._advance()
        elif name == "aside" and self.steps_in_phase > 15 and x_there:
            self._advance()
        elif name == "forward" and self.steps_in_phase > 15 and (
            xy_there or (self.steps_in_phase > 40 and abs(float(eef[0] - target[0])) < 0.02)
        ):
            self._advance()
        elif name == "lower" and self.steps_in_phase > 10 and low_enough and x_there:
            self._advance()
        elif name == "retreat" and (reached or (self.steps_in_phase > 20 and float(eef[2]) > 0.97)):
            self._advance()
        elif name == "push" and self._on_pad(cube):
            self._advance()
        return self.cmd, self.hand, self.name == "done"


class BimanualSortPolicy:
    def __init__(self, cfg: dict, hand_index: dict, hand_open: np.ndarray, hand_closed: np.ndarray):
        del hand_closed
        self.hand_pose = hand_open
        li, ri = hand_index["left"], hand_index["right"]
        self.left = HandFSM("left", cfg, hand_open[li])
        self.right = HandFSM("right", cfg, hand_open[ri])
        self.hand_index = hand_index

    def reset(self):
        self.left.reset()
        self.right.reset()
        self.right_hold = None

    def __call__(self, obs: dict, hand_vel: np.ndarray) -> tuple[np.ndarray, bool]:
        del hand_vel
        li, ri = self.hand_index["left"], self.hand_index["right"]
        lpos, lhand, ldone = self.left(obs)
        # Both hands lift together. After that the right hand holds, clear of the table, until the
        # left has finished: reaching across the body rolls the shared waist and locks the other arm.
        if self.right_hold is None or self.right.name == "lift" or self.left.name in ("retreat", "done"):
            rpos, rhand, rdone = self.right(obs)
            self.right_hold = np.array(rpos, np.float32).copy()
        else:
            rpos, rhand, rdone = self.right_hold, self.right.hand, False
        hand = np.zeros(14, np.float32)
        hand[li] = lhand
        hand[ri] = rhand
        action = np.concatenate([lpos, self.left.quat, rpos, self.right.quat, hand]).astype(np.float32)
        return action, ldone and rdone
