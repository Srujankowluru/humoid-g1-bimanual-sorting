"""Collect scripted G1 cube-sorting demonstrations (Isaac Sim 6.2 + Isaac Lab 3.0).

    python humanoid/simulation/collect_demonstrations.py --num_episodes 60
    python humanoid/simulation/collect_demonstrations.py --debug      # one episode, frames + phase log

Every control step records the 28-D action, arm/hand joint state, both wrist poses, both cube
positions (for scoring only) and the chest / left wrist / right wrist RGB frames. Only episodes
where both cubes end on their pads are saved. Output layout: the shared Isaac Lab-style HDF5
(`dataset/demo_io.py` in the repository root), source=isaaclab_scripted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HUMANOID_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HUMANOID_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from humanoid.simulation.g1_sort_env import (  # noqa: E402
    CAMERAS,
    cube_on_pad,
    hand_layout,
    load_sim_config,
    make_env_cfg,
    read_showcase,
    read_step_observation,
)

parser = argparse.ArgumentParser(description="Collect scripted G1 sorting demonstrations.")
parser.add_argument("--config", default=str(HUMANOID_ROOT / "configs" / "simulation.yaml"))
parser.add_argument("--num_episodes", type=int, default=None)
parser.add_argument("--output", default=None)
parser.add_argument("--debug", action="store_true", help="one attempt, save showcase frames, print phases")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from dataset.demo_io import DemoWriter, Episode  # noqa: E402
from humanoid.simulation.scripted_policy import BimanualSortPolicy  # noqa: E402

RECORDED = (
    "arm_joint_pos",
    "hand_joint_pos",
    "left_eef_pos",
    "left_eef_quat",
    "right_eef_pos",
    "right_eef_quat",
    "red_cube_pos",
    "blue_cube_pos",
) + CAMERAS


def print_table_bboxes():
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    root = stage.GetPrimAtPath("/World/envs/env_0/PackingTable")
    for prim in root.GetChildren():
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        print(f"[bbox] {prim.GetPath()} min={np.round(np.array(r.GetMin()), 3)} max={np.round(np.array(r.GetMax()), 3)}")


def debug_contacts(env):
    def t(x):
        return x.torch if hasattr(x, "torch") else x

    scene = env.unwrapped.scene
    for side in ("left", "right"):
        sensor = scene[f"{side}_hand_contact"]
        forces = t(sensor.data.net_forces_w)[0].norm(dim=-1).cpu().numpy()
        hits = {n: round(float(f), 2) for n, f in zip(sensor.body_names, forces) if f > 0.05}
        print(f"[contact] {side}: {hits}")
    robot = scene["robot"]
    lim = t(robot.data.joint_pos_limits)[0].cpu().numpy()
    jp = t(robot.data.joint_pos)[0].cpu().numpy()
    near = {
        n: (round(float(jp[i]), 3), round(float(lim[i, 0]), 3), round(float(lim[i, 1]), 3))
        for i, n in enumerate(robot.data.joint_names)
        if "hand" not in n and (jp[i] - lim[i, 0] < 0.05 or lim[i, 1] - jp[i] < 0.05)
    }
    print(f"[limits] joints within 0.05 rad of a limit: {near}")
    names = list(env.unwrapped.action_manager.get_term("upper_body_ik").cfg.hand_joint_names)
    tgt = t(robot.data.joint_pos_target)[0].cpu().numpy()
    for n in names:
        i = robot.data.joint_names.index(n)
        print(f"[hand] {n:28s} pos={jp[i]:+.3f} target={tgt[i]:+.3f} limits=[{lim[i, 0]:+.3f},{lim[i, 1]:+.3f}]")


def run_attempt(env, cfg, policy, debug_dir: Path | None):
    dev = env.unwrapped.device
    col = cfg["collection"]
    env.reset()
    policy.reset()
    obs = read_step_observation(env, cfg)
    # Let the cubes land before the arms move. The first control step used to start while they
    # were still falling, and a rising forearm would bat them across the table.
    from humanoid.simulation.scripted_policy import PALM_DOWN_LEFT, PALM_DOWN_RIGHT

    hold = np.concatenate(
        [obs["left_eef_pos"], PALM_DOWN_LEFT, obs["right_eef_pos"], PALM_DOWN_RIGHT, policy.hand_pose]
    )
    for _ in range(25):
        with torch.no_grad():
            env.step(torch.as_tensor(hold, device=dev, dtype=torch.float32).unsqueeze(0))
    obs = read_step_observation(env, cfg)
    rows, actions, steps, done_steps = {}, [], 0, 0
    goals = (np.array(cfg["scene"]["red_goal"]), np.array(cfg["scene"]["blue_goal"]))
    while True:
        act, done = policy(obs, obs["hand_joint_vel"])
        for k in RECORDED:
            rows.setdefault(k, []).append(obs[k])
        actions.append(act)
        if debug_dir is not None and steps in (80, 300):
            debug_contacts(env)
        if debug_dir is not None and steps % 25 == 0:
            Image.fromarray(read_showcase(env)).save(debug_dir / f"showcase_{steps:04d}.png")
            print(
                f"[debug] t={steps:4d} L={policy.left.name:8s} R={policy.right.name:8s} "
                f"Leef={np.round(obs['left_eef_pos'], 3)} Lcmd={np.round(act[:3], 3)} red={np.round(obs['red_cube_pos'], 3)} "
                f"Reef={np.round(obs['right_eef_pos'], 3)} Rcmd={np.round(act[7:10], 3)} blue={np.round(obs['blue_cube_pos'], 3)}"
            )
        with torch.no_grad():
            _, _, term, trunc, _ = env.step(torch.as_tensor(act, device=dev).unsqueeze(0))
        steps += 1
        if bool(term[0]) or bool(trunc[0]):
            outcome = "time_out" if bool(trunc[0]) else "task_termination"
            break
        obs = read_step_observation(env, cfg)
        if done:
            done_steps += 1
            if done_steps >= int(col["settle_steps"]):
                ok = cube_on_pad(obs["red_cube_pos"], goals[0], cfg) and cube_on_pad(obs["blue_cube_pos"], goals[1], cfg)
                outcome = "success" if ok else "misplaced"
                break
    info = {
        "red_final": obs["red_cube_pos"].round(3).tolist(),
        "blue_final": obs["blue_cube_pos"].round(3).tolist(),
        "phases": (policy.left.name, policy.right.name),
    }
    return outcome, steps, rows, actions, info


def main():
    cfg = load_sim_config(args_cli.config)
    col = cfg["collection"]
    env_cfg = make_env_cfg(cfg, showcase=args_cli.debug)
    debug_dir = None
    if args_cli.debug:
        debug_dir = HUMANOID_ROOT / "outputs" / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(cfg["task"]["base_id"], cfg=env_cfg)
        env.reset()
        if args_cli.debug:
            print_table_bboxes()
        index, opened, closed = hand_layout(env)
        policy = BimanualSortPolicy(cfg, index, opened, closed)
        if args_cli.debug:
            outcome, steps, _, _, info = run_attempt(env, cfg, policy, debug_dir)
            print(f"[debug] outcome={outcome} steps={steps} {info}")
            env.close()
            return
        output = REPO_ROOT / (args_cli.output or col["output_hdf5"])
        n_target = args_cli.num_episodes or int(col["num_episodes"])
        dt = env.unwrapped.step_dt
        writer = DemoWriter(
            output,
            source="isaaclab_scripted",
            task="Unitree G1 sorting: red cube to the left pad, blue cube to the right pad",
            fps=round(1.0 / dt),
            env_args={"sim_config": cfg, "cameras": list(CAMERAS)},
        )
        saved = attempts = 0
        while saved < n_target and attempts < int(col["max_attempts"]):
            attempts += 1
            outcome, steps, rows, actions, info = run_attempt(env, cfg, policy, None)
            print(f"[collect] attempt {attempts}: {outcome} after {steps} steps {info}")
            if outcome != "success":
                continue
            writer.add_episode(
                Episode(
                    actions=np.stack(actions),
                    obs={k: np.stack(v) for k, v in rows.items()},
                    success=True,
                    termination=outcome,
                )
            )
            saved += 1
            print(f"[collect] saved {saved}/{n_target}")
        writer.close()
        print(f"[collect] {saved} successful episodes from {attempts} attempts -> {output}")
        env.close()


if __name__ == "__main__":
    main()
