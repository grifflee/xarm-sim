"""Phase 0 for the eval suite: weld-free grasp feasibility sweep.

Runs the scripted lift policy WITHOUT the grasp weld (``env.grasp_lock()`` is never
called) so the cube is held by finger friction alone, and sweeps the finger close
setpoint. A trained policy evaluated in sim gets no weld, so this measures whether
the physics can support honest grasping — and which close target the eval harness
should map "gripper closed" to.

Slip is measured, not just eyeballed: from grasp-settle to release the cube pose is
tracked in the TCP frame; any relative translation/rotation is slip even when the task
still completes (grifflee: visible slip = not a pass).

    uv run python scripts/test_friction_grasp.py --backend gpu
    uv run python scripts/test_friction_grasp.py --setpoints 0.58 --n-episodes 3 \
        --videos-per-setpoint 3   # contact-sheet videos for visual review
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
from typing import Literal

import cv2
import numpy as np
import torch
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import genesis as gs  # noqa: E402

from xsim.task_env import TaskEnv, TaskEnvCfg  # noqa: E402
from xsim.grasp_slip import cube_rel_tcp  # noqa: E402

from generate_task_dataset import (  # noqa: E402
    Config as GenConfig,
    _episode_result,
    _make_policy,
    contact_sheet,
)


@dataclass
class Config:
    backend: Literal["gpu", "cpu"] = "gpu"
    n_episodes: int = 20
    # 0.58 is the demo grasp dof (fingers at cube task width); higher = more squeeze
    setpoints: tuple[float, ...] = (0.58, 0.62, 0.66, 0.70)
    seed: int = 50000
    steps_per_segment: int = 108
    videos_per_setpoint: int = 0  # record the first N episodes of each setpoint
    video_dir: Path = PROJECT_ROOT / "outputs" / "friction_grasp"
    video_fps: float = 30.0
    video_every_steps: int = 4  # 4 = normal 30 Hz capture; 1 = 120 Hz slow-motion at 30 fps
    pre_lift_hold_s: float = 0.0  # diagnostic: hold closed before lifting so contacts can settle
    lift_slowdown: float = 1.0  # diagnostic: multiply only the lift segment duration
    ramp_close: bool = False  # diagnostic: ramp finger target across the close segment
    carry_hold_s: float = 0.0   # diagnostic: hold the grasp before release so slip is visible
    # slip gate: cube motion relative to the TCP during the carry (post-settle to release)
    slip_settle_steps: int = 12   # 0.1 s after the close completes before the reference pose
    slip_mm_tol: float = 3.0
    slip_deg_tol: float = 5.0
    env: TaskEnvCfg = field(default_factory=TaskEnvCfg)


def _write_video_frame(env: TaskEnv, cfg: Config, video_writer, step_idx: int,
                       phase: str, slip_mm: float, slip_deg: float) -> None:
    if video_writer is None:
        return
    every = max(1, int(cfg.video_every_steps))
    if step_idx % every != 0:
        return
    sheet = contact_sheet(env.render(), f"{step_idx:04d}")
    label = f"{phase}  slip={slip_mm:.2f}mm  rot={slip_deg:.2f}deg"
    cv2.putText(sheet, label, (12, sheet.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(sheet, label, (12, sheet.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)
    video_writer.write(cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def _go_to_goal_with_finger(env: TaskEnv, goal_pose: torch.Tensor, finger_dof: float) -> None:
    robot = env.robot
    init_qpos = None
    if robot._args.get("ik_init_at_home", False):
        init_qpos = robot._init_qpos.unsqueeze(0).expand(goal_pose.shape[0], -1)
    q_pos = robot._robot_entity.inverse_kinematics(
        link=robot._ee_link,
        pos=goal_pose[:, :3],
        quat=goal_pose[:, 3:7],
        init_qpos=init_qpos,
        max_samples=robot._args.get("ik_max_samples", 50),
        max_solver_iters=robot._args.get("ik_max_solver_iters", 20),
        damping=robot._args.get("ik_damping", 0.01),
        dofs_idx_local=robot._arm_dof_idx,
    )
    q_pos[:, robot._fingers_dof] = float(finger_dof)
    robot._robot_entity.control_dofs_position(position=q_pos)


def run_trial(env: TaskEnv, cfg: Config, gen_cfg: GenConfig, seed: int, video_writer=None) -> dict:
    """One weld-free scripted lift episode; returns _episode_result stats + slip metrics."""
    env.reset(seed=seed)
    policy = _make_policy(env, gen_cfg)
    if cfg.lift_slowdown != 1.0 and hasattr(policy, "segment_weights"):
        weights = list(policy.segment_weights)
        if len(weights) >= 4:
            weights[3] = weights[3] * max(0.1, float(cfg.lift_slowdown))
            policy.segment_weights = tuple(weights)
    policy.reset()
    cube_start = env.cube_pos().copy()
    max_rise = 0.0
    pre_lift_hold_steps = max(0, int(round(cfg.pre_lift_hold_s / env.cfg.physics_dt)))

    def shifted(policy_step: int) -> int:
        if policy_step >= policy.grasp_lock_step:
            return policy_step + pre_lift_hold_steps
        return policy_step

    slip_start = shifted(policy.grasp_lock_step) + cfg.slip_settle_steps
    # carry-phase boundaries for the slip timeline: end of lift, transport, settle
    seg = policy._segment_steps
    boundaries = {'lift_end': shifted(1 + sum(seg[:4])),
                  'transport_end': shifted(1 + sum(seg[:5])),
                  'pre_release': shifted(policy.release_step) - 1}
    slip_timeline: dict[str, float] = {}
    close_start = 1 + sum(seg[:2])
    close_end = policy.grasp_lock_step
    early_lift_start = shifted(policy.grasp_lock_step)
    early_lift_end = min(shifted(policy.release_step) - 1, early_lift_start + 20)
    last_cube_pos = cube_start.copy()
    close_max_step_mm = 0.0
    close_max_step = -1
    pre_lift_hold_max_step_mm = 0.0
    pre_lift_hold_max_step = -1
    early_lift_max_step_mm = 0.0
    early_lift_max_step = -1
    rel0 = None
    slip_mm = 0.0
    slip_deg = 0.0
    slip_step = -1
    slip_vec = np.zeros(3)
    release_tail = max(1, int(round(gen_cfg.release_tail_s / env.cfg.physics_dt)))
    carry_hold_steps = max(0, int(round(cfg.carry_hold_s / env.cfg.physics_dt)))

    def track_slip(step_idx: int) -> float | None:
        nonlocal rel0, slip_mm, slip_deg, slip_step, slip_vec
        if step_idx < slip_start:
            return None
        rel_pos, rel_quat = cube_rel_tcp(env)
        if rel0 is None:
            rel0 = (rel_pos, rel_quat)
            return 0.0
        d_mm = float(np.linalg.norm(rel_pos - rel0[0])) * 1000.0
        if d_mm > slip_mm:
            slip_mm = d_mm
            slip_step = step_idx
            slip_vec = (rel_pos - rel0[0]) * 1000.0
        dot = min(1.0, abs(float(np.dot(rel_quat, rel0[1]))))
        slip_deg = max(slip_deg, math.degrees(2.0 * math.acos(dot)))
        for name, b in boundaries.items():
            if step_idx == b:
                slip_timeline[name] = d_mm
        return d_mm

    def track_grab_motion(step_idx: int) -> None:
        nonlocal last_cube_pos, close_max_step_mm, close_max_step
        nonlocal pre_lift_hold_max_step_mm, pre_lift_hold_max_step
        nonlocal early_lift_max_step_mm, early_lift_max_step
        cube = env.cube_pos().copy()
        d_mm = float(np.linalg.norm(cube - last_cube_pos)) * 1000.0
        if close_start <= step_idx <= close_end and d_mm > close_max_step_mm:
            close_max_step_mm = d_mm
            close_max_step = step_idx
        if close_end < step_idx <= early_lift_start and d_mm > pre_lift_hold_max_step_mm:
            pre_lift_hold_max_step_mm = d_mm
            pre_lift_hold_max_step = step_idx
        if early_lift_start < step_idx <= early_lift_end and d_mm > early_lift_max_step_mm:
            early_lift_max_step_mm = d_mm
            early_lift_max_step = step_idx
        last_cube_pos = cube

    def close_target_for_step(policy_step: int) -> float | None:
        if not cfg.ramp_close or not (close_start <= policy_step <= close_end):
            return None
        denom = max(1, close_end - close_start)
        alpha = min(1.0, max(0.0, (policy_step - close_start) / denom))
        robot = env.robot
        return float(robot._gripper_open_dof + alpha * (robot._gripper_grasp_dof - robot._gripper_open_dof))

    def drive_cmd(policy_step: int, cmd) -> None:
        finger_dof = close_target_for_step(policy_step)
        if finger_dof is None:
            env.robot.go_to_goal(cmd.pose, open_gripper=cmd.open_gripper)
        else:
            _go_to_goal_with_finger(env, cmd.pose, finger_dof)

    def phase_for_step(step_idx: int) -> str:
        if pre_lift_hold_steps and close_end < step_idx <= early_lift_start:
            return 'pre_lift_hold'
        if step_idx < slip_start:
            return "approach/close"
        if step_idx < boundaries["lift_end"]:
            return "lift"
        if step_idx < boundaries["transport_end"]:
            return "transport"
        if step_idx < policy.release_step:
            return "settle"
        if step_idx < policy.release_step + carry_hold_steps:
            return "diagnostic_hold"
        return "release"

    with torch.no_grad():
        last_closed_cmd = None
        actual_step = 0
        for policy_step in range(policy.release_step):
            if policy_step == policy.grasp_lock_step and pre_lift_hold_steps and last_closed_cmd is not None:
                for _ in range(pre_lift_hold_steps):
                    env.robot.go_to_goal(last_closed_cmd.pose, open_gripper=False)
                    env.step()
                    max_rise = max(max_rise, float(env.cube_pos()[2] - cube_start[2]))
                    track_grab_motion(actual_step)
                    track_slip(actual_step)
                    _write_video_frame(env, cfg, video_writer, actual_step,
                                       phase_for_step(actual_step), slip_mm, slip_deg)
                    actual_step += 1
            cmd = policy.step()
            last_closed_cmd = cmd
            # no grasp_lock()/grasp_release(): friction only
            drive_cmd(policy_step, cmd)
            env.step()
            max_rise = max(max_rise, float(env.cube_pos()[2] - cube_start[2]))
            track_grab_motion(actual_step)
            track_slip(actual_step)
            _write_video_frame(env, cfg, video_writer, actual_step,
                               phase_for_step(actual_step), slip_mm, slip_deg)
            actual_step += 1

        if carry_hold_steps and last_closed_cmd is not None:
            for hold_i in range(carry_hold_steps):
                step_idx = actual_step + hold_i
                env.robot.go_to_goal(last_closed_cmd.pose, open_gripper=False)
                env.step()
                max_rise = max(max_rise, float(env.cube_pos()[2] - cube_start[2]))
                track_grab_motion(step_idx)
                d_mm = track_slip(step_idx)
                if hold_i == carry_hold_steps - 1 and d_mm is not None:
                    slip_timeline["hold_end"] = d_mm
                _write_video_frame(env, cfg, video_writer, step_idx, phase_for_step(step_idx), slip_mm, slip_deg)

        for tail_i in range(release_tail):
            step_idx = actual_step + carry_hold_steps + tail_i
            cmd = policy.step()
            env.robot.go_to_goal(cmd.pose, open_gripper=cmd.open_gripper)
            env.step()
            max_rise = max(max_rise, float(env.cube_pos()[2] - cube_start[2]))
            track_grab_motion(step_idx)
            _write_video_frame(env, cfg, video_writer, step_idx, phase_for_step(step_idx), slip_mm, slip_deg)

        for _ in range(gen_cfg.hold_steps):
            cmd = policy.step()
            env.robot.go_to_goal(cmd.pose, open_gripper=cmd.open_gripper)
            env.step()
    res = _episode_result(env, gen_cfg, max_rise)
    res["slip_mm"] = slip_mm
    res["slip_deg"] = slip_deg
    res["slip_step"] = slip_step
    res["slip_vec_mm"] = [round(float(v), 2) for v in slip_vec]  # TCP frame
    res["slip_timeline"] = {k: round(v, 2) for k, v in slip_timeline.items()}
    res["close_max_step_mm"] = close_max_step_mm
    res["close_max_step"] = close_max_step
    res['pre_lift_hold_s'] = cfg.pre_lift_hold_s
    res['pre_lift_hold_steps'] = pre_lift_hold_steps
    res['pre_lift_hold_max_step_mm'] = pre_lift_hold_max_step_mm
    res['pre_lift_hold_max_step'] = pre_lift_hold_max_step
    res["lift_slowdown"] = cfg.lift_slowdown
    res["ramp_close"] = cfg.ramp_close
    res["early_lift_max_step_mm"] = early_lift_max_step_mm
    res["early_lift_max_step"] = early_lift_max_step
    res["slip_pass"] = slip_mm <= cfg.slip_mm_tol and slip_deg <= cfg.slip_deg_tol
    res["pass"] = bool(res["success"] and res["slip_pass"])
    return res


def main(cfg: Config) -> None:
    cfg.env.task = "lift"
    gs.init(backend=gs.gpu if cfg.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    env = TaskEnv(cfg.env)
    gen_cfg = GenConfig(task="lift", steps_per_segment=cfg.steps_per_segment, env=cfg.env)

    sheet_size = None
    rows = []
    for sp in cfg.setpoints:
        env.robot._gripper_grasp_dof = sp
        stats = []
        for ep in range(cfg.n_episodes):
            video_writer = None
            video_path = None
            if ep < cfg.videos_per_setpoint:
                cfg.video_dir.mkdir(parents=True, exist_ok=True)
                video_path = cfg.video_dir / f"noweld_sp{int(round(sp * 100)):02d}_ep{ep:02d}.mp4"
                if sheet_size is None:
                    probe = contact_sheet(env.render(), "probe")
                    sheet_size = (probe.shape[1], probe.shape[0])
                video_writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                    cfg.video_fps, sheet_size,
                )
            res = run_trial(env, cfg, gen_cfg, cfg.seed + ep, video_writer)
            if video_writer is not None:
                video_writer.release()
                print(f"wrote video to {video_path}")
            stats.append(res)
            print(f"setpoint={sp:.2f} ep={ep:02d} seed={cfg.seed + ep} "
                  f"lifted={res['lifted']} delivered={res['delivered']} "
                  f"slip_mm={res['slip_mm']:.2f} slip_deg={res['slip_deg']:.2f} "
                  f"pass={res['pass']} max_rise={res['max_rise']:.3f} "
                  f"deliver_dist={res['deliver_dist']:.3f} "
                  f"close_step_mm={res['close_max_step_mm']:.2f}@{res['close_max_step']} "
                  f"pre_lift_hold_step_mm={res['pre_lift_hold_max_step_mm']:.2f}@{res['pre_lift_hold_max_step']} "
                  f"lift_slowdown={res['lift_slowdown']:.2f} "
                  f"ramp_close={res['ramp_close']} "
                  f"early_lift_step_mm={res['early_lift_max_step_mm']:.2f}@{res['early_lift_max_step']} "
                  f"slip@step={res['slip_step']} vec={res['slip_vec_mm']} "
                  f"timeline={res['slip_timeline']}")
        rows.append({
            "setpoint": sp,
            "lifted": sum(s["lifted"] for s in stats),
            "delivered": sum(s["delivered"] for s in stats),
            "success": sum(s["success"] for s in stats),
            "pass": sum(s["pass"] for s in stats),
            "max_slip_mm": max(s["slip_mm"] for s in stats),
            "max_slip_deg": max(s["slip_deg"] for s in stats),
            "mean_rise": float(np.mean([s["max_rise"] for s in stats])),
        })

    n = cfg.n_episodes
    print(f"\n== weld-free grasp sweep (pass = success AND slip <= "
          f"{cfg.slip_mm_tol:g}mm/{cfg.slip_deg_tol:g}deg) ==")
    print(f"{'setpoint':>8} {'lifted':>8} {'delivered':>10} {'success':>8} {'PASS':>8} "
          f"{'max_slip':>12} {'mean_rise':>10}")
    for r in rows:
        print(f"{r['setpoint']:>8.2f} {r['lifted']:>5d}/{n:<2d} {r['delivered']:>7d}/{n:<2d} "
              f"{r['success']:>5d}/{n:<2d} {r['pass']:>5d}/{n:<2d} "
              f"{r['max_slip_mm']:>6.2f}mm {r['max_slip_deg']:>5.2f}° {r['mean_rise']:>9.3f}m")


if __name__ == "__main__":
    main(tyro.cli(Config))
