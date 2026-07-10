"""DAgger hybrid-episode collection: student rollouts corrected by the scripted teacher.

Per scene (a seeded grid point; the SAME seed+options for both phases, so the scenes are
identical — appearance, lighting, cameras, arm jitter, and cube placement all re-draw
deterministically):

  A. **reference** — the teacher (``xsim.teacher.ScriptedTeacherPolicy``) rolls alone
     through the plain gym loop; its TCP/cube/gripper/segment trace is recorded. A
     teacher failure skips the scene.
  B. **hybrid** — ``xsim.dagger.DAggerPolicy(student, teacher)`` rolls on the identical
     scene: the student drives until the ``DivergenceMonitor`` fires against the phase-A
     trace, then the teacher re-plans from the live state and finishes the task.

Every rollout lands in ``results.jsonl`` and (with ``--video``) an mp4 with a phase strip:
gray = reference, green = student on-path, red flash = divergence detected, blue = teacher
recovery. The strip is video-only (never in obs or recorded data). Scenes whose outcome is
``corrected`` are the DAgger correction episodes; writing those as training MCAP is a
follow-up once the takeover behavior is visually verified.

    # offline validation (no server): a scripted student lied to about the cube position,
    # so it plunges/grasps off-target and the monitor must fire
    uv run python scripts/dagger.py --student perturbed --grid-nx 2 --grid-ny 2
    # sanity: student == teacher; must complete with no divergence
    uv run python scripts/dagger.py --student expert --grid-nx 2 --grid-ny 2
    # the served crossformer (grainlike serving stack)
    uv run python scripts/dagger.py --student remote --host localhost --port 8001
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time
from typing import Literal

import numpy as np
import torch
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import genesis as gs  # noqa: E402

from xsim.task_env import TaskEnv, TaskEnvCfg  # noqa: E402
from xsim.dagger import (  # noqa: E402
    DAggerPolicy, DaggerThresholds, ExpertReference, ModeStripWrapper, ReferenceRecorder,
)
from xsim.teacher import ScriptedTeacherPolicy  # noqa: E402
from xsim.wrappers import GenesisGymAdapter, VideoRecordWrapper  # noqa: E402

from eval_grid import build_grid, grid_ranges  # noqa: E402  (same-directory import)


@dataclass
class Config:
    task: Literal["lift"] = "lift"     # stack: after the lift pilot is verified
    student: Literal["remote", "expert", "perturbed"] = "remote"
    host: str = "localhost"            # webpolicy inference server (grainlike serve)
    port: int = 8001
    chunk_h: int = 50                  # student actions executed per policy inference
    grid_nx: int = 3
    grid_ny: int = 3
    reps: int = 1
    seed: int = 61000                  # clear of training (9k/20k/30k+) and eval (51k) ranges
    cube_yaw: float = 0.0
    perturb_xy: float = 0.05           # cube-position lie (m) for --student perturbed
    # hybrid budget: student segment + takeover recovery, so ~2x the eval time limit
    max_control_steps: int = 1200      # 40 s at 30 Hz
    sim_hz: int = 120
    control_hz: int = 30
    close_setpoint: float = 0.58       # closed-finger dof (training value)
    backend: Literal["gpu", "cpu"] = "gpu"
    video: bool = True
    out: Path | None = None            # default: PROJECT_ROOT/outputs/dagger/<task>
    teacher_steps_per_segment: int = 27  # 108 @ 120 Hz -> 27 @ 30 Hz, same real speed
    thresholds: DaggerThresholds = field(default_factory=DaggerThresholds)
    env: TaskEnvCfg = field(default_factory=lambda: TaskEnvCfg(noslip_iterations=10))


class _CubeLiar:
    """Env proxy misreporting the cube position by a fixed xy offset.

    Given to a ScriptedTeacherPolicy this makes a *plausibly wrong* student: it plans and
    executes a clean trajectory toward a spot ``offset`` away from the real cube, so the
    grasp closes on air — the exact failure the monitor's cube tracking must catch.
    """

    def __init__(self, env, offset_xy: tuple[float, float]):
        self._env = env
        self._offset = np.asarray(offset_xy, dtype=np.float64)

    def cube_pos(self) -> np.ndarray:
        pos = np.asarray(self._env.cube_pos(), dtype=np.float64).copy()
        pos[:2] += self._offset
        return pos

    def __getattr__(self, name):
        return getattr(self._env, name)


def build_student(cfg: Config, env):
    if cfg.student == "remote":
        from webpolicy.client import Client

        from eval import ChunkClient  # scripts/eval.py

        return ChunkClient(Client(cfg.host, cfg.port))
    if cfg.student == "expert":
        return ScriptedTeacherPolicy(env, steps_per_segment=cfg.teacher_steps_per_segment)
    return ScriptedTeacherPolicy(
        _CubeLiar(env, (cfg.perturb_xy, cfg.perturb_xy)),
        steps_per_segment=cfg.teacher_steps_per_segment)


# ---------------------------------------------------------------------------------------
# Rollouts (both are the plain gym loop; only the policy differs)
# ---------------------------------------------------------------------------------------


def _tcp(env) -> np.ndarray:
    return np.asarray(env.proprio()[3], dtype=np.float64).reshape(-1)[:3]


def run_reference(env, teacher, seed: int, options: dict, control_dt: float,
                  grace: int = 30) -> tuple[ExpertReference, dict]:
    """Roll the teacher alone and record its trace. ``grace`` extra steps let the cube
    finish dropping after the trajectory ends so the env can register success."""
    obs = env.reset(seed=seed, options=options)
    teacher.reset()
    recorder = ReferenceRecorder(control_dt)
    done, info, extra = False, {}, 0
    while not done:
        action = teacher.step(obs)
        obs, _, done, info = env.step(action)
        recorder.append(_tcp(env), env.cube_pos(), bool(action[7] > 0.5), teacher.segment)
        if teacher.done:
            extra += 1
            if extra > grace:
                break
    return recorder.finalize(), info


def run_hybrid(env, policy: DAggerPolicy, seed: int, options: dict,
               grace: int = 30) -> tuple[dict, int]:
    obs = env.reset(seed=seed, options=options)
    policy.reset()
    done, info, extra, steps = False, {}, 0, 0
    while not done:
        action = policy.step(obs)
        obs, _, done, info = env.step(action)
        steps += 1
        if policy.mode == "teacher" and policy.teacher.done:
            extra += 1
            if extra > grace:
                break
    return info, steps


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------


def _outcome(policy: DAggerPolicy, info: dict) -> str:
    if policy.switch is None:
        return "student_success" if info.get("success") else "student_failed_undetected"
    return "corrected" if info.get("success") else "correction_failed"


def main(cfg: Config) -> None:
    out = cfg.out if cfg.out is not None else PROJECT_ROOT / "outputs" / "dagger" / cfg.task
    out.mkdir(parents=True, exist_ok=True)

    cfg.env.task = cfg.task
    cfg.env.physics_dt = 1.0 / cfg.sim_hz
    control_every = max(1, round(cfg.sim_hz / cfg.control_hz))
    control_dt = control_every * cfg.env.physics_dt

    x_range, y_range = grid_ranges(cfg, cfg.env)
    points, _, _ = build_grid(cfg.task, cfg.grid_nx, cfg.grid_ny, x_range, y_range)

    gs.init(backend=gs.gpu if cfg.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    adapter = GenesisGymAdapter(
        TaskEnv(cfg.env), control_every=control_every,
        max_control_steps=cfg.max_control_steps, close_setpoint=cfg.close_setpoint)
    strip = ModeStripWrapper(adapter)
    env = (VideoRecordWrapper(strip, out / "videos", capture_every=1,
                              fps=float(cfg.control_hz), name_prefix="rollout")
           if cfg.video else strip)

    teacher = ScriptedTeacherPolicy(env, steps_per_segment=cfg.teacher_steps_per_segment)
    student = build_student(cfg, env)
    policy = DAggerPolicy(student, teacher, env,
                          thresholds=cfg.thresholds, chunk_h=cfg.chunk_h)

    phase = {"name": ""}
    strip.mode_fn = lambda: phase["name"] if phase["name"] == "reference" else policy.mode

    counts: dict[str, int] = {}
    records: list[dict] = []
    with open(out / "results.jsonl", "a") as results_fh, torch.no_grad():
        for rep in range(cfg.reps):
            for gp in points:
                seed = cfg.seed + rep * 10007 + gp.grid_idx
                options = {"cube_xy": gp.cube_xy, "cube_yaw": cfg.cube_yaw,
                           "drop_xy": (0.35, 0.0)}
                record = {
                    "rep": rep, "grid_idx": gp.grid_idx,
                    "cube_xy": [float(gp.cube_xy[0]), float(gp.cube_xy[1])], "seed": seed,
                }
                t0 = time.monotonic()

                phase["name"] = "reference"
                video_id = getattr(env, "_episode_id", -1) + 1
                reference, ref_info = run_reference(env, teacher, seed, options, control_dt)
                record.update(ref_len=len(reference), ref_video=video_id)
                if not ref_info.get("success"):
                    record.update(outcome="reference_failed", wall_s=round(time.monotonic() - t0, 1))
                    counts["reference_failed"] = counts.get("reference_failed", 0) + 1
                    results_fh.write(json.dumps(record) + "\n")
                    results_fh.flush()
                    records.append(record)
                    print(f"rep{rep} grid{gp.grid_idx:03d} REFERENCE FAILED — scene skipped")
                    continue

                phase["name"] = "hybrid"
                policy.reference = reference
                info, steps = run_hybrid(env, policy, seed, options)
                outcome = _outcome(policy, info)
                counts[outcome] = counts.get(outcome, 0) + 1
                record.update(
                    outcome=outcome,
                    hybrid_video=video_id + 1,
                    steps=steps,
                    success=bool(info.get("success")),
                    max_rise=round(float(info.get("max_rise", 0.0)), 4),
                    deliver_dist=round(float(info.get("deliver_dist", -1.0)), 4),
                    fell=bool(info.get("fell")), flew=bool(info.get("flew")),
                    timeout=bool(info.get("timeout")),
                    wall_s=round(time.monotonic() - t0, 1),
                )
                if policy.switch is not None:
                    v = policy.switch
                    record["switch"] = {
                        "step": policy.switch_step, "reason": v.reason,
                        "ref_idx": v.ref_idx, "segment": v.segment,
                        "tcp_err": round(v.tcp_err, 4), "cube_err": round(v.cube_err, 4),
                        "progress": round(v.progress, 3),
                    }
                results_fh.write(json.dumps(record) + "\n")
                results_fh.flush()
                records.append(record)
                sw = (f"switch@{policy.switch_step} {policy.switch.reason} "
                      f"seg={policy.switch.segment} prog={policy.switch.progress:.2f}"
                      if policy.switch else "no divergence")
                print(f"rep{rep} grid{gp.grid_idx:03d} xy=({gp.cube_xy[0]:.3f},{gp.cube_xy[1]:.3f}) "
                      f"{outcome} ({sw}) steps={steps} [{record['wall_s']}s]", flush=True)

    env.close()
    reasons: dict[str, int] = {}
    for r in records:
        if "switch" in r:
            reasons[r["switch"]["reason"]] = reasons.get(r["switch"]["reason"], 0) + 1
    summary = {
        "student": cfg.student, "n_scenes": len(records), "outcomes": counts,
        "divergence_reasons": reasons, "seed": cfg.seed,
        "thresholds": {k: getattr(cfg.thresholds, k) for k in
                       ("corridor", "cube_tol_pre", "cube_tol_post", "patience",
                        "max_advance_s", "stall_window_s", "stall_min_advance_s")},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ndagger done: {counts} -> {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
