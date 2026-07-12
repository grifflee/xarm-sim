"""DAgger hybrid-episode collection: student rollouts corrected by the scripted teacher.

Per scene (a seeded grid point; the SAME seed+options for both phases, so the scenes are
identical — appearance, lighting, cameras, arm jitter, and cube placement all re-draw
deterministically):

  A. **reference** — the teacher (``xsim.teacher.ScriptedTeacherPolicy``) rolls alone
     through the plain gym loop; its TCP/cube/gripper/segment trace is recorded. A
     teacher failure skips the scene.
  B. **hybrid** — ``xsim.dagger.DAggerPolicy(student, teacher)`` rolls on the identical
     scene: the student drives, and on every control step whose error against the phase-A
     trace exceeds the thresholds the teacher's action is executed instead (re-planned
     from live state at the start of each intervention). Control returns to the student
     on the first clean step — per-step interventions, no latched takeover. A persistent
     failure keeps the error high and so keeps the teacher acting until it is repaired.

Every rollout lands in ``results.jsonl`` and (with ``--video``) an mp4 with a phase strip:
gray = reference, green = student driving, red = teacher intervention step. The strip is
video-only (never in obs or recorded data). Scenes whose outcome is ``corrected``
(intervened AND ended in success) are the DAgger correction episodes: they are written to
``--mcap-dir`` as training MCAP (``<model>_episode_<seed>.mcap`` + merged
``manifest.json``, same format as the generator batches, trimmed to end at the release
like the demonstration protocol). Training actions are derived from the recorded measured
joint trajectory, so the data is the executed hybrid motion itself — student behavior
held inside the reference corridor by the teacher's knocks.

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
from xsim.wrappers import GenesisGymAdapter, McapRecordWrapper, VideoRecordWrapper  # noqa: E402

from eval_grid import _git_provenance, _jsonable, build_grid, grid_ranges  # noqa: E402  (same-directory import)


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
    # `corrected` hybrids are written here as training MCAP (<model>_episode_<seed>.mcap
    # + manifest.json, same format as the generator batches). None disables MCAP output.
    mcap_dir: Path | None = Path("/data/store/griffen_sim_mcaps/dagger_mcaps")
    # names the corrected model in MCAP filenames/manifest; REQUIRED for --student remote
    # (e.g. 0707_iconic-spaceship-1191), defaults to the student kind for scripted ones
    model_name: str | None = None
    teacher_steps_per_segment: int = 27  # 108 @ 120 Hz -> 27 @ 30 Hz, same real speed
    thresholds: DaggerThresholds = field(default_factory=DaggerThresholds)
    # nyx rendering is REQUIRED for training data: the student was trained on nyx images
    # (splat background, colored robot); raster frames are out-of-domain for it
    env: TaskEnvCfg = field(default_factory=lambda: TaskEnvCfg(
        noslip_iterations=10, render_backend="nyx"))


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


def _model_name(cfg: Config) -> str:
    if cfg.model_name:
        return cfg.model_name
    if cfg.student != "remote":
        return cfg.student
    raise SystemExit("--model-name is required with --student remote so the MCAP "
                     "filenames say which model the corrections were run on")


def _outcome(policy: DAggerPolicy, info: dict) -> str:
    if not policy.interventions:
        return "student_success" if info.get("success") else "student_failed_undetected"
    return "corrected" if info.get("success") else "correction_failed"


def main(cfg: Config) -> None:
    out = cfg.out if cfg.out is not None else PROJECT_ROOT / "outputs" / "dagger" / cfg.task
    out.mkdir(parents=True, exist_ok=True)
    model = _model_name(cfg)  # fail fast, before the env spins up

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
    # recorder sits below the strip so buffered MCAP images are clean of the overlay
    recorder = McapRecordWrapper(adapter, record_dt=control_dt)
    strip = ModeStripWrapper(recorder)
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
    mcap_episodes: list[dict] = []
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
                recorder.enabled = False
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
                recorder.enabled = cfg.mcap_dir is not None
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
                if policy.interventions:
                    record["interventions"] = policy.interventions
                    record["teacher_steps"] = policy.teacher_steps
                if cfg.mcap_dir is not None and outcome == "corrected":
                    cfg.mcap_dir.mkdir(parents=True, exist_ok=True)
                    mcap_path = cfg.mcap_dir / f"{model}_episode_{seed:06d}.mcap"
                    frames = recorder.save(mcap_path)["frames"]
                    record["mcap"] = str(mcap_path)
                    entry = {
                        "episode": seed, "frames": frames, "seed": seed, "kept": True,
                        "model": model,
                        "cube_yaw": cfg.cube_yaw, "grid_idx": gp.grid_idx,
                        "cube_xy": record["cube_xy"], "outcome": outcome,
                        "interventions": record.get("interventions"),
                        "teacher_steps": record.get("teacher_steps"),
                        "extrinsics": {k: np.asarray(v).tolist()
                                       for k, v in adapter.episode_extrinsics.items()},
                    }
                    for key in ("max_rise", "lifted", "deliver_dist", "delivered",
                                "success", "drop_target"):
                        if key in info:
                            entry[key] = _jsonable(info[key])
                    mcap_episodes.append(entry)
                else:
                    recorder.discard()
                results_fh.write(json.dumps(record) + "\n")
                results_fh.flush()
                records.append(record)
                if policy.interventions:
                    first = policy.interventions[0]
                    sw = (f"{len(policy.interventions)} interventions / "
                          f"{policy.teacher_steps} teacher steps, first "
                          f"{first['reason']}@{first['start']} seg={first['segment']}")
                else:
                    sw = "no interventions"
                print(f"rep{rep} grid{gp.grid_idx:03d} xy=({gp.cube_xy[0]:.3f},{gp.cube_xy[1]:.3f}) "
                      f"{outcome} ({sw}) steps={steps} [{record['wall_s']}s]", flush=True)

    env.close()
    reasons: dict[str, int] = {}
    for r in records:
        for iv in r.get("interventions", []):
            reasons[iv["reason"]] = reasons.get(iv["reason"], 0) + 1
    summary = {
        "student": cfg.student, "n_scenes": len(records), "outcomes": counts,
        "intervention_reasons": reasons, "seed": cfg.seed,
        "thresholds": {k: getattr(cfg.thresholds, k) for k in
                       ("corridor", "cube_tol_pre", "cube_tol_post",
                        "max_advance_s", "stall_window_s", "stall_min_advance_s")},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    if cfg.mcap_dir is not None and mcap_episodes:
        # merge with any existing manifest so successive dagger runs (other models,
        # other grids) accumulate in one directory. `config` reflects the latest run.
        manifest_path = cfg.mcap_dir / "manifest.json"
        existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        by_ep = {(e.get("model"), e["episode"]): e for e in existing.get("episodes", [])}
        for e in mcap_episodes:
            by_ep[(e["model"], e["episode"])] = e
        sha, dirty = _git_provenance()
        manifest_path.write_text(json.dumps({
            "source": "scripts/dagger.py",
            "git_sha": sha, "git_dirty": dirty,
            "config": _jsonable(cfg),
            "success_rate": 1.0,  # only corrected (successful) hybrids are kept
            "episodes": sorted(by_ep.values(), key=lambda e: (e.get("model", ""), e["episode"])),
        }, indent=2))
        print(f"mcap: {len(mcap_episodes)} corrected episodes -> {cfg.mcap_dir} "
              f"(manifest total {len(by_ep)})")
    print(f"\ndagger done: {counts} -> {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
