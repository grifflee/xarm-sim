"""Paper-DAgger data collection for a served crossformer student (Ross et al. 2010).

Replaces the reference-corridor pipeline of ``scripts/dagger.py`` (which stays untouched)
with the beta-mixture loop of Algorithm 3.1 (arXiv:1011.0686):

  - The reactive expert (``xsim.lift_expert.LiftExpertPolicy``) is a *labeler*: at EVERY
    control tick it computes ``label = expert.act()`` from the live state, and that label
    is what enters the dataset (recorded to the ``/teacher/*`` channels), regardless of who
    actually drives the arm that tick.
  - Control is a per-window beta mixture. At each replan boundary (every ``chunk_h`` ticks,
    matching the student's open-loop horizon) we draw ``use_teacher = rng.random() < beta``
    for the WHOLE window. A teacher window executes the label per tick; a student window
    queries the served ``ChunkClient`` once and plays its 50 rows open-loop. ``beta >= 1.0``
    never constructs or contacts the student client (round 0 runs with no server).
  - EVERY episode is saved (keep_all recording, no success gate, no length gate, no trim):
    the student's failure states are exactly the states DAgger exists to label.

Iteration is the outer loop: run round 0 at ``--beta 1.0`` (pure teacher), train, then run
round N at ``--beta-from <eval summary.json>`` on the new checkpoint, aggregate, retrain.

    # round 0: pure-teacher labels on the grid, no server needed
    uv run python scripts/dagger_rounds.py --round 0 --beta 1.0 \
        --model-name 0715_still-star-1213 --grid-nx 3 --grid-ny 3
    # round 1: beta from a prior eval of the student, querying the served model
    uv run python scripts/dagger_rounds.py --round 1 \
        --beta-from /data/store/griffen_sim_mcaps/evals/lift/summary.json \
        --model-name 0715_still-star-1213 --host 127.0.0.1 --port 9001
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
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
from xsim.lift_expert import LiftExpertPolicy  # noqa: E402
from xsim.dagger import ModeStripWrapper  # noqa: E402  (video overlay only)
from xsim.wrappers import GenesisGymAdapter, McapRecordWrapper, VideoRecordWrapper  # noqa: E402

from eval_grid import (  # noqa: E402  (same-directory import)
    GridPoint, _git_provenance, _jsonable, build_grid, grid_ranges,
)


@dataclass
class Config:
    round: int              # DAgger iteration; named in the generation-run dir (REQUIRED)
    task: Literal["lift"] = "lift"
    # Control mixing weight. Exactly one of --beta / --beta-from. beta >= 1.0 is pure
    # teacher and never contacts the student server (use for round 0).
    beta: float | None = None
    # Path to a prior eval summary.json; beta = clip(1 - 1.2 * overall_success_rate, 0.2, 1.0).
    beta_from: Path | None = None
    host: str = "localhost"            # webpolicy inference server (grainlike serve)
    port: int = 9001                   # grifflee: use 9xxx ports for model serving
    chunk_h: int = 50                  # student open-loop horizon == mixing window length
    grid_nx: int = 3
    grid_ny: int = 3
    reps: int = 1
    seed: int = 61000                  # 61000-61999 reserved for dagger; clear of training/eval
    # Gap targeting: rerun the cells of a previous eval run instead of a uniform grid.
    cells_from: Path | None = None
    cells: Literal["failed", "all"] = "failed"
    cells_limit: int | None = None
    cube_yaw: float = 0.0
    # Rollout cap in CONTROL steps (episode ends at success, cube off-table, or here).
    max_control_steps: int = 1200      # 40 s at 30 Hz
    sim_hz: int = 120
    control_hz: int = 30
    close_setpoint: float = 0.58       # closed-finger dof (training value)
    backend: Literal["gpu", "cpu"] = "gpu"
    video: bool = True
    out: Path | None = None            # explicit diagnostics dir; None uses the run layout
    # Root for the DAgger training MCAP runs:
    #   <mcap-dir>/<model>/<dagger-version>/<generation-run>/
    mcap_dir: Path | None = Path("/data/store/griffen_sim_mcaps/dagger_mcaps")
    # names the student model in MCAP filenames/manifest (REQUIRED: the path needs it)
    model_name: str | None = None
    dagger_version: str = "paper-v1"
    generation_run: str | None = None
    # New-lineage model data always uses Genesis foregrounds composited over gsplat.
    env: TaskEnvCfg = field(default_factory=lambda: TaskEnvCfg(
        noslip_iterations=10, render_backend="raster", splat_bg=True))


# ---------------------------------------------------------------------------------------
# beta / naming / cells helpers (mirrors scripts/dagger.py conventions)
# ---------------------------------------------------------------------------------------


def _resolve_beta(cfg: Config) -> float:
    if (cfg.beta is None) == (cfg.beta_from is None):
        raise SystemExit("provide exactly one of --beta or --beta-from")
    if cfg.beta_from is not None:
        summary = json.loads(Path(cfg.beta_from).read_text())
        if "overall_success_rate" not in summary:  # eval_grid.write_summary field
            raise SystemExit(f"--beta-from {cfg.beta_from}: no 'overall_success_rate' key")
        success = float(summary["overall_success_rate"])
        return float(np.clip(1.0 - 1.2 * success, 0.2, 1.0))
    return float(cfg.beta)


def _model_name(cfg: Config) -> str:
    if not cfg.model_name:
        raise SystemExit("--model-name is required: it names the student in the MCAP path")
    return cfg.model_name


def _path_component(value: str, label: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    if not component:
        raise SystemExit(f"--{label} must contain at least one letter or number")
    return component


def _generation_run(cfg: Config, beta: float, n_cells: int) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    shape = f"cells{n_cells}" if cfg.cells_from is not None else f"{cfg.grid_nx}x{cfg.grid_ny}"
    name = f"round{cfg.round}_beta{beta:.2f}_{stamp}_seed{cfg.seed}_{shape}_r{cfg.reps}"
    if cfg.generation_run:
        name = _path_component(cfg.generation_run, "generation-run")
    return name


def _cells_from_eval(path: Path, which: str) -> list[GridPoint]:
    """Grid cells of a previous eval run, from its results.jsonl (deduped by grid_idx;
    ``failed`` keeps cells that failed in at least one rep). Copied from scripts/dagger.py."""
    points: dict[int, GridPoint] = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("grid_idx") in points or (which == "failed" and r.get("success")):
                continue
            points[r["grid_idx"]] = GridPoint(
                grid_idx=int(r["grid_idx"]), ix=-1, iy=-1,
                cube_xy=(float(r["cube_xy"][0]), float(r["cube_xy"][1])))
    if not points:
        raise SystemExit(f"--cells-from {path}: no matching cells (--cells {which})")
    return [points[i] for i in sorted(points)]


def _run_dir(root: Path, model: str, cfg: Config, generation_run: str) -> Path:
    return (root / _path_component(model, "model-name")
            / _path_component(cfg.dagger_version, "dagger-version")
            / generation_run)


# ---------------------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------------------


def build_student(cfg: Config):
    """The served crossformer, returning an ``(H, A)`` chunk per query. Only called when
    beta < 1.0, so a pure-teacher round never imports webpolicy or contacts a server."""
    from webpolicy.client import Client

    from eval import ChunkClient  # scripts/eval.py

    return ChunkClient(Client(cfg.host, cfg.port))


def _student_chunk(student, obs, chunk_h: int) -> deque:
    chunk = np.atleast_2d(np.asarray(student.step(obs), dtype=np.float32))
    return deque(chunk[:chunk_h])


# ---------------------------------------------------------------------------------------
# Episode: per-window beta mixture, expert labels every tick
# ---------------------------------------------------------------------------------------


def run_episode(env, recorder, expert, student, seed: int, options: dict, beta: float,
                chunk_h: int, max_control_steps: int, phase: dict, rng: np.random.Generator):
    """Roll one episode. Returns (info, ticks, windows) where ``windows`` is the per-window
    ``use_teacher`` draws. The expert labels EVERY tick; control is teacher- or student-driven
    per window."""
    obs = env.reset(seed=seed, options=options)
    expert.reset()
    if student is not None:
        student.reset()

    done, info = False, {}
    use_teacher = True
    chunk: deque = deque()
    windows: list[bool] = []
    tick = 0
    while not done and tick < max_control_steps:
        label = np.asarray(expert.act(), dtype=np.float32)   # ALWAYS: this is the label
        recorder.set_teacher(label[:7], expert.last_pos_cmd, expert.last_quat_cmd, float(label[7]))

        if tick % chunk_h == 0:  # replan boundary: draw the window's controller
            use_teacher = beta >= 1.0 or float(rng.random()) < beta
            windows.append(bool(use_teacher))
            if not use_teacher:  # query the student ONCE for the whole window
                chunk = _student_chunk(student, obs, chunk_h)

        if use_teacher:
            action = label
            phase["mode"] = "teacher"
        else:
            if not chunk:  # window outran the returned chunk: re-infer from live obs
                chunk = _student_chunk(student, obs, chunk_h)
            action = chunk.popleft()
            phase["mode"] = "student"

        obs, _, done, info = env.step(action)
        tick += 1
    return info, tick, windows


def _outcome(info: dict) -> str:
    if info.get("success"):
        return "success"
    if info.get("fell"):
        return "cube_fell"
    if info.get("flew"):
        return "cube_flew"
    if info.get("timeout"):
        return "timeout"
    return "ended"


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------


def main(cfg: Config) -> None:
    beta = _resolve_beta(cfg)          # fail fast, before the env spins up
    model = _model_name(cfg)

    cfg.env.task = cfg.task
    cfg.env.physics_dt = 1.0 / cfg.sim_hz
    control_every = max(1, round(cfg.sim_hz / cfg.control_hz))
    control_dt = control_every * cfg.env.physics_dt

    if cfg.cells_from is not None:
        points = _cells_from_eval(cfg.cells_from, cfg.cells)
        if cfg.cells_limit is not None and cfg.cells_limit < len(points):
            points = points[:: max(1, len(points) // cfg.cells_limit)][: cfg.cells_limit]
        print(f"cells: {len(points)} {cfg.cells} cells from {cfg.cells_from}")
    else:
        x_range, y_range = grid_ranges(cfg, cfg.env)
        points, _, _ = build_grid(cfg.task, cfg.grid_nx, cfg.grid_ny, x_range, y_range)

    generation_run = _generation_run(cfg, beta, len(points))
    default_out_root = Path("/data/store/griffen_sim_mcaps/dagger_runs")
    out = cfg.out if cfg.out is not None else _run_dir(default_out_root, model, cfg, generation_run)
    mcap_run_dir = (_run_dir(cfg.mcap_dir, model, cfg, generation_run)
                    if cfg.mcap_dir is not None else None)
    out.mkdir(parents=True, exist_ok=True)
    if mcap_run_dir is not None:
        mcap_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"round {cfg.round}: beta={beta:.3f} ({'pure teacher' if beta >= 1.0 else 'mixture'}) "
          f"-> {generation_run}")

    gs.init(backend=gs.gpu if cfg.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    adapter = GenesisGymAdapter(
        TaskEnv(cfg.env), control_every=control_every,
        max_control_steps=cfg.max_control_steps, close_setpoint=cfg.close_setpoint)
    # recorder sits below the strip so buffered MCAP images are clean of the overlay;
    # keep_all: DAgger keeps every visited state, no release trim.
    recorder = McapRecordWrapper(adapter, record_dt=control_dt, keep_all=True)
    strip = ModeStripWrapper(recorder)
    env = (VideoRecordWrapper(strip, out / "videos", capture_every=1,
                              fps=float(cfg.control_hz), name_prefix="rollout")
           if cfg.video else strip)

    expert = LiftExpertPolicy(env)
    student = build_student(cfg) if beta < 1.0 else None  # never construct client at beta>=1
    phase = {"mode": "student"}
    strip.mode_fn = lambda: phase["mode"]

    counts: dict[str, int] = {}
    records: list[dict] = []
    mcap_episodes: list[dict] = []
    with open(out / "results.jsonl", "a") as results_fh, torch.no_grad():
        for rep in range(cfg.reps):
            for gp in points:
                seed = cfg.seed + rep * 10007 + gp.grid_idx
                options = {"cube_xy": gp.cube_xy, "cube_yaw": cfg.cube_yaw,
                           "drop_xy": (0.35, 0.0)}
                # Bernoulli rng seeded from the episode seed for reproducible mixing.
                rng = np.random.default_rng(seed)
                t0 = time.monotonic()
                video_id = getattr(env, "_episode_id", -1) + 1

                info, ticks, windows = run_episode(
                    env, recorder, expert, student, seed, options, beta,
                    cfg.chunk_h, cfg.max_control_steps, phase, rng)
                outcome = _outcome(info)
                counts[outcome] = counts.get(outcome, 0) + 1
                teacher_windows = int(sum(windows))
                student_windows = len(windows) - teacher_windows

                mcap_path = mcap_run_dir / f"{model}_episode_{seed:06d}.mcap" \
                    if mcap_run_dir is not None else None
                frames = recorder.save(mcap_path)["frames"] if mcap_path is not None \
                    else recorder.trimmed_frames
                if mcap_path is None:
                    recorder.discard()

                record = {
                    "rep": rep, "grid_idx": gp.grid_idx,
                    "cube_xy": [float(gp.cube_xy[0]), float(gp.cube_xy[1])], "seed": seed,
                    "round": cfg.round, "beta": round(beta, 4),
                    "outcome": outcome, "frames": frames, "ticks": ticks,
                    "n_windows": len(windows),
                    "teacher_windows": teacher_windows, "student_windows": student_windows,
                    "success": bool(info.get("success")),
                    "max_rise": round(float(info.get("max_rise", 0.0)), 4),
                    "deliver_dist": round(float(info.get("deliver_dist", -1.0)), 4),
                    "fell": bool(info.get("fell")), "flew": bool(info.get("flew")),
                    "timeout": bool(info.get("timeout")),
                    "video": video_id, "wall_s": round(time.monotonic() - t0, 1),
                }
                if mcap_path is not None:
                    record["mcap"] = str(mcap_path)
                    entry = {
                        "episode": seed, "frames": frames, "seed": seed, "kept": True,
                        "model": model, "round": cfg.round, "beta": round(beta, 4),
                        "cube_yaw": cfg.cube_yaw, "grid_idx": gp.grid_idx,
                        "cube_xy": record["cube_xy"], "outcome": outcome,
                        "teacher_windows": teacher_windows, "student_windows": student_windows,
                        "extrinsics": {k: np.asarray(v).tolist()
                                       for k, v in adapter.episode_extrinsics.items()},
                    }
                    for key in ("max_rise", "lifted", "deliver_dist", "delivered",
                                "success", "drop_target"):
                        if key in info:
                            entry[key] = _jsonable(info[key])
                    mcap_episodes.append(entry)

                results_fh.write(json.dumps(record) + "\n")
                results_fh.flush()
                records.append(record)
                print(f"rep{rep} grid{gp.grid_idx:03d} xy=({gp.cube_xy[0]:.3f},{gp.cube_xy[1]:.3f}) "
                      f"{outcome} windows={teacher_windows}T/{student_windows}S frames={frames} "
                      f"ticks={ticks} [{record['wall_s']}s]", flush=True)

    env.close()
    sha, dirty = _git_provenance()
    summary = {
        "source": "scripts/dagger_rounds.py",
        "model": model, "round": cfg.round, "beta": beta,
        "beta_from": str(cfg.beta_from) if cfg.beta_from is not None else None,
        "dagger_version": cfg.dagger_version, "generation_run": generation_run,
        "mcap_run_dir": str(mcap_run_dir) if mcap_run_dir is not None else None,
        "n_episodes": len(records), "outcomes": counts,
        "cells_from": str(cfg.cells_from) if cfg.cells_from is not None else None,
        "cells": cfg.cells if cfg.cells_from is not None else None,
        "seed": cfg.seed, "chunk_h": cfg.chunk_h,
        "max_control_steps": cfg.max_control_steps,
        "git_sha": sha, "git_dirty": dirty,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    if mcap_run_dir is not None and mcap_episodes:
        # A run directory owns one manifest; a re-run of an explicit generation_run merges
        # by (model, episode).
        manifest_path = mcap_run_dir / "manifest.json"
        existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        by_ep = {(e.get("model"), e["episode"]): e for e in existing.get("episodes", [])}
        for e in mcap_episodes:
            by_ep[(e["model"], e["episode"])] = e
        manifest_path.write_text(json.dumps({
            "source": "scripts/dagger_rounds.py",
            "git_sha": sha, "git_dirty": dirty,
            "model": model, "round": cfg.round, "beta": beta,
            "dagger_version": cfg.dagger_version, "generation_run": generation_run,
            "mcap_run_dir": str(mcap_run_dir), "config": _jsonable(cfg),
            "episodes": sorted(by_ep.values(), key=lambda e: (e.get("model", ""), e["episode"])),
        }, indent=2))
        print(f"mcap: {len(mcap_episodes)} episodes -> {mcap_run_dir} "
              f"(manifest total {len(by_ep)})")
    print(f"\ndagger round {cfg.round} done: {counts} -> {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
