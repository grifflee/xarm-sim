"""Spawn-feasibility sweep: measure WHERE on the table a cube can be picked and delivered.

``TaskEnvCfg.rectangle_x`` is the lift task's cube spawn range and was chosen by hand
(0.20-0.40 m). This script measures the real feasible region instead: it walks a regular
x/y grid over the table, pins the spawn to each cell (the ``eval_grid.py`` idiom --
``rectangle_x = (x, x)``), runs the scripted lift policy once (or ``--yaw-reps`` times with
different cube yaw / arm jitter), and scores it with the shared ``xsim.success`` rule.

This is a physics/IK question, not a rendering one, so the sweep runs with the splat
background OFF, the raster backend, and **never** calls ``env.render()``. One ``TaskEnv`` is
built and reused across every cell (~1-2 s/cell instead of ~7 s).

    # coarse pass (~5 min)
    uv run python scripts/spawn_feasibility.py --pitch 0.05
    # full-table sweep at 3 cm pitch (~540 cells)
    uv run python scripts/spawn_feasibility.py --x-range 0.02 0.82 --y-range -0.288 0.288 --pitch 0.03
    # redraw the figure from an existing results.jsonl without touching Genesis
    uv run python scripts/spawn_feasibility.py --plot-only

Outputs (under ``--out-dir``, default on /data/store -- never the repo's outputs/):
``results.jsonl`` (one record per trial), ``summary.json`` (per-cell / per-column success
plus a data-driven rectangle_x recommendation) and ``feasibility.png`` (the heatmap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import time
from typing import Literal

import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import genesis as gs  # noqa: E402

from xsim.task_env import TaskEnv, TaskEnvCfg  # noqa: E402
from xsim.scripted_lift_policy import ScriptedLiftPolicy  # noqa: E402
from xsim.success import episode_result  # noqa: E402

# The legacy rectangle this sweep exists to challenge, drawn on the figure for reference.
CURRENT_RECT_X = (0.20, 0.40)
CURRENT_RECT_Y = (-0.288, 0.288)
# Production proposal chosen from this sweep.  The rectangle is only the rejection
# sampler's proposal box; accepted points must also fall inside the radial band.
CHOSEN_RECT_X = (0.0, 0.445)
CHOSEN_RECT_Y = (-0.288, 0.288)
CHOSEN_SPAWN_RADIUS = (0.25, 0.445)
M_PER_IN = 0.0254


@dataclass
class Config:
    # --- sweep geometry ---
    x_range: tuple[float, float] = (0.02, 0.82)     # cube x span to probe (m, robot frame)
    y_range: tuple[float, float] = (-0.288, 0.288)  # cube y span to probe (m)
    pitch: float = 0.03                             # grid spacing (m) in both axes
    yaw_reps: int = 1                 # trials per cell; each redraws cube yaw + arm jitter
    seed: int = 60000                 # base seed, clear of the reserved batch ranges
    # --- output ---
    out_dir: Path = Path("/data/store/griffen_sim_mcaps/spawn_feasibility")
    resume: bool = True               # skip (cell, rep) pairs already in results.jsonl
    plot_only: bool = False           # rebuild summary.json + the PNG from results.jsonl
    # --- episode protocol (mirrors scripts/generate_task_dataset.py) ---
    backend: Literal["gpu", "cpu"] = "gpu"
    # "weld": grasp_lock/grasp_release exactly like the data generator (what will actually
    # consume a widened rectangle). "free": friction-only, needs noslip_iterations=10.
    grasp_mode: Literal["weld", "free"] = "weld"
    close_setpoint: float = 0.58      # closed-finger dof, only applied in grasp_mode=free
    steps_per_segment: int = 108      # generator default (no per-episode tempo jitter here)
    hold_steps: int = 48              # unrecorded settle after release
    release_tail_s: float = 0.3
    grasp_tcp_offset: float = 0.018
    drop_xy: tuple[float, float] = (0.35, 0.0)  # pinned drop target: isolates spawn feasibility
    # A welded grasp can drag a cube the gripper never actually reached (unreachable
    # spawn -> IK saturates -> the weld snaps the cube on from 20 cm away and "delivers"
    # it). A trial only counts as CLEAN when the EE was within this xy distance of the
    # cube at close time. Applied at aggregation time, so --plot-only can re-derive it
    # from an existing results.jsonl without re-running the sweep.
    close_xy_tol: float = 0.03
    # --- success thresholds (duck-typed into xsim.success.episode_result) ---
    task: Literal["lift"] = "lift"
    lift_threshold: float = 0.05
    deliver_radius: float = 0.12
    stack_xy_tol: float = 0.02        # unused for lift; present for the duck-typed cfg
    stack_z_tol: float = 0.008
    # --- recommendation ---
    column_pass: float = 0.80         # min per-column success rate for a column to be "in"
    # splat_bg=False is the whole point: it skips loading the 328 MB .ply and every
    # per-step gsplat render. Physics/IK are untouched by it, and render() is never called.
    env: TaskEnvCfg = field(default_factory=lambda: TaskEnvCfg(
        splat_bg=False,
        render_backend="raster",
        noslip_iterations=0,
        # Preserve the environment used to collect the historical grid. Each cell is
        # pinned below, and the sweep intentionally excludes new arm/camera domains.
        spawn_radius=None,
        arm_start_mode="home",
        camera_mode="fixed",
    ))


# ---------------------------------------------------------------------------------------
# Grid (pure)
# ---------------------------------------------------------------------------------------


def build_axis(lo: float, hi: float, pitch: float) -> list[float]:
    """Inclusive-of-lo grid coords stepping by ``pitch`` and never exceeding ``hi``."""
    n = int(math.floor((hi - lo) / pitch + 1e-9)) + 1
    return [round(lo + i * pitch, 6) for i in range(max(1, n))]


def build_cells(cfg: Config) -> tuple[list[tuple[int, int, float, float]], list[float], list[float]]:
    xs = build_axis(cfg.x_range[0], cfg.x_range[1], cfg.pitch)
    ys = build_axis(cfg.y_range[0], cfg.y_range[1], cfg.pitch)
    cells = [(ix, iy, x, y) for ix, x in enumerate(xs) for iy, y in enumerate(ys)]
    return cells, xs, ys


# ---------------------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------------------


def run_cell(env, cfg: Config, ix: int, iy: int, x: float, y: float, rep: int) -> dict:
    """Pin the spawn to (x, y), run one scripted lift, and score it."""
    import torch

    seed = cfg.seed + rep * 100003 + ix * 211 + iy
    t0 = time.monotonic()
    rec = {
        "ix": ix, "iy": iy, "x": float(x), "y": float(y), "rep": rep, "seed": seed,
        "ok": False, "error": None,
    }
    try:
        # eval_grid.py idiom: uniform(a, a) == a, so reset() draws exactly this cell
        env.cfg.rectangle_x = (x, x)
        env.cfg.rectangle_y = (y, y)
        env.reset(seed=seed)
        env.current_drop_xy = (float(cfg.drop_xy[0]), float(cfg.drop_xy[1]))

        if cfg.grasp_mode == "free":
            env.robot._gripper_grasp_dof = cfg.close_setpoint
        # constructed after reset+drop pin: the policy caches the cube pose and drop target
        policy = ScriptedLiftPolicy(env, steps_per_segment=cfg.steps_per_segment,
                                    grasp_tcp_offset=cfg.grasp_tcp_offset)
        policy.reset()

        cube_start = env.cube_pos().copy()
        max_rise = 0.0
        min_ee_cube = float("inf")
        close_xy_err = float("nan")
        release_tail = max(1, int(round(cfg.release_tail_s / env.cfg.physics_dt)))
        record_until = policy.release_step + release_tail

        with torch.no_grad():
            for i in range(record_until):
                cmd = policy.step()
                if i == policy.grasp_lock_step:
                    # measured BEFORE any weld: how close the EE actually got in xy.
                    # A welded grasp can otherwise drag a cube the gripper never reached.
                    ee = np.asarray(env.robot.ee_pose.detach().cpu()).reshape(-1)[:3]
                    cube_now = np.asarray(env.cube_pos(), dtype=np.float64).reshape(-1)
                    close_xy_err = float(np.linalg.norm(ee[:2] - cube_now[:2]))
                    if cfg.grasp_mode == "weld":
                        env.grasp_lock()
                if i == policy.release_step:
                    env.grasp_release()
                env.robot.go_to_goal(cmd.pose, open_gripper=cmd.open_gripper)
                env.step()
                cube = env.cube_pos()
                max_rise = max(max_rise, float(cube[2] - cube_start[2]))
                if i % 30 == 0:
                    ee = np.asarray(env.robot.ee_pose.detach().cpu()).reshape(-1)[:3]
                    min_ee_cube = min(min_ee_cube, float(np.linalg.norm(
                        ee - np.asarray(cube, dtype=np.float64).reshape(-1))))
            for _ in range(cfg.hold_steps):
                cmd = policy.step()
                env.robot.go_to_goal(cmd.pose, open_gripper=cmd.open_gripper)
                env.step()

        res = episode_result(env, cfg, max_rise)
        cube_end = np.asarray(env.cube_pos(), dtype=np.float64).reshape(-1)
        rec.update(res)
        rec.update({
            "ok": True,
            "cube_yaw": float(env.cube_yaw()),
            "cube_end_xy": [float(cube_end[0]), float(cube_end[1])],
            "cube_end_z": float(cube_end[2]),
            "close_xy_err": close_xy_err,
            "min_ee_cube": (min_ee_cube if math.isfinite(min_ee_cube) else None),
            # the honest verdict: scored success AND the gripper was really on the cube
            "clean_success": bool(res["success"]
                                  and math.isfinite(close_xy_err)
                                  and close_xy_err <= cfg.close_xy_tol),
        })
    except Exception as exc:  # unreachable IK / solver blowups must not kill the sweep
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec.update({"max_rise": 0.0, "lifted": False, "deliver_dist": None,
                    "delivered": False, "success": False, "clean_success": False,
                    "close_xy_err": None, "min_ee_cube": None,
                    "cube_end_xy": None, "cube_end_z": None})
    rec["wall_s"] = round(time.monotonic() - t0, 3)
    return rec


# ---------------------------------------------------------------------------------------
# Aggregation / summary / figure
# ---------------------------------------------------------------------------------------


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def is_clean(cfg: Config, r: dict) -> bool:
    """Scored success AND the gripper was really on the cube when it closed."""
    err = r.get("close_xy_err")
    return bool(r.get("success")) and err is not None and math.isfinite(err) and err <= cfg.close_xy_tol


def _grids(cfg: Config, xs: list[float], ys: list[float],
           records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(success rate, clean success rate, n trials) grids shaped [ny, nx]."""
    ny, nx = len(ys), len(xs)
    s = np.zeros((ny, nx))
    c = np.zeros((ny, nx))
    n = np.zeros((ny, nx))
    for r in records:
        ix, iy = r["ix"], r["iy"]
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        n[iy, ix] += 1
        s[iy, ix] += float(bool(r.get("success")))
        c[iy, ix] += float(is_clean(cfg, r))
    with np.errstate(invalid="ignore"):
        rate = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        clean = np.where(n > 0, c / np.maximum(n, 1), np.nan)
    return rate, clean, n


def recommend_rect(cfg: Config, xs: list[float], clean: np.ndarray) -> dict:
    """Largest contiguous run of x-columns whose mean clean-success clears --column-pass."""
    col = np.nanmean(clean, axis=0)  # mean over y per x column
    good = [bool(np.isfinite(v) and v >= cfg.column_pass) for v in col]
    best = (0, -1)
    i = 0
    while i < len(good):
        if not good[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(good) and good[j + 1]:
            j += 1
        if (j - i) > (best[1] - best[0]):
            best = (i, j)
        i = j + 1
    lo_i, hi_i = best
    if hi_i < lo_i:
        return {"feasible": False, "column_clean_success": [float(v) for v in col]}
    inside = clean[:, lo_i:hi_i + 1]
    return {
        "feasible": True,
        "rectangle_x": [float(xs[lo_i]), float(xs[hi_i])],
        "rectangle_x_inches": [float(xs[lo_i] / M_PER_IN), float(xs[hi_i] / M_PER_IN)],
        "clean_success_inside": float(np.nanmean(inside)),
        "column_pass_threshold": cfg.column_pass,
        "column_x": [float(v) for v in xs],
        "column_clean_success": [float(v) for v in col],
    }


def write_summary(cfg: Config, out_dir: Path, xs: list[float], ys: list[float],
                  records: list[dict]) -> dict:
    rate, clean, n = _grids(cfg, xs, ys, records)
    rec = recommend_rect(cfg, xs, clean)
    summary = {
        "n_trials": len(records),
        "n_cells": len(xs) * len(ys),
        "x_coords": xs,
        "y_coords": ys,
        "pitch": cfg.pitch,
        "yaw_reps": cfg.yaw_reps,
        "grasp_mode": cfg.grasp_mode,
        "close_xy_tol": cfg.close_xy_tol,
        "drop_xy": list(cfg.drop_xy),
        "overall_success": float(np.nanmean(rate)) if np.any(n > 0) else 0.0,
        "overall_clean_success": float(np.nanmean(clean)) if np.any(n > 0) else 0.0,
        "n_errors": sum(1 for r in records if r.get("error")),
        "success_grid": np.where(np.isnan(rate), None, rate).tolist(),
        "clean_success_grid": np.where(np.isnan(clean), None, clean).tolist(),
        "trials_grid": n.astype(int).tolist(),
        "recommendation": rec,
        "current_rectangle_x": list(CURRENT_RECT_X),
        "current_rectangle_y": list(CURRENT_RECT_Y),
        "chosen_rectangle_x": list(CHOSEN_RECT_X),
        "chosen_rectangle_y": list(CHOSEN_RECT_Y),
        "chosen_spawn_radius": list(CHOSEN_SPAWN_RADIUS),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_heatmap(cfg: Config, out_dir: Path, xs: list[float], ys: list[float],
                  records: list[dict], summary: dict, table) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle, Rectangle

    clean = np.array(
        [[np.nan if v is None else v for v in row] for row in summary["clean_success_grid"]],
        dtype=float,
    )
    pitch = cfg.pitch
    x_edges = np.array([x - pitch / 2 for x in xs] + [xs[-1] + pitch / 2])
    y_edges = np.array([y - pitch / 2 for y in ys] + [ys[-1] + pitch / 2])

    # sequential = ONE hue, light -> dark (never a rainbow); light end starts above pure
    # white so a measured 0% cell is still visibly a cell, and no-data cells get the
    # explicit gray "bad" colour instead.
    cmap = LinearSegmentedColormap.from_list(
        "success", ["#f1f6f2", "#cbe3d4", "#8bc7a3", "#3d9866", "#11603c"])
    cmap.set_bad("#e2e2e4")

    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(clean),
                         cmap=cmap, vmin=0.0, vmax=1.0, edgecolors="#ffffff", linewidth=0.35)

    # table outline (top-down footprint of the real cart)
    tx0 = table.center_xy[0] - table.size_xy[0] / 2
    ty0 = table.center_xy[1] - table.size_xy[1] / 2
    ty1 = ty0 + table.size_xy[1]
    ax.add_patch(Rectangle((tx0, ty0), table.size_xy[0], table.size_xy[1],
                           fill=False, edgecolor="#26262b", linewidth=2.0, zorder=4))
    ax.text(tx0 - 0.008, ty1, "table top", fontsize=9, color="#26262b",
            ha="right", va="center", zorder=5)

    # the CURRENT spawn rectangle, for reference
    cx0, cx1 = CURRENT_RECT_X
    cy0, cy1 = CURRENT_RECT_Y
    ax.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False,
                           edgecolor="#c2410c", linewidth=2.2, linestyle="--", zorder=6))
    ax.text((cx0 + cx1) / 2, ty1 + 0.016,
            f"current rectangle_x = ({cx0:.2f}, {cx1:.2f})  [dashed]",
            fontsize=10.5, color="#c2410c", ha="center", va="bottom", zorder=6)

    # The selected production domain: proposal rectangle intersected with an annulus.
    # Circles make the radial reach constraint legible without obscuring measured cells.
    px0, px1 = CHOSEN_RECT_X
    py0, py1 = CHOSEN_RECT_Y
    inner_r, outer_r = CHOSEN_SPAWN_RADIUS
    ax.add_patch(Rectangle((px0, py0), px1 - px0, py1 - py0, fill=False,
                           edgecolor="#7c3aed", linewidth=1.8, linestyle=":", zorder=6))
    ax.add_patch(Circle((0.0, 0.0), inner_r, fill=False, edgecolor="#7c3aed",
                        linewidth=2.4, linestyle="-.", zorder=6))
    ax.add_patch(Circle((0.0, 0.0), outer_r, fill=False, edgecolor="#7c3aed",
                        linewidth=2.4, linestyle="-.", zorder=6))
    ax.text(0.445, -0.245,
            "chosen production domain\n"
            "proposal box ∩ annulus r = 0.250–0.445 m  [purple]",
            fontsize=9.5, color="#6d28d9", ha="right", va="top", zorder=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2})

    # the measured recommendation
    rec = summary["recommendation"]
    if rec.get("feasible"):
        rx0, rx1 = rec["rectangle_x"]
        ry0, ry1 = ys[0] - pitch / 2, ys[-1] + pitch / 2
        ax.add_patch(Rectangle((rx0 - pitch / 2, ry0), (rx1 - rx0) + pitch, ry1 - ry0,
                               fill=False, edgecolor="#1d4ed8", linewidth=2.4, zorder=6))
        ax.text((rx0 + rx1) / 2, ty0 - 0.018,
                f"measured feasible x = ({rx0:.2f}, {rx1:.2f}) m = "
                f"({rx0/M_PER_IN:.1f}, {rx1/M_PER_IN:.1f}) in  "
                f"[{rec['clean_success_inside']:.0%} success inside]  [solid]",
                fontsize=10.5, color="#1d4ed8", ha="center", va="top", zorder=6)

    # robot base
    ax.plot([0.0], [0.0], marker="o", markersize=11, color="#26262b", zorder=7)
    ax.plot([0.0, 0.075], [0.0, 0.0], color="#26262b", linewidth=2.0, zorder=7)
    ax.annotate("robot base (0, 0)\nfacing +x", xy=(0.0, 0.0), xytext=(0.005, -0.075),
                fontsize=10, color="#26262b", ha="left", va="top", zorder=7)

    ax.set_xlabel("cube spawn x  (m from robot base, +x = away from robot)")
    ax.set_ylabel("cube spawn y  (m)")
    reps = cfg.yaw_reps
    ax.set_title(
        f"Cube spawn feasibility: scripted lift + deliver, {reps} trial(s)/cell, "
        f"{pitch*100:.0f} cm pitch, grasp={cfg.grasp_mode}\n"
        f"colour = fraction of trials that lifted >= {cfg.lift_threshold:g} m AND landed "
        f"within {cfg.deliver_radius:g} m of the drop target, with the gripper actually on the cube",
        fontsize=11.5, pad=26,
    )
    ax.set_aspect("equal")
    ax.set_xlim(min(-0.10, x_edges[0] - 0.02), max(0.90, x_edges[-1] + 0.02))
    ax.set_ylim(min(y_edges[0], ty0) - 0.075, max(y_edges[-1], ty1) + 0.055)
    ax.set_xticks(np.arange(-0.1, 0.95, 0.1))
    ax.set_yticks(np.arange(-0.3, 0.35, 0.1))
    ax.tick_params(labelsize=9)

    # secondary axis in inches from the robot origin -- how grifflee thinks about reach
    sec = ax.secondary_xaxis("top", functions=(lambda m: m / M_PER_IN,
                                               lambda i: i * M_PER_IN))
    sec.set_xlabel("distance from robot base along x  (inches)", fontsize=10)
    sec.set_xticks(np.arange(-4, 36, 2))
    sec.tick_params(labelsize=9)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("pick + deliver success rate per cell", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    path = out_dir / "feasibility.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------


def main(cfg: Config) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    cells, xs, ys = build_cells(cfg)
    print(f"grid: {len(xs)} x-cols x {len(ys)} y-rows = {len(cells)} cells "
          f"x {cfg.yaw_reps} rep(s) = {len(cells) * cfg.yaw_reps} trials")

    if cfg.plot_only:
        records = _load_records(results_path)
        summary = write_summary(cfg, out_dir, xs, ys, records)
        png = write_heatmap(cfg, out_dir, xs, ys, records, summary, cfg.env.table)
        print(f"replotted {len(records)} trials -> {png}")
        return

    cfg.env.task = "lift"
    if cfg.grasp_mode == "free" and cfg.env.noslip_iterations == 0:
        cfg.env.noslip_iterations = 10  # friction-only grasping creeps without it
    if cfg.env.splat_bg:
        raise SystemExit("refusing to run with splat_bg=True: the sweep must not render")

    records = _load_records(results_path) if cfg.resume else []
    done = {(r["ix"], r["iy"], r["rep"]) for r in records}
    if records:
        print(f"[resume] {len(records)} trials already in {results_path}")

    t_init = time.monotonic()
    gs.init(backend=gs.gpu if cfg.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    env = TaskEnv(cfg.env)
    print(f"env built in {time.monotonic() - t_init:.1f} s "
          f"(splat_bg={cfg.env.splat_bg}, render_backend={cfg.env.render_backend})", flush=True)

    todo = [(ix, iy, x, y, rep) for rep in range(cfg.yaw_reps) for (ix, iy, x, y) in cells
            if (ix, iy, rep) not in done]
    n_done = 0
    n_clean = sum(1 for r in records if r.get("clean_success"))
    t_sweep = time.monotonic()
    with open(results_path, "a") as fh:
        for ix, iy, x, y, rep in todo:
            r = run_cell(env, cfg, ix, iy, x, y, rep)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            records.append(r)
            n_done += 1
            n_clean += int(is_clean(cfg, r))
            elapsed = time.monotonic() - t_sweep
            eta = (elapsed / n_done) * (len(todo) - n_done)
            err = f" ERROR {r['error']}" if r.get("error") else ""
            print(f"[{n_done}/{len(todo)}] x={x:+.3f} y={y:+.3f} rep{rep} "
                  f"rise={r.get('max_rise') or 0.0:.3f} "
                  f"deliver={(r.get('deliver_dist') if r.get('deliver_dist') is not None else float('nan')):.3f} "
                  f"closeErr={(r.get('close_xy_err') if r.get('close_xy_err') is not None else float('nan')):.3f} "
                  f"clean={bool(r.get('clean_success'))} "
                  f"| {r['wall_s']:.2f}s avg={elapsed/n_done:.2f}s eta={eta/60:.1f}m"
                  f" | clean {n_clean}/{len(records)}{err}", flush=True)
            if n_done % 25 == 0:
                write_summary(cfg, out_dir, xs, ys, records)

    summary = write_summary(cfg, out_dir, xs, ys, records)
    png = write_heatmap(cfg, out_dir, xs, ys, records, summary, cfg.env.table)
    rec = summary["recommendation"]
    print(f"\nsweep done: {len(records)} trials, "
          f"clean success {summary['overall_clean_success']:.1%} "
          f"(raw {summary['overall_success']:.1%}), errors {summary['n_errors']}")
    if rec.get("feasible"):
        rx0, rx1 = rec["rectangle_x"]
        print(f"recommended rectangle_x = ({rx0:.3f}, {rx1:.3f}) m "
              f"= ({rx0/M_PER_IN:.1f}, {rx1/M_PER_IN:.1f}) in from base; "
              f"{rec['clean_success_inside']:.1%} success inside")
    else:
        print("no contiguous x-column band cleared --column-pass; inspect the heatmap")
    print(f"-> {png}")


if __name__ == "__main__":
    main(tyro.cli(Config))
