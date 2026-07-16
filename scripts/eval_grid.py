"""Grid-evaluation harness: benchmark a lift/stack policy over a table-position grid.

Places the (red) cube at ``grid_nx x grid_ny`` positions spanning the training spawn
range, gives the policy **one try per position** (pick -> relocate/drop for lift; pick ->
stack on green for stack), repeats the whole grid ``reps`` times, and reports an average
success score plus a per-position heatmap. Each rep re-draws appearance/lighting/camera/
arm-start jitter (seed varies per rep) while the cube grid positions stay identical, so the
average measures robustness to visual variation.

The policy is either a crossformer served over the webpolicy websocket (``--policy remote``)
or one of the repo's scripted waypoint policies run **weld-free** (``--policy scripted``, a
validation baseline). Both are driven through the uniform adapters in ``xsim.eval_policy``.

    # scripted baseline smoke test (fast raster backend, tiny grid):
    uv run python scripts/eval_grid.py --task lift --policy scripted \
        --env.render-backend raster --grid-nx 3 --grid-ny 3 --reps 1 --backend gpu
    # real eval of a served model (grainlike/bela serving stack) on the nyx domain;
    # obs/action defaults match that server so no extra flags are needed:
    uv run python scripts/eval_grid.py --task lift --policy remote --host localhost \
        --port 8001 --env.render-backend nyx --backend gpu --video-every 10

Grasping runs weld-free and requires the noslip post-pass (``--env.noslip-iterations``,
defaulted to 10 here) or the cube creeps down the fingers; the closed-finger setpoint
defaults to the training value 0.58 (``--close-setpoint``).
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
import torch
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import genesis as gs  # noqa: E402

from xsim.task_env import TaskEnv, TaskEnvCfg  # noqa: E402
from xsim.eval_policy import ActionSpec, ObsSpec, RemotePolicy, ScriptedEvalPolicy  # noqa: E402
from xsim.success import episode_result  # noqa: E402


# Five fixed green (base) anchor positions for the stack task. All lie inside the free_green
# ranges and OUTSIDE the side-camera keep-out (x<0.38 and y<-0.10), and are spread so every
# red grid point has at least one anchor within the [0.10, 0.34] m distance constraint.
STACK_GREEN_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.40, -0.08),
    (0.44, 0.08),
    (0.50, 0.00),
    (0.56, -0.10),
    (0.60, 0.10),
)


@dataclass
class Config:
    task: Literal["lift", "stack"] = "lift"
    policy: Literal["remote", "scripted"] = "scripted"  # scripted = validation/baseline
    host: str = "localhost"
    port: int = 8001
    grid_nx: int = 10
    grid_ny: int = 10
    reps: int = 10
    seed: int = 51000                 # eval base seed, clear of training ranges
    max_seconds: float = 20.0         # sim-time cap per trial
    out: Path | None = None           # default: PROJECT_ROOT/outputs/eval/<task>
    backend: Literal["gpu", "cpu"] = "gpu"
    close_setpoint: float = 0.58      # closed-finger dof (training value)
    resume: bool = True
    cube_yaw: float = 0.0             # deterministic cube (and green) yaw at spawn
    video_every: int = 0              # save an mp4 of every Nth trial (0 = off, 1 = all)
    # warm start: drive the TCP to a hover above the cube BEFORE the policy takes over,
    # commanding arm joints only (the fingers keep reset's command; the policy owns the
    # gripper). Separates "can't find the cube" from "can't execute the grasp".
    warm_start: bool = False
    warm_start_height: float = 0.09   # hover height above the cube center (m)
    warm_start_steps: int = 240       # physics steps to settle before handing over (2 s)
    # success thresholds -> duck-typed into xsim.success.episode_result
    lift_threshold: float = 0.05      # min cube rise (m) for a lift
    deliver_radius: float = 0.12      # max xy dist (m) from the drop target for a delivery
    stack_xy_tol: float = 0.02        # max xy offset (m) red-vs-green for a stack
    stack_z_tol: float = 0.008        # max |z error| (m) from ideal stacked height
    # noslip_iterations defaults to 10 here (TaskEnvCfg's own default is 0): weld-free
    # friction grasping needs it or the cube creeps down the fingers. Override on the CLI.
    # nyx is REQUIRED for any remote-policy eval: the model is trained on nyx images
    # (splat background, colored robot); raster frames are out-of-domain and invalidate
    # the numbers. Same default as scripts/dagger.py — never rely on a CLI flag for this.
    env: TaskEnvCfg = field(default_factory=lambda: TaskEnvCfg(
        noslip_iterations=10, render_backend="nyx"))
    obs: ObsSpec = field(default_factory=ObsSpec)        # remote-policy obs mapping
    action: ActionSpec = field(default_factory=ActionSpec)  # remote-policy action mapping


# ---------------------------------------------------------------------------------------
# Grid generation (pure functions: no Genesis, unit-testable standalone)
# ---------------------------------------------------------------------------------------


@dataclass
class GridPoint:
    grid_idx: int
    ix: int
    iy: int
    cube_xy: tuple[float, float]                 # red cube (both tasks)
    green_xy: tuple[float, float] | None = None  # stack only: base-cube anchor
    green_anchor_index: int | None = None        # stack only: which anchor was used


def _anchor_violation(red, anchor, min_dist: float, max_dist: float) -> float:
    """How far the red-green distance falls outside [min_dist, max_dist] (0.0 = feasible)."""
    d = math.hypot(red[0] - anchor[0], red[1] - anchor[1])
    clamped = min(max(d, min_dist), max_dist)
    return abs(d - clamped)


def build_grid(
    task: str,
    grid_nx: int,
    grid_ny: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    anchors: tuple[tuple[float, float], ...] = STACK_GREEN_ANCHORS,
    min_dist: float = 0.10,
    max_dist: float = 0.34,
) -> tuple[list[GridPoint], list[float], list[float]]:
    """Build the trial grid. Returns (points, x_coords, y_coords).

    ``grid_idx = ix * grid_ny + iy`` and ``points`` is ordered by grid_idx. For ``stack`` the
    green anchor for a grid point is ``anchors[grid_idx % len(anchors)]`` when that satisfies
    the distance constraint, else the anchor minimizing the constraint violation.
    """
    xs = [float(v) for v in np.linspace(x_range[0], x_range[1], grid_nx)]
    ys = [float(v) for v in np.linspace(y_range[0], y_range[1], grid_ny)]
    points: list[GridPoint] = []
    for ix in range(grid_nx):
        for iy in range(grid_ny):
            grid_idx = ix * grid_ny + iy
            x, y = xs[ix], ys[iy]
            if task == "stack":
                assigned = grid_idx % len(anchors)
                if _anchor_violation((x, y), anchors[assigned], min_dist, max_dist) == 0.0:
                    ai = assigned
                else:
                    ai = min(
                        range(len(anchors)),
                        key=lambda i: _anchor_violation((x, y), anchors[i], min_dist, max_dist),
                    )
                points.append(GridPoint(grid_idx, ix, iy, (x, y),
                                        (float(anchors[ai][0]), float(anchors[ai][1])), ai))
            else:
                points.append(GridPoint(grid_idx, ix, iy, (x, y)))
    return points, xs, ys


def grid_ranges(cfg: Config, env_cfg: TaskEnvCfg) -> tuple[tuple[float, float], tuple[float, float]]:
    """Read the spawn ranges the grid spans from a TaskEnvCfg (never hard-coded)."""
    if cfg.task == "stack":
        return env_cfg.stack.free_red_x, env_cfg.stack.free_red_y
    return env_cfg.rectangle_x, env_cfg.rectangle_y


# ---------------------------------------------------------------------------------------
# Trial protocol
# ---------------------------------------------------------------------------------------


def _pin_spawn(env: TaskEnv, cfg: Config, gp: GridPoint) -> None:
    """Pin the reset() sampling ranges so reset draws exactly this grid point.

    Done *before* reset (not by moving cubes after) so reset's stack camera-visibility
    redraw runs against the true cube positions. uniform(a, a) == a.
    """
    x, y = gp.cube_xy
    if cfg.task == "stack":
        gx, gy = gp.green_xy
        env.cfg.stack.free_placement = False
        env.cfg.stack.green_x = (gx, gx)
        env.cfg.stack.green_y = (gy, gy)
        env.cfg.stack.red_dx = (x - gx, x - gx)
        env.cfg.stack.red_dy = (y - gy, y - gy)
    else:
        env.cfg.rectangle_x = (x, x)
        env.cfg.rectangle_y = (y, y)


def _capture_frame(env: TaskEnv, video_frames: list[np.ndarray]) -> None:
    frames = env.render()
    video_frames.append(np.concatenate(
        [np.ascontiguousarray(frames[k]) for k in ("low", "side", "wrist")], axis=1))


def _warm_start(env: TaskEnv, cfg: Config, video_frames: list[np.ndarray] | None) -> None:
    """Drive the TCP to a top-down hover above the cube, arm joints only.

    One IK solve, then position control on the arm dofs alone — the finger dofs are
    deliberately never commanded here, so the policy inherits the gripper exactly as
    ``env.reset()`` left it.
    """
    from xsim.scripted_lift_policy import _yawed_top_down_quat

    robot = env.robot
    cube = env.cube_pos()
    pose = torch.as_tensor(
        [[float(cube[0]), float(cube[1]), float(cube[2]) + cfg.warm_start_height,
          *_yawed_top_down_quat(0.0)]],
        dtype=torch.float32, device=env.device,
    )
    q = robot._robot_entity.inverse_kinematics(
        link=robot._ee_link, pos=pose[:, :3], quat=pose[:, 3:7],
        dofs_idx_local=robot._arm_dof_idx,
    )
    robot._robot_entity.control_dofs_position(
        position=q[:, robot._arm_dof_idx], dofs_idx_local=robot._arm_dof_idx)
    for step in range(cfg.warm_start_steps):
        if video_frames is not None and step % env.cfg.record_every == 0:
            _capture_frame(env, video_frames)
        env.step()


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import cv2

    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame[:, :, ::-1])  # RGB -> BGR
    writer.release()


def run_trial(env: TaskEnv, cfg: Config, policy, rep: int, gp: GridPoint,
              video_path: Path | None = None) -> dict:
    """Run one (rep, grid point) trial weld-free and score it."""
    x, y = gp.cube_xy
    seed = cfg.seed + rep * 10007 + gp.grid_idx
    t0 = time.monotonic()

    _pin_spawn(env, cfg, gp)
    env.reset(seed=seed)
    # re-place for deterministic yaw at the SAME xy (visibility unaffected)
    env._place_cube(env.cube, x, y, cfg.cube_yaw)
    env._cube_yaw = cfg.cube_yaw
    if cfg.task == "stack":
        gx, gy = gp.green_xy
        env._place_cube(env.cube2, gx, gy, cfg.cube_yaw)
        env._green_yaw = cfg.cube_yaw
    else:
        env.current_drop_xy = (0.35, 0.0)  # pin the lift drop target for consistent scoring

    video_frames: list[np.ndarray] = []
    if cfg.warm_start:
        _warm_start(env, cfg, video_frames if video_path is not None else None)

    policy.reset()  # after cube placement (ScriptedEvalPolicy caches the cube pose here)

    start_z = float(env.cube_pos()[2])
    table_top = env.cfg.table.top_z
    max_rise = 0.0
    min_tcp_cube = float("inf")
    max_steps = int(cfg.max_seconds / env.cfg.physics_dt)
    n_steps = 0
    with torch.no_grad():
        for step in range(max_steps):
            if step % policy.control_every == 0 and not policy.done:
                policy.act(env)
            if video_path is not None and step % env.cfg.record_every == 0:
                _capture_frame(env, video_frames)
            env.step()
            cube = env.cube_pos()
            max_rise = max(max_rise, float(cube[2] - start_z))
            n_steps = step + 1
            if step % 30 == 0:
                # objective approach metric: how close the TCP ever got to the cube
                _, _, _, ee_pose = env.proprio()
                tcp = np.asarray(ee_pose, dtype=np.float64).reshape(-1)[:3]
                d = float(np.linalg.norm(tcp - np.asarray(cube, dtype=np.float64)))
                min_tcp_cube = min(min_tcp_cube, d)
                fell = float(cube[2]) < table_top - 0.05
                flew = float(math.hypot(cube[0], cube[1])) > 0.9
                if fell or flew:
                    break  # fail-fast: cube left the table
                if policy.done:
                    break  # scripted trajectory finished
                if cfg.policy == "remote":
                    res = episode_result(env, cfg, max_rise)
                    if res["success"] and env.gripper_norm() > 0.9:
                        break  # success achieved and gripper released
        # unrecorded settle holding the last command (no further policy stepping)
        for _ in range(48):
            env.step()

    if video_path is not None and video_frames:
        _write_video(video_path, video_frames, fps=1.0 / (env.cfg.physics_dt * env.cfg.record_every))

    res = episode_result(env, cfg, max_rise)
    if not res["lifted"]:
        failure_stage = "never_lifted"
    elif res["success"]:
        failure_stage = "success"
    else:
        failure_stage = "dropped_in_transit"

    record = {
        "task": cfg.task,
        "rep": rep,
        "grid_idx": gp.grid_idx,
        "ix": gp.ix,
        "iy": gp.iy,
        "cube_xy": [float(x), float(y)],
        "seed": seed,
        "seed_components": {"base": cfg.seed, "rep_term": rep * 10007, "grid_idx": gp.grid_idx},
        "failure_stage": failure_stage,
        "min_tcp_cube": (round(min_tcp_cube, 4) if math.isfinite(min_tcp_cube) else None),
        "n_steps": n_steps,
        "wall_s": round(time.monotonic() - t0, 3),
        "close_setpoint": cfg.close_setpoint,
        "noslip_iterations": env.cfg.noslip_iterations,
    }
    if cfg.task == "stack":
        record["green_anchor_xy"] = [float(gp.green_xy[0]), float(gp.green_xy[1])]
        record["green_anchor_index"] = gp.green_anchor_index
    record.update(res)
    return record


# ---------------------------------------------------------------------------------------
# Results / resume / summary / heatmap
# ---------------------------------------------------------------------------------------


def _jsonable(obj):
    import dataclasses

    if dataclasses.is_dataclass(obj):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def _git_provenance() -> tuple[str, bool]:
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        return "unknown", True
    return (sha or "unknown"), dirty


def _load_records(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    records = []
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def write_summary(out_dir: Path, cfg: Config, resolved_out: Path, points: list[GridPoint],
                  records: list[dict]) -> None:
    from collections import defaultdict

    n = len(records)
    overall = float(np.mean([1.0 if r["success"] else 0.0 for r in records])) if n else 0.0

    per_rep: dict[int, list[float]] = defaultdict(list)
    per_pos: dict[int, list[float]] = defaultdict(list)
    for r in records:
        s = 1.0 if r["success"] else 0.0
        per_rep[r["rep"]].append(s)
        per_pos[r["grid_idx"]].append(s)

    sha, dirty = _git_provenance()
    cfg_json = _jsonable(cfg)
    cfg_json["out"] = str(resolved_out)
    summary = {
        "task": cfg.task,
        "policy": cfg.policy,
        "n_trials": n,
        "n_expected": cfg.reps * len(points),
        "overall_success_rate": overall,
        "per_rep_success_rate": {str(k): float(np.mean(v)) for k, v in sorted(per_rep.items())},
        "per_position_mean_success": {
            str(gp.grid_idx): (float(np.mean(per_pos[gp.grid_idx])) if gp.grid_idx in per_pos else None)
            for gp in points
        },
        "close_setpoint": cfg.close_setpoint,
        "noslip_iterations": cfg.env.noslip_iterations,
        "git_sha": sha,
        "git_dirty": dirty,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg_json,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def write_grid_layout(out_dir: Path, cfg: Config, points: list[GridPoint],
                      xs: list[float], ys: list[float]) -> None:
    """Write a labeled top-down grid layout for visual audit of trial positions."""
    manifest = {
        "task": cfg.task,
        "grid_nx": cfg.grid_nx,
        "grid_ny": cfg.grid_ny,
        "x_coords": [float(v) for v in xs],
        "y_coords": [float(v) for v in ys],
        "points": [_jsonable(gp) for gp in points],
    }
    (out_dir / "grid_layout.json").write_text(json.dumps(manifest, indent=2))

    path = out_dir / "grid_layout.png"
    x_pad = max(0.02, (xs[-1] - xs[0]) * 0.08 if len(xs) > 1 else 0.04)
    y_pad = max(0.02, (ys[-1] - ys[0]) * 0.08 if len(ys) > 1 else 0.04)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title(f"{cfg.task} eval grid ({cfg.grid_nx}x{cfg.grid_ny})")
        ax.set_xlabel("cube x (m)")
        ax.set_ylabel("cube y (m)")
        ax.set_xlim(xs[0] - x_pad, xs[-1] + x_pad)
        ax.set_ylim(ys[0] - y_pad, ys[-1] + y_pad)
        ax.set_xticks(xs, [f"{v:.2f}" for v in xs], rotation=90, fontsize=7)
        ax.set_yticks(ys, [f"{v:.2f}" for v in ys], fontsize=7)
        ax.grid(True, alpha=0.25)

        if cfg.task == "stack":
            anchors = {gp.green_anchor_index: gp.green_xy for gp in points if gp.green_anchor_index is not None}
            for ai, xy in sorted(anchors.items()):
                ax.scatter([xy[0]], [xy[1]], marker="s", s=80, c="#1a7f37", edgecolor="black", zorder=2)
                ax.text(xy[0], xy[1], f"G{ai}", color="white", ha="center", va="center", fontsize=7, zorder=3)

        for gp in points:
            x, y = gp.cube_xy
            ax.scatter([x], [y], c="#d62728", edgecolor="black", s=46, zorder=4)
            label = f"{gp.grid_idx}" if cfg.task != "stack" else f"{gp.grid_idx}/G{gp.green_anchor_index}"
            ax.text(x, y + y_pad * 0.16, label, ha="center", va="bottom", fontsize=7, zorder=5)

        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
    except ImportError:
        _grid_layout_cv2(path, cfg, points, xs, ys)


def _grid_layout_cv2(path: Path, cfg: Config, points: list[GridPoint], xs: list[float], ys: list[float]) -> None:
    """cv2 fallback for the labeled grid layout."""
    import cv2

    w = max(420, cfg.grid_nx * 64 + 120)
    h = max(360, cfg.grid_ny * 58 + 100)
    margin_l, margin_r, margin_t, margin_b = 64, 28, 46, 48
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    x0, x1 = xs[0], xs[-1]
    y0, y1 = ys[0], ys[-1]
    xr = max(x1 - x0, 1e-6)
    yr = max(y1 - y0, 1e-6)

    def px(x, y):
        u = margin_l + int(round((x - x0) / xr * (w - margin_l - margin_r)))
        v = h - margin_b - int(round((y - y0) / yr * (h - margin_t - margin_b)))
        return u, v

    cv2.putText(img, f"{cfg.task} eval grid ({cfg.grid_nx}x{cfg.grid_ny})", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
    for x in xs:
        u, _ = px(x, y0)
        cv2.line(img, (u, margin_t), (u, h - margin_b), (210, 210, 210), 1)
        cv2.putText(img, f"{x:.2f}", (u - 18, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
    for y in ys:
        _, v = px(x0, y)
        cv2.line(img, (margin_l, v), (w - margin_r, v), (210, 210, 210), 1)
        cv2.putText(img, f"{y:.2f}", (8, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)

    if cfg.task == "stack":
        anchors = {gp.green_anchor_index: gp.green_xy for gp in points if gp.green_anchor_index is not None}
        for ai, xy in sorted(anchors.items()):
            u, v = px(*xy)
            cv2.rectangle(img, (u - 12, v - 12), (u + 12, v + 12), (55, 127, 26), -1)
            cv2.putText(img, f"G{ai}", (u - 9, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    for gp in points:
        u, v = px(*gp.cube_xy)
        cv2.circle(img, (u, v), 9, (40, 40, 210), -1)
        cv2.circle(img, (u, v), 9, (0, 0, 0), 1)
        label = f"{gp.grid_idx}" if cfg.task != "stack" else f"{gp.grid_idx}/G{gp.green_anchor_index}"
        cv2.putText(img, label, (u - 15, v - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1)
    cv2.imwrite(str(path), img)


def write_heatmap(out_dir: Path, cfg: Config, points: list[GridPoint],
                  xs: list[float], ys: list[float], records: list[dict]) -> None:
    from collections import defaultdict

    per_pos: dict[int, list[float]] = defaultdict(list)
    for r in records:
        per_pos[r["grid_idx"]].append(1.0 if r["success"] else 0.0)

    grid = np.full((cfg.grid_ny, cfg.grid_nx), np.nan)
    for gp in points:
        if gp.grid_idx in per_pos:
            grid[gp.iy, gp.ix] = float(np.mean(per_pos[gp.grid_idx]))
    overall = float(np.nanmean(grid)) if np.any(~np.isnan(grid)) else 0.0
    title = f"{cfg.task} per-position success (overall {overall:.0%}, {cfg.policy})"
    x_min, x_max = xs[0], xs[-1]
    y_min, y_max = ys[0], ys[-1]
    path = out_dir / "heatmap.png"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0,
                       extent=[x_min, x_max, y_min, y_max])
        ax.set_xlabel("cube x (m)")
        ax.set_ylabel("cube y (m)")
        ax.set_xticks([round(v, 3) for v in xs], [f"{v:.2f}" for v in xs], rotation=90, fontsize=7)
        ax.set_yticks([round(v, 3) for v in ys], [f"{v:.2f}" for v in ys], fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="mean success")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except ImportError:
        _heatmap_cv2(path, grid, title)


def _heatmap_cv2(path: Path, grid: np.ndarray, title: str) -> None:
    """Fallback heatmap with cv2 (no matplotlib): colored cells + per-cell success text."""
    import cv2

    ny, nx = grid.shape
    cell = 48
    margin_top = 40
    img = np.full((margin_top + ny * cell, nx * cell, 3), 30, dtype=np.uint8)
    for iy in range(ny):
        for ix in range(nx):
            v = grid[iy, ix]
            # origin lower: iy=0 at the bottom row
            row = ny - 1 - iy
            y0 = margin_top + row * cell
            x0 = ix * cell
            if np.isnan(v):
                color = (60, 60, 60)
                text = "-"
            else:
                color = (int(40 + 40 * (1 - v)), int(60 + 160 * v), int(200 * (1 - v)))
                text = f"{v:.1f}"
            cv2.rectangle(img, (x0, y0), (x0 + cell - 1, y0 + cell - 1), color, -1)
            cv2.putText(img, text, (x0 + 6, y0 + cell // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------


def _build_policy(env: TaskEnv, cfg: Config):
    if cfg.policy == "remote":
        if cfg.action.close_setpoint is None:
            cfg.action.close_setpoint = cfg.close_setpoint
        return RemotePolicy(cfg.host, cfg.port, cfg.obs, cfg.action,
                            control_every=env.cfg.record_every)
    return ScriptedEvalPolicy(env, task=cfg.task, close_setpoint=cfg.close_setpoint)


def main(cfg: Config) -> None:
    cfg.env.task = cfg.task  # single --task flag drives both env and policy
    resolved_out = cfg.out if cfg.out is not None else PROJECT_ROOT / "outputs" / "eval" / cfg.task
    resolved_out.mkdir(parents=True, exist_ok=True)
    results_path = resolved_out / "results.jsonl"

    x_range, y_range = grid_ranges(cfg, cfg.env)
    points, xs, ys = build_grid(cfg.task, cfg.grid_nx, cfg.grid_ny, x_range, y_range)
    write_grid_layout(resolved_out, cfg, points, xs, ys)

    records = _load_records(results_path) if cfg.resume else []
    done_pairs = {(r["rep"], r["grid_idx"]) for r in records}
    if records:
        print(f"[resume] loaded {len(records)} completed trials from {results_path}")

    gs.init(backend=gs.gpu if cfg.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    env = TaskEnv(cfg.env)
    policy = _build_policy(env, cfg)

    n_success = 0
    for r in records:
        n_success += int(bool(r["success"]))

    with open(results_path, "a") as results_fh:
        for rep in range(cfg.reps):
            for gp in points:
                if (rep, gp.grid_idx) in done_pairs:
                    continue
                ordinal = rep * len(points) + gp.grid_idx
                video_path = (
                    resolved_out / "videos" / f"rep{rep}_g{gp.grid_idx:03d}.mp4"
                    if cfg.video_every > 0 and ordinal % cfg.video_every == 0
                    else None
                )
                rec = run_trial(env, cfg, policy, rep, gp, video_path=video_path)
                results_fh.write(json.dumps(rec) + "\n")
                results_fh.flush()
                records.append(rec)
                n_success += int(bool(rec["success"]))
                detail = (f"xy_err={rec['stack_xy_err']:.3f} stacked={rec['stacked']}"
                          if cfg.task == "stack"
                          else f"deliver={rec['deliver_dist']:.3f} delivered={rec['delivered']}")
                print(f"rep{rep} grid{gp.grid_idx:03d} xy=({gp.cube_xy[0]:.3f},{gp.cube_xy[1]:.3f}) "
                      f"rise={rec['max_rise']:.3f} {detail} stage={rec['failure_stage']} "
                      f"[{n_success}/{len(records)}]", flush=True)
            # rep boundary: cheap summary refresh
            write_summary(resolved_out, cfg, resolved_out, points, records)

    write_summary(resolved_out, cfg, resolved_out, points, records)
    write_heatmap(resolved_out, cfg, points, xs, ys, records)
    overall = n_success / len(records) if records else 0.0
    print(f"\n{cfg.task} eval done: {n_success}/{len(records)} success "
          f"({overall:.1%})  -> {resolved_out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
