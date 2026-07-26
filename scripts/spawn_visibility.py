"""Audit whether a lift spawn is VISIBLE to both static cameras.

The lift spawn region is bounded only by "the cube stays physically on the table"
(``rectangle_y = +-0.288`` is the table half-width minus the cube half-extent). Nothing
checks that the policy can SEE the cube. The stack task does enforce this -- see
``_sample_free_stack_xy``'s side-camera keep-out wedge, derived from a 7x7 render audit --
but lift has no equivalent, so it can spawn cubes that are out of frame and still score a
success, producing demonstrations whose input never contained the object.

Cameras jitter per episode (``camera_mode="jitter"``, 15 deg / 5 cm), so visibility is a
distribution, not a yes/no. This measures, per candidate spawn cell, the fraction of
camera draws in which the cube centre projects inside BOTH static frames with margin.

Projection is analytic (calibrated intrinsics + the episode's own jittered c2w), so a fine
grid over many draws costs seconds rather than a render per cell.

    uv run python scripts/spawn_visibility.py --draws 200
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import genesis as gs  # noqa: E402

from xsim.task_env import BLOCK_SIZE, TableCfg, TaskEnv, TaskEnvCfg  # noqa: E402

STATIC_CAMS = ("low", "side")


@dataclass
class Cfg:
    draws: int = 200
    """Camera-jitter draws. Visibility is reported as the fraction of draws in frame."""
    pitch: float = 0.02
    x_range: tuple[float, float] = (-0.06, 0.72)
    y_range: tuple[float, float] = (-0.31, 0.31)
    margin_px: float = 24.0
    """How far inside the frame border the cube centre must land. A cube clipped at the
    edge is not usefully visible, so centre-in-frame alone is too generous."""
    out_dir: Path = Path("/nas/glee10/sim_mcaps/spawn_visibility")
    backend: str = "gpu"


def project(pts: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points (N,3) through an OpenCV c2w. Returns (uv (N,2), z (N,))."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    cam = (pts - t) @ R  # == (R.T @ (p - t)).T
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * cam[:, 1] / safe + K[1, 2]
    return np.stack([u, v], axis=1), z


def main(cfg: Cfg) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu if cfg.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    env = TaskEnv(TaskEnvCfg(noslip_iterations=10))
    W, H = env.cfg.res

    xs = np.arange(cfg.x_range[0], cfg.x_range[1] + 1e-9, cfg.pitch)
    ys = np.arange(cfg.y_range[0], cfg.y_range[1] + 1e-9, cfg.pitch)
    gx, gy = np.meshgrid(xs, ys)
    cube_z = env.cfg.table.top_z + BLOCK_SIZE / 2.0
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, cube_z)], axis=1)

    Ks = {n: np.asarray(env.intrinsics(n), dtype=np.float64) for n in STATIC_CAMS}
    visible = np.zeros(gx.size, dtype=np.float64)

    visible_any = np.zeros(gx.size, dtype=np.float64)
    for i in range(cfg.draws):
        env.reset(seed=900_000 + i)  # a range used for nothing else; only cameras matter
        per_cam = []
        for name in STATIC_CAMS:
            c2w = np.asarray(env.episode_extrinsics[name], dtype=np.float64)
            uv, z = project(pts, Ks[name], c2w)
            per_cam.append(
                (z > 0)
                & (uv[:, 0] >= cfg.margin_px) & (uv[:, 0] < W - cfg.margin_px)
                & (uv[:, 1] >= cfg.margin_px) & (uv[:, 1] < H - cfg.margin_px)
            )
        # BOTH is the stack task's standard (redundancy); ANY is the weaker bar of "the
        # policy can see the cube somewhere", since it also receives the wrist view.
        visible += per_cam[0] & per_cam[1]
        visible_any += per_cam[0] | per_cam[1]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{cfg.draws} draws", flush=True)

    frac = (visible / cfg.draws).reshape(gx.shape)
    frac_any = (visible_any / cfg.draws).reshape(gx.shape)

    table = env.cfg.table
    radius = np.hypot(gx, gy)
    on_table = (
        (gx >= table.center_xy[0] - table.size_xy[0] / 2 + BLOCK_SIZE / 2)
        & (gx <= table.center_xy[0] + table.size_xy[0] / 2 - BLOCK_SIZE / 2)
        & (np.abs(gy) <= table.size_xy[1] / 2 - BLOCK_SIZE / 2)
    )

    def summarize(lo: float, hi: float, ycap: float = 1.0) -> dict:
        m = on_table & (radius >= lo) & (radius <= hi) & (gy <= ycap)
        if not m.any():
            return {}
        return {
            "annulus": [lo, hi],
            "y_cap": ycap,
            "cells": int(m.sum()),
            "both_mean": round(float(frac[m].mean()), 3),
            "both_min": round(float(frac[m].min()), 3),
            "both_always_frac": round(float((frac[m] >= 0.999).mean()), 3),
            "any_mean": round(float(frac_any[m].mean()), 3),
            "any_min": round(float(frac_any[m].min()), 3),
            "any_always_frac": round(float((frac_any[m] >= 0.999).mean()), 3),
        }

    print()
    rows = [
        ("current  r<=0.445, y<=0.288", summarize(0.25, 0.445)),
        ("wider    r<=0.550, y<=0.288", summarize(0.25, 0.55)),
        ("wider    r<=0.550, y<=0.200", summarize(0.25, 0.55, 0.20)),
        ("wider    r<=0.550, y<=0.150", summarize(0.25, 0.55, 0.15)),
        ("wider    r<=0.550, y<=0.100", summarize(0.25, 0.55, 0.10)),
    ]
    for lab, s in rows:
        if s:
            print(f"{lab}: BOTH mean={s['both_mean']:.3f} min={s['both_min']:.3f} "
                  f"always={s['both_always_frac']:.3f} | ANY mean={s['any_mean']:.3f} "
                  f"min={s['any_min']:.3f} always={s['any_always_frac']:.3f}  cells={s['cells']}")
    current, proposed = rows[0][1], rows[1][1]

    out = {
        "draws": cfg.draws,
        "pitch": cfg.pitch,
        "margin_px": cfg.margin_px,
        "res": [W, H],
        "cube_z": cube_z,
        "x": xs.tolist(),
        "y": ys.tolist(),
        "visibility": frac.tolist(),
        "summary": {"current": current, "proposed": proposed},
    }
    (cfg.out_dir / "visibility.json").write_text(json.dumps(out))
    _figure(cfg, xs, ys, frac, table)
    print(f"wrote {cfg.out_dir}/visibility.json and visibility.png")


def _figure(cfg: Cfg, xs, ys, frac, table: TableCfg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle, Rectangle

    # same single-hue sequential ramp as scripts/spawn_feasibility.py
    cmap = LinearSegmentedColormap.from_list(
        "vis", ["#f1f6f2", "#cbe3d4", "#8bc7a3", "#3d9866", "#11603c"])
    p = cfg.pitch
    x_edges = np.array([x - p / 2 for x in xs] + [xs[-1] + p / 2])
    y_edges = np.array([y - p / 2 for y in ys] + [ys[-1] + p / 2])

    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    mesh = ax.pcolormesh(x_edges, y_edges, frac, cmap=cmap, vmin=0.0, vmax=1.0,
                         edgecolors="#ffffff", linewidth=0.3)
    tx0 = table.center_xy[0] - table.size_xy[0] / 2
    ty0 = table.center_xy[1] - table.size_xy[1] / 2
    ax.add_patch(Rectangle((tx0, ty0), table.size_xy[0], table.size_xy[1],
                           fill=False, ec="#0b0b0b", lw=2.0, label="table top"))
    for r, style, lab in ((0.25, "-", "r=0.250 (inner)"),
                          (0.445, "--", "r=0.445 (current outer)"),
                          (0.55, "-.", "r=0.550 (proposed outer)")):
        ax.add_patch(Circle((0, 0), r, fill=False, ec="#c2410c", lw=1.8, ls=style, label=lab))
    ax.plot([0], [0], marker="o", ms=9, color="#0b0b0b")
    ax.annotate("robot base", (0, 0), textcoords="offset points", xytext=(8, 10))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, +x away from robot)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Lift spawn visibility in BOTH static cameras "
                 f"({cfg.draws} camera-jitter draws, {cfg.margin_px:.0f}px margin)")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.colorbar(mesh, ax=ax, fraction=0.030, pad=0.02, label="fraction of draws visible")
    fig.tight_layout()
    fig.savefig(cfg.out_dir / "visibility.png", dpi=140)


if __name__ == "__main__":
    main(tyro.cli(Cfg))
