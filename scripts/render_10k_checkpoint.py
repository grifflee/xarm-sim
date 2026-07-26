"""Render the boundary-focused visual checkpoint for the 10k lift dataset.

The ten seeds were selected from 100000..100999 by measured reset state, not by
eyeballing frames. Together they cover the annulus boundaries, both y edges, a
beside-the-base spawn, all four arm-start buckets, and capped long approaches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import cv2
import genesis as gs
import numpy as np
import tyro

from generate_task_dataset import (
    Config as DatasetConfig,
    _make_policy,
    _save_rgb_png,
    contact_sheet,
    run_video,
)
from xsim.task_env import TaskEnv, TaskEnvCfg


CASES = (
    ("nearest_spawn", 100628),
    ("farthest_spawn", 100165),
    ("positive_y_edge", 100914),
    ("negative_y_edge", 100408),
    ("beside_robot_min_x", 100612),
    ("far_start_backward", 100251),
    ("post_drop_start", 100320),
    ("broad_long_approach", 100255),
    ("representative_broad", 100000),
    ("additional_distribution_draw", 100002),
)


@dataclass
class Config:
    out_dir: Path = Path("/data/store/griffen_sim_mcaps/lift_10k_checkpoint/visual_batch")
    render_backend: str = "batch"
    use_rasterizer: bool = False
    nyx_spp: int = 8
    video_fps: float = 30.0
    reset_only: bool = False

    res: tuple[int, int] = (640, 480)
    """Review videos render LARGER than the training data (TaskEnvCfg.res = 96x72), because
    96x72 is unwatchable for a human and what is being reviewed here is geometry -- spawn
    positions, arm starts, approach paths -- not pixel fidelity. Physics, distributions and
    seeds are identical at any resolution, so the episode you watch is the episode that
    gets generated."""

    seed_start: int | None = None
    """Base seed for the pool/sequential modes. None keeps the curated CASES."""

    select_pool: int = 0
    """If >0, probe this many resets from `seed_start` (cheap: reset + policy plan, no
    render) and SELECT `n_seeds` of them by measured state. 0 = plain sequential range."""

    n_seeds: int = 100
    n_featured: int = 12
    """Reviewers do a smell test, not an audit. The first `n_featured` renders are ordered
    to span the distribution deliberately -- neutral/typical draws first, then the edges --
    so watching only those is a fair sample. Filenames are rank-prefixed, so sorted order
    is review order."""

    label_prefix: str = "ep"
    sheet_rows: int = 10
    """Cases per contact sheet (2 tiles/row), so a 100-seed run emits readable sheets
    rather than one 12000px image."""


def _probe(env, base, cfg: Config) -> list[dict]:
    """Measured reset state per seed. No rendering, so this is cheap enough to scan a
    pool far larger than we intend to render."""
    rows = []
    for i in range(cfg.select_pool):
        seed = cfg.seed_start + i
        env.reset(seed=seed)
        policy = _make_policy(env, base)
        policy.reset()
        spawn = env.episode_spawn
        start = env.episode_arm_start
        tcp = np.asarray(start.get("achieved_tcp") or [0.0, 0.0, 0.0], dtype=float)
        cx, cy = float(spawn["red_xy"][0]), float(spawn["red_xy"][1])
        rows.append({
            "seed": seed,
            "radius": float(spawn["radius"]),
            "x": cx,
            "y": cy,
            "bucket": start.get("bucket", "unknown"),
            "approach": float(policy.approach_distance),
            # arm start position, and how it sits RELATIVE to the cube -- the edge cases
            # that matter are combinations (cube far / arm near, etc), not each alone
            "tcp_r": float(np.hypot(tcp[0], tcp[1])),
            "tcp_z": float(tcp[2]),
            "sep": float(np.hypot(tcp[0] - cx, tcp[1] - cy)),
            "lateral": float(abs(tcp[1] - cy)),
            "radial_gap": float(np.hypot(tcp[0], tcp[1]) - float(spawn["radius"])),
        })
    return rows


def _select(rows: list[dict], cfg: Config) -> list[tuple[str, int]]:
    """Featured smell-test set first, then a stratified fill.

    Featured deliberately leads with NEUTRAL draws: an approval reel made only of corner
    cases misrepresents what 10,000 episodes actually look like.
    """
    by = lambda key, rev=False: sorted(rows, key=lambda r: r[key], reverse=rev)
    med_r = sorted(r["radius"] for r in rows)[len(rows) // 2]

    def typical(bucket: str) -> dict | None:
        cand = [r for r in rows if r["bucket"] == bucket]
        if not cand:
            return None
        return min(cand, key=lambda r: (abs(r["radius"] - med_r), abs(r["y"])))

    far_rows = [r for r in rows if r["bucket"] == "far"]
    rmed = sorted(r["radius"] for r in rows)[len(rows) // 2]
    tmed = sorted(r["tcp_r"] for r in rows)[len(rows) // 2]

    def best(pred, key, rev=False):
        cand = [r for r in rows if pred(r)]
        return max(cand, key=key) if (cand and rev) else (min(cand, key=key) if cand else None)

    featured: list[tuple[str, dict | None]] = [
        # two neutral draws first -- a reel of only corner cases misrepresents the batch
        ("neutral_home", typical("home")),
        ("neutral_post_drop", typical("post_drop")),
        # --- the COMBINATIONS: what matters is arm and cube relative to each other ---
        # cube out at the annulus edge while the arm starts tucked in near the base
        ("cubeFAR_armNEAR", best(lambda r: r["radius"] > rmed, lambda r: r["tcp_r"])),
        # arm starts fully extended while the cube sits close in -- the reach-back case
        ("cubeNEAR_armFAR", best(lambda r: r["radius"] < rmed, lambda r: r["tcp_r"], rev=True)),
        # arm and cube at similar distance from the base but offset sideways: the arm has
        # to translate across rather than in/out
        ("sideBYside_lateral", best(lambda r: abs(r["radial_gap"]) < 0.06, lambda r: r["lateral"], rev=True)),
        # arm essentially on top of the cube already: shortest possible approach
        ("armONcube_shortest", by("sep")[0]),
        # the two furthest apart
        ("maxSeparation", by("sep", True)[0]),
        ("far_start_backward", max(far_rows, key=lambda r: r["approach"]) if far_rows else None),
        # --- absolute extremes of the spawn region ---
        ("farthest_spawn", by("radius", True)[0]),
        ("nearest_spawn", by("radius")[0]),
        ("positive_y_edge", by("y", True)[0]),
        ("negative_y_edge", by("y")[0]),
    ]

    picked: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, row in featured[: cfg.n_featured]:
        if row is not None and row["seed"] not in seen:
            picked.append((label, row["seed"]))
            seen.add(row["seed"])

    # Fill: stratify by arm-start bucket in its observed proportion, spreading each
    # bucket's picks evenly over its radius range so the fill is not all mid-table.
    remaining = [r for r in rows if r["seed"] not in seen]
    need = max(0, cfg.n_seeds - len(picked))
    buckets: dict[str, list[dict]] = {}
    for r in remaining:
        buckets.setdefault(r["bucket"], []).append(r)
    for name, group in buckets.items():
        share = round(need * len(group) / max(1, len(remaining)))
        group.sort(key=lambda r: r["radius"])
        if share <= 0:
            continue
        step = max(1, len(group) // share)
        for r in group[::step][:share]:
            picked.append((f"{name}_r{r['radius']:.3f}".replace(".", "p"), r["seed"]))

    return picked[: cfg.n_seeds]


def main(cfg: Config) -> None:
    if cfg.render_backend not in ("raster", "nyx", "batch"):
        raise ValueError(f"unknown render backend: {cfg.render_backend!r}")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    env_cfg = TaskEnvCfg(
        render_backend=cfg.render_backend,
        use_rasterizer=cfg.use_rasterizer,
        nyx_spp=cfg.nyx_spp,
        noslip_iterations=10,
        res=cfg.res,
    )
    env = TaskEnv(env_cfg)
    base = DatasetConfig(env=env_cfg, mode="video", video_fps=cfg.video_fps)
    metadata = []
    tiles = []

    if cfg.select_pool > 0:
        if cfg.seed_start is None:
            raise ValueError("--select-pool requires --seed-start")
        print(f"probing {cfg.select_pool} resets from {cfg.seed_start} ...", flush=True)
        cases = _select(_probe(env, base, cfg), cfg)
        print(f"selected {len(cases)}; first {cfg.n_featured} are the review set")
    elif cfg.seed_start is not None:
        cases = [(f"{cfg.label_prefix}{i:03d}", cfg.seed_start + i) for i in range(cfg.n_seeds)]
    else:
        cases = list(CASES)

    for rank, (label, seed) in enumerate(cases):
        # rank prefix so sorted order == review order; the featured set sorts first
        label = f"{rank:03d}_{label}"
        env.reset(seed=seed)
        policy = _make_policy(env, base)
        policy.reset()
        images = env.render()
        sheet = contact_sheet(images, f"{label} seed={seed}")
        cv2.putText(
            sheet,
            (
                f"spawn={env.episode_spawn['red_xy']} r={env.episode_spawn['radius']:.3f}m "
                f"start={env.episode_arm_start['bucket']} approach={policy.approach_distance:.3f}m"
            ),
            (16, sheet.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        reset_path = cfg.out_dir / f"{label}_seed{seed}_reset.png"
        _save_rgb_png(reset_path, sheet)
        tiles.append(cv2.resize(sheet, (960, 240), interpolation=cv2.INTER_AREA))

        video_path = None
        res = {}
        if not cfg.reset_only:
            video_path = cfg.out_dir / f"{label}_seed{seed}_{cfg.render_backend}.mp4"
            run_cfg = replace(base, seed=seed, video_path=video_path)
            res = run_video(env, run_cfg) or {}
        metadata.append(
            {
                "label": label,
                "seed": seed,
                "featured": rank < cfg.n_featured,
                "success": res.get("success"),
                "delivered": res.get("delivered"),
                "close_xy_err": res.get("close_xy_err"),
                "abort_reason": res.get("abort_reason"),
                "video": str(video_path) if video_path is not None else None,
                "reset_png": str(reset_path),
                "spawn": env.episode_spawn,
                "arm_start": env.episode_arm_start,
                "approach_distance": policy.approach_distance,
                "approach_scale": policy.approach_scale,
            }
        )

    tag = "" if cfg.seed_start is None else f"_{cases[0][1]}_{cases[-1][1]}"
    rows = [np.concatenate(tiles[i : i + 2], axis=1) for i in range(0, len(tiles), 2)]
    per_sheet = max(1, cfg.sheet_rows)
    sheets = []
    for n, i in enumerate(range(0, len(rows), per_sheet)):
        suffix = "" if len(rows) <= per_sheet else f"_{n:02d}"
        path = cfg.out_dir / f"checkpoint_contact_sheet{tag}{suffix}.png"
        _save_rgb_png(path, np.concatenate(rows[i : i + per_sheet], axis=0))
        sheets.append(path.name)
    (cfg.out_dir / f"checkpoint_cases{tag}.json").write_text(json.dumps(metadata, indent=2))
    kind = "reset cases" if cfg.reset_only else "videos"
    n_ok = sum(1 for m in metadata if m.get("success"))
    print(f"wrote {len(metadata)} {kind} ({n_ok} success) + {len(sheets)} contact "
          f"sheet(s) to {cfg.out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Config))
