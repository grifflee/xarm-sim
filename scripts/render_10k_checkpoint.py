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


def main(cfg: Config) -> None:
    if cfg.render_backend not in ("raster", "nyx", "batch"):
        raise ValueError(f"unknown render backend: {cfg.render_backend!r}")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    env_cfg = TaskEnvCfg(
        render_backend=cfg.render_backend,
        use_rasterizer=cfg.use_rasterizer,
        nyx_spp=cfg.nyx_spp,
    )
    env = TaskEnv(env_cfg)
    base = DatasetConfig(env=env_cfg, mode="video", video_fps=cfg.video_fps)
    metadata = []
    tiles = []

    for label, seed in CASES:
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
        if not cfg.reset_only:
            video_path = cfg.out_dir / f"{label}_seed{seed}_{cfg.render_backend}.mp4"
            run_cfg = replace(base, seed=seed, video_path=video_path)
            run_video(env, run_cfg)
        metadata.append(
            {
                "label": label,
                "seed": seed,
                "video": str(video_path) if video_path is not None else None,
                "reset_png": str(reset_path),
                "spawn": env.episode_spawn,
                "arm_start": env.episode_arm_start,
                "approach_distance": policy.approach_distance,
                "approach_scale": policy.approach_scale,
            }
        )

    rows = [np.concatenate(tiles[i : i + 2], axis=1) for i in range(0, len(tiles), 2)]
    _save_rgb_png(cfg.out_dir / "checkpoint_contact_sheet.png", np.concatenate(rows, axis=0))
    (cfg.out_dir / "checkpoint_cases.json").write_text(json.dumps(metadata, indent=2))
    kind = "reset cases" if cfg.reset_only else "videos"
    print(f"wrote {len(metadata)} {kind} + checkpoint_contact_sheet.png to {cfg.out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Config))
