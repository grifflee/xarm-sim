"""Render side-by-side MP4s showing the range of the new lighting/shadow variation.

Judging domain randomization from separate clips is unreliable -- you cannot hold the
previous one in your head. So every variant here plays SIMULTANEOUSLY in one tiled video,
driven by the SAME seed, so the robot motion and cube placement are identical and the only
thing differing between panels is the thing being judged.

Rendered at two resolutions on purpose:
  * `--res 640 480` so a human can actually see what the shadow is doing;
  * `--res 96 72` (nearest-upscaled for viewing) because that is what the network receives,
    and an effect that is obvious at 640x480 can be invisible by the time it is downsampled.
Approving the pretty one alone would approve something the model never sees.

    uv run python scripts/render_lighting_range.py --mode shadow_blur
    uv run python scripts/render_lighting_range.py --mode lighting --res 96 72
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys

import cv2
import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import genesis as gs  # noqa: E402

from xsim.task_env import TaskEnv, TaskEnvCfg  # noqa: E402
from generate_task_dataset import Config as DatasetConfig, _make_policy  # noqa: E402


@dataclass
class Cfg:
    mode: str = "lighting"
    """`lighting` = light direction/intensity + shadow-strength jitter across draws.
    `shadow_blur` = the resolution-scaling fix, old vs new, at fixed lighting."""
    seed: int = 100051
    res: tuple[int, int] = (640, 480)
    camera: str = "side"
    """The side camera sees the most tabletop, so shadows read best there."""
    variants: int = 6
    steps: int = 240
    fps: float = 30.0
    tile_w: int = 360
    out_dir: Path = Path("/nas/glee10/sim_mcaps/lighting_range")
    single: str = ""
    """Internal: render one variant to .npy. Genesis is a process singleton, so each
    variant needs its own process (same pattern as scripts/res_sweep.py)."""


def variant_cfgs(cfg: Cfg) -> list[tuple[str, dict]]:
    """(label, TaskEnvCfg overrides) per panel."""
    if cfg.mode == "shadow_blur":
        return [
            ("BEFORE blur=3.0px absolute", {"batch_shadow_blur_ref_w": 0.0}),  # 0 -> no scaling
            ("AFTER  blur scaled to width", {}),
            ("no shadow catcher", {"batch_shadow_catcher": False}),
        ]
    out: list[tuple[str, dict]] = [("nominal (no jitter)", {})]
    for i in range(cfg.variants - 1):
        out.append((f"jitter draw {i}", {
            "batch_light_dir_jitter_deg": 12.0,
            "batch_light_intensity_jitter": 0.25,
            "batch_shadow_strength_jitter": 0.15,
            "appearance_seed": 7000 + i,
        }))
    return out


def render_variant(cfg: Cfg, overrides: dict) -> np.ndarray:
    base = TaskEnvCfg(render_backend="batch", noslip_iterations=10, res=cfg.res)
    env_cfg = replace(base, **{k: v for k, v in overrides.items() if hasattr(base, k)})
    env = TaskEnv(env_cfg)
    env.reset(seed=cfg.seed)
    policy = _make_policy(env, DatasetConfig(env=env_cfg, mode="video"))
    policy.reset()
    frames = []
    import torch
    with torch.no_grad():
        for i in range(cfg.steps):
            cmd = policy.step()
            env.robot.go_to_goal(cmd.pose, open_gripper=cmd.open_gripper, ik_from_current=True)
            env.step()
            if i % env.cfg.record_every == 0:
                frames.append(np.asarray(env.render()[cfg.camera]).copy())
    return np.stack(frames)


def main(cfg: Cfg) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.single:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        idx = int(cfg.single)
        label, ov = variant_cfgs(cfg)[idx]
        np.save(cfg.out_dir / f"_v{idx}.npy", render_variant(cfg, ov))
        return

    import subprocess
    variants = variant_cfgs(cfg)
    stacks = []
    for i, (label, _) in enumerate(variants):
        path = cfg.out_dir / f"_v{i}.npy"
        if not path.exists():
            print(f"  rendering [{i}] {label}", flush=True)
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--mode", cfg.mode, "--seed", str(cfg.seed),
                 "--res", str(cfg.res[0]), str(cfg.res[1]),
                 "--camera", cfg.camera, "--variants", str(cfg.variants),
                 "--steps", str(cfg.steps), "--out-dir", str(cfg.out_dir),
                 "--single", str(i)], check=True, capture_output=True)
        stacks.append(np.load(path))

    n = min(len(s) for s in stacks)
    tw = cfg.tile_w
    th = int(round(tw * cfg.res[1] / cfg.res[0]))
    cols = min(3, len(stacks))
    rows = (len(stacks) + cols - 1) // cols
    tag = f"{cfg.mode}_{cfg.res[0]}x{cfg.res[1]}"
    out_path = cfg.out_dir / f"{tag}_seed{cfg.seed}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             cfg.fps, (cols * tw, rows * (th + 22)))
    for f in range(n):
        tiles = []
        for i, (label, _) in enumerate(variants):
            # NEAREST so 96x72 is judged as the coarse image it is, not smoothed by the
            # upscaler into looking better than the network's actual input
            img = cv2.resize(stacks[i][f], (tw, th), interpolation=cv2.INTER_NEAREST)
            pad = np.zeros((th + 22, tw, 3), np.uint8)
            pad[22:] = img
            cv2.putText(pad, label[:46], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(pad)
        while len(tiles) < rows * cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1)
                               for r in range(rows)], axis=0)
        writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    writer.release()
    for p in cfg.out_dir.glob("_v*.npy"):
        p.unlink()
    print(f"wrote {out_path}  ({n} frames, {len(variants)} panels)")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
