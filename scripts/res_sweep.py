"""Find the smallest render resolution that survives the training downsample.

crossformer's loader resizes every frame with a plain ``cv2.resize(f, (64, 64))``
(``crossformer/data/grain/loader.py:imresize``) -- no crop -- so the network sees 64x64
whatever we render. Render resolution therefore buys exactly one thing: downsample
quality. Below some size the render is already so coarse that shrinking it to 64x64
loses detail a larger render would have preserved; above it, extra pixels are discarded.

So "minimum viable" is measurable: render one scene at several 4:3 sizes, resize each to
64x64 exactly as training does, and compare against a high-resolution reference put
through the same resize. The smallest size whose 64x64 output is indistinguishable from
the reference is the answer.

4:3 is preserved throughout because the real rig is 4:3 and the square resize squashes
sim and real identically only if both share an aspect ratio.

    uv run python scripts/res_sweep.py --seed 100051
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import genesis as gs  # noqa: E402

from xsim.task_env import TaskEnv, TaskEnvCfg  # noqa: E402

# 4:3 candidates, coarse -> fine. 88x66 is about the floor that avoids UPsampling either
# axis on the way to 64x64.
CANDIDATES = ((88, 66), (96, 72), (128, 96), (160, 120), (192, 144), (256, 192), (320, 240))
REFERENCE = (640, 480)
TRAIN_SIZE = (64, 64)


@dataclass
class Cfg:
    seed: int = 100051
    """Default is the checkpoint's farthest-spawn case: the cube is smallest on screen
    there, so it is the worst case for losing it to downsampling."""
    render_backend: str = "batch"
    out_dir: Path = Path("/nas/glee10/sim_mcaps/res_sweep")
    single: tuple[int, int] | None = None
    """Internal: render just this resolution and dump .npy. Genesis is a process-global
    singleton, so a second TaskEnv in one process fails -- each resolution needs its own
    process and the default mode drives them as subprocesses."""


def train_resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, TRAIN_SIZE)


def render_at(res: tuple[int, int], cfg: Cfg) -> dict[str, np.ndarray]:
    env = TaskEnv(TaskEnvCfg(render_backend=cfg.render_backend, noslip_iterations=10, res=res))
    env.reset(seed=cfg.seed)
    return {k: np.asarray(v) for k, v in env.render().items()}


def _npy(cfg: Cfg, res: tuple[int, int]) -> Path:
    return cfg.out_dir / f"raw_{res[0]}x{res[1]}.npz"


def _render_subprocess(cfg: Cfg, res: tuple[int, int]) -> dict[str, np.ndarray]:
    import subprocess
    path = _npy(cfg, res)
    if not path.exists():
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--seed", str(cfg.seed), "--out-dir", str(cfg.out_dir),
             "--render-backend", cfg.render_backend,
             "--single", str(res[0]), str(res[1])],
            check=True, capture_output=True,
        )
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main(cfg: Cfg) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.single is not None:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        imgs = render_at(tuple(cfg.single), cfg)
        np.savez_compressed(_npy(cfg, tuple(cfg.single)), **imgs)
        return

    print(f"reference {REFERENCE[0]}x{REFERENCE[1]} (seed {cfg.seed})", flush=True)
    ref = {k: train_resize(v) for k, v in _render_subprocess(cfg, REFERENCE).items()}

    rows = []
    strips: list[np.ndarray] = []
    for res in CANDIDATES:
        imgs = _render_subprocess(cfg, res)
        per_cam = {}
        for name, img in imgs.items():
            if name not in ref:
                continue
            small = train_resize(img)
            diff = small.astype(np.float64) - ref[name].astype(np.float64)
            mse = float((diff ** 2).mean())
            psnr = float("inf") if mse == 0 else 10 * np.log10((255.0 ** 2) / mse)
            per_cam[name] = psnr
        px = res[0] * res[1]
        rows.append((res, px / (REFERENCE[0] * REFERENCE[1]), per_cam))
        print(f"  {res[0]:>3}x{res[1]:<3} ({px/(REFERENCE[0]*REFERENCE[1]):5.1%} of ref px)  "
              + "  ".join(f"{n}={v:5.2f}dB" for n, v in per_cam.items()), flush=True)

        # visual strip: what the network actually sees, upscaled for human inspection
        cam = "low" if "low" in imgs else sorted(imgs)[0]
        vis = cv2.resize(train_resize(imgs[cam]), (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.putText(vis, f"{res[0]}x{res[1]}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        strips.append(vis)

    ref_vis = cv2.resize(ref["low"], (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.putText(ref_vis, "640x480 ref", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    sheet = np.concatenate(strips + [ref_vis], axis=1)
    cv2.imwrite(str(cfg.out_dir / "res_sweep_64px.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    print(f"\nAll rows are what the NETWORK sees (64x64). Higher dB = closer to the "
          f"640x480 render's 64x64.\nwrote {cfg.out_dir}/res_sweep_64px.png")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
