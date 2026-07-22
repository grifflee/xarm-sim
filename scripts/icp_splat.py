"""Refine the fork's splat->world alignment with annealed trimmed ICP.

Adapted from upstream-main scripts/icp_splat.py. This fork has no ``xsim.suite``
package and no standalone gsplat rasterizer, so the two suite dependencies are
re-pointed to their flat ``xsim`` equivalents:

* splat centers come straight from the raw scan PLY (numpy) and are lifted to
  world by ``xsim.task_env.splat_world_transform`` — the fork's committed
  RANSAC/ICP solve (``DEFAULT_SPLAT_POS/QUAT/SCALE``) — instead of
  ``suite.renderers.splat_bg.SplatBackground``.
* the ICP target mesh is built from ``xsim.task_env``'s ``TableCfg`` /
  ``BaseDecorCfg`` fields, the fork equivalents of ``suite`` ``TableArena`` /
  ``PlateMount``.

Seeds from the committed solve, crops the world-frame splat centers to a band
around the expected tabletop, and runs point-to-mesh ICP against a trimesh of
the table slab + base plate + the arm's two lowest links. Prints the corrected
splat pos/quat (paste into ``scripts/clean_splat.py``'s ``RAW_SCAN_SOLVE`` and
re-bake, then eyeball through the nyx pipeline). No before/after render: the
fork renders splats only through nyx, which needs the full Genesis pipeline.

    uv run python scripts/icp_splat.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import trimesh
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xsim.task_env import (  # noqa: E402
    DEFAULT_SPLAT_POS,
    DEFAULT_SPLAT_QUAT,
    DEFAULT_SPLAT_SCALE,
    BaseDecorCfg,
    TableCfg,
    splat_world_transform,
)

# task_env builds the visual table slab at a fixed 0.72 m height (local `slab_h`
# in TaskEnv._build_scene); not a TableCfg field, so mirror it here.
SLAB_HEIGHT = 0.72
# link1's URDF joint-1 origin (z, metres); the default ready pose has joint1 = 0.
LINK1_ORIGIN_Z = 0.267


def rot_from_quat_xyzw(q) -> np.ndarray:
    """xyzw quaternion -> 3x3 rotation (matches suite.models.cameras)."""
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_ply_means(path: Path) -> np.ndarray:
    """(N, 3) splat centers from a binary-little-endian 3DGS PLY (scan frame)."""
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode()
    n = int(next(ln for ln in header.splitlines() if ln.startswith("element vertex")).split()[-1])
    props = sum(1 for ln in header.splitlines() if ln.startswith("property "))
    data = np.frombuffer(raw[end:], dtype=np.float32).reshape(n, props)
    return data[:, :3].astype(np.float64)


@dataclass
class Config:
    src: Path = Path("/data/store/lab.ply")  # raw lab scan (scan frame)
    # crop band around the expected table/plate surfaces, sim frame
    z_band: tuple[float, float] = (-0.07, 0.05)  # only the floor cutoff is used now
    xy_pad: float = 0.1  # metres beyond the slab footprint
    trim: float = 0.06  # round-1 outlier cutoff; decays each round
    trim_decay: float = 0.75  # trim multiplier per round: anneal toward the surface
    rounds: int = 10  # trim -> fit cycles; 0.06 * 0.75^9 ~ 4.5 mm final trim
    n_icp: int = 20000  # subsampled points fed to ICP
    max_iterations: int = 60
    seed: int = 0


def target_mesh() -> trimesh.Trimesh:
    """Table slab + base plate + the arm's two lowest links at their sim pose,
    from the same fields/files the fork's TaskEnv builds from."""
    tf = trimesh.transformations.translation_matrix
    table = TableCfg()
    decor = BaseDecorCfg()
    slab = trimesh.creation.box(
        extents=(table.size_xy[0], table.size_xy[1], SLAB_HEIGHT),
        transform=tf((table.center_xy[0], table.center_xy[1], table.top_z - SLAB_HEIGHT / 2)),
    )
    plate = trimesh.creation.box(
        extents=decor.plate_size,
        transform=tf((*decor.plate_center_xy, -decor.plate_size[2] / 2)),
    )
    base = trimesh.load(PROJECT_ROOT / "assets" / "link_base.stl")
    link1 = trimesh.load(PROJECT_ROOT / "assets" / "link1.stl")
    link1.apply_transform(tf((0.0, 0.0, LINK1_ORIGIN_Z)))
    return trimesh.util.concatenate([slab, plate, base, link1])


def main(cfg: Config) -> None:
    # seed: the fork's committed splat->world solve
    seed_pos = np.asarray(DEFAULT_SPLAT_POS, dtype=np.float64)
    seed_quat = DEFAULT_SPLAT_QUAT
    seed_scale = float(DEFAULT_SPLAT_SCALE)

    Ts = splat_world_transform(seed_pos, seed_quat, seed_scale)
    raw = read_ply_means(cfg.src)
    pts = raw @ Ts[:3, :3].T + Ts[:3, 3]  # world/sim-frame centers
    mesh = target_mesh()

    pad = np.array([cfg.xy_pad, cfg.xy_pad, cfg.xy_pad])
    bounds = mesh.bounds + np.stack([-pad, pad])
    bounds[0, 2] = cfg.z_band[0]  # keep the floor out
    sel = np.all((pts >= bounds[0]) & (pts <= bounds[1]), axis=1)
    print(f"{sel.sum():,} / {len(pts):,} splat centers in the target bbox")

    rng = np.random.default_rng(cfg.seed)
    sub = pts[sel][rng.permutation(sel.sum())[: 2 * cfg.n_icp]]
    T = np.eye(4)
    for r in range(cfg.rounds):
        trim = cfg.trim * cfg.trim_decay**r
        cur = trimesh.transform_points(sub, T)
        d = trimesh.proximity.closest_point(mesh, cur)[1]
        keep = cur[d <= trim][: cfg.n_icp]  # clutter/ghost-arm outliers out
        Tr, _, cost = trimesh.registration.icp(
            keep, mesh, max_iterations=cfg.max_iterations, reflection=False, scale=False
        )
        d1 = trimesh.proximity.closest_point(mesh, trimesh.transform_points(keep, Tr))[1]
        print(
            f"round {r + 1}: {len(keep):,} pts within {trim * 1e3:.1f} mm, "
            f"mean |dist|: {d[d <= trim].mean() * 1e3:.1f} -> {d1.mean() * 1e3:.1f} mm, "
            f"cost {cost:.6f}"
        )
        T = Tr @ T
    print("ICP correction (applied in sim frame):")
    print(f"  translation: {T[:3, 3].round(4)}")
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    print(f"  rotation: {ang:.3f} deg")

    # compose the correction onto the seed: p'' = T @ (s R p + t)
    #   -> R_new = T R, pos_new = T @ t + t_delta, scale unchanged
    R_new = T[:3, :3] @ rot_from_quat_xyzw(seed_quat)
    t_new = T[:3, :3] @ seed_pos + T[:3, 3]
    qw, qx, qy, qz = trimesh.transformations.quaternion_from_matrix(
        np.block([[R_new, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]])
    )
    print("\nsplat solve before -> after (paste into clean_splat.RAW_SCAN_SOLVE):")
    print(f"  pos:       {tuple(float(v) for v in seed_pos)}")
    print(f"          -> {tuple(float(v) for v in t_new.round(4))}")
    print(f"  quat_xyzw: {tuple(seed_quat)}")
    print(f"          -> ({qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f})")
    print(f"  scale:     {seed_scale}  (unchanged)")


if __name__ == "__main__":
    main(tyro.cli(Config))
