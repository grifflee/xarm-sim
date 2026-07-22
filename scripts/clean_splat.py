"""Bake the solved alignment into the lab splat and crop the table volume.

Transforms every gaussian by the committed splat->world solve (so the output PLY
is world-frame: load it with an identity splat pose), then cleans the table
region. Two modes:

Default: empties an axis-aligned box over the table (z from just below the
tabletop to 3 ft above, xy the table footprint plus margin), removing the
scanned (ghost) robot, cube and tabletop fuzz that fight the sim's own meshes.
Gaussians whose center is outside the box but whose volume pokes in are not
dropped or shrunk uniformly (that thins the table surface into holes) — each is
replaced by the largest spheroid contained in the original gaussian that stays
outside the box: squashed only along the offending box axis, with the center
pushed away from the face when that preserves more volume. Solved per gaussian
in the whitened frame, where the optimum depends only on the clearance/extent
ratio. Writes ``assets/lab_aligned.ply``.

``--keep-table``: keeps the scanned cart/table and under-table region, removing
only above-table clutter/baked robot points from the same footprint. Use that
with ``--env.table-transparent`` when the desired view is the real splat table
without the sim mesh slab. Writes ``assets/lab_aligned_w_table.ply``.

Both outputs are world-frame — load with splat_pos=(0,0,0),
splat_quat=(0,0,0,1), splat_scale=1.0 (the fork's nyx light-field loader still
applies its z-up->y-up conversion to that identity pose, matching the meshes).

    uv run python scripts/clean_splat.py
    uv run python scripts/clean_splat.py --keep-table
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALIGNED_SPLAT = PROJECT_ROOT / "assets" / "lab_aligned.ply"
DEFAULT_ALIGNED_W_TABLE_SPLAT = PROJECT_ROOT / "assets" / "lab_aligned_w_table.ply"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xsim.task_env import BaseDecorCfg, TableCfg  # noqa: E402  (task-generic module)

# splat->world solve for the raw lab scan: align_ransac.py seed refined by
# scripts/icp_splat.py. Copied literally from xsim.task_env DEFAULT_SPLAT_*
# (2026-07-22); UPDATE from icp_splat.py's printed "after" pose and re-bake.
# Kept local (not imported live) on purpose: once lab_aligned.ply is the loaded
# splat, task_env's DEFAULT_SPLAT_* becomes the identity pose, so reading them
# here would bake a no-op. Semantics: p_world = scale * R(quat_xyzw) * p + pos.
RAW_SCAN_SOLVE = dict(
    pos=(-0.2088, 0.7866, 0.1831),
    quat_xyzw=(-0.517954, 0.490916, -0.489502, 0.501112),
    scale=0.9966,
)

# standard 3DGS PLY layout: x y z nx ny nz f_dc(3) f_rest(45) opacity scale(3) rot(4)
_XYZ = slice(0, 3)
_F_DC = slice(6, 9)
_F_REST = slice(9, 54)
_OPACITY = 54
_SCALE = slice(55, 58)
_ROT = slice(58, 62)  # wxyz

# Synthetic tabletop plate (reverse-engineered from upstream's released assets-v1
# lab_aligned.ply, which is NOT reproducible from upstream's committed scripts alone):
# the scanned tabletop rasterizes as gray mush, so the surface band is deleted and
# replaced with a uniform grid of flat gaussians — the splat-space equivalent of a
# clean visual slab, renderer-agnostic because it rasterizes with everything else.
# All constants measured from the released asset: grid 228x152 centered on the table,
# pitch == xy sigma (tiles seamlessly), sunk 1 cm below the physical tabletop, 1 mm
# thin, identity rotation, opacity sigmoid(6)~0.998, uniform bluish gray.
PLATE_GRID = (228, 152)
PLATE_PITCH = float(np.exp(-5.563525))  # == xy log-scale below -> ~3.83 mm
PLATE_Z_BELOW_TOP = 0.01
PLATE_F_DC = (-0.62135345, -0.62102026, -0.48242748)  # rgb ~ (0.325, 0.325, 0.364)
PLATE_OPACITY = 6.0
PLATE_LOG_SCALE = (-5.563525, -5.563525, -6.9077554)
PLATE_JITTER = 1e-6  # break exact coplanarity (z-fighting in the rasterizer)


def make_table_plate(table, n_props: int) -> np.ndarray:
    """Rows (N, n_props) for the synthetic tabletop plate, in vertex-column layout."""
    nx, ny = PLATE_GRID
    cx, cy = table.center_xy
    xs = cx + (np.arange(nx) - (nx - 1) / 2.0) * PLATE_PITCH
    ys = cy + (np.arange(ny) - (ny - 1) / 2.0) * PLATE_PITCH
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    rows = np.zeros((nx * ny, n_props), dtype=np.float32)
    rows[:, _XYZ] = np.stack(
        [gx.ravel(), gy.ravel(), np.full(nx * ny, table.top_z - PLATE_Z_BELOW_TOP)], axis=-1
    )
    rows[:, _XYZ] += np.random.default_rng(0).uniform(0, PLATE_JITTER, (nx * ny, 3))
    rows[:, _F_DC] = PLATE_F_DC
    rows[:, _OPACITY] = PLATE_OPACITY
    rows[:, _SCALE] = PLATE_LOG_SCALE
    rows[:, _ROT] = (1.0, 0.0, 0.0, 0.0)
    return rows


def rot_from_quat_xyzw(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rots_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """(B, 4) wxyz -> (B, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product; q1 is (4,), q2 is (N, 4), both wxyz."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


@dataclass
class Cfg:
    src: Path = Path("/data/store/lab.ply")  # the full raw lab scan (scan frame)
    dst: Path = DEFAULT_ALIGNED_SPLAT
    keep_table: bool = False
    # default-mode crop box: tabletop to 3 ft above it, xy = table footprint + margin
    z_range: tuple[float, float] = (-0.01, 0.9144)
    xy_margin: float = 0.10  # fraction of the table half-extents
    sigma: float = 2.0  # support radius (in sigmas) used for the boundary reach test
    # keep-table mode: preserve the scanned cart/table and under-table region, but
    # remove baked robot/clamps/poles sitting above the tabletop inside the footprint.
    keep_table_remove_above_z: float = 0.04
    # also drop giant haze/floater gaussians: metre-scale semi-transparent smears
    max_radius: float = 0.10
    # DC-only colour: sim cameras sit far off the scan trajectory where higher-order
    # SH extrapolates into streak garbage (and baking the alignment rotation into
    # SH>0 isn't implemented)
    flatten_sh: bool = True
    # replace the scanned tabletop mush with the synthetic uniform plate (default
    # mode only; keep-table mode preserves the real scanned surface instead)
    table_plate: bool = True


def main(c: Cfg) -> None:
    raw = c.src.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode()
    n = int(next(ln for ln in header.splitlines() if ln.startswith("element vertex")).split()[-1])
    props = sum(1 for ln in header.splitlines() if ln.startswith("property "))
    data = np.frombuffer(raw[end:], dtype=np.float32).reshape(n, props).copy()

    # bake the solved splat->world alignment into positions and rotations
    pos = np.asarray(RAW_SCAN_SOLVE["pos"], dtype=np.float64)
    quat_xyzw = RAW_SCAN_SOLVE["quat_xyzw"]
    scale = float(RAW_SCAN_SOLVE["scale"])
    R = rot_from_quat_xyzw(quat_xyzw)
    data[:, _XYZ] = (scale * data[:, _XYZ].astype(np.float64) @ R.T + pos).astype(np.float32)
    qx, qy, qz, qw = quat_xyzw
    q = data[:, _ROT].astype(np.float64)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    data[:, _ROT] = quat_mul_wxyz(np.array([qw, qx, qy, qz]), q).astype(np.float32)
    data[:, _SCALE] += np.log(scale)

    # table footprint box (world frame), from the fork's TableCfg
    table = TableCfg()
    hx, hy = (1 + c.xy_margin) * np.asarray(table.size_xy) / 2
    lo = np.array([table.center_xy[0] - hx, table.center_xy[1] - hy, c.z_range[0]])
    hi = np.array([table.center_xy[0] + hx, table.center_xy[1] + hy, c.z_range[1]])
    pw = data[:, _XYZ].astype(np.float64)

    if c.keep_table:
        # keep cart/table + under-table; drop only above-table clutter in the footprint
        inside_xy = np.all((pw[:, :2] >= lo[:2]) & (pw[:, :2] <= hi[:2]), axis=1)
        inside = inside_xy & (pw[:, 2] > c.keep_table_remove_above_z) & (pw[:, 2] <= hi[2])
        inside_label = "above-table clutter"
        shrink = np.zeros(n, dtype=bool)  # no boundary squash: we keep the table surface
        kept_vol = 1.0
        dst = DEFAULT_ALIGNED_W_TABLE_SPLAT if c.dst == DEFAULT_ALIGNED_SPLAT else c.dst
    else:
        inside = np.all((pw >= lo) & (pw <= hi), axis=1)
        inside_label = "in the table box"
        dst = c.dst

        # boundary gaussians: center outside, but sigma-support reaches the box.
        # world half-extent along axis i is sigma * ||B[i, :]|| for B = R_g diag(s);
        # the gaussian is separated from the box iff extent < clearance on an axis
        # where the center is out. For the rest, work in the whitened frame (old
        # ellipsoid -> unit ball, box face -> plane at distance d = clear/extent):
        # replace the ball with the max-volume spheroid inside both the ball and the
        # half-space — semi-axis a along the whitened face normal w, r perp, center
        # shifted t away from the face. a = d + t makes the face constraint tight, and
        # containment in the unit ball gives r(t) in closed form; the best t is a 1-D
        # search shared across gaussians.
        clear = np.maximum(np.maximum(lo - pw, pw - hi), 0.0)
        B = rots_from_quat_wxyz(data[:, _ROT].astype(np.float64)) * np.exp(
            data[:, _SCALE].astype(np.float64)
        )[:, None, :]
        extent = c.sigma * np.linalg.norm(B, axis=-1)
        ratio = np.where(clear > 0, clear / extent, -np.inf).max(axis=1)
        shrink = ~inside & (ratio < 1.0)

        idx = np.flatnonzero(shrink)
        k = np.argmax(np.where(clear[idx] > 0, clear[idx] / extent[idx], -np.inf), axis=1)
        d = ratio[idx]
        sign = np.where(pw[idx, k] < lo[k], 1.0, -1.0)  # toward-box direction is sign * e_k
        Btn = B[idx, k, :] * sign[:, None]  # B^T (sign e_k): rows of B
        w = Btn / np.linalg.norm(Btn, axis=-1, keepdims=True)

        grid = np.linspace(0.0, 1.0, 65)
        t = grid[None, :] * ((1.0 - d) / 2.0)[:, None]  # a + t <= 1 keeps the ball bound
        a = d[:, None] + t
        c1 = 1.0 - t**2 - a**2
        beta = (c1 + np.sqrt(np.maximum(c1**2 - 4.0 * a**2 * t**2, 0.0))) / 2.0
        r2 = a**2 + beta
        best = np.argmax(a * r2, axis=1)
        ar = np.arange(len(idx))
        t, a, r = t[ar, best], a[ar, best], np.sqrt(r2[ar, best])

        Bw = np.einsum("mij,mj->mi", B[idx], w)
        data[idx, _XYZ] -= ((c.sigma * t)[:, None] * Bw).astype(np.float32)
        # B' = B (r I + (a - r) w w^T): squash along w, full extent elsewhere
        Bp = r[:, None, None] * B[idx] + (a - r)[:, None, None] * Bw[:, :, None] * w[:, None, :]
        evals, evecs = np.linalg.eigh(Bp @ Bp.transpose(0, 2, 1))
        data[idx, _SCALE] = np.log(np.sqrt(np.maximum(evals, 1e-18))).astype(np.float32)
        evecs[:, :, 0] *= np.sign(np.linalg.det(evecs))[:, None]  # rotations, not reflections
        from scipy.spatial.transform import Rotation

        data[idx, _ROT] = Rotation.from_matrix(evecs).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
        kept_vol = float(np.mean(a * r**2)) if len(idx) else 1.0

    giant = np.exp(data[:, _SCALE]).max(axis=1) > c.max_radius
    surface = np.zeros(n, dtype=bool)
    plate = None
    if c.table_plate and not c.keep_table:
        # clear the residual scanned fuzz between the plate plane and the crop plane
        # (the crop box starts AT the tabletop, so near-surface mush below it survives
        # and would float above the plate), then lay the synthetic plate.
        z_plate = table.top_z - PLATE_Z_BELOW_TOP
        half = np.array([PLATE_GRID[0], PLATE_GRID[1]]) * PLATE_PITCH / 2 + 0.01
        surface = (
            (np.abs(pw[:, 0] - table.center_xy[0]) < half[0])
            & (np.abs(pw[:, 1] - table.center_xy[1]) < half[1])
            & (pw[:, 2] > z_plate - 0.005)
            & (pw[:, 2] <= c.z_range[0])
        )
        plate = make_table_plate(table, data.shape[1])
    kept = data[~(inside | giant | surface)].copy()
    if c.flatten_sh:
        kept[:, _F_REST] = 0.0
    if plate is not None:
        kept = np.concatenate([kept, plate])
    print(
        f"{n} gaussians: {inside.sum()} {inside_label}, {shrink.sum()} squashed at the "
        f"boundary (mean {100 * kept_vol:.0f}% volume kept), {giant.sum()} giant, "
        f"{surface.sum()} surface fuzz cleared, "
        f"+{0 if plate is None else len(plate)} synthetic plate, keeping {len(kept)}"
    )

    new_header = header.replace(f"element vertex {n}", f"element vertex {len(kept)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(new_header.encode())
        kept.astype(np.float32).tofile(f)
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.0f} MB)")
    print("output is world-frame: load with splat_pos=(0,0,0), splat_quat=(0,0,0,1), splat_scale=1.0")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
