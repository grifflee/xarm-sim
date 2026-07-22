"""Gsplat-rendered backgrounds for compositing behind Genesis raster frames.

The aligned splat is rasterized directly from OpenCV camera poses.  Only SH
degree zero is used because the baked world-frame asset does not rotate the
higher-order coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SH_C0 = 0.28209479177387814

T_GL_TO_CV = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SplatAsset:
    """Gaussian splat plus its splat-to-world alignment."""

    uri: Path
    pos: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]
    scale: float = 1.0


_PLY_DTYPES = {
    b"float": "<f4",
    b"float32": "<f4",
    b"double": "<f8",
    b"float64": "<f8",
    b"int": "<i4",
    b"int32": "<i4",
    b"uint": "<u4",
    b"uint32": "<u4",
    b"short": "<i2",
    b"int16": "<i2",
    b"ushort": "<u2",
    b"uint16": "<u2",
    b"char": "i1",
    b"int8": "i1",
    b"uchar": "u1",
    b"uint8": "u1",
}


def read_ply_vertices(path: Path) -> np.ndarray:
    """Read a binary-little-endian PLY vertex element."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = None
        count = 0
        fields: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected EOF in PLY header")
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == b"end_header":
                break
            if tokens[0] == b"format":
                fmt = tokens[1]
            elif tokens[0] == b"element":
                in_vertex = tokens[1] == b"vertex"
                if in_vertex:
                    count = int(tokens[2])
            elif tokens[0] == b"property" and in_vertex:
                if tokens[1] == b"list":
                    raise ValueError("list properties on vertices are not supported")
                fields.append((tokens[-1].decode(), _PLY_DTYPES[tokens[1]]))
        if fmt != b"binary_little_endian":
            raise ValueError(f"only binary_little_endian PLY is supported, got {fmt!r}")
        return np.fromfile(f, dtype=np.dtype(fields), count=count)


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
    """Convert a batch of wxyz quaternions to rotation matrices."""
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
    """Hamilton product; ``q1`` is (4,), ``q2`` is (N, 4), both wxyz."""
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


def viewmats_cv(pos, lookat, up) -> np.ndarray:
    """Batched OpenCV world-to-camera transforms.

    Inputs may be (3,) or (B, 3); output is always (B, 4, 4).  The optical
    convention is x right, y down, +z forward.
    """
    pos, lookat, up = np.atleast_2d(pos, lookat, up)
    z = lookat - pos
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    x = np.cross(z, np.broadcast_to(up, z.shape))
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = np.cross(z, x)
    T = np.tile(np.eye(4), (len(z), 1, 1))
    T[:, :3, :3] = np.stack([x, y, z], axis=-2)
    T[:, :3, 3] = -(T[:, :3, :3] @ pos[..., None])[..., 0]
    return T


def invert_rigid(T: np.ndarray) -> np.ndarray:
    """Invert a batch of rigid 4x4 transforms."""
    R = T[:, :3, :3]
    out = np.tile(np.eye(4), (len(T), 1, 1))
    out[:, :3, :3] = R.transpose(0, 2, 1)
    out[:, :3, 3] = -(R.transpose(0, 2, 1) @ T[:, :3, 3, None])[..., 0]
    return out


def load_world_splat(
    asset: SplatAsset, ply: Path | None = None, device: str = "cuda"
) -> dict[str, torch.Tensor]:
    """Load PLY gaussians and apply the asset's splat-to-world alignment."""
    v = read_ply_vertices(ply or Path(asset.uri).expanduser())
    means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)
    Ra = rot_from_quat_xyzw(asset.quat_xyzw)
    means = asset.scale * means @ Ra.T + np.asarray(asset.pos)

    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float64)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    qx, qy, qz, qw = asset.quat_xyzw
    quats = quat_mul_wxyz(np.array([qw, qx, qy, qz]), quats)

    scales = asset.scale * np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    colors = np.clip(
        0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1),
        0.0,
        1.0,
    )
    to_device = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)
    return {
        "means": to_device(means),
        "quats": to_device(quats),
        "scales": to_device(scales),
        "opacities": to_device(opacities),
        "colors": to_device(colors),
    }


class SplatBackground:
    """Chunked gsplat rasterizer returning uint8 RGB backgrounds."""

    def __init__(
        self,
        asset: SplatAsset,
        ply: Path | None = None,
        device: str = "cuda",
        chunk: int = 256,
        prune_opacity: float = 0.0,
    ):
        self.device = device
        self.chunk = chunk
        self.splat = load_world_splat(asset, ply, device)
        if prune_opacity > 0:
            keep = self.splat["opacities"] >= prune_opacity
            self.splat = {key: value[keep] for key, value in self.splat.items()}

    def render(
        self,
        viewmats_cv: np.ndarray,
        Ks: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Rasterize ``(C,4,4)`` OpenCV view matrices to ``(C,H,W,3)``."""
        import gsplat

        vm = torch.from_numpy(np.asarray(viewmats_cv)).float().to(self.device)
        K_batch = torch.from_numpy(np.broadcast_to(Ks, (len(vm), 3, 3)).copy()).float().to(self.device)
        out = np.empty((len(vm), height, width, 3), dtype=np.uint8)
        for start in range(0, len(vm), self.chunk):
            end = start + self.chunk
            rgb, _, _ = gsplat.rasterization(
                self.splat["means"],
                self.splat["quats"],
                self.splat["scales"],
                self.splat["opacities"],
                self.splat["colors"],
                vm[start:end],
                K_batch[start:end],
                width,
                height,
            )
            out[start:end] = (rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out
