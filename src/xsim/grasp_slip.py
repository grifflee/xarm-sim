"""Cube-vs-TCP slip measurement, shared by the dataset generator and the friction sweep.

A held cube should be motionless *in the TCP frame*. Tracking its relative pose turns
"the grasp is slipping" into a number, and separates a slipping grasp from a brisk
transport in which cube and hand move together.

Tolerances are the ones swept in ``scripts/test_friction_grasp.py`` against grifflee's
"visible slip = not a pass" standard; the generator reuses them so one definition governs
both the diagnostic sweep and production gating (same reasoning as ``xsim.success``).
"""

from __future__ import annotations

import math

import numpy as np

SLIP_SETTLE_STEPS = 12
"""Physics steps after the close completes before latching the reference pose (~0.1 s)."""

SLIP_MM_TOL = 3.0
SLIP_DEG_TOL = 5.0


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def cube_rel_tcp(env) -> tuple[np.ndarray, np.ndarray]:
    """Cube pose expressed in the TCP frame (pos in metres, quat wxyz)."""
    p_c = env.cube.get_pos().cpu().numpy().reshape(-1)[:3].astype(np.float64)
    q_c = env.cube.get_quat().cpu().numpy().reshape(-1)[:4].astype(np.float64)
    p_t = env._tcp_link.get_pos().cpu().numpy().reshape(-1)[:3].astype(np.float64)
    q_t = env._tcp_link.get_quat().cpu().numpy().reshape(-1)[:4].astype(np.float64)
    rel_pos = quat_wxyz_to_rot(q_t).T @ (p_c - p_t)
    rel_quat = quat_mul(quat_conj(q_t), q_c)
    return rel_pos, rel_quat / np.linalg.norm(rel_quat)


def slip_since(
    ref: tuple[np.ndarray, np.ndarray], cur: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float]:
    """(translation mm, rotation deg) of the cube in the TCP frame vs a reference pose."""
    d_mm = float(np.linalg.norm(cur[0] - ref[0])) * 1000.0
    dot = min(1.0, abs(float(np.dot(cur[1], ref[1]))))
    return d_mm, math.degrees(2.0 * math.acos(dot))
