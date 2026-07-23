"""Madrona batch-renderer configuration for TaskEnv.

Madrona does not consume Genesis scene lights, so the key/fill pair is an
explicit part of the observation domain rather than an optional decoration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchLight:
    direction: tuple[float, float, float]
    intensity: float
    castshadow: bool = False
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    position: tuple[float, float, float] = (0.0, 0.0, 3.0)


@dataclass(frozen=True)
class BatchConfig:
    use_rasterizer: bool = False
    lights: tuple[BatchLight, ...] = (
        BatchLight(direction=(-0.4, -0.4, -0.8), intensity=1.7, castshadow=True),
        BatchLight(direction=(0.5, 0.3, -0.6), intensity=0.85),
    )
