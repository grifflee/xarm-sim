"""Madrona batch-renderer configuration for TaskEnv.

Madrona does not consume Genesis scene lights, so the key/fill pair is an
explicit part of the observation domain rather than an optional decoration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


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


def light_elevation_deg(direction) -> float:
    """Degrees the light travels BELOW horizontal. 0 = horizontal, 90 = straight down.

    Genesis light directions point from the fixture toward the scene, so a downward
    light has negative z. The nominal rig measures 54.7 deg (key) and 45.8 deg (fill).
    """
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    return math.degrees(math.asin(float(np.clip(-d[2], -1.0, 1.0))))


def clamp_light_elevation(direction, min_elevation_deg: float) -> tuple[float, float, float]:
    """Lift a direction back to `min_elevation_deg` below horizontal, keeping azimuth.

    Re-projection rather than rejection-and-redraw on purpose: a redraw would consume a
    variable number of rng values and make the episode stream depend on how often the
    clamp fired, which is exactly the kind of silent seed instability the reset() draw
    order exists to prevent. Re-projecting is deterministic and costs no draws.
    """
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    if light_elevation_deg(d) >= min_elevation_deg:
        return tuple(float(v) for v in d)
    horiz = d[:2]
    norm = float(np.linalg.norm(horiz))
    if norm < 1e-9:
        return tuple(float(v) for v in d)  # already straight down; nothing shallower to fix
    phi = math.radians(min_elevation_deg)
    horiz = horiz / norm * math.cos(phi)
    return (float(horiz[0]), float(horiz[1]), float(-math.sin(phi)))


def jitter_lights(
    lights: tuple[BatchLight, ...],
    rng: np.random.Generator,
    dir_jitter_deg: float,
    intensity_jitter: float,
    min_elevation_deg: float,
) -> tuple[BatchLight, ...]:
    """Draw a per-episode variation of the key/fill rig.

    Each light is perturbed independently: the pair is a stand-in for the lab's
    real fixtures, which do not move together, and independent draws cover more
    of the (key azimuth, fill azimuth, contrast) space than a rigid rotation of
    the whole rig would.

    Direction is tilted about an axis sampled in the plane PERPENDICULAR to the
    light — rotating a direction vector about itself is a no-op, so a freely
    sampled 3D axis would silently realize less than the requested angle. With a
    perpendicular axis the drawn angle is exactly the angular change, which is
    what makes ``dir_jitter_deg`` mean what it says.

    Every result is then forced to at least `min_elevation_deg` below horizontal.
    This lab is lit from the CEILING; a light arriving horizontally or from below
    is not a lighting condition the real cell can produce, and training on it
    teaches a world that cannot occur. The clamp is unconditional so that the
    bound holds for ANY jitter magnitude, not just small default ones.
    """
    if dir_jitter_deg <= 0.0 and intensity_jitter <= 0.0:
        return tuple(lights)
    out = []
    for light in lights:
        d = np.asarray(light.direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        if dir_jitter_deg > 0.0:
            axis = rng.normal(size=3)
            axis -= float(np.dot(axis, d)) * d
            norm = float(np.linalg.norm(axis))
            if norm > 1e-9:
                axis = axis / norm
                theta = math.radians(float(rng.uniform(-dir_jitter_deg, dir_jitter_deg)))
                d = d * math.cos(theta) + np.cross(axis, d) * math.sin(theta)
        intensity = float(light.intensity)
        if intensity_jitter > 0.0:
            intensity *= 1.0 + float(rng.uniform(-intensity_jitter, intensity_jitter))
        out.append(replace(
            light,
            direction=clamp_light_elevation(d, min_elevation_deg),
            intensity=intensity,
        ))
    return tuple(out)
