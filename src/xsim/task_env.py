"""TaskEnv: xArm7 red-block pickup env for synthetic MCAP data generation.

A purpose-built Genesis env (reuses ``Manipulator`` from ``grasp_env`` but does not touch
``GraspEnv``) with:

- a flat **collision-plane table** at z=0, aligned with the splat's real table top,
- a **0.03175 m red cube** (1.25 in) spawned in a configurable table rectangle,
- three cameras matching the real MCAP rig: ``low``/``side`` static and ``wrist``
  mounted on the EE; each defaults to 640x480.
- **physics dt vs record decimation** decoupling.

Frames/axes: Genesis cameras use an OpenGL convention internally.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch

import genesis as gs
import gs_nyx.nyx_py_renderer as npr
import gs_nyx.nyx_py_sdk as nps
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

from xsim.grasp_env import Manipulator, ROBOT_VISUAL_MATERIALS, _robot_material_name, _set_vgeom_surface
from xsim.splat_bg import SplatAsset, SplatBackground, T_GL_TO_CV, invert_rigid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_URDF_PATH = PROJECT_ROOT / "xarm7_standalone.urdf"
# World-frame splat baked by scripts/clean_splat.py. TaskEnv fails early if
# this prerequisite is absent instead of silently changing the visual domain.
DEFAULT_SPLAT_PATH = PROJECT_ROOT / "assets" / "lab_aligned.ply"

BLOCK_SIZE = 0.03175  # 1.25 inch cube edge (m)
BLOCK_COLOR = (0.48, 0.05, 0.04)  # saturated red; brighter albedos wash to salmon under the nyx light
DEFAULT_NYX_LIGHT_DIR = (-0.4, -0.4, -0.8)
DEFAULT_NYX_CEILING_LIGHT_X = (0.05, 0.75)
DEFAULT_NYX_CEILING_LIGHT_Y = (-0.30, 0.30)
DEFAULT_NYX_CEILING_LIGHT_Z = 1.85
DEFAULT_NYX_CEILING_TARGET_X = (0.28, 0.55)
DEFAULT_NYX_CEILING_TARGET_Y = (-0.12, 0.12)
ROBOT_BASE_ROUGHNESS = {"White": 0.28, "Black": 0.35, "Aluminum": 0.22}

def _rot_from_rpy_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    x, y, z, w = quat_xyzw_from_rpy_deg(roll, pitch, yaw)
    return _quat_wxyz_to_rot((w, x, y, z))


def _unit(v) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(arr)
    return arr if n == 0.0 else arr / n


def _jitter_direction(base, jitter_deg: float, rng: np.random.Generator | None) -> tuple[float, float, float]:
    if jitter_deg <= 0.0 or rng is None:
        return tuple(float(v) for v in base)
    base = _unit(base)
    axis = rng.normal(size=3)
    axis = axis - float(np.dot(axis, base)) * base
    axis = _unit(axis)
    if np.linalg.norm(axis) == 0.0:
        return tuple(float(v) for v in base)
    angle = math.radians(float(rng.uniform(-jitter_deg, jitter_deg)))
    c, s = math.cos(angle), math.sin(angle)
    rotated = base * c + np.cross(axis, base) * s + axis * float(np.dot(axis, base)) * (1.0 - c)
    return tuple(float(v) for v in _unit(rotated))


def _jitter_color_hsv(
    base: tuple[float, float, float],
    hue_jitter_deg: float,
    value_jitter: float,
    rng: np.random.Generator | None,
) -> tuple[float, float, float]:
    if rng is None or (hue_jitter_deg <= 0.0 and value_jitter <= 0.0):
        return tuple(float(v) for v in base)
    h, s, v = colorsys.rgb_to_hsv(*base)
    if hue_jitter_deg > 0.0:
        h = (h + float(rng.uniform(-hue_jitter_deg, hue_jitter_deg)) / 360.0) % 1.0
    if value_jitter > 0.0:
        v *= float(rng.uniform(max(0.0, 1.0 - value_jitter), 1.0 + value_jitter))
    return tuple(float(np.clip(c, 0.0, 1.0)) for c in colorsys.hsv_to_rgb(h, s, np.clip(v, 0.0, 1.0)))


def _c2w_gl_from_view(pos, lookat, up) -> np.ndarray:
    """OpenGL camera-to-world (x right, y up, −z forward) from a pos/lookat/up view."""
    pos = np.asarray(pos, dtype=np.float64)
    forward = np.asarray(lookat, dtype=np.float64) - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, np.cross(right, forward), -forward, pos
    return T


def quat_xyzw_from_rpy_deg(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    roll, pitch, yaw = math.radians(roll), math.radians(pitch), math.radians(yaw)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _make_light_field(
    uri: Path,
    position: tuple[float, float, float] | None,
    rotation_xyzw: tuple[float, float, float, float],
    scale: float | tuple[float, float, float] | None,
):
    light_field = nps.LightFieldAsset()
    light_field.type = nps.ELightFieldType.GaussianField
    light_field.uri = str(uri.expanduser())
    # The nyx scene exporter converts every mesh instance from Genesis z-up to Nyx y-up
    # (float3_z_up_to_y_up_a / quaternion_z_up_to_y_up_a, see nyx_scene_exporter.py) but
    # passes LightFieldAssets through raw — so we must apply the same world conversion
    # here or the splat lands in a different frame than the cameras and meshes.
    if position is not None:
        light_field.position = nps.float3_z_up_to_y_up_a(nps.float3(*position))
    light_field.rotation = nps.quaternion_z_up_to_y_up_a(nps.quaternion(*rotation_xyzw))
    if scale is not None:
        if isinstance(scale, (int, float)):
            scale = (float(scale), float(scale), float(scale))
        light_field.scale = nps.float3(*scale)
    return light_field


def _as_single_np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _quat_wxyz_to_rot(quat) -> np.ndarray:
    w, x, y, z = _as_single_np(quat)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_to_T(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_rot(quat_wxyz)
    T[:3, 3] = _as_single_np(pos)
    return T


XARM7_ROBOT_CFG: dict = {
    "robot_morph": "urdf",
    "robot_file": str(ROBOT_URDF_PATH),
    "robot_fixed": True,
    "merge_fixed_links": False,
    # Skip the Nyx material-URDF rewrite (writes to a shared /tmp path); colors are set by
    # apply_robot_visual_surfaces and Genesis resolves relative mesh paths itself.
    "rewrite_robot_visual_urdf": False,
    "ee_link_name": "link_tcp",
    "gripper_link_names": ["left_finger", "right_finger"],
    "arm_dof_dim": 7,
    "gripper_dof_dim": 6,
    # low ready pose matching the real demos' start (TCP ≈ (0.34, 0, 0.10), top-down) —
    # IK-solved; the old high home (TCP z=0.29) gave episodes a different opening style
    # than the real recordings (see compare_batches report)
    "default_arm_dof": [math.radians(v) for v in [0.0, -26.2, 0.0, 13.5, 0.0, 25.0, 90.0]],
    # xArm gripper joint convention (verified by finger separation): 0.0 = open (fingers
    # apart), 0.85 = hard fully closed. For a 31.75 mm cube, command a tighter
    # task grasp target instead of the hard stop; this holds the block without
    # driving as deeply through it as full closure.
    "default_gripper_dof": [0.0] * 6,
    "gripper_open_dof": 0.0,
    "gripper_close_dof": 0.85,
    # 0.58 -> recorded norm floor 0.32 (real demos read ~0.37, but rigid sim fingers need
    # the extra squeeze; 0.53 matches the real reading exactly and drops the cube)
    "gripper_grasp_dof": 0.58,
    "dofs_kp": [4500, 4500, 3500, 3500, 2000, 2000, 2000, 350, 350, 350, 350, 350, 350],
    "dofs_kv": [450, 450, 350, 350, 200, 200, 200, 35, 35, 35, 35, 35, 35],
    "dofs_force_lower": [-50] * 13,
    "dofs_force_upper": [50] * 13,
    "ik_method": "dls_ik",
    "ik_init_at_home": True,
    "ik_max_samples": 50,
    "ik_max_solver_iters": 40,
}


@dataclass
class CameraView:
    """Placement for one camera. Static cams use pos/lookat; the wrist cam attaches to a link."""

    name: str
    pos: tuple[float, float, float] | None = None
    lookat: tuple[float, float, float] | None = None
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_deg: float | None = None          # vertical FOV; falls back to cfg.fov_deg
    attach_link: str | None = None        # e.g. "link_tcp" for the wrist cam
    attach_offset: tuple = field(default=None)  # 4x4 offset_T from link frame to camera


def view_from_c2w_cv(name: str, c2w: np.ndarray | tuple, fov_deg: float | None = None) -> CameraView:
    """CameraView from a calibrated OpenCV camera-to-world (robot-base) pose."""
    T = np.asarray(c2w, dtype=np.float64)
    pos = T[:3, 3]
    return CameraView(
        name,
        pos=tuple(pos),
        lookat=tuple(pos + T[:3, 2]),  # CV optical +z = view direction
        up=tuple(-T[:3, 1]),           # CV optical +y points down
        fov_deg=fov_deg,
    )


def _look_offset_T(back=0.12, side=0.0, lift=0.0, pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0) -> np.ndarray:
    """Offset transform mounting the wrist camera on ``link_tcp``.

    The TCP approach axis is +z (points out of the gripper / downward at home). A Genesis
    camera looks along its own −z, so a 180°-about-x rotation aims the camera along +z_tcp
    (down the tool toward the grasp point). The camera is set ``back`` metres up the tool
    axis (−z_tcp) so the fingertips and workspace are in view. ``pitch_deg`` tilts the view
    about the camera x-axis; negative values push the gripper toward the bottom of the
    image like the real EE-mounted RealSense. ``yaw_deg`` tilts about the camera y-axis
    (aims a side-mounted camera back toward the tool axis). ``roll_deg`` spins the image
    about the optical axis.
    """
    R0 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # 180° about x
    th = math.radians(pitch_deg)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(th), -math.sin(th)], [0.0, math.sin(th), math.cos(th)]])
    ps = math.radians(yaw_deg)
    Ry = np.array([[math.cos(ps), 0.0, math.sin(ps)], [0.0, 1.0, 0.0], [-math.sin(ps), 0.0, math.cos(ps)]])
    ph = math.radians(roll_deg)
    Rz = np.array([[math.cos(ph), -math.sin(ph), 0.0], [math.sin(ph), math.cos(ph), 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R0 @ Rx @ Ry @ Rz
    T[:3, 3] = (side, lift, -back)  # lift: off the gripper body, along y_tcp
    return T


# Logitech extrinsics from /data/store/opencv_calibrated (dream_sam_roboreg, pose_c2w_cv,
# robot-base frame): low = 1e9c6aae (/dev/video8), side = ad3f052e (/dev/video10). Both
# were solved with approximate intrinsics fx=fy=515, cx=320, cy=240 → vFOV ≈ 49.98°, so
# the sim cameras must render with that same FOV for the extrinsics to be consistent.
LOGITECH_FOV_DEG = math.degrees(2.0 * math.atan(240.0 / 515.0))
# RealSense D435 colour at 640×480: fx ≈ 617 → vFOV ≈ 42.6° (no calibration data; guess).
REALSENSE_FOV_DEG = 42.5

LOW_C2W_CV = (
    (-0.6532074213027954, 0.09281682223081589, -0.7514687180519104, 1.0390468835830688),
    (0.7571737170219421, 0.07631510496139526, -0.6487403512001038, 0.48672395944595337),
    (-0.0028656369540840387, -0.9927542209625244, -0.12012804299592972, 0.23500925302505493),
    (0.0, 0.0, 0.0, 1.0),
)
SIDE_C2W_CV = (
    (-0.9901586174964905, 0.07804439961910248, -0.11616794764995575, 0.4850386679172516),
    (0.13690687716007233, 0.7123172879219055, -0.6883754134178162, 0.6458088159561157),
    (0.02902454137802124, -0.697504997253418, -0.7159919738769531, 0.9215802550315857),
    (0.0, 0.0, 0.0, 1.0),
)

DEFAULT_CAMERAS: tuple[CameraView, ...] = (
    view_from_c2w_cv("low", LOW_C2W_CV, fov_deg=LOGITECH_FOV_DEG),
    view_from_c2w_cv("side", SIDE_C2W_CV, fov_deg=LOGITECH_FOV_DEG),
    # Matched against the real wrist stream (no calibration data): the real RealSense is
    # side-mounted, so the finger axis runs near-horizontal, the fingers enter from the
    # frame bottom with the assembly parked on the right half, and the white housing peeks
    # in at the bottom. Candidate "P1" from the 2026-07-02 iterative mount sweep
    # (outputs/wrist_mount/wrist_mount_final_P.png), verified frame-by-frame by grifflee.
    CameraView(
        "wrist",
        fov_deg=REALSENSE_FOV_DEG,
        attach_link="link_tcp",
        attach_offset=_look_offset_T(back=0.14, side=0.085, lift=-0.03, pitch_deg=-5.0, yaw_deg=25.0, roll_deg=-90.0),
    ),
)


# Splat (lab.ply) → world alignment, solved 2026-07-01 by scripts/align_ransac.py:
# RANSAC geometry on the ZED fused point cloud (human-verified table/robot landmarks,
# checkpoint CP1), closed-form fused→robot solve (table rect center agrees with the
# calibrated-camera IPM measurement to 5 cm, CP2 human-verified), photometric refine,
# then scaled ICP splat→fused (1.1 cm RMS; scale 0.9966 — the splat is metric).
# Semantics: p_world = scale · R(quat) · p_splat + pos.
DEFAULT_SPLAT_POS = (-0.2237, 0.7717, 0.1711)
DEFAULT_SPLAT_QUAT = (-0.501119, 0.487918, -0.50087, 0.509849)  # xyzw
DEFAULT_SPLAT_SCALE = 0.9966


def splat_world_transform(pos=DEFAULT_SPLAT_POS, quat=DEFAULT_SPLAT_QUAT, scale=DEFAULT_SPLAT_SCALE):
    """(4x4 world-from-splat transform incl. scale) for cropping/analysis tooling."""
    x, y, z, w = quat
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = pos
    return T


def _apply_robot_shine(robot_entity, roughness_scale: float) -> None:
    """Rebind robot visual surfaces with scaled roughness; lower roughness = shinier."""
    roughness_scale = float(np.clip(roughness_scale, 0.2, 3.0))
    surfaces = {
        name: gs.surfaces.BSDF(
            color=ROBOT_VISUAL_MATERIALS[name][:3],
            metallic=1.0 if name == "Aluminum" else 0.0,
            roughness=float(np.clip(ROBOT_BASE_ROUGHNESS[name] * roughness_scale, 0.02, 0.95)),
        )
        for name in ROBOT_BASE_ROUGHNESS
    }
    for vgeom in robot_entity.vgeoms:
        material_name = _robot_material_name(vgeom.link.name)
        _set_vgeom_surface(vgeom, surfaces[material_name], ROBOT_VISUAL_MATERIALS[material_name])


@dataclass
class TableCfg:
    # the robot base sits on a 1 cm mounting plate, so the table top is 1 cm below the
    # robot-base origin (grifflee, 2026-07-02)
    top_z: float = -0.01
    # center measured by inverse-perspective-mapping the calibrated cap.npz photos onto
    # the table plane (robot base at one end of the table); size is the real cart's
    # 3 ft x 2 ft top (grifflee) — the IPM estimate read (0.93, 0.62)
    center_xy: tuple[float, float] = (0.375, 0.01)
    size_xy: tuple[float, float] = (0.9144, 0.6096)
    color: tuple[float, float, float] = (0.13, 0.14, 0.17)  # dark slate like the real cart


@dataclass
class BaseDecorCfg:
    """Visual-only stand-in for the robot's mounting plate.

    The real base sits on a flat light-metal rectangle (the plate is why the table top
    is 1 cm below the base origin). The scanned splat renders that area as translucent
    mush, so model it with simple geometry like the cart slab. Nothing here collides.
    NOTE: an earlier revision also modeled an "E-stop" next to the base — that red blob
    in the real frames was the cube itself sitting near the plate. Do not re-add it.
    """

    enabled: bool = True
    # flat metal rectangle under the base, top flush with the robot-base origin (z=0),
    # centered on the base (grifflee): along x (away from the table edge) it is only as
    # long as the base's outer ring (~13 cm); along y (toward the cameras/wall) it
    # sticks out ~1 inch past the ring on each side, just enough for the blue clamps.
    plate_size_xy: tuple[float, float] = (0.13, 0.18)
    plate_center_xy: tuple[float, float] = (0.0, 0.0)
    plate_color: tuple[float, float, float] = (0.55, 0.55, 0.58)


@dataclass
class StackCfg:
    """Second (green) cube + spawn geometry for the stack task (red stacks onto green).

    Protocol reference: /data/store/griffen/stack_right human demos — the green cube
    stays fixed, the red cube starts to its right (viewer = the main cameras) and is
    placed on top. Image-right of the calibrated low/side cameras is roughly world −x
    (toward the robot base), so the red offset is sampled with negative Δx.
    """

    green_x: tuple[float, float] = (0.47, 0.56)
    green_y: tuple[float, float] = (-0.08, 0.10)
    red_dx: tuple[float, float] = (-0.20, -0.12)   # red = green + Δ, Δx < 0 = image-right
    red_dy: tuple[float, float] = (-0.04, 0.04)
    # Opt-in broader stack placement: red and green are sampled independently, then
    # rejection-sampled so they are not overlapping and not so far apart that transport
    # becomes a different task. Defaults keep the verified real-demo-like layout above.
    free_placement: bool = False
    # y ceiling 0.14 (not 0.18): under the production 15deg/5cm camera jitter a cube
    # at y=+0.18 can leave the side camera's frame (occlusion audit, 3 jitter draws
    # per cell); the wedge behind the arm is rejected in _sample_free_stack_xy
    free_green_x: tuple[float, float] = (0.34, 0.62)
    free_green_y: tuple[float, float] = (-0.18, 0.14)
    free_red_x: tuple[float, float] = (0.26, 0.62)
    free_red_y: tuple[float, float] = (-0.18, 0.14)
    free_min_dist: float = 0.10
    free_max_dist: float = 0.34
    free_max_tries: int = 100
    green_color: tuple[float, float, float] = (0.05, 0.30, 0.06)  # darkened like BLOCK_COLOR
    # clearance between the red cube's bottom and the green cube's top at release
    place_clearance: float = 0.003


@dataclass
class TaskEnvCfg:
    task: Literal["lift", "stack"] = "lift"
    stack: StackCfg = field(default_factory=StackCfg)
    res: tuple[int, int] = (640, 480)
    fov_deg: float = 42.0                 # fallback vertical FOV → intrinsics
    physics_dt: float = 1.0 / 120.0       # stable sim step; ×record_every → 30 Hz like real
    record_every: int = 4                 # emit every k-th step → record_dt = physics_dt*k
    # noslip post-pass (MuJoCo-style): 0 = off, matching all approved training batches.
    # Without it the friction-cone regularization lets a held cube creep ~1 mm/s down the
    # fingers regardless of squeeze force; weld-free eval needs it on (measured in
    # scripts/test_friction_grasp.py)
    noslip_iterations: int = 0
    # None preserves Genesis' default robot collision import. Set to 0.0 to force
    # convex decomposition of robot meshes instead of coarse per-mesh hulls.
    robot_decompose_robot_error_threshold: float | None = None
    rectangle_x: tuple[float, float] = (0.35, 0.58)   # cube spawn range (m)
    rectangle_y: tuple[float, float] = (-0.15, 0.15)
    # drop target: "middle of the table" — x sampled per episode, y fixed on the centerline.
    # The release happens at the transport height (no lowering); the cube free-falls.
    drop_x_range: tuple[float, float] = (0.30, 0.40)
    drop_y: float = 0.0
    # per-episode start-pose jitter: each arm joint gets a uniform ±deg offset from the
    # fixed IK-solved home before the episode starts (the policy reads the actual TCP at
    # reset, so the trajectory adapts). 0 = every episode starts from the identical pose.
    arm_start_jitter_deg: float = 3.0
    table: TableCfg = field(default_factory=TableCfg)
    base_decor: BaseDecorCfg = field(default_factory=BaseDecorCfg)
    table_mode: Literal["slab", "plane"] = "slab"  # plane = visible infinite tabletop, no finite cart slab
    table_transparent: bool = True         # baked splat supplies table pixels; collision stays live
    show_viewer: bool = False
    render_backend: Literal["raster", "nyx"] = "raster"
    splat_bg: bool = True
    splat_uri: Path | None = DEFAULT_SPLAT_PATH
    splat_pos: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    splat_rot_rpy_deg: tuple[float, float, float] | None = None
    splat_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    splat_scale: float | None = 1.0
    splat_chunk: int = 256
    splat_prune_opacity: float = 0.15
    splat_resplat_every: int = 3
    nyx_spp: int = 8
    nyx_light_type: Literal["directional", "ceiling_panel"] = "directional"
    nyx_light_dir: tuple[float, float, float] = DEFAULT_NYX_LIGHT_DIR
    # Realistic randomized lighting: a broad overhead spot sampled from the ceiling
    # panel area and aimed at the work surface. The production default stays directional.
    nyx_ceiling_light_x: tuple[float, float] = DEFAULT_NYX_CEILING_LIGHT_X
    nyx_ceiling_light_y: tuple[float, float] = DEFAULT_NYX_CEILING_LIGHT_Y
    nyx_ceiling_light_z: float = DEFAULT_NYX_CEILING_LIGHT_Z
    nyx_ceiling_target_x: tuple[float, float] = DEFAULT_NYX_CEILING_TARGET_X
    nyx_ceiling_target_y: tuple[float, float] = DEFAULT_NYX_CEILING_TARGET_Y
    nyx_ceiling_inner_angle_deg: float = 55.0
    nyx_ceiling_outer_angle_deg: float = 85.0
    nyx_light_range: float = 5.0
    nyx_light_intensity: float = 2.0  # 5.0 washed out the mesh entities vs the dim splat
    # Per-episode appearance jitter. In Nyx these are baked into the exported scene, so
    # generate_task_dataset.py rebuilds the env per episode when any of these are nonzero.
    nyx_light_dir_jitter_deg: float = 0.0
    nyx_light_intensity_jitter: float = 0.0  # multiplicative +/- fraction around nyx_light_intensity
    # shadow dial: fraction of the light that casts shadows (the rest becomes a
    # coincident shadowless fill, so overall brightness is unchanged). 1.0 = the
    # approved fully-shadowed look, 0.0 = shadowless. The range, when set, samples
    # uniformly per episode (real lab shows little shadow -> bias low).
    nyx_shadow_strength: float = 1.0
    nyx_shadow_strength_range: tuple[float, float] | None = None
    robot_roughness_jitter: float = 0.0      # multiplicative +/- fraction; lower roughness = shinier
    cube_hue_jitter_deg: float = 0.0
    cube_value_jitter: float = 0.0           # multiplicative +/- fraction in HSV value
    appearance_seed: int | None = None       # set by the generator for reproducible appearance samples
    # per-episode camera jitter, applied in reset() around the calibrated nominal poses
    # (the nominals themselves never move); the actual sampled poses are exposed via
    # episode_extrinsics so batch manifests can record them. 0 = off.
    cam_jitter_deg: float = 0.0    # low/side: ± per-axis rpy, in the camera frame (deg)
    cam_jitter_cm: float = 0.0     # low/side: ± per-axis world xyz (cm)
    wrist_jitter_deg: float = 0.0  # wrist mount offset: ± per-axis rpy (deg)
    wrist_jitter_cm: float = 0.0   # wrist mount offset: ± per-axis xyz (cm)


class TaskEnv:
    def __init__(self, cfg: TaskEnvCfg | None = None, robot_cfg: dict | None = None, cameras=DEFAULT_CAMERAS):
        self.cfg = cfg or TaskEnvCfg()
        self.robot_cfg = dict(robot_cfg or XARM7_ROBOT_CFG)
        if self.cfg.robot_decompose_robot_error_threshold is not None:
            self.robot_cfg["decompose_robot_error_threshold"] = (
                self.cfg.robot_decompose_robot_error_threshold
            )
        self.camera_views = list(cameras)
        self.device = gs.device
        self.res = self.cfg.res
        self.record_dt = self.cfg.physics_dt * self.cfg.record_every
        self.episode_appearance = self._sample_appearance(self.cfg.appearance_seed)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.cfg.physics_dt, substeps=4),
            rigid_options=gs.options.RigidOptions(
                dt=self.cfg.physics_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                noslip_iterations=self.cfg.noslip_iterations,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=self.cfg.show_viewer,
        )

        # flat table plane: always provides collision at the aligned table-top height
        # (z=0). In plane mode it is also the visible infinite tabletop.
        t = self.cfg.table
        table_surface = gs.surfaces.Plastic(color=t.color, roughness=0.8)
        # The surface must be set even in slab mode: Nyx exports collision-only
        # primitives regardless of visualization=False, so without it the infinite
        # plane renders in Nyx's default light gray and shows up as a bright sheet
        # where the dark room floor belongs. With the dark slate surface it blends
        # into the room floor exactly as in all verified batches.
        self.table = self.scene.add_entity(
            gs.morphs.Plane(
                pos=(0.0, 0.0, t.top_z),
                visualization=self.cfg.table_mode == "plane" and not self.cfg.table_transparent,
                collision=True,
            ),
            surface=table_surface,
        )
        if self.cfg.table_mode == "slab":
            if not self.cfg.table_transparent:
                # visual-only cart body with the real cart's measured footprint: the real dark
                # metal cart scans as sparse see-through mush in the splat (scripts/clean_splat.py
                # crops that region out), so this box stands in for it — full depth down to the
                # floor so it occludes the under-table region from every camera angle, exactly
                # like the real cart does
                slab_h = 0.72
                self.scene.add_entity(
                    gs.morphs.Box(
                        size=(t.size_xy[0], t.size_xy[1], slab_h),
                        pos=(t.center_xy[0], t.center_xy[1], t.top_z - slab_h / 2.0),
                        fixed=True,
                        visualization=True,
                        collision=False,
                    ),
                    surface=table_surface,
                )
        elif self.cfg.table_mode != "plane":
            raise ValueError(f"unknown table_mode: {self.cfg.table_mode!r}")

        d = self.cfg.base_decor
        if d.enabled and not (self.cfg.splat_bg and self.cfg.render_backend == "raster"):
            plate_h = max(-t.top_z, 0.004)  # plate top flush with the robot-base origin
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(*d.plate_size_xy, plate_h),
                    pos=(*d.plate_center_xy, t.top_z + plate_h / 2.0),
                    fixed=True,
                    visualization=True,
                    collision=False,
                ),
                surface=gs.surfaces.Plastic(color=d.plate_color, roughness=0.35),
            )

        # robot (base at world origin, on the table top)
        self.robot = Manipulator(num_envs=1, scene=self.scene, args=self.robot_cfg, device=gs.device)
        _apply_robot_shine(self.robot._robot_entity, self.episode_appearance["robot_roughness_scale"])

        # red cube (high friction so the gripper can hold it)
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), fixed=False),
            material=gs.materials.Rigid(friction=2.0),
            surface=gs.surfaces.Plastic(color=self.episode_appearance["cube_color"], roughness=0.6),
        )

        # stack task: green target cube the red cube gets placed onto (same size;
        # same friction so the stacked pair doesn't slide apart during the settle)
        self.cube2 = None
        if self.cfg.task == "stack":
            self.cube2 = self.scene.add_entity(
                gs.morphs.Box(size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), fixed=False),
                material=gs.materials.Rigid(friction=2.0),
                surface=gs.surfaces.Plastic(color=self.episode_appearance["green_color"], roughness=0.6),
            )

        self.cams = {}
        self._manual_attached_cams = []
        self._rig_attached_camera_names = set()
        self._splat_renderer: SplatBackground | None = None
        self._splat_bg_frames: dict[str, np.ndarray] = {}
        self._splat_steps = 0
        self._add_cameras()

        self.scene.build(n_envs=1)
        self.robot.set_pd_gains()
        self._tcp_link = self.robot._robot_entity.get_link("link_tcp")
        self._grasp_welded = False
        self._cube_yaw = 0.0
        self.current_drop_xy = (float(np.mean(self.cfg.drop_x_range)), self.cfg.drop_y)

        # place static cams + attach wrist cam; keep the nominal poses that per-episode
        # jitter centers on, and the attach machinery so reset() can re-pose everything
        self._nominal_c2w_gl = {
            v.name: _c2w_gl_from_view(v.pos, v.lookat, v.up) for v in self.camera_views if v.attach_link is None
        }
        self._attach_links = {}
        self._attach_offsets = {}
        self.episode_extrinsics: dict[str, np.ndarray] = {}
        for view in self.camera_views:
            cam = self.cams[view.name]
            if view.attach_link is not None:
                link = self.robot._robot_entity.get_link(view.attach_link)
                self._attach_links[view.name] = link
                self._attach_offsets[view.name] = np.asarray(view.attach_offset, dtype=np.float64)
                if hasattr(cam, "attach"):
                    cam.attach(link, view.attach_offset)
                    self._rig_attached_camera_names.add(view.name)
                else:
                    self._manual_attached_cams.append((view.name, cam, link))
            elif hasattr(cam, "set_pose"):
                cam.set_pose(pos=view.pos, lookat=view.lookat, up=view.up)

        self._setup_splat_bg()
        self.reset()

    def _sample_appearance(self, seed: int | None) -> dict:
        rng = np.random.default_rng(seed) if seed is not None else None
        intensity = self.cfg.nyx_light_intensity
        if rng is not None and self.cfg.nyx_light_intensity_jitter > 0.0:
            lo = max(0.0, 1.0 - self.cfg.nyx_light_intensity_jitter)
            hi = 1.0 + self.cfg.nyx_light_intensity_jitter
            intensity *= float(rng.uniform(lo, hi))
        roughness_scale = 1.0
        if rng is not None and self.cfg.robot_roughness_jitter > 0.0:
            lo = max(0.05, 1.0 - self.cfg.robot_roughness_jitter)
            hi = 1.0 + self.cfg.robot_roughness_jitter
            roughness_scale = float(rng.uniform(lo, hi))

        light_dir = tuple(float(v) for v in self.cfg.nyx_light_dir)
        light_pos = None
        light_target = None
        if self.cfg.nyx_light_type == "ceiling_panel":
            if rng is None:
                lx = float(np.mean(self.cfg.nyx_ceiling_light_x))
                ly = float(np.mean(self.cfg.nyx_ceiling_light_y))
                tx = float(np.mean(self.cfg.nyx_ceiling_target_x))
                ty = float(np.mean(self.cfg.nyx_ceiling_target_y))
            else:
                lx = float(rng.uniform(*self.cfg.nyx_ceiling_light_x))
                ly = float(rng.uniform(*self.cfg.nyx_ceiling_light_y))
                tx = float(rng.uniform(*self.cfg.nyx_ceiling_target_x))
                ty = float(rng.uniform(*self.cfg.nyx_ceiling_target_y))
            light_pos = (lx, ly, float(self.cfg.nyx_ceiling_light_z))
            light_target = (tx, ty, float(self.cfg.table.top_z))
            light_dir = tuple(float(v) for v in _unit(np.asarray(light_target) - np.asarray(light_pos)))
        else:
            light_dir = _jitter_direction(self.cfg.nyx_light_dir, self.cfg.nyx_light_dir_jitter_deg, rng)

        app = {
            "seed": seed,
            "light_type": self.cfg.nyx_light_type,
            "light_dir": light_dir,
            "light_pos": light_pos,
            "light_target": light_target,
            "light_range": float(self.cfg.nyx_light_range),
            "ceiling_inner_angle_deg": float(self.cfg.nyx_ceiling_inner_angle_deg),
            "ceiling_outer_angle_deg": float(self.cfg.nyx_ceiling_outer_angle_deg),
            "light_intensity": float(intensity),
            "robot_roughness_scale": float(roughness_scale),
            "cube_color": _jitter_color_hsv(BLOCK_COLOR, self.cfg.cube_hue_jitter_deg, self.cfg.cube_value_jitter, rng),
            "green_color": _jitter_color_hsv(
                self.cfg.stack.green_color, self.cfg.cube_hue_jitter_deg, self.cfg.cube_value_jitter, rng
            ),
        }
        # drawn last so enabling the shadow dial doesn't shift the draws above
        shadow_strength = float(self.cfg.nyx_shadow_strength)
        if rng is not None and self.cfg.nyx_shadow_strength_range is not None:
            shadow_strength = float(rng.uniform(*self.cfg.nyx_shadow_strength_range))
        app["shadow_strength"] = float(np.clip(shadow_strength, 0.0, 1.0))
        return app

    def _splat_light_fields(self):
        if self.cfg.splat_uri is None:
            return ()
        splat_uri = Path(self.cfg.splat_uri).expanduser()
        if not splat_uri.exists():
            raise FileNotFoundError(f"splat file does not exist: {splat_uri}")
        rotation = self.cfg.splat_quat
        if self.cfg.splat_rot_rpy_deg is not None:
            rotation = quat_xyzw_from_rpy_deg(*self.cfg.splat_rot_rpy_deg)
        return (_make_light_field(splat_uri, self.cfg.splat_pos, rotation, self.cfg.splat_scale),)

    def _setup_splat_bg(self) -> None:
        if not self.cfg.splat_bg or self.cfg.render_backend == "nyx":
            return
        if self.cfg.splat_uri is None:
            raise ValueError("splat_bg=True requires splat_uri")
        splat_uri = Path(self.cfg.splat_uri).expanduser()
        if not splat_uri.exists():
            raise FileNotFoundError(f"splat file does not exist: {splat_uri}")
        if self.cfg.splat_pos is None or self.cfg.splat_scale is None:
            raise ValueError("composite splats require explicit pos and scale")
        rotation = self.cfg.splat_quat
        if self.cfg.splat_rot_rpy_deg is not None:
            rotation = quat_xyzw_from_rpy_deg(*self.cfg.splat_rot_rpy_deg)
        asset = SplatAsset(
            uri=splat_uri,
            pos=self.cfg.splat_pos,
            quat_xyzw=rotation,
            scale=self.cfg.splat_scale,
        )
        self._splat_renderer = SplatBackground(
            asset,
            device=str(self.device),
            chunk=self.cfg.splat_chunk,
            prune_opacity=self.cfg.splat_prune_opacity,
        )

    def _add_cameras(self) -> None:
        if self.cfg.render_backend == "nyx":
            # the exporter concatenates lights/light_fields across ALL sensors, so
            # historically the one light was baked 3x (once per camera). Lights sum
            # linearly (verified pixel-equivalent), hence the x3 below keeps every
            # approved look while the assets are now attached to the first sensor only.
            effective_intensity = self.episode_appearance["light_intensity"] * 3.0
            base = {"color": (1.0, 1.0, 1.0)}
            if self.episode_appearance["light_type"] == "ceiling_panel":
                base.update({
                    "type": "spot",
                    "pos": self.episode_appearance["light_pos"],
                    "dir": self.episode_appearance["light_dir"],
                    "range": self.episode_appearance["light_range"],
                    "inner_angle": self.episode_appearance["ceiling_inner_angle_deg"],
                    "outer_angle": self.episode_appearance["ceiling_outer_angle_deg"],
                })
            else:
                base.update({"type": "directional", "dir": self.episode_appearance["light_dir"]})
            # shadow dial: split the SAME light into a shadow-casting key and a
            # shadowless fill so shadow density scales with s while total
            # illumination stays constant. s=1 -> single shadowed light (the
            # approved look); s=0 -> fully shadowless.
            s = float(np.clip(self.episode_appearance.get("shadow_strength", 1.0), 0.0, 1.0))
            lights = []
            if s > 0.0:
                lights.append(dict(base, intensity=effective_intensity * s, shadow=True))
            if s < 1.0:
                lights.append(dict(base, intensity=effective_intensity * (1.0 - s), shadow=False))
            light_fields = self._splat_light_fields()
            for i, view in enumerate(self.camera_views):
                self.cams[view.name] = self.scene.add_sensor(
                    NyxCameraOptions(
                        res=self.res,
                        fov=view.fov_deg or self.cfg.fov_deg,
                        pos=view.pos or (1.0, 0.0, 0.5),
                        lookat=view.lookat or (0.0, 0.0, 0.0),
                        up=view.up,
                        near=0.02,
                        far=50.0,
                        spp=self.cfg.nyx_spp,
                        render_mode=npr.ERenderMode.FastPathTracer,
                        lights=lights if i == 0 else [],
                        light_fields=light_fields if i == 0 else [],
                    )
                )
            return

        for view in self.camera_views:
            self.cams[view.name] = self.scene.add_camera(
                res=self.res, fov=view.fov_deg or self.cfg.fov_deg, GUI=False,
                pos=view.pos or (1.0, 0.0, 0.5), lookat=view.lookat or (0.0, 0.0, 0.0),
                near=0.02, far=50.0,  # default near=0.1 clips the wrist cam's own gripper
            )

    def _place_cube(self, cube, x: float, y: float, yaw: float) -> None:
        z = self.cfg.table.top_z + BLOCK_SIZE / 2.0
        pos = torch.tensor([[x, y, z]], device=self.device, dtype=gs.tc_float)
        quat = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], device=self.device, dtype=gs.tc_float)
        cube.set_pos(pos, skip_forward=True)
        cube.set_quat(quat, skip_forward=True)

    def _sample_free_stack_xy(self, rng: np.random.Generator) -> tuple[float, float, float, float]:
        s = self.cfg.stack
        gx = gy = x = y = 0.0
        for _ in range(max(1, s.free_max_tries)):
            gx = float(rng.uniform(*s.free_green_x))
            gy = float(rng.uniform(*s.free_green_y))
            x = float(rng.uniform(*s.free_red_x))
            y = float(rng.uniform(*s.free_red_y))
            dist = math.hypot(x - gx, y - gy)
            if not (s.free_min_dist <= dist <= s.free_max_dist):
                continue
            # side-camera keep-out: cubes behind/under the arm's home pose (low x,
            # negative y) are occluded or in deep shadow from the side view, so the
            # "visible in both static cams at spawn" guarantee would break
            if (x < 0.38 and y < -0.10) or (gx < 0.38 and gy < -0.10):
                continue
            return gx, gy, x, y
        return gx, gy, x, y

    # -- lifecycle --
    def reset(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        if self.cfg.task == "stack":
            s = self.cfg.stack
            if s.free_placement:
                gx, gy, x, y = self._sample_free_stack_xy(rng)
            else:
                gx = float(rng.uniform(*s.green_x))
                gy = float(rng.uniform(*s.green_y))
                x = gx + float(rng.uniform(*s.red_dx))
                y = gy + float(rng.uniform(*s.red_dy))
            gyaw = float(rng.uniform(-math.pi / 4, math.pi / 4))
            yaw = float(rng.uniform(-math.pi / 4, math.pi / 4))
            self._place_cube(self.cube2, gx, gy, gyaw)
            self._green_yaw = gyaw
            self.episode_spawn = {
                "free_placement": bool(s.free_placement),
                "red_xy": [float(x), float(y)],
                "green_xy": [float(gx), float(gy)],
                "red_green_dist": float(math.hypot(x - gx, y - gy)),
            }
        else:
            x = float(rng.uniform(*self.cfg.rectangle_x))
            y = float(rng.uniform(*self.cfg.rectangle_y))
            yaw = float(rng.uniform(-math.pi / 4, math.pi / 4))
            self.episode_spawn = {"red_xy": [float(x), float(y)]}
        self._place_cube(self.cube, x, y, yaw)
        self._cube_yaw = yaw
        self.grasp_release()  # clear any weld left from a previous episode
        # draw order matters for seed reproducibility: cube first, then cameras, then
        # drop, then start joints (new draws go last so earlier streams stay stable)
        self._randomize_cameras(rng)
        if self.cfg.task == "stack":
            # redraw the camera jitter until both cubes project inside both static
            # frames: a bad +/-15 deg pitch draw can push the cube area out of the
            # low cam's view entirely. Stack only — lift keeps the single draw so
            # approved lift batches stay seed-stable. Redraws are deterministic per
            # seed (extra draws simply shift the later joint-jitter stream).
            for _ in range(50):
                if self._spawn_visible_in_static_cams():
                    break
                self._randomize_cameras(rng)
        if self.cfg.task == "stack":
            # the "drop" target is the green cube; no rng draw so camera/joint streams
            # stay aligned with the cube draws above
            self.current_drop_xy = (gx, gy)
        else:
            self.current_drop_xy = (float(rng.uniform(*self.cfg.drop_x_range)), self.cfg.drop_y)
        arm_offset = None
        if self.cfg.arm_start_jitter_deg > 0.0:
            arm_offset = rng.uniform(-1.0, 1.0, 7) * math.radians(self.cfg.arm_start_jitter_deg)
        # skip_forward=False: this is the episode's one FK pass, so the wrist-cam sync
        # below sees the jittered pose
        self.robot.reset(envs_idx=None, skip_forward=False, arm_qpos_offset=arm_offset)
        self._sync_attached_cams()
        self._splat_steps = 0
        self._render_splat_bg()

    # -- grasp weld: while gripped, the cube must not be able to slip (grifflee) --
    def grasp_lock(self) -> None:
        """Weld the cube to link_tcp at its current pose. Call once the close completes."""
        if self._grasp_welded:
            return
        solver = self.scene.rigid_solver
        solver.add_weld_constraint(self.cube.links[0].idx, self._tcp_link.idx)
        self._grasp_welded = True

    def grasp_release(self) -> None:
        """Delete the grasp weld (cube free-falls). Call at the open command."""
        if not getattr(self, "_grasp_welded", False):
            self._grasp_welded = False
            return
        solver = self.scene.rigid_solver
        solver.delete_weld_constraint(self.cube.links[0].idx, self._tcp_link.idx)
        self._grasp_welded = False

    def cube_yaw(self) -> float:
        """Cube yaw (rad) sampled at reset; used to align the grasp to the cube faces."""
        return self._cube_yaw

    def _randomize_cameras(self, rng: np.random.Generator) -> None:
        """Per-episode camera jitter around the nominal poses; records actual extrinsics.

        Static cams get an orientation delta (± cam_jitter_deg per rpy axis, applied in
        the camera frame) and a world-frame position delta (± cam_jitter_cm per axis).
        The wrist mount offset gets the same treatment in its own frame when wrist jitter
        is enabled. ``episode_extrinsics`` then holds camera(optical CV)→world for static
        cams — the same convention as LOW_C2W_CV/SIDE_C2W_CV — and link_tcp→camera
        (optical CV) under ``wrist_mount``.
        """
        self.episode_extrinsics = {}
        for view in self.camera_views:
            cam = self.cams[view.name]
            if view.attach_link is None:
                c2w = self._nominal_c2w_gl[view.name].copy()
                d_rpy = rng.uniform(-1.0, 1.0, 3) * self.cfg.cam_jitter_deg
                d_xyz = rng.uniform(-1.0, 1.0, 3) * (self.cfg.cam_jitter_cm / 100.0)
                c2w[:3, :3] = c2w[:3, :3] @ _rot_from_rpy_deg(*d_rpy)
                c2w[:3, 3] += d_xyz
                pos = tuple(c2w[:3, 3])
                lookat = tuple(c2w[:3, 3] - c2w[:3, 2])
                up = tuple(c2w[:3, 1])
                if hasattr(cam, "set_pose"):
                    cam.set_pose(pos=pos, lookat=lookat, up=up)
                else:
                    cam.update_camera_pose(pos=pos, lookat=lookat, up=up)
                self.episode_extrinsics[view.name] = c2w @ T_GL_TO_CV
            else:
                offset = np.asarray(view.attach_offset, dtype=np.float64).copy()
                if self.cfg.wrist_jitter_deg or self.cfg.wrist_jitter_cm:
                    delta = np.eye(4)
                    delta[:3, :3] = _rot_from_rpy_deg(*(rng.uniform(-1.0, 1.0, 3) * self.cfg.wrist_jitter_deg))
                    delta[:3, 3] = rng.uniform(-1.0, 1.0, 3) * (self.cfg.wrist_jitter_cm / 100.0)
                    offset = offset @ delta
                    if view.name in self._rig_attached_camera_names:
                        cam.attach(self._attach_links[view.name], offset)
                self._attach_offsets[view.name] = offset
                self.episode_extrinsics[f"{view.name}_mount"] = offset @ T_GL_TO_CV

    def step(self) -> None:
        self.scene.step()
        self._sync_attached_cams()
        if self._splat_renderer is not None and self.cfg.splat_resplat_every > 0:
            self._splat_steps += 1
            if self._splat_steps % self.cfg.splat_resplat_every == 0:
                moving = [v.name for v in self.camera_views if v.attach_link is not None]
                self._render_splat_bg(moving)

    def _sync_attached_cams(self) -> None:
        for view in self.camera_views:
            if view.name in self._rig_attached_camera_names:
                self.cams[view.name].move_to_attach()
        for name, cam, link in self._manual_attached_cams:
            link_T = _pose_to_T(link.get_pos(), link.get_quat())
            cam_T = link_T @ self._attach_offsets[name]
            pos = cam_T[:3, 3]
            lookat = pos - cam_T[:3, 2]
            up = cam_T[:3, 1]
            cam.update_camera_pose(pos=tuple(pos), lookat=tuple(lookat), up=tuple(up))

    def _camera_viewmat_cv(self, name: str) -> np.ndarray:
        view = next(v for v in self.camera_views if v.name == name)
        if view.attach_link is not None:
            link_T = _pose_to_T(self._attach_links[name].get_pos(), self._attach_links[name].get_quat())
            c2w_cv = link_T @ (self._attach_offsets[name] @ T_GL_TO_CV)
        else:
            c2w_cv = self.episode_extrinsics[name]
        return invert_rigid(np.asarray(c2w_cv, dtype=np.float64)[None])[0]

    def _render_splat_bg(self, names: list[str] | None = None) -> None:
        if self._splat_renderer is None:
            return
        names = list(self.cams) if names is None else names
        if not names:
            return
        viewmats = np.stack([self._camera_viewmat_cv(name) for name in names])
        Ks = np.stack([self.intrinsics(name) for name in names])
        width, height = self.res
        frames = self._splat_renderer.render(
            viewmats,
            Ks,
            width,
            height,
        )
        for name, frame in zip(names, frames, strict=True):
            self._splat_bg_frames[name] = frame


    # -- observations --
    def render(self) -> dict[str, np.ndarray]:
        out = {}
        for name, cam in self.cams.items():
            bg = self._splat_bg_frames.get(name)
            if hasattr(cam, "render"):
                if bg is not None:
                    rgb, _, seg, _ = cam.render(rgb=True, segmentation=True)
                else:
                    rgb = cam.render(rgb=True)[0]
            else:
                rgb = cam.read(envs_idx=0).rgb
                seg = None
            if hasattr(rgb, "detach"):
                rgb = rgb.detach().cpu().numpy()
            else:
                rgb = np.asarray(rgb)
            if rgb.ndim == 4:
                rgb = rgb[0]
            if bg is not None and seg is not None:
                if hasattr(seg, "detach"):
                    seg = seg.detach().cpu().numpy()
                else:
                    seg = np.asarray(seg)
                if seg.ndim == 3 and seg.shape[0] == 1:
                    seg = seg[0]
                rgb = np.where((seg == 0)[..., None], bg, rgb[..., :3])
            out[name] = np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)
        return out

    STATIC_CAM_MARGIN_PX = 30.0  # ~1.5 cube widths inside the frame edge

    def _spawn_visible_in_static_cams(self) -> bool:
        """Both cubes' centers project inside the low AND side frames with margin.

        Uses the episode's actual (jittered) extrinsics, validated against rendered
        frames (projection error < 10 px). Occlusion by the arm is handled separately
        by the spawn keep-out in _sample_free_stack_xy.
        """
        z = self.cfg.table.top_z + BLOCK_SIZE / 2.0
        points = [(*self.episode_spawn["red_xy"], z), (*self.episode_spawn["green_xy"], z)]
        w, h = self.res
        m = self.STATIC_CAM_MARGIN_PX
        for name in ("low", "side"):
            K = self.intrinsics(name)
            w2c = np.linalg.inv(np.asarray(self.episode_extrinsics[name]))
            for p in points:
                pc = w2c[:3, :3] @ np.asarray(p, dtype=np.float64) + w2c[:3, 3]
                if pc[2] <= 0.05:
                    return False
                uv = K @ (pc / pc[2])
                if not (m <= uv[0] <= w - m and m <= uv[1] <= h - m):
                    return False
        return True

    def intrinsics(self, name: str) -> np.ndarray:
        cam = self.cams[name]
        if hasattr(cam, "intrinsics"):
            return np.asarray(cam.intrinsics, dtype=np.float64)
        # nyx sensors don't expose K; derive it from the view's vertical FOV
        view = next(v for v in self.camera_views if v.name == name)
        w, h = self.res
        fy = (h / 2.0) / math.tan(math.radians(view.fov_deg or self.cfg.fov_deg) / 2.0)
        return np.array([[fy, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])

    def extrinsic_base_cam(self, name: str) -> np.ndarray:
        """4x4 camera(optical)-to-base transform for FrameTransform (base → cam)."""
        cam_to_world_gl = np.asarray(self.cams[name].transform, dtype=np.float64)
        if cam_to_world_gl.ndim == 3:
            cam_to_world_gl = cam_to_world_gl[0]
        return cam_to_world_gl @ T_GL_TO_CV

    def proprio(self):
        """Return (joint_pos, joint_vel, joint_eff) for the 7 arm joints and the EE pose."""
        ent = self.robot._robot_entity
        pos = np.asarray(ent.get_dofs_position().cpu()).reshape(-1)[:7]
        vel = np.asarray(ent.get_dofs_velocity().cpu()).reshape(-1)[:7]
        force = np.asarray(ent.get_dofs_force().cpu()).reshape(-1)[:7]
        ee = np.asarray(self.robot.ee_pose.cpu()).reshape(-1)  # [x,y,z, qw,qx,qy,qz]
        return pos, vel, force, ee

    def gripper_norm(self) -> float:
        """Normalized gripper opening in [0,1] (1=open, 0=closed), matching bela convention."""
        g = float(np.asarray(self.robot._robot_entity.get_dofs_position().cpu()).reshape(-1)[self.robot._arm_dof_dim])
        close = float(self.robot_cfg["gripper_close_dof"]) or 0.85
        return float(np.clip(1.0 - g / close, 0.0, 1.0))

    def cube_pos(self) -> np.ndarray:
        return np.asarray(self.cube.get_pos().cpu()).reshape(-1)

    def green_pos(self) -> np.ndarray:
        """Green target cube position (stack task only)."""
        if self.cube2 is None:
            raise RuntimeError("green cube only exists when cfg.task == 'stack'")
        return np.asarray(self.cube2.get_pos().cpu()).reshape(-1)

    def green_yaw(self) -> float:
        """Green cube yaw (rad) sampled at reset; used to align the placed cube's faces."""
        if self.cube2 is None:
            raise RuntimeError("green cube only exists when cfg.task == 'stack'")
        return self._green_yaw

    def camera_specs(self):
        """Return {name: (width, height, fx, fy, cx, cy)} for the MCAP CameraSpecs."""
        specs = {}
        for name in self.cams:
            K = self.intrinsics(name)
            specs[name] = (self.res[0], self.res[1], float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        return specs
