"""Generate synthetic block-lift episodes as Foxglove MCAP.

Drives ``TaskEnv`` with ``ScriptedLiftPolicy``. In generate mode it records at a
fixed rate and writes one ``<episode>.mcap`` per rollout via ``EpisodeMcapWriter``,
matching the real lift MCAP topic/schema layout under ``/data/store/mcaps/single/lift``.
Preview/video modes render inspection artifacts without writing MCAP. Grasp success is
computed per episode; by default only successful episodes are kept.

    uv run python scripts/generate_task_dataset.py --n-episodes 3 --backend gpu
    uv run python scripts/generate_task_dataset.py --mode preview --backend cpu
    uv run python scripts/generate_task_dataset.py --mode video --backend gpu --env.render-backend nyx
    uv run python scripts/generate_task_dataset.py --mode video --backend gpu --env.render-backend nyx --env.table-transparent
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import os
from pathlib import Path
import sys
import time
from typing import Literal

# phase timing for generation-architecture profiling: XSIM_TIMING=1 prints a
# per-phase wall-clock breakdown at exit (imports happen before main, so the
# marks must start here)
_TIMING = bool(os.environ.get("XSIM_TIMING"))
_MARKS: list[tuple[str, float]] = [("stdlib_imports", time.monotonic())]


def _mark(name: str) -> None:
    if _TIMING:
        _MARKS.append((name, time.monotonic()))


def _print_timing() -> None:
    prev = _MARKS[0][1]
    print("== XSIM_TIMING phase breakdown ==")
    for name, t in _MARKS[1:]:
        print(f"  {name:24s} {t - prev:7.2f} s")
        prev = t
    print(f"  {'TOTAL':24s} {prev - _MARKS[0][1]:7.2f} s")


import cv2
import numpy as np
import torch
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM_MCAP_ROOT = Path("/data/store/griffen_sim_mcaps")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_mark("thirdparty_imports")

import genesis as gs  # noqa: E402

_mark("genesis_import")

from xsim.task_env import BaseDecorCfg, TaskEnv, TaskEnvCfg, StackCfg, TableCfg  # noqa: E402
from xsim.mcap_writer import CameraSpec, EpisodeMcapWriter  # noqa: E402
from xsim.scripted_lift_policy import (  # noqa: E402
    GRASP_R,
    GRASP_TOL_XY,
    GRASP_TOL_Z,
    ScriptedLiftPolicy,
)
from xsim.grasp_slip import (  # noqa: E402
    SLIP_DEG_TOL,
    SLIP_MM_TOL,
    SLIP_SETTLE_STEPS,
    cube_rel_tcp,
    slip_since,
)
from xsim.scripted_stack_policy import ScriptedStackPolicy  # noqa: E402
# success scoring lives in xsim.success so the eval harness shares one definition;
# alias keeps the existing _episode_result call sites (and scripts importing it) working
from xsim.success import episode_result as _episode_result  # noqa: E402


@dataclass
class Config:
    task: Literal["lift", "stack"] = "lift"
    mode: Literal["generate", "preview", "video"] = "generate"
    out_dir: Path = SIM_MCAP_ROOT / "lift"
    preview_dir: Path = PROJECT_ROOT / "outputs" / "sim_preview" / "task_env"
    video_path: Path = PROJECT_ROOT / "outputs" / "sim_preview" / "task_current.mp4"
    video_fps: float = 30.0
    n_episodes: int = 1
    episode_offset: int = 0  # output episode numbering offset for subprocess/sharded generation
    appearance_child: bool = False  # internal: one-episode subprocess for Nyx appearance randomization
    # concurrent appearance children: the Nyx render only holds the GPU ~36% of an
    # episode's wall time (the rest is CPU scene setup), so 2-3 children overlap well
    pool_workers: int = 1
    manifest_name: str = "manifest.json"  # internal: children write private manifests
    # Use completed child manifests to skip episodes after an interrupted pooled run.
    # Child manifests are removed only after the parent writes the final manifest.
    resume: bool = False
    backend: Literal["gpu", "cpu"] = "gpu"
    seed: int = 0
    steps_per_segment: int = 108    # 5 weighted segments at 120 Hz → ~6 s episodes
    hold_steps: int = 48            # unrecorded settle steps after release (success eval only)
    release_tail_s: float = 0.3     # recorded tail after the open command (fingers opening)
    lift_threshold: float = 0.05    # min cube rise (m) for a successful grasp
    deliver_radius: float = 0.12    # max xy dist (m) from the drop target after settling
    grasp_tcp_offset: float = 0.018 # TCP target height above table while closing (m)
    save_failures: bool = False

    # Slip telemetry: cube motion in the TCP frame between close-complete and the open
    # command. RECORD-ONLY by default (inf thresholds) -- slip_mm/slip_deg always land in
    # the manifest, but never abort an episode.
    #
    # Acquisition (GRASP_TOL_XY etc.) is what actually prevents carrying a bad grasp, and
    # it is validated: it rejects the 18.4 mm edge grasp that dropped a cube while keeping
    # all 79 good episodes. A slip abort would guard against a good grasp degrading
    # mid-transport, which has not been observed in any episode -- so it stays off rather
    # than risk destroying real episodes on a false positive. Set a finite value to opt in
    # once there is data justifying a threshold.
    slip_abort_mm: float = math.inf
    slip_abort_deg: float = math.inf
    stack_xy_tol: float = 0.02      # max xy offset (m) red-vs-green center for a stack
    stack_z_tol: float = 0.008      # max |z error| (m) from the ideal stacked height
    # Lift generation follows current upstream: physical contact only, with the
    # no-slip solver stabilizing the carry. main() enforces the value for lift.
    env: TaskEnvCfg = field(default_factory=lambda: TaskEnvCfg(noslip_iterations=10))


def _make_policy(env: TaskEnv, cfg: Config, steps_per_segment: int | None = None):
    cls = ScriptedStackPolicy if cfg.task == "stack" else ScriptedLiftPolicy
    return cls(env, steps_per_segment=steps_per_segment or cfg.steps_per_segment,
               grasp_tcp_offset=cfg.grasp_tcp_offset)


class GraspIntegrity:
    """Permanent physical-acquisition diagnostics (stack alone retains its weld)."""

    def __init__(self, env: TaskEnv, cfg: Config, policy) -> None:
        self.env = env
        self.cfg = cfg
        self.policy = policy
        self.min_ee_cube = float("inf")
        self.close_xy_err: float | None = None
        self.close_z_err: float | None = None
        self.weld_fired = False
        self.weld_step: int | None = None
        self.weld_tcp_cube_dist: float | None = None
        self.weld_gripper_norm: float | None = None
        self.close_ticks = 0
        self.physical_grasp_detected = False
        self.physical_grasp_step: int | None = None
        self.physical_grasp_tcp_cube_dist: float | None = None
        self.physical_grasp_gripper_norm: float | None = None
        # Acquisition/slip gating: the cube must never be carried on a grasp we have
        # already measured as bad. abort_reason ends the episode early; run_episode then
        # reports success=False and the existing keep-gate deletes the partial MCAP.
        self.abort_reason: str | None = None
        self.abort_step: int | None = None
        self._slip_ref: tuple[np.ndarray, np.ndarray] | None = None
        self._slip_ref_step: int | None = None
        self.slip_mm = 0.0
        self.slip_deg = 0.0

    def observe(self, step_idx: int, cmd) -> None:
        ee = np.asarray(self.env.robot.ee_pose.detach().cpu(), dtype=np.float64).reshape(-1)[:3]
        cube = np.asarray(self.env.cube_pos(), dtype=np.float64).reshape(-1)[:3]
        dist = float(np.linalg.norm(ee - cube))
        xy_err = float(np.linalg.norm(ee[:2] - cube[:2]))
        z_err = abs(float(ee[2] - (self.env.cfg.table.top_z + self.cfg.grasp_tcp_offset)))
        gripper_norm = float(self.env.gripper_norm())
        if not self.weld_fired:
            self.min_ee_cube = min(self.min_ee_cube, dist)

        if self.cfg.task == "stack":
            # Stack retains its verified fixed close timing; the lift gate below is
            # the distribution being changed for the 10k lift dataset.
            if step_idx == self.policy.grasp_lock_step:
                self.env.grasp_lock()
                self._record_weld(step_idx, dist, xy_err, z_err)
            return

        deadline = self.policy.grasp_lock_step
        in_close_window = self.policy.close_start_step <= step_idx <= deadline
        near_cube = dist <= GRASP_R and xy_err <= GRASP_TOL_XY and z_err <= GRASP_TOL_Z
        if in_close_window and not cmd.open_gripper:
            self.close_ticks += 1
        # Upstream counts 12 dwell ticks at its 30 Hz control rate. This runner
        # advances at 120 Hz, so preserve the same physical dwell duration.
        close_ticks_min = 12 * self.env.cfg.record_every
        seated = 0.20 < gripper_norm < 0.85 and self.close_ticks >= close_ticks_min
        acquired = near_cube and seated
        if acquired and not self.physical_grasp_detected:
            self.physical_grasp_detected = True
            self.physical_grasp_step = step_idx
            self.physical_grasp_tcp_cube_dist = dist
            self.physical_grasp_gripper_norm = gripper_norm
            self.close_xy_err = xy_err
            self.close_z_err = z_err
        elif step_idx == deadline and self.close_xy_err is None:
            self.close_xy_err = xy_err
            self.close_z_err = z_err

        # (a) no acquisition by the deadline -> abort before the transport begins.
        if (
            step_idx >= deadline
            and not self.physical_grasp_detected
            and self.abort_reason is None
        ):
            self._abort("no_grasp", step_idx)
            return

        # (b) acquired, but the cube moves in the TCP frame -> the grasp is slipping.
        #
        # The reference is latched after the CLOSE COMPLETES (grasp_lock_step), matching
        # scripts/test_friction_grasp.py. Latching at physical_grasp_step instead is
        # wrong: acquisition is detected mid-close (e.g. step 519 against a 566 lock), so
        # the reference lands while the cube is still seating into the fingers and normal
        # seating then reads as slip -- measured 3.5-4.0 mm on grasps that were fine.
        # The window CLOSES at the open command: after release the cube is meant to leave
        # the gripper, so its motion relative to the TCP is the intended drop, not slip.
        # Measuring through the release reads the 0.089 m lift height back as ~92 mm of
        # "slip" on episodes that delivered to within 3 mm.
        if self.physical_grasp_detected and self.abort_reason is None and not cmd.open_gripper:
            settled_at = self.policy.grasp_lock_step + SLIP_SETTLE_STEPS
            if step_idx == settled_at:
                self._slip_ref = cube_rel_tcp(self.env)
                self._slip_ref_step = step_idx
            elif self._slip_ref is not None and step_idx > settled_at:
                mm, deg = slip_since(self._slip_ref, cube_rel_tcp(self.env))
                self.slip_mm = max(self.slip_mm, mm)
                self.slip_deg = max(self.slip_deg, deg)
                if mm > self.cfg.slip_abort_mm or deg > self.cfg.slip_abort_deg:
                    self._abort("slip", step_idx)

    def _abort(self, reason: str, step_idx: int) -> None:
        self.abort_reason = reason
        self.abort_step = step_idx

    def _record_weld(
        self,
        step_idx: int,
        dist: float,
        xy_err: float,
        z_err: float,
        gripper_norm: float | None = None,
    ) -> None:
        self.weld_fired = True
        self.weld_step = step_idx
        self.weld_tcp_cube_dist = dist
        self.close_xy_err = xy_err
        self.close_z_err = z_err
        self.weld_gripper_norm = gripper_norm

    def stats(self) -> dict:
        stats = {
            "grasp_mode": "stack_weld" if self.cfg.task == "stack" else "physical",
            "min_ee_cube": self.min_ee_cube if np.isfinite(self.min_ee_cube) else None,
            "close_xy_err": self.close_xy_err,
            "close_z_err": self.close_z_err,
            "weld_fired": self.weld_fired,
            "weld_step": self.weld_step,
            "weld_tcp_cube_dist": self.weld_tcp_cube_dist,
            "weld_gripper_norm": self.weld_gripper_norm,
            "physical_grasp_detected": self.physical_grasp_detected,
            "physical_grasp_step": self.physical_grasp_step,
            "physical_grasp_tcp_cube_dist": self.physical_grasp_tcp_cube_dist,
            "physical_grasp_gripper_norm": self.physical_grasp_gripper_norm,
            "close_ticks": self.close_ticks,
            "close_control_ticks": self.close_ticks / self.env.cfg.record_every,
            "grasp_timeout_step": self.policy.grasp_lock_step,
            "approach_distance": getattr(self.policy, "approach_distance", None),
            "approach_scale": getattr(self.policy, "approach_scale", None),
            # why a dropped episode was dropped: None | "no_grasp" | "slip".
            # Without this a drop is only visible as delivered=False.
            "abort_reason": self.abort_reason,
            "abort_step": self.abort_step,
            "slip_mm": self.slip_mm,
            "slip_deg": self.slip_deg,
        }
        return stats


def _advance_policy_step(
    env: TaskEnv,
    cfg: Config,
    policy,
    integrity: GraspIntegrity,
    step_idx: int,
    cmd,
) -> None:
    if cfg.task == "stack" and step_idx == policy.release_step:
        env.grasp_release()
    env.robot.go_to_goal(
        cmd.pose,
        open_gripper=cmd.open_gripper,
        ik_from_current=cfg.task == "lift",
    )
    env.step()
    integrity.observe(step_idx, cmd)


APPEARANCE_JITTER_FIELDS = (
    "nyx_light_dir_jitter_deg",
    "nyx_light_intensity_jitter",
    "robot_roughness_jitter",
    "cube_hue_jitter_deg",
    "cube_value_jitter",
)

SOURCE_FINGERPRINT_FILES = (
    "scripts/generate_task_dataset.py",
    "src/xsim/batch_renderer.py",
    "src/xsim/task_env.py",
    "src/xsim/scripted_lift_policy.py",
    "src/xsim/scripted_stack_policy.py",
    "src/xsim/mcap_writer.py",
)


def _source_fingerprint() -> str:
    import hashlib

    h = hashlib.sha256()
    for rel in SOURCE_FINGERPRINT_FILES:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update((PROJECT_ROOT / rel).read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _appearance_randomization_enabled(env_cfg: TaskEnvCfg) -> bool:
    return (
        env_cfg.nyx_light_type == "ceiling_panel"
        or any(abs(float(getattr(env_cfg, name))) > 0.0 for name in APPEARANCE_JITTER_FIELDS)
    )


def _env_cfg_for_episode(env_cfg: TaskEnvCfg, episode_seed: int) -> TaskEnvCfg:
    # decorrelate from the reset() stream: default_rng(seed) and default_rng(seed)
    # emit identical uniforms, so seeding appearance with the bare episode seed made
    # light intensity track cube placement (r=0.85 in batch 33300)
    return replace(env_cfg, appearance_seed=episode_seed * 9973 + 3)


def _config_to_jsonable(obj):
    import dataclasses

    if dataclasses.is_dataclass(obj):
        return {k: _config_to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_config_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _config_to_jsonable(v) for k, v in obj.items()}
    return obj


def _config_from_jsonable(raw: dict) -> Config:
    import dataclasses

    raw = dict(raw)
    env_raw = dict(raw.pop("env"))
    env_fields = {f.name for f in dataclasses.fields(TaskEnvCfg)}
    env_raw = {k: v for k, v in env_raw.items() if k in env_fields}
    env_raw["stack"] = StackCfg(**env_raw["stack"])
    env_raw["table"] = TableCfg(**env_raw["table"])
    env_raw["base_decor"] = BaseDecorCfg(**env_raw["base_decor"])
    if env_raw.get("splat_uri") is not None:
        env_raw["splat_uri"] = Path(env_raw["splat_uri"])
    raw["out_dir"] = Path(raw["out_dir"])
    raw["preview_dir"] = Path(raw["preview_dir"])
    raw["video_path"] = Path(raw["video_path"])
    return Config(env=TaskEnvCfg(**env_raw), **raw)


def _run_config_subprocess(child_cfg: Config) -> None:
    import json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(_config_to_jsonable(child_cfg), f)
        config_path = Path(f.name)
    try:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--config-json", str(config_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    finally:
        config_path.unlink(missing_ok=True)


def _episode_manifest_name(global_ep: int) -> str:
    return f".manifest_ep{global_ep:06d}.json"


def _episode_mcap_path(cfg: Config, global_ep: int) -> Path:
    return cfg.out_dir / f"episode_{global_ep:06d}.mcap"


def _child_config_for_episode(cfg: Config, ep: int) -> Config:
    episode_seed = cfg.seed + ep
    global_ep = cfg.episode_offset + ep
    return replace(
        cfg,
        n_episodes=1,
        episode_offset=global_ep,
        seed=episode_seed,
        appearance_child=True,
        pool_workers=1,
        manifest_name=_episode_manifest_name(global_ep),
        resume=False,
        env=_env_cfg_for_episode(cfg.env, episode_seed),
    )


def _load_completed_child(cfg: Config, ep: int) -> dict | None:
    """Recover one already-finished child only if code and config still match."""
    import json

    episode_seed = cfg.seed + ep
    global_ep = cfg.episode_offset + ep
    manifest_path = cfg.out_dir / _episode_manifest_name(global_ep)
    if not manifest_path.exists() or not _episode_mcap_path(cfg, global_ep).exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    expected_cfg = _child_config_for_episode(cfg, ep)
    if manifest.get("source_fingerprint") != _source_fingerprint():
        return None
    if manifest.get("config") != _config_to_jsonable(expected_cfg):
        return None
    episodes = manifest.get("episodes") or []
    if len(episodes) != 1:
        return None
    stats = episodes[0]
    if stats.get("episode") != global_ep or stats.get("seed") != episode_seed:
        return None
    return stats


def _run_episode_subprocess(cfg: Config, ep: int) -> dict:
    """One appearance child: private manifest so concurrent children can't clobber."""
    import json

    global_ep = cfg.episode_offset + ep
    if cfg.resume and (stats := _load_completed_child(cfg, ep)) is not None:
        print(f"[resume] ep{global_ep}: using matching child manifest + MCAP", flush=True)
        return stats

    child_cfg = _child_config_for_episode(cfg, ep)
    _run_config_subprocess(child_cfg)
    manifest_path = cfg.out_dir / child_cfg.manifest_name
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("episodes"):
        raise RuntimeError(f"child episode {ep} wrote no manifest episode stats")
    return manifest["episodes"][0]


def _cleanup_child_manifests(cfg: Config) -> None:
    for path in cfg.out_dir.glob(".manifest_ep*.json"):
        path.unlink(missing_ok=True)


def _run_appearance_subprocess_batch(cfg: Config) -> None:
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, cfg.pool_workers)
    print(f"appearance randomization enabled: one Nyx subprocess per episode "
          f"for fresh lights/materials ({workers} in flight)")
    all_stats: list[dict | None] = [None] * cfg.n_episodes
    completed = 0
    next_ep = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        in_flight = {}

        def submit_more() -> None:
            nonlocal next_ep
            while next_ep < cfg.n_episodes and len(in_flight) < workers:
                fut = pool.submit(_run_episode_subprocess, cfg, next_ep)
                in_flight[fut] = next_ep
                next_ep += 1

        submit_more()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                ep = in_flight.pop(fut)
                try:
                    stats = fut.result()
                except BaseException:
                    for pending in in_flight:
                        pending.cancel()
                    raise
                all_stats[ep] = stats
                completed += 1
                print(
                    f"[parent] {completed}/{cfg.n_episodes} "
                    f"ep{stats.get('episode', cfg.episode_offset + ep)} "
                    f"success={stats.get('success')}",
                    flush=True,
                )
            submit_more()

    ordered_stats = [s for s in all_stats if s is not None]
    if len(ordered_stats) != cfg.n_episodes:
        raise RuntimeError(f"only collected {len(ordered_stats)}/{cfg.n_episodes} episode stats")
    n_success = sum(int(bool(s.get("success"))) for s in ordered_stats)
    _write_manifest(cfg, None, ordered_stats, n_success)
    _cleanup_child_manifests(cfg)
    print(f"\nsuccess rate: {n_success}/{cfg.n_episodes}  -> {cfg.out_dir}")

def _spec_dict(env: TaskEnv) -> dict[str, CameraSpec]:
    specs = {}
    for name, (w, h, fx, fy, cx, cy) in env.camera_specs().items():
        specs[name] = CameraSpec(name=name, width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
    return specs


PREVIEW_CAMERA_ORDER = ("low", "side", "wrist", "over")


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)


def _save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"failed to write {path}")


def _preview_camera_names(images: dict[str, np.ndarray]) -> list[str]:
    ordered = [name for name in PREVIEW_CAMERA_ORDER if name in images]
    return ordered + [name for name in images if name not in ordered]


def save_preview_frame(env: TaskEnv, preview_dir: Path, step_idx: int, label: str) -> None:
    images = env.render()
    safe = _safe_label(label)
    names = _preview_camera_names(images)
    for name in names:
        _save_rgb_png(preview_dir / f"{step_idx:04d}_{safe}_{name}.png", images[name])

    annotated = []
    for name in names:
        frame = images[name].copy()
        cv2.putText(
            frame,
            f"{step_idx:04d} {label} {name}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        annotated.append(frame)
    _save_rgb_png(preview_dir / f"{step_idx:04d}_{safe}_contact.png", np.concatenate(annotated, axis=1))


def contact_sheet(images: dict[str, np.ndarray], label: str) -> np.ndarray:
    frames = []
    for name in _preview_camera_names(images):
        frame = images[name].copy()
        cv2.putText(
            frame,
            f"{label} {name}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(frame)
    return np.concatenate(frames, axis=1)


def run_preview(env: TaskEnv, cfg: Config) -> None:
    cfg.preview_dir.mkdir(parents=True, exist_ok=True)
    env.reset(seed=cfg.seed)
    policy = _make_policy(env, cfg)
    policy.reset()
    integrity = GraspIntegrity(env, cfg, policy)

    save_preview_frame(env, cfg.preview_dir, 0, "reset")

    waypoint_names = policy.waypoint_names or ["home"]
    milestone_steps = {0: waypoint_names[0]}
    cumulative = 0
    for seg_steps, name in zip(policy._segment_steps, waypoint_names[1:], strict=True):
        cumulative += seg_steps
        milestone_steps[cumulative] = name

    release_tail = max(1, int(round(cfg.release_tail_s / env.cfg.physics_dt)))
    total = policy.release_step + release_tail
    with torch.no_grad():
        for step_idx in range(total):
            cmd = policy.step()
            _advance_policy_step(env, cfg, policy, integrity, step_idx, cmd)
            if step_idx in milestone_steps:
                save_preview_frame(env, cfg.preview_dir, step_idx, milestone_steps[step_idx])

    save_preview_frame(env, cfg.preview_dir, total - 1, "release")
    print(f"wrote preview frames to {cfg.preview_dir}")


def run_video(env: TaskEnv, cfg: Config) -> None:
    cfg.video_path.parent.mkdir(parents=True, exist_ok=True)
    env.reset(seed=cfg.seed)
    policy = _make_policy(env, cfg)
    policy.reset()
    integrity = GraspIntegrity(env, cfg, policy)
    cube_start = env.cube_pos().copy()
    max_rise = 0.0

    first = contact_sheet(env.render(), f"0000 reset seed={cfg.seed}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(cfg.video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        cfg.video_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise OSError(f"failed to open video writer for {cfg.video_path}")

    release_tail = max(1, int(round(cfg.release_tail_s / env.cfg.physics_dt)))
    record_until = policy.release_step + release_tail
    frame_idx = 0
    try:
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        with torch.no_grad():
            for step_idx in range(record_until):
                cmd = policy.step()
                _advance_policy_step(env, cfg, policy, integrity, step_idx, cmd)
                cube = env.cube_pos()
                max_rise = max(max_rise, float(cube[2] - cube_start[2]))
                if step_idx % env.cfg.record_every == 0:
                    sheet = contact_sheet(env.render(), f"{frame_idx:04d}")
                    writer.write(cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
                    frame_idx += 1
            # unrecorded settle for the success stats, matching run_episode
            for _ in range(cfg.hold_steps):
                cmd = policy.step()
                env.robot.go_to_goal(cmd.pose, open_gripper=cmd.open_gripper)
                env.step()
    finally:
        writer.release()

    res = _episode_result(env, cfg, max_rise)
    print(f"wrote video ({frame_idx + 1} frames @ {cfg.video_fps:g} fps) to {cfg.video_path}")
    print("video stats: " + " ".join(f"{k}={v}" for k, v in res.items() if k != "green_pos"))
    print("grasp stats: " + " ".join(f"{k}={v}" for k, v in integrity.stats().items()))
    return {**res, **integrity.stats()}


def run_episode(env: TaskEnv, cfg: Config, episode_idx: int, path: Path) -> dict:
    global_episode = cfg.episode_offset + episode_idx
    env.reset(seed=cfg.seed + episode_idx)
    # vary episode tempo per seed; the new release-at-drop protocol runs roughly 5-7 s
    tempo_rng = np.random.default_rng((cfg.seed + episode_idx) * 7919 + 1)
    sps = int(round(cfg.steps_per_segment * tempo_rng.uniform(0.85, 1.30)))
    policy = _make_policy(env, cfg, steps_per_segment=sps)
    policy.reset()
    integrity = GraspIntegrity(env, cfg, policy)

    cube_start = env.cube_pos().copy()
    max_rise = 0.0
    record_dt_ns = int(round(env.record_dt * 1e9))
    base_ns = 1_000_000_000
    rec = 0

    specs = _spec_dict(env)
    # recording ends shortly after the open command: the fingers opening are in the data,
    # the robot moving away from the dropped cube never is (grifflee's protocol)
    release_tail = max(1, int(round(cfg.release_tail_s / env.cfg.physics_dt)))
    record_until = policy.release_step + release_tail
    with EpisodeMcapWriter(path, specs) as writer:
        writer.log_calibration(base_ns, env.episode_extrinsics)
        with torch.no_grad():
            for i in range(record_until):
                cmd = policy.step()
                _advance_policy_step(env, cfg, policy, integrity, i, cmd)
                cube = env.cube_pos()
                max_rise = max(max_rise, float(cube[2] - cube_start[2]))
                if i % env.cfg.record_every == 0:
                    imgs = env.render()
                    pos, vel, eff, ee = env.proprio()
                    writer.log_step(base_ns + rec * record_dt_ns, imgs, pos, vel, eff, None,
                                    ee_pose=ee, gripper_norm=env.gripper_norm())
                    rec += 1
                if integrity.abort_reason is not None:
                    # Measured-bad grasp: stop before carrying it. The episode scores as a
                    # failure below and the keep-gate deletes this partial MCAP.
                    break
            if integrity.abort_reason is None:
                # unrecorded settle: let the cube land for the success evaluation
                for _ in range(cfg.hold_steps):
                    cmd = policy.step()
                    env.robot.go_to_goal(cmd.pose, open_gripper=cmd.open_gripper)
                    env.step()

    stats = {
        "episode": global_episode, "frames": rec, "cube_yaw": float(env.cube_yaw()),
        # actual (jittered) camera poses this episode: c2w OpenCV for low/side, plus the
        # link_tcp->camera(optical) wrist mount; also written into the MCAP itself
        # (camera_info + /tf) via log_calibration
        "extrinsics": {k: np.asarray(v).tolist() for k, v in env.episode_extrinsics.items()},
        "appearance": env.episode_appearance,
        "spawn": env.episode_spawn,
        "arm_start": env.episode_arm_start,
    }
    stats.update(integrity.stats())
    stats.update(_episode_result(env, cfg, max_rise))
    if integrity.abort_reason is not None:
        # An aborted episode never reached the settle, so _episode_result scored a
        # partial trajectory. Force the failure rather than trusting delivered=False.
        stats["success"] = False
    return stats


def main(cfg: Config) -> None:
    cfg.env.task = cfg.task  # single --task flag drives both the env and the policy
    if cfg.task == "lift" and cfg.env.noslip_iterations != 10:
        raise ValueError("lift generation is physical/no-weld and requires --env.noslip-iterations 10")
    appearance_randomized = _appearance_randomization_enabled(cfg.env)
    if cfg.mode == "generate" and appearance_randomized and not cfg.appearance_child and cfg.n_episodes > 1:
        _run_appearance_subprocess_batch(cfg)
        return

    backend = gs.gpu if cfg.backend == "gpu" else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning")
    _mark("gs_init")

    initial_env_cfg = _env_cfg_for_episode(cfg.env, cfg.seed) if appearance_randomized else cfg.env
    env = TaskEnv(initial_env_cfg)
    _mark("env_build")
    if cfg.mode == "preview":
        run_preview(env, cfg)
        return
    if cfg.mode == "video":
        run_video(env, cfg)
        return

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    n_success = 0
    all_stats = []
    manifest_env = env

    for ep in range(cfg.n_episodes):
        path = cfg.out_dir / f"episode_{cfg.episode_offset + ep:06d}.mcap"
        stats = run_episode(env, cfg, ep, path)
        if cfg.task == "lift" and stats.get("weld_fired"):
            raise RuntimeError("retired lift weld fired; refusing to keep the episode")
        _mark(f"episode_{ep}")
        keep = stats["success"] or cfg.save_failures
        if not keep:
            path.unlink(missing_ok=True)
        stats["kept"] = keep
        stats["seed"] = cfg.seed + ep
        all_stats.append(stats)
        n_success += int(stats["success"])
        flag = "OK " if stats["success"] else ("kept" if keep else "drop")
        detail = (f"xy_err={stats['stack_xy_err']:.3f} z_err={stats['stack_z_err']:+.3f} stacked={stats['stacked']}"
                  if cfg.task == "stack" else
                  f"deliver={stats['deliver_dist']:.3f} delivered={stats['delivered']}")
        print(f"[{flag}] ep{ep}: frames={stats['frames']} rise={stats['max_rise']:.3f} "
              f"lifted={stats['lifted']} {detail}")

    _write_manifest(cfg, manifest_env, all_stats, n_success)
    _mark("manifest_write")
    if _TIMING:
        _print_timing()
    print(f"\nsuccess rate: {n_success}/{cfg.n_episodes}  -> {cfg.out_dir}")


def _write_manifest(cfg: Config, env: TaskEnv | None, all_stats: list[dict], n_success: int) -> None:
    """Provenance sidecar: enough to regenerate any episode bit-for-bit."""
    import dataclasses
    import hashlib
    import json
    import subprocess

    def _jsonable(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        return obj

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        sha, dirty = "unknown", True

    env_cfg = env.cfg if env is not None else cfg.env
    manifest_cfg = replace(cfg, env=env_cfg)
    splat = Path(env_cfg.splat_uri).expanduser() if env_cfg.splat_uri else None
    splat_md5 = None
    if splat is not None and splat.exists():
        h = hashlib.md5()
        with open(splat, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        splat_md5 = h.hexdigest()

    manifest = {
        "git_sha": sha, "git_dirty": dirty,
        "source_fingerprint": _source_fingerprint(),
        "source_fingerprint_files": list(SOURCE_FINGERPRINT_FILES),
        "config": _jsonable(manifest_cfg),
        "splat_file": str(splat) if splat else None,
        "splat_md5": splat_md5,
        "success_rate": n_success / cfg.n_episodes if cfg.n_episodes else 0.0,
        "episodes": all_stats,
    }
    (cfg.out_dir / cfg.manifest_name).write_text(json.dumps(manifest, indent=2))
    print(f"wrote {cfg.out_dir / cfg.manifest_name}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--config-json":
        import json

        main(_config_from_jsonable(json.loads(Path(sys.argv[2]).read_text())))
    else:
        main(tyro.cli(Config))
