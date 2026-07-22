"""Buffered training-MCAP recorder for gym rollouts.

Records each control step of an episode (images, proprio, gripper, action) into memory
and writes a Foxglove MCAP **byte-compatible with the training batches** (same topics,
encodings, and 30 Hz cadence as ``scripts/generate_task_dataset.py``, via
``xsim.mcap_writer.EpisodeMcapWriter``) — but only when told to. The keep/drop decision
is made *after* the episode by the caller (e.g. the dagger driver keeps only
diverged-and-recovered hybrids), so the wrapper buffers and exposes
``save(path)`` / ``discard()`` instead of writing eagerly.

Stack placement: directly above the ``GenesisGymAdapter`` and **below** any cosmetic
wrappers (``ModeStripWrapper``) so buffered images are clean. ``render()`` returns the
frame captured for the current step, so an outer ``VideoRecordWrapper`` reuses it and no
extra render happens.

Recording follows grifflee's demonstration protocol: on ``save`` the buffer is trimmed to
the last close->open gripper transition (the release) plus ``release_tail_steps``, so the
episode ends with the fingers opening — the cube landing and the arm idling at the drop
target are never in the data. Set ``enabled = False`` to make the wrapper a passthrough
(e.g. during reference rollouts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from xsim.wrappers.base import Wrapper


class McapRecordWrapper(Wrapper):
    def __init__(self, env: Any, record_dt: float, release_tail_steps: int = 9,
                 keep_all: bool = False):
        super().__init__(env)
        self.record_dt = record_dt
        self.release_tail_steps = release_tail_steps
        # keep_all bypasses the release trim so DAgger episodes save full length
        # (paper-DAgger keeps every visited state, including post-release/failure ones).
        self.keep_all = keep_all
        self.enabled = True
        self._steps: list[dict] = []
        self._last_render: dict[str, np.ndarray] | None = None
        self._pending_teacher: dict | None = None

    def reset(self, **kwargs) -> Any:
        self._steps = []
        self._last_render = None
        self._pending_teacher = None
        return self.env.reset(**kwargs)

    def set_teacher(self, joints7, pos_cmd, quat_cmd, grip_norm) -> None:
        """Attach the expert's label to the NEXT recorded step (call before ``step``).

        joints7: 7 commanded arm-joint targets. pos_cmd/quat_cmd: commanded EE pose (xyz
        metres, wxyz). grip_norm: commanded gripper (1=open/0=closed). ``joints7=None``
        omits the teacher channels for the step. The label is consumed by the following
        ``step`` and reset to None, so a step with no ``set_teacher`` records no teacher data.
        """
        if joints7 is None:
            self._pending_teacher = None
            return
        pos = np.asarray(pos_cmd, dtype=np.float64).reshape(-1)[:3]
        quat = np.asarray(quat_cmd, dtype=np.float64).reshape(-1)[:4]
        self._pending_teacher = dict(
            joint_pos=np.asarray(joints7, dtype=np.float64).reshape(-1)[:7].copy(),
            ee_pose=np.concatenate([pos, quat]).copy(),  # [x,y,z,qw,qx,qy,qz]
            gripper_norm=float(grip_norm),
        )

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        if self.enabled:
            images = {k: np.ascontiguousarray(v[..., :3]).copy()
                      for k, v in self.env.render().items()}
            self._last_render = images
            pos, vel, eff, ee = self.env.proprio()
            self._steps.append(dict(
                images=images,
                joint_pos=np.asarray(pos, dtype=np.float64).copy(),
                joint_vel=np.asarray(vel, dtype=np.float64).copy(),
                joint_eff=np.asarray(eff, dtype=np.float64).copy(),
                ee_pose=np.asarray(ee, dtype=np.float64).reshape(-1)[:7].copy(),
                gripper_norm=float(self.env.gripper_norm()),
                gripper_open_cmd=_gripper_open(action),
                teacher=self._pending_teacher,
            ))
        else:
            self._last_render = None
        self._pending_teacher = None
        return obs, reward, done, info

    def render(self) -> dict:
        return self._last_render if self._last_render is not None else self.env.render()

    # -- keep/drop API --
    @property
    def trimmed_frames(self) -> int:
        """Frames save() would keep, without writing anything (full length under keep_all)."""
        if not self._steps:
            return 0
        return len(self._steps) if self.keep_all else self._trim_index()

    def save(self, path: str | Path) -> dict:
        """Write the buffered episode as a training MCAP; returns {"frames": n}."""
        from xsim.mcap_writer import CameraSpec, EpisodeMcapWriter

        if not self._steps:
            raise ValueError("McapRecordWrapper.save called with an empty buffer")
        steps = self._steps if self.keep_all else self._steps[: self._trim_index()]

        specs = {}
        for name, (w, h, fx, fy, cx, cy) in self.env.camera_specs().items():
            specs[name] = CameraSpec(name=name, width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)

        base_ns = 1_000_000_000
        record_dt_ns = int(round(self.record_dt * 1e9))
        with EpisodeMcapWriter(path, specs) as writer:
            writer.log_calibration(base_ns, self.env.episode_extrinsics)
            for i, s in enumerate(steps):
                t = s.get("teacher")
                writer.log_step(
                    base_ns + i * record_dt_ns, s["images"],
                    s["joint_pos"], s["joint_vel"], s["joint_eff"], None,
                    ee_pose=s["ee_pose"], gripper_norm=s["gripper_norm"],
                    teacher_joint_pos=(t["joint_pos"] if t else None),
                    teacher_ee_pose=(t["ee_pose"] if t else None),
                    teacher_gripper_norm=(t["gripper_norm"] if t else None),
                )
        self.discard()
        return {"frames": len(steps)}

    def discard(self) -> None:
        self._steps = []

    def _trim_index(self) -> int:
        """End of the kept range: last close->open command transition + the tail."""
        opens = [s["gripper_open_cmd"] for s in self._steps]
        release = None
        for i in range(1, len(opens)):
            if opens[i] and not opens[i - 1]:
                release = i
        if release is None:
            return len(self._steps)
        return min(len(self._steps), release + self.release_tail_steps)


def _gripper_open(action: Any) -> bool:
    """Commanded gripper state from a [q0..q6, gripper] action (>0.5 = open)."""
    vec = np.asarray(action, dtype=np.float64).reshape(-1)
    return bool(vec[7] > 0.5) if vec.size > 7 else True
