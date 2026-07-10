"""MP4 recording wrapper.

Captures the env's rendered camera views each step and writes one mp4 per episode, so
eval/rollout code never hand-rolls frame buffers or ``cv2.VideoWriter`` again. Views are
tiled left-to-right into a single frame (matching the eval harness's ``low|side|wrist``
strip). Which episodes get recorded is controlled by ``episode_trigger`` (à la
``gymnasium.wrappers.RecordVideo``); the default records every episode.

Place this *below* :class:`~xsim.wrappers.action_chunk.ActionChunkWrapper` in the stack so
it sees every physics step, not just one frame per action chunk::

    env = VideoRecordWrapper(TaskEnv(cfg), "outputs/videos")
    env = ActionChunkWrapper(env, h=50)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from xsim.wrappers.base import Wrapper


class VideoRecordWrapper(Wrapper):
    def __init__(
        self,
        env: Any,
        video_folder: str | Path,
        *,
        view_keys: Sequence[str] | None = ("low", "side", "wrist"),
        name_prefix: str = "episode",
        episode_trigger: Callable[[int], bool] | None = None,
        capture_every: int | None = None,
        fps: float | None = None,
    ):
        super().__init__(env)
        self.video_folder = Path(video_folder)
        self.view_keys = view_keys
        self.name_prefix = name_prefix
        self.episode_trigger = episode_trigger or (lambda _ep: True)

        # default capture cadence / fps from the base env's sim config when present, so a
        # per-physics-step video plays back in real time.
        cfg = getattr(self.unwrapped, "cfg", None)
        self.capture_every = capture_every if capture_every is not None else int(
            getattr(cfg, "record_every", 1) or 1)
        if fps is not None:
            self.fps = float(fps)
        elif cfg is not None and getattr(cfg, "physics_dt", None):
            self.fps = 1.0 / (cfg.physics_dt * self.capture_every)
        else:
            self.fps = 30.0

        self._episode_id = -1
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._inner_step = 0

    def reset(self, **kwargs) -> Any:
        self._flush()  # write the previous episode if it never hit `done`
        obs = self.env.reset(**kwargs)
        self._episode_id += 1
        self._recording = bool(self.episode_trigger(self._episode_id))
        self._frames = []
        self._inner_step = 0
        if self._recording:
            self._capture()
        return obs

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        if self._recording:
            self._inner_step += 1
            if self._inner_step % self.capture_every == 0:
                self._capture()
            if done:
                self._flush()
        return obs, reward, done, info

    def close(self) -> None:
        self._flush()
        super().close()

    # -- internals --
    def _capture(self) -> None:
        frames = self.env.render()
        keys = [k for k in (self.view_keys or list(frames)) if k in frames]
        tiled = np.concatenate(
            [np.ascontiguousarray(frames[k]) for k in keys], axis=1)
        self._frames.append(tiled)

    def _flush(self) -> None:
        if self._recording and self._frames:
            path = self.video_folder / f"{self.name_prefix}_{self._episode_id:04d}.mp4"
            _write_mp4(path, self._frames, self.fps)
        self._frames = []
        self._recording = False


def _write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import cv2

    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame[:, :, ::-1])  # RGB -> BGR
    writer.release()
