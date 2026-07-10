"""DAgger correction: a student policy monitored against a scripted-teacher reference.

Everything composes through the Gymnasium contract used by ``scripts/eval.py``
(``policy.step(obs) -> action``, ``env.step(action) -> (obs, reward, done, info)``), so
the collection loop in ``scripts/dagger.py`` is the plain gym loop for every phase:

    teacher   = ScriptedTeacherPolicy(env)              # xsim.teacher
    student   = ChunkClient(Client(host, port))          # the served crossformer
    policy    = DAggerPolicy(student, teacher, env)

    # phase A: reference — roll the teacher alone, record its trace
    # phase B: hybrid    — roll DAggerPolicy on the identical scene (same seed/options)

:class:`DAggerPolicy` runs the student until it diverges from the reference, then hands
control to the teacher, whose ``reset()`` re-plans from the live (failed) state. Its
``step()`` returns ONE control-step action: student chunks are buffered internally and
doled out one per step (subsuming ``ActionChunkWrapper``), which is what lets the
:class:`DivergenceMonitor` check every control step instead of once per 50-step chunk.

The monitor's comparison is geometric, not time-indexed: the student is matched to the
nearest point on the teacher's TCP path *ahead of its last matched point* (monotonic
progress, advancing at most ``max_advance_s`` of reference time per control step so a
lurch can't "match" a far-forward path point), so a student that moves slower than the
teacher but along the right line stays at error ~0. Three triggers fire the takeover:

- ``tcp_off_corridor`` — TCP outside the segment's corridor radius (loose on approach,
  tight at the plunge/grasp, looser in transport) for ``patience`` consecutive steps;
- ``cube_disturbed`` / ``cube_off_track`` — the cube strays from the reference cube trace:
  pre-grasp the student knocked it; post-grasp it catches a missed grasp (the reference
  cube rises, the real one doesn't) even when the TCP is perfectly on-path;
- ``stalled`` — the progress index stops advancing (an obs-insensitive policy hovering
  in-corridor forever would otherwise never trigger).

The monitor/reference classes are pure numpy (no Genesis) and unit-testable standalone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from xsim.teacher import LIFT_SEGMENTS
from xsim.wrappers.base import Wrapper


# ---------------------------------------------------------------------------------------
# Reference trace
# ---------------------------------------------------------------------------------------


@dataclass
class ExpertReference:
    """Per-control-step trace of one successful teacher rollout."""

    tcp: np.ndarray            # (N, 3) TCP position per step
    cube: np.ndarray           # (N, 3) cube position per step
    gripper_open: np.ndarray   # (N,) bool commanded gripper state per step
    segment: np.ndarray        # (N,) int index into segment_names
    segment_names: tuple[str, ...]
    grasp_idx: int             # first step the gripper was commanded closed
    dt: float                  # control period the trace was sampled at

    def __len__(self) -> int:
        return len(self.tcp)


class ReferenceRecorder:
    """Accumulates the teacher trace during the reference rollout (one append per step)."""

    def __init__(self, dt: float, segment_names: tuple[str, ...] = LIFT_SEGMENTS):
        self._dt = dt
        self._names = tuple(segment_names)
        self._index = {name: i for i, name in enumerate(self._names)}
        self._tcp: list[np.ndarray] = []
        self._cube: list[np.ndarray] = []
        self._open: list[bool] = []
        self._segment: list[int] = []

    def append(self, tcp, cube, gripper_open: bool, segment: str) -> None:
        self._tcp.append(np.asarray(tcp, dtype=np.float64).reshape(-1)[:3].copy())
        self._cube.append(np.asarray(cube, dtype=np.float64).reshape(-1)[:3].copy())
        self._open.append(bool(gripper_open))
        self._segment.append(self._index.get(segment, len(self._names) - 1))

    def finalize(self) -> ExpertReference:
        gripper_open = np.asarray(self._open, dtype=bool)
        closed = np.flatnonzero(~gripper_open)
        return ExpertReference(
            tcp=np.asarray(self._tcp),
            cube=np.asarray(self._cube),
            gripper_open=gripper_open,
            segment=np.asarray(self._segment, dtype=np.int64),
            segment_names=self._names,
            grasp_idx=int(closed[0]) if len(closed) else len(gripper_open),
            dt=self._dt,
        )


# ---------------------------------------------------------------------------------------
# Divergence monitor
# ---------------------------------------------------------------------------------------


@dataclass
class DaggerThresholds:
    # corridor radius (m) around the reference TCP path, one per scripted segment (see
    # xsim.teacher.LIFT_SEGMENTS): loose approach, tight plunge/grasp, looser after
    corridor: tuple[float, ...] = (0.06, 0.03, 0.03, 0.05, 0.07, 0.07)
    cube_tol_pre: float = 0.03     # cube may not move before the teacher's grasp point
    cube_tol_post: float = 0.08    # cube must track the reference cube trace after it
    patience: int = 5              # consecutive breached control steps before takeover
    # max reference-time the matched point may advance per control step. On-pace progress
    # is 1 control step of reference time; the headroom allows a faster-than-teacher
    # student but stops a lurch from "matching" a far-forward path point and reporting
    # inflated progress (the lurch then measures against the local segment and breaches).
    max_advance_s: float = 0.5
    stall_window_s: float = 3.0    # progress lookback horizon
    stall_min_advance_s: float = 0.25  # min reference-time advance over the stall window


@dataclass
class Verdict:
    diverged: bool
    reason: str | None      # tcp_off_corridor | cube_disturbed | cube_off_track | stalled
    ref_idx: int            # matched reference step (monotonic)
    segment: str            # segment name at the matched step
    tcp_err: float          # distance to the matched reference TCP point (m)
    cube_err: float         # distance to the matched reference cube point (m)
    progress: float         # ref_idx / (len(reference) - 1)


class DivergenceMonitor:
    """Call :meth:`update` once per student CONTROL step; fires once and stays fired."""

    def __init__(self, ref: ExpertReference, thresholds: DaggerThresholds | None = None):
        self.ref = ref
        self.thr = thresholds or DaggerThresholds()
        self._idx = 0
        self._tcp_breach = 0
        self._cube_breach = 0
        self._max_advance = max(1, int(round(self.thr.max_advance_s / ref.dt)))
        self._stall_min_advance = max(1, int(round(self.thr.stall_min_advance_s / ref.dt)))
        window = max(2, int(round(self.thr.stall_window_s / ref.dt)))
        self._progress_hist: deque[int] = deque(maxlen=window)
        self._fired: Verdict | None = None

    def update(self, tcp, cube) -> Verdict:
        if self._fired is not None:
            return self._fired
        ref, thr = self.ref, self.thr
        tcp = np.asarray(tcp, dtype=np.float64).reshape(-1)[:3]
        cube = np.asarray(cube, dtype=np.float64).reshape(-1)[:3]

        lo, hi = self._idx, min(len(ref), self._idx + 1 + self._max_advance)
        d = np.linalg.norm(ref.tcp[lo:hi] - tcp, axis=1)
        j = int(np.argmin(d))
        idx = lo + j
        self._idx = idx  # monotonic progress
        tcp_err = float(d[j])
        cube_err = float(np.linalg.norm(ref.cube[idx] - cube))
        seg = int(ref.segment[idx])
        pre_grasp = idx < ref.grasp_idx

        reason = None
        radius = thr.corridor[min(seg, len(thr.corridor) - 1)]
        self._tcp_breach = self._tcp_breach + 1 if tcp_err > radius else 0
        cube_tol = thr.cube_tol_pre if pre_grasp else thr.cube_tol_post
        self._cube_breach = self._cube_breach + 1 if cube_err > cube_tol else 0
        if self._tcp_breach >= thr.patience:
            reason = "tcp_off_corridor"
        elif self._cube_breach >= thr.patience:
            reason = "cube_disturbed" if pre_grasp else "cube_off_track"

        self._progress_hist.append(idx)
        full = len(self._progress_hist) == self._progress_hist.maxlen
        at_end = idx >= len(ref) - self._stall_min_advance
        if reason is None and full and not at_end \
                and idx - self._progress_hist[0] < self._stall_min_advance:
            reason = "stalled"

        verdict = Verdict(
            diverged=reason is not None, reason=reason, ref_idx=idx,
            segment=ref.segment_names[seg], tcp_err=tcp_err, cube_err=cube_err,
            progress=idx / max(1, len(ref) - 1),
        )
        if verdict.diverged:
            self._fired = verdict
        return verdict


# ---------------------------------------------------------------------------------------
# DAgger policy: student until divergence, then teacher
# ---------------------------------------------------------------------------------------


class DAggerPolicy:
    """GymPolicy composing a student and a teacher: ``step(obs)`` -> one joint action.

    ``student.step(obs)`` may return an ``(H, A)`` chunk (a served crossformer) or a
    single ``(A,)`` action; chunks are buffered and consumed one action per step. The
    monitor checks the env state before every student action, and on divergence the
    teacher is ``reset()`` (it re-plans from the live state) and drives from then on.

    Privileged like the teacher: reads TCP/cube from ``env`` directly, not from obs.
    Set :attr:`reference` (the scene's teacher trace) before each :meth:`reset`.
    """

    def __init__(self, student, teacher, env,
                 thresholds: DaggerThresholds | None = None, chunk_h: int = 50):
        self.student = student
        self.teacher = teacher
        self.env = env
        self.thresholds = thresholds or DaggerThresholds()
        self.chunk_h = chunk_h
        self.reference: ExpertReference | None = None
        self.monitor: DivergenceMonitor | None = None
        self.mode = "student"           # "student" | "teacher"
        self.switch: Verdict | None = None   # the verdict that triggered the takeover
        self.switch_step: int | None = None  # control step the takeover happened at
        self._steps = 0
        self._chunk: deque[np.ndarray] = deque()

    def reset(self) -> None:
        if self.reference is None:
            raise ValueError("DAggerPolicy.reference must be set before reset()")
        self.monitor = DivergenceMonitor(self.reference, self.thresholds)
        self.mode = "student"
        self.switch = None
        self.switch_step = None
        self._steps = 0
        self._chunk.clear()
        self.student.reset()

    def step(self, obs) -> np.ndarray:
        if self.mode == "student":
            _, _, _, ee = self.env.proprio()
            verdict = self.monitor.update(ee[:3], self.env.cube_pos())
            if verdict.diverged:
                self.mode = "teacher"
                self.switch = verdict
                self.switch_step = self._steps
                self._chunk.clear()
                self.teacher.reset()    # re-plan from the live (failed) state
            else:
                if not self._chunk:
                    chunk = np.atleast_2d(
                        np.asarray(self.student.step(obs), dtype=np.float32))
                    self._chunk.extend(chunk[: self.chunk_h])
                self._steps += 1
                return self._chunk.popleft()
        self._steps += 1
        return self.teacher.step(obs)


# ---------------------------------------------------------------------------------------
# Video annotation
# ---------------------------------------------------------------------------------------


class ModeStripWrapper(Wrapper):
    """Tints a strip at the top of each rendered view by the current rollout phase, so
    the takeover moment is visible in the mp4s. Sits between the adapter and
    ``VideoRecordWrapper``; set ``mode_fn`` to a callable returning the phase name.
    The student->teacher transition shows the ``failure`` color for ``flash_frames``
    captures before settling on the teacher color. Video-only: observations and any
    recorded episode data are built below this wrapper and never see the strip."""

    COLORS = {
        "reference": (160, 160, 160),  # gray: teacher reference rollout
        "student": (40, 200, 60),      # green: student driving, on-path
        "failure": (220, 50, 40),      # red: divergence detected (flash at takeover)
        "teacher": (60, 120, 230),     # blue: teacher recovery driving
    }

    def __init__(self, env, mode_fn=None, strip_px: int = 12, flash_frames: int = 30):
        super().__init__(env)
        self.mode_fn = mode_fn or (lambda: "")
        self.strip_px = strip_px
        self.flash_frames = flash_frames
        self._last_mode = ""
        self._flash = 0

    def reset(self, **kwargs):
        self._last_mode = ""
        self._flash = 0
        return self.env.reset(**kwargs)

    def render(self) -> dict:
        mode = self.mode_fn()
        if mode == "teacher" and self._last_mode == "student":
            self._flash = self.flash_frames
        self._last_mode = mode
        if self._flash > 0 and mode == "teacher":
            self._flash -= 1
            mode = "failure"
        frames = self.env.render()
        color = self.COLORS.get(mode)
        if color is None:
            return frames
        out = {}
        for key, frame in frames.items():
            frame = np.ascontiguousarray(frame).copy()
            frame[: self.strip_px, :, :3] = color
            out[key] = frame
        return out
