"""Policy adapters for the grid-evaluation harness.

These adapters present one uniform interface so the eval harness can drive either

- a **remote served crossformer** over the webpolicy websocket (:class:`RemotePolicy`), or
- one of the repo's **scripted waypoint policies** (:class:`ScriptedEvalPolicy`), used as
  the weld-free baseline,

without the harness caring which is running. The single interface is :class:`EvalPolicy`:
``reset()`` prepares for a fresh episode, ``act(env)`` observes the env and *applies* one
control-tick action to it, and ``done`` reports whether the policy's trajectory has ended.

Having ``act(env)`` apply the action directly (rather than return one) keeps all
gripper/weld handling inside the adapter: the harness never calls ``env.grasp_lock()`` for
eval, so the scripted baseline runs **weld-free** and relies on finger friction to hold the
cube (per the Phase-0 feasibility test). Both adapters honor a configurable finger
``close_setpoint`` by mutating ``env.robot._gripper_grasp_dof`` (see the notes on each).

Import stays light on purpose: only numpy/torch at module top level. ``webpolicy`` is
imported lazily inside :meth:`RemotePolicy.__init__` (it is not a dependency of this repo),
and the scripted policies are imported lazily inside :meth:`ScriptedEvalPolicy.reset` so
that importing this module never pulls in Genesis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class EvalPolicy(Protocol):
    """Uniform driver interface the eval harness calls."""

    def reset(self) -> None:
        """Prepare for a fresh episode (call *after* the cube has been placed)."""

    def act(self, env) -> None:
        """Observe ``env``, compute one action, and apply it as a single control tick."""

    @property
    def done(self) -> bool:
        """True once the policy's trajectory has ended (remote policies are never done)."""


# CrossFormer xflow DOF ids: j0..j6 + gripper. See grifflee/crossformer:crossformer/embody.py.
_DEFAULT_DOF_IDS = (1, 2, 3, 4, 5, 6, 7, 8)
_DEFAULT_CHUNK_STEPS = tuple(float(i) for i in range(20))

# The "single" embodiment's 140-slot state layout, extracted from
# Embodiment.REGISTRY["single"] (grifflee/crossformer dev, crossformer/embody.py):
# slots 0-6 arm joints (ids 1-7), 7 gripper (8), 8-10 cart_pos (9-11), 11-13 cart_ori
# (12-14), then kp3dc x3 views (ids 258-299, view 1/2/3). MASK_ID=0 pads nothing here.
# XStateEncoder consumes ONLY observation.state.{base,id,view}+mask.state.base — the
# proprio_* keys are pipeline raw material the model never sees, so the client must pack
# (and normalize) this block itself; kp3dc slots are sent masked-out (no FK keypoints).
_STATE_SLOTS = 140
_STATE_IDS = tuple(range(1, 15)) + tuple(range(258, 300)) * 3
_STATE_VIEWS = (0,) * 14 + (1,) * 42 + (2,) * 42 + (3,) * 42
_STATE_PART_SLOTS = {"joints": (0, 7), "gripper": (7, 8), "position": (8, 11), "orientation": (11, 14)}


@dataclass
class ObsSpec:
    """Maps the sim observation to a served crossformer's wire format.

    Two wire formats exist (docs/INFERENCE_CONTRACT.md):

    ``wire_format="grainlike"`` (default) — the ``scripts/serve/bela.py`` stack
    (``ModelPolicy -> ActionDenormWrapper -> GrainlikeWrapper``, mhyatt000/crossformer dev).
    The client sends a RAW arec-shaped sample and the server replays the exact training
    transforms (proprio normalization, image resize, state-block packing) and returns
    DENORMALIZED actions as named parts (``actions.joints`` etc.):

    - ``observation.image``: uint8 (1, V, H, W, 3), native resolution (the server resizes
      to the trained size with the training augmax chain — do not pre-resize), views in
      sorted-image-topic order ``[low, side, wrist]``.
    - ``observation.proprio.{joints,gripper,position,orientation}``: raw values with a
      leading batch dim of 1; kp3d keypoint parts omitted (masked downstream).
    - ``info.{step,episode}``: episode-relative step index for ``observation.timestep``.
    - no ``dof_ids``/``chunk_steps`` (``ModelPolicy`` hardcodes the full query grid) and
      no client-side normalization; leave ``ActionSpec.denorm_stats`` unset.

    ``wire_format="v1"`` — the fork's ``scripts/server.py`` v1 path (verified 2026-07-08
    against the checkpoint's example_batch.msgpack):

    - ``image``: uint8 (1, 1, V, 64, 64, 3), stacked views resized client-side because the
      v1 server only auto-resizes per-view ``image_*`` keys, not the stacked key.
    - ``proprio_joints`` (1,1,7) rad, ``proprio_gripper`` (1,1,1) norm 1=open,
      ``proprio_position`` (1,1,3) TCP metres, ``proprio_orientation`` (1,1,3) scipy
      euler "xyz" of the TCP quat — the server normalizes these with the dataset stats.
    - ``view_mask`` (1,1,V) all-True; obs wrapped under "observation" plus
      ``dof_ids``/``chunk_steps``; set ``state_stats`` (client-side state packing) and
      ``ActionSpec.denorm_stats`` (the flow head predicts in normalized space).
    """

    wire_format: Literal["grainlike", "v1"] = "grainlike"
    image_key: str = "image"
    view_order: tuple[str, ...] = ("low", "side", "wrist")
    image_hw: tuple[int, int] = (64, 64)
    include_view_mask: bool = True
    proprio_keys: dict = field(
        default_factory=lambda: {
            "joints": "proprio_joints",
            "gripper": "proprio_gripper",
            "position": "proprio_position",
            "orientation": "proprio_orientation",
        }
    )
    payload_key: str | None = "observation"
    include_xflow_control: bool = True
    dof_ids: tuple[int, ...] = _DEFAULT_DOF_IDS
    chunk_steps: tuple[float, ...] = _DEFAULT_CHUNK_STEPS
    # dataset_statistics.json path; when set, a normalized state/{base,id,view} block is
    # packed client-side (training normalizes proprio BEFORE embody_transform packs it)
    state_stats: str | None = None
    state_dataset: str = "xarm_sim"

    def build(self, env, step: int = 0) -> dict:
        """Render cameras and assemble a CrossFormer webpolicy payload."""
        if self.wire_format == "grainlike":
            return self._build_grainlike(env, step)
        return self._build_v1(env)

    def _build_grainlike(self, env, step: int) -> dict:
        """Raw arec-shaped sample: the server replays the training transforms."""
        frames = env.render()
        views = [np.ascontiguousarray(frames[name]) for name in self.view_order]
        return {
            "observation": {
                "image": np.stack(views).astype(np.uint8)[None],  # (1,V,H,W,3) native res
                "proprio": {
                    part: self._proprio_part(env, part).astype(np.float32)[None]
                    for part in self.proprio_keys
                },
            },
            "info": {
                "step": np.asarray([step], dtype=np.int64),
                "episode": np.asarray([0], dtype=np.int64),
            },
        }

    def _build_v1(self, env) -> dict:
        import cv2  # deferred: keeps module import light

        frames = env.render()
        h, w = self.image_hw
        views = [
            cv2.resize(np.ascontiguousarray(frames[name]), (w, h), interpolation=cv2.INTER_AREA)
            for name in self.view_order
        ]
        obs: dict = {
            # (V,H,W,3) -> (1,1,V,H,W,3): batch and window dims the model expects
            self.image_key: np.stack(views).astype(np.uint8)[None, None],
        }
        if self.include_view_mask:
            obs["view_mask"] = np.ones((1, 1, len(views)), dtype=bool)
        for part, key in self.proprio_keys.items():
            obs[key] = self._proprio_part(env, part)[None, None].astype(np.float32)
        if self.state_stats is not None:
            obs["state"], obs["mask"] = self._state_block(env)

        payload = obs if self.payload_key is None else {self.payload_key: obs}
        if self.include_xflow_control:
            payload["dof_ids"] = np.asarray(self.dof_ids, dtype=np.int32)
            payload["chunk_steps"] = np.asarray(self.chunk_steps, dtype=np.float32)
        return payload

    def _state_block(self, env) -> tuple[dict, dict]:
        """Pack the normalized 140-slot state the XStateEncoder consumes."""
        if not hasattr(self, "_state_norm"):
            import json

            stats = json.load(open(self.state_stats))[self.state_dataset]["proprio"]
            self._state_norm = {
                part: (np.asarray(stats[part]["mean"]).reshape(-1),
                       np.asarray(stats[part]["std"]).reshape(-1))
                for part in _STATE_PART_SLOTS
            }
        base = np.zeros(_STATE_SLOTS, dtype=np.float32)
        mask = np.zeros(_STATE_SLOTS, dtype=bool)
        for part, (lo, hi) in _STATE_PART_SLOTS.items():
            mean, std = self._state_norm[part]
            raw = self._proprio_part(env, part)
            base[lo:hi] = (raw - mean) / std
            mask[lo:hi] = True
        state = {
            "base": base[None, None],
            "id": np.asarray(_STATE_IDS, dtype=np.int32)[None, None],
            "view": np.asarray(_STATE_VIEWS, dtype=np.int32)[None, None],
        }
        return state, {"state": {"base": mask[None, None]}}

    def _proprio_part(self, env, part: str) -> np.ndarray:
        joint_pos, _joint_vel, _joint_eff, ee_pose = env.proprio()
        ee = np.asarray(ee_pose, dtype=np.float64).reshape(-1)
        if part == "joints":
            return np.asarray(joint_pos, dtype=np.float64).reshape(-1)
        if part == "gripper":
            return np.asarray([env.gripper_norm()], dtype=np.float64)
        if part == "position":
            return ee[:3]  # metres (the dataset builder converts the MCAP mm; sim is native m)
        if part == "orientation":
            from scipy.spatial.transform import Rotation

            w, x, y, z = ee[3:7]  # sim proprio quat is wxyz; scipy wants xyzw
            return Rotation.from_quat([x, y, z, w]).as_euler("xyz")
        raise ValueError(f"unknown proprio part {part!r}")


@dataclass
class ActionSpec:
    """How to interpret the action the server returns and apply it to the env.

    ``denorm_stats``: only for ``ObsSpec(wire_format="v1")`` servers — their flow head
    predicts in NORMALIZED action space (verified empirically 2026-07-08: raw j6 ≈ 0.7 vs
    actual 1.6 rad). Point it at the checkpoint's ``dataset_statistics.json`` to apply
    ``a*std + mean`` per channel where the stats' ``mask`` is True (joints AND gripper for
    xarm_sim). Leave None (no-op) for grainlike servers: their ``ActionDenormWrapper``
    already returns denormalized named parts (``actions.joints`` (1,1,H,7),
    ``actions.gripper`` (1,1,H,1)), which the dict branch of ``_joint_action_vector``
    consumes directly.
    """

    mode: Literal["ee_abs", "ee_delta", "joint_abs"] = "joint_abs"
    denorm_stats: str | None = None
    denorm_dataset: str = "xarm_sim"
    # key into the returned dict holding the action array/dict; None means the return IS it
    key: str | None = "actions"
    fallback_keys: tuple[str, ...] = (
        "action", "joint_action", "joint_actions", "proprio_single", "vector", "qpos", "joints",
    )
    # If the server returns an action chunk shaped [H, D], execute up to this many rows
    # open-loop (one control tick each) before querying the server again — the same
    # one-inference-one-trajectory semantics as eval.py's ActionChunkWrapper. 1 restores
    # the legacy closed-loop behavior (re-query every tick, execute row ``chunk_index``
    # only), which dithers in place with flow-sampled chunks: row 0 ≈ the current pose
    # every query, plus fresh sampling noise.
    open_loop_h: int = 50
    # Row executed when ``open_loop_h`` == 1 (legacy closed-loop mode).
    chunk_index: int = 0
    # scale applied to the ee position channels in "ee_abs" (mm->m by default; 1.0 if the
    # server already emits meters). Deltas ("ee_delta") are passed through unscaled.
    ee_pos_scale: float = 0.001
    # which element of the flattened action vector carries the gripper norm; None-able if
    # the server returns the gripper on a separate dict key (gripper_key)
    gripper_index: int | None = -1
    gripper_key: str | None = None
    gripper_open_threshold: float = 0.5  # gripper norm > threshold -> open (1=open)
    close_setpoint: float | None = None  # finger dof when closed; None -> robot's grasp dof

    def denorm_joint_action(self, vec8: np.ndarray) -> np.ndarray:
        """Denormalize [j0..j6, gripper] per the dataset stats (no-op when unconfigured)."""
        if self.denorm_stats is None:
            return vec8
        if not hasattr(self, "_denorm_cache"):
            import json

            stats = json.load(open(self.denorm_stats))[self.denorm_dataset]["action"]
            mean = np.concatenate([stats["joints"]["mean"], stats["gripper"]["mean"]])
            std = np.concatenate([stats["joints"]["std"], stats["gripper"]["std"]])
            mask = np.concatenate([stats["joints"]["mask"], stats["gripper"]["mask"]])
            self._denorm_cache = (mean, std, np.asarray(mask, dtype=bool))
        mean, std, mask = self._denorm_cache
        out = np.asarray(vec8, dtype=np.float64).copy()
        out[mask] = out[mask] * std[mask] + mean[mask]
        return out


class RemotePolicy:
    """Drives a crossformer served over the webpolicy websocket.

    Chunk execution: one server query buffers up to ``ActionSpec.open_loop_h`` rows of the
    returned action chunk; each :meth:`act` call applies the next buffered row and the
    server is only queried again once the buffer runs out. ``info.step`` advances one per
    :meth:`act` (i.e. per control tick), so the timestep sent at each query matches the
    episode-relative frame index the model saw in training.

    ``close_setpoint`` handling: :meth:`Manipulator.go_to_goal` / ``apply_action`` hard-code
    the closed finger dof to ``_gripper_grasp_dof``, so to honor a custom setpoint we mutate
    ``env.robot._gripper_grasp_dof`` **once** on the first :meth:`act` after a reset (reset
    itself has no env handle), not per step. ``joint_abs`` reads the same attribute for its
    closed finger target, so all three modes stay consistent.
    """

    def __init__(
        self,
        host: str,
        port: int,
        obs_spec: ObsSpec,
        action_spec: ActionSpec,
        control_every: int = 4,
        client_factory: Callable[[str, int], object] | None = None,
    ):
        if client_factory is None:
            try:
                from webpolicy.client import Client
            except ImportError as exc:  # webpolicy is not a dependency of xarm-sim
                raise ImportError(
                    "webpolicy is required for RemotePolicy; add it to xarm-sim deps: "
                    "uv add git+https://github.com/mhyatt000/webpolicy"
                ) from exc
            client_factory = Client
        self._client = client_factory(host, port)
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        # the harness steps this policy every ``control_every``-th physics step (the
        # position controller holds the last target in between), so a served model runs
        # at the 30 Hz training cadence (record_every=4 physics steps) rather than 120 Hz
        self.control_every = control_every
        self._setpoint_applied = False
        self._t = 0  # episode-relative control tick, sent as info.step in grainlike payloads
        self._pending: list[tuple[np.ndarray, bool]] = []  # buffered (action, is_open) rows

    def reset(self) -> None:
        self._client.reset()
        self._setpoint_applied = False  # re-apply close_setpoint on the next act
        self._t = 0
        self._pending = []

    def act(self, env) -> None:
        spec = self.action_spec
        if spec.close_setpoint is not None and not self._setpoint_applied:
            env.robot._gripper_grasp_dof = spec.close_setpoint
            self._setpoint_applied = True

        if not self._pending:
            result = self._client.step(self.obs_spec.build(env, self._t))
            self._pending = self._chunk_rows(result)
        action, is_open = self._pending.pop(0)
        self._t += 1

        if spec.mode == "ee_abs":
            pos = action[:3] * spec.ee_pos_scale
            quat = action[3:7]
            pose = torch.as_tensor(
                np.concatenate([pos, quat])[None, :], dtype=torch.float32, device=env.device
            )
            env.robot.go_to_goal(pose, open_gripper=is_open)
        elif spec.mode == "ee_delta":
            delta = torch.as_tensor(action[:6][None, :], dtype=torch.float32, device=env.device)
            env.robot.apply_action(delta, open_gripper=is_open)
        elif spec.mode == "joint_abs":
            robot = env.robot
            q_pos = robot._robot_entity.get_qpos()
            q_pos = (q_pos.unsqueeze(0) if q_pos.ndim == 1 else q_pos).clone()
            q_pos[:, robot._arm_dof_idx] = torch.as_tensor(
                action[:7], dtype=q_pos.dtype, device=q_pos.device
            )
            finger = robot._gripper_open_dof if is_open else robot._gripper_grasp_dof
            q_pos[:, robot._fingers_dof] = finger
            robot._robot_entity.control_dofs_position(position=q_pos)
        else:
            raise ValueError(f"unknown action mode {spec.mode!r}")

    @property
    def done(self) -> bool:
        return False

    def _chunk_rows(self, result) -> list[tuple[np.ndarray, bool]]:
        """Rows of the returned chunk to execute open-loop, as (action, gripper_open)."""
        spec = self.action_spec
        if spec.open_loop_h <= 1:
            indices: list[int] = [spec.chunk_index]  # legacy closed-loop: one fixed row
        else:
            indices = list(range(min(spec.open_loop_h, self._n_rows(result))))
        rows = []
        for idx in indices:
            action = self._action_vector(result, idx)
            rows.append((action, self._gripper_open(result, action, idx)))
        return rows

    def _n_rows(self, result) -> int:
        payload = self._action_payload(result)
        if isinstance(payload, dict):
            payload = payload.get("joints", next(iter(payload.values())))
        arr = np.asarray(payload)
        return arr.reshape(-1, arr.shape[-1]).shape[0] if arr.ndim >= 2 else 1

    def _action_vector(self, result, idx: int) -> np.ndarray:
        if self.action_spec.mode == "joint_abs":
            return self.action_spec.denorm_joint_action(self._joint_action_vector(result, idx))
        return self._row_vector(self._action_payload(result), idx)

    def _action_payload(self, result):
        spec = self.action_spec
        if not isinstance(result, dict) or spec.key is None:
            return result
        if spec.key in result:
            return result[spec.key]
        if "joints" in result or "gripper" in result:
            return result
        for key in spec.fallback_keys:
            if key in result:
                return result[key]
        raise KeyError(
            f"action key {spec.key!r} not found; tried fallbacks {spec.fallback_keys}; "
            f"available keys={sorted(result)}"
        )

    def _row_vector(self, value, idx: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim >= 2:
            rows = arr.reshape(-1, arr.shape[-1])
            if not -len(rows) <= idx < len(rows):
                raise IndexError(f"row {idx} out of range for action chunk with {len(rows)} rows")
            return rows[idx].reshape(-1)
        return arr.reshape(-1)

    def _joint_action_vector(self, result, idx: int) -> np.ndarray:
        payload = self._action_payload(result)
        if isinstance(payload, dict):
            if "joints" in payload:
                joints = self._row_vector(payload["joints"], idx)
            else:
                joints = np.asarray([self._row_vector(payload[f"joint{i}"], idx)[0] for i in range(1, 8)])
            if "gripper" in payload:
                gripper = self._row_vector(payload["gripper"], idx)
            elif self.action_spec.gripper_key and self.action_spec.gripper_key in payload:
                gripper = self._row_vector(payload[self.action_spec.gripper_key], idx)
            else:
                gripper = np.asarray([1.0])
            return np.concatenate([joints[:7], [float(gripper[0])]])

        slots = self._row_vector(payload, idx)
        dof_ids = result.get("dof_ids") if isinstance(result, dict) else None
        if dof_ids is not None:
            ids = np.asarray(dof_ids).reshape(-1)
            ordered = np.full(8, np.nan, dtype=np.float64)
            for slot, dof_id in enumerate(ids[: len(slots)]):
                did = int(dof_id)
                if 1 <= did <= 7:
                    ordered[did - 1] = slots[slot]
                elif did == 8:
                    ordered[7] = slots[slot]
            if np.isfinite(ordered).all():
                return ordered

        if slots.size < 8:
            raise ValueError(f"joint_abs action needs 8 values, got shape {np.asarray(payload).shape}")
        return slots[:8]

    def _gripper_open(self, result, action: np.ndarray, idx: int) -> bool:
        spec = self.action_spec
        if spec.gripper_key is not None and isinstance(result, dict):
            payload = self._action_payload(result)
            if isinstance(payload, dict) and spec.gripper_key in payload:
                g = float(self._row_vector(payload[spec.gripper_key], idx)[0])
            elif spec.gripper_key in result:
                g = float(self._row_vector(result[spec.gripper_key], idx)[0])
            else:
                g = float(action[spec.gripper_index]) if spec.gripper_index is not None else 1.0
        elif spec.gripper_index is not None:
            g = float(action[spec.gripper_index])
        else:
            return True  # no gripper channel configured -> leave the gripper open
        return g > spec.gripper_open_threshold


class ScriptedEvalPolicy:
    """Weld-free wrapper around the repo's scripted lift/stack policies.

    Runs the scripted waypoint trajectory with **no** ``env.grasp_lock()`` call — friction
    holds the cube (Phase-0 feasibility test) — behind the same interface as
    :class:`RemotePolicy`. ``close_setpoint`` is applied by setting
    ``env.robot._gripper_grasp_dof`` in :meth:`reset`, the same knob ``go_to_goal`` reads.

    Contract: the scripted policy caches the cube pose when its own ``reset()`` runs, and the
    harness moves the cube *after* ``env.reset()``. So the inner policy is (re-)constructed
    inside :meth:`reset`, and :meth:`reset` must be called **after** the cube is placed —
    once per episode.
    """

    def __init__(
        self,
        env,
        task: Literal["lift", "stack"],
        steps_per_segment: int = 108,
        grasp_tcp_offset: float = 0.018,
        close_setpoint: float | None = None,
        tail_steps: int = 36,  # ~0.3 s at 120 Hz, matching the generator's release_tail_s=0.3
    ):
        self._env = env
        self.task = task
        self.steps_per_segment = steps_per_segment
        self.grasp_tcp_offset = grasp_tcp_offset
        self.close_setpoint = close_setpoint
        self.tail_steps = tail_steps
        # the scripted waypoint trajectory is authored at the physics rate, so the harness
        # steps it every physics step (unlike RemotePolicy's decimated 30 Hz cadence)
        self.control_every = 1
        self._policy = None
        self._step_count = 0
        self._end_step = 0

    def reset(self) -> None:
        env = self._env
        if self.close_setpoint is not None:
            env.robot._gripper_grasp_dof = self.close_setpoint
        # lazy import keeps this module free of Genesis until an episode actually runs
        if self.task == "lift":
            from xsim.scripted_lift_policy import ScriptedLiftPolicy as Policy
        elif self.task == "stack":
            from xsim.scripted_stack_policy import ScriptedStackPolicy as Policy
        else:
            raise ValueError(f"unknown task {self.task!r}; expected 'lift' or 'stack'")
        # constructed here (not in __init__) so it reads the freshly placed cube pose
        self._policy = Policy(
            env, steps_per_segment=self.steps_per_segment, grasp_tcp_offset=self.grasp_tcp_offset
        )
        self._policy.reset()
        self._step_count = 0
        self._end_step = self._policy.release_step + self.tail_steps

    def act(self, env) -> None:
        cmd = self._policy.step()
        env.robot.go_to_goal(cmd.pose, open_gripper=cmd.open_gripper)
        self._step_count += 1

    @property
    def done(self) -> bool:
        return self._policy is not None and self._step_count > self._end_step
