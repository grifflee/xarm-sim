"""Temporary Gymnasium adapter for the Genesis ``TaskEnv``.

``TaskEnv`` is not gym-conformant yet: ``reset()`` returns ``None`` and ``step()`` takes no
action and returns ``None``. Rather than rewrite it now, this adapter presents a gym face
over it *without touching TaskEnv's code* — a stopgap until the real refactor:

    reset(seed=, options=) -> obs           # obs = served-policy wire format (ObsSpec)
    step(action)           -> (obs, reward, done, info)

Reuse is deliberately minimal:

- :class:`~xsim.eval_policy.ObsSpec` builds the observation the served crossformer expects,
- :func:`xsim.success.episode_result` decides success for reward/done.

The action is applied directly here (joint-position control) — no ``RemotePolicy``. The
served policy's action is expected as either a ``{"joints": (...,7), "gripper": (...,1)}``
dict (the grainlike server's denormalized parts) or a flat ``[j0..j6, gripper]`` vector;
the closed-finger setpoint is honored via ``robot._gripper_grasp_dof``.

Composed eval stack::

    env = GenesisGymAdapter(TaskEnv(cfg))
    env = VideoRecordWrapper(env, ...)      # optional
    env = ActionChunkWrapper(env, h=...)

Caveat: ``step`` executes one control tick per call. A served crossformer returns a chunk
shaped ``(1, 1, H, D)`` whose leading axis is the batch, not ``H``, so wrapping this in
``ActionChunkWrapper`` runs one action per inference, not ``H`` open-loop. Real open-loop
chunk playback needs the act dict transposed to a horizon-leading axis — a refactor concern.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from xsim.eval_policy import ObsSpec


class GenesisGymAdapter:
    def __init__(
        self,
        env,
        obs_spec: ObsSpec | None = None,
        *,
        control_every: int | None = None,
        max_control_steps: int | None = 600,   # episode time limit in CONTROL steps; None = off
        close_setpoint: float = 0.58,
        gripper_open_threshold: float = 0.5,  # gripper norm > threshold -> open (1 = open)
        lift_threshold: float = 0.05,
        deliver_radius: float = 0.12,
        stack_xy_tol: float = 0.02,
        stack_z_tol: float = 0.008,
    ):
        self.env = env
        self.obs_spec = obs_spec or ObsSpec()
        self.close_setpoint = close_setpoint
        self.gripper_open_threshold = gripper_open_threshold

        self._control_every = int(control_every or env.cfg.record_every)
        # limit is in control steps (== adapter.step calls). ActionChunkWrapper calls step
        # once per action in the chunk, so this counts individual actions, not policy queries.
        self._max_steps = max_control_steps
        # duck-typed cfg for xsim.success.episode_result (only these fields are read)
        self._score_cfg = SimpleNamespace(
            task=env.cfg.task, lift_threshold=lift_threshold, deliver_radius=deliver_radius,
            stack_xy_tol=stack_xy_tol, stack_z_tol=stack_z_tol,
        )
        self._t = 0
        self._steps = 0
        self._start_z = 0.0
        self._max_rise = 0.0
        self._success = False
        self._setpoint_applied = False

    # -- gym API --
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Any:
        options = options or {}
        self._pin_spawn(options)          # before reset so camera-visibility redraw is correct
        self.env.reset(seed=seed)
        self._place_deterministic(options)
        self._t = self._steps = 0
        self._start_z = float(self.env.cube_pos()[2])
        self._max_rise = 0.0
        self._success = False
        self._setpoint_applied = False
        return self.obs_spec.build(self.env, self._t)

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        self._apply(action)                             # one control tick
        for _ in range(self._control_every):            # hold the command over the sim substeps
            self.env.step()
        self._t += 1
        self._steps += 1

        cube = self.env.cube_pos()
        self._max_rise = max(self._max_rise, float(cube[2]) - self._start_z)
        obs = self.obs_spec.build(self.env, self._t)
        reward, done, info = self._score(cube)
        return obs, reward, done, info

    def render(self) -> dict:
        return self.env.render()

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    # -- action application (joint-position control, no RemotePolicy) --
    def _apply(self, action: Any) -> None:
        robot = self.env.robot
        if not self._setpoint_applied:  # go_to_goal/joint control read this as the closed dof
            robot._gripper_grasp_dof = self.close_setpoint
            self._setpoint_applied = True

        joints, gripper = _joints_and_gripper(action)
        is_open = gripper > self.gripper_open_threshold

        q_pos = robot._robot_entity.get_qpos()
        q_pos = (q_pos.unsqueeze(0) if q_pos.ndim == 1 else q_pos).clone()
        q_pos[:, robot._arm_dof_idx] = torch.as_tensor(
            joints[:7], dtype=q_pos.dtype, device=q_pos.device)
        q_pos[:, robot._fingers_dof] = robot._gripper_open_dof if is_open else robot._gripper_grasp_dof
        robot._robot_entity.control_dofs_position(position=q_pos)

    # -- internals --
    def _score(self, cube) -> tuple[float, bool, dict]:
        from xsim.success import episode_result  # lazy: pulls in Genesis via task_env

        res = episode_result(self.env, self._score_cfg, self._max_rise)
        success = bool(res["success"])
        fell = float(cube[2]) < self.env.cfg.table.top_z - 0.05
        flew = math.hypot(float(cube[0]), float(cube[1])) > 0.9
        timeout = self._max_steps is not None and self._steps >= self._max_steps
        reward = float(success and not self._success)   # fire once, on first success
        self._success = self._success or success
        info = dict(res)
        info.update(success=success, fell=fell, flew=flew, timeout=timeout)
        return reward, (success or fell or flew or timeout), info

    def _pin_spawn(self, options: dict) -> None:
        """Force reset's sampling ranges to a specific grid point (uniform(a, a) == a)."""
        if "cube_xy" not in options:
            return
        x, y = options["cube_xy"]
        if self.env.cfg.task == "stack" and "green_xy" in options:
            gx, gy = options["green_xy"]
            self.env.cfg.stack.free_placement = False
            self.env.cfg.stack.green_x = (gx, gx)
            self.env.cfg.stack.green_y = (gy, gy)
            self.env.cfg.stack.red_dx = (x - gx, x - gx)
            self.env.cfg.stack.red_dy = (y - gy, y - gy)
        else:
            self.env.cfg.rectangle_x = (x, x)
            self.env.cfg.rectangle_y = (y, y)

    def _place_deterministic(self, options: dict) -> None:
        """After reset, re-place cubes at a fixed yaw and pin the lift drop target."""
        if "cube_xy" in options and "cube_yaw" in options:
            x, y = options["cube_xy"]
            self.env._place_cube(self.env.cube, x, y, options["cube_yaw"])
            self.env._cube_yaw = options["cube_yaw"]
            if self.env.cfg.task == "stack" and "green_xy" in options:
                gx, gy = options["green_xy"]
                self.env._place_cube(self.env.cube2, gx, gy, options.get("green_yaw", 0.0))
                self.env._green_yaw = options.get("green_yaw", 0.0)
        if self.env.cfg.task != "stack" and "drop_xy" in options:
            self.env.current_drop_xy = tuple(options["drop_xy"])

    def __getattr__(self, name: str) -> Any:
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)


def _joints_and_gripper(action: Any) -> tuple[np.ndarray, float]:
    """Pull 7 arm-joint targets + a scalar gripper norm from the served action.

    Accepts a ``{"joints": (...,7), "gripper": (...,1)}`` dict (grainlike parts) or a flat
    ``[j0..j6, gripper]`` vector. When a horizon axis is present, the first control step is
    used (``reshape(-1, 7)[0]``).
    """
    if isinstance(action, dict):
        joints = np.asarray(action["joints"], dtype=np.float64).reshape(-1, 7)[0]
        g = action.get("gripper")
        gripper = float(np.asarray(g, dtype=np.float64).reshape(-1)[0]) if g is not None else 1.0
        return joints, gripper
    vec = np.asarray(action, dtype=np.float64).reshape(-1)
    return vec[:7], (float(vec[7]) if vec.size > 7 else 1.0)
