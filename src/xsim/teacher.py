"""Scripted teacher behind the Gymnasium policy face.

Wraps :class:`~xsim.scripted_lift_policy.ScriptedLiftPolicy` so the expert speaks the same
contract as the served student (``reset() -> None``, ``step(obs) -> action``) and emits the
same action space — one ``[q0..q6, gripper]`` joint-position vector per CONTROL step —
so teacher and student actions flow through the same ``GenesisGymAdapter.step()``. The
Cartesian waypoint command is converted to joints with the same IK call ``go_to_goal``
uses; ``gripper`` is 1.0 (open) / 0.0 (closed), matching the adapter's threshold.

The teacher is *privileged*: it plans from live env state (TCP, cube pose) and ignores
``obs``. ``reset()`` plans from the CURRENT scene state, which gives it both roles:

- **reference** — reset right after ``env.reset()`` plans the clean demonstration;
- **recovery** (DAgger takeover) — reset mid-episode re-plans from wherever the student
  left things. If the cube is held in the air at reset, a release preamble runs first
  (open, retreat upward, let the cube settle) and the grasp is then planned fresh from
  the settled cube pose. The live cube yaw is re-read from the cube quaternion so a
  rotated cube is still grasped face-on.

The scripted trajectory is authored per-step; at the 30 Hz control rate,
``steps_per_segment=27`` reproduces the speed of the generator's 108 @ 120 Hz.
"""

from __future__ import annotations

import numpy as np
import torch

from xsim.scripted_lift_policy import ScriptedLiftPolicy
from xsim.success import _yaw_from_quat_wxyz

# names for the scripted lift policy's 6 waypoint segments, in generator order
LIFT_SEGMENTS = ("approach", "plunge", "close", "lift", "transport", "hold")


class ScriptedTeacherPolicy:
    def __init__(
        self,
        env,
        steps_per_segment: int = 27,
        grasp_tcp_offset: float = 0.018,
        tail_steps: int = 9,        # open-gripper hold after release (~0.3 s at 30 Hz)
        release_steps: int = 15,    # recovery preamble: hold open in place (~0.5 s)
        retreat_steps: int = 30,    # recovery preamble: retreat up + cube settle (~1 s)
        retreat_height: float = 0.12,
        held_cube_dist: float = 0.08,   # cube within this of the TCP counts as held
    ):
        self.env = env
        self.steps_per_segment = steps_per_segment
        self.grasp_tcp_offset = grasp_tcp_offset
        self.tail_steps = tail_steps
        self.release_steps = release_steps
        self.retreat_steps = retreat_steps
        self.retreat_height = retreat_height
        self.held_cube_dist = held_cube_dist
        self._policy: ScriptedLiftPolicy | None = None
        self._phase = "task"          # release -> retreat -> task (preamble only on takeover)
        self._phase_step = 0
        self._task_step = 0
        self._end_step = 0
        self._hold_action: np.ndarray | None = None
        self.segment = "approach"     # segment of the most recent step() command

    # -- GymPolicy API --
    def reset(self) -> None:
        self._phase_step = 0
        if self._cube_held():
            # can't re-plan a grasp around a cube that is in the gripper: release it,
            # retreat clear, and plan fresh once it has settled (at the end of `retreat`)
            self._phase = "release"
            self._policy = None
        else:
            self._phase = "task"
            self._plan()

    def step(self, obs=None) -> np.ndarray:
        if self._phase == "release":
            self.segment = "release"
            if self._hold_action is None:
                # hold the current arm pose, gripper open — the cube drops in place
                arm = np.asarray(self.env.proprio()[0], dtype=np.float32).reshape(-1)[:7]
                self._hold_action = np.concatenate([arm, [1.0]]).astype(np.float32)
            action = self._hold_action
            self._phase_step += 1
            if self._phase_step >= self.release_steps:
                self._phase, self._phase_step = "retreat", 0
                pose = self.env.robot.ee_pose.clone().reshape(1, 7)
                pose[:, 2] += self.retreat_height
                self._hold_action = self._ik_action(pose, open_gripper=True)
            return action
        if self._phase == "retreat":
            self.segment = "retreat"
            self._phase_step += 1
            action = self._hold_action
            if self._phase_step >= self.retreat_steps:  # cube settled: plan the real task
                self._phase, self._hold_action = "task", None
                self._plan()
            return action

        cmd = self._policy.step()
        self._task_step += 1
        seg = int(np.searchsorted(self._bounds, max(self._task_step, 1), side="left"))
        self.segment = LIFT_SEGMENTS[min(seg, len(LIFT_SEGMENTS) - 1)]
        return self._ik_action(cmd.pose, open_gripper=cmd.open_gripper)

    @property
    def done(self) -> bool:
        """Trajectory (incl. the open tail) fully played out. The env owns episode `done`."""
        return self._phase == "task" and self._policy is not None \
            and self._task_step > self._end_step

    # -- internals --
    def _plan(self) -> None:
        env = self.env
        # ScriptedLiftPolicy aligns the grasp to env.cube_yaw(), which is only set at
        # placement; after the student shoved the cube around, re-read the live yaw
        env._cube_yaw = float(_yaw_from_quat_wxyz(env.cube.get_quat().cpu()))
        self._policy = ScriptedLiftPolicy(
            env, steps_per_segment=self.steps_per_segment,
            grasp_tcp_offset=self.grasp_tcp_offset)
        self._policy.reset()
        self._bounds = np.cumsum(self._policy._segment_steps)
        self._task_step = 0
        self._end_step = self._policy.release_step + self.tail_steps

    def _ik_action(self, pose: torch.Tensor, open_gripper: bool) -> np.ndarray:
        """The same IK solve as Manipulator.go_to_goal, returned as a joint action."""
        robot = self.env.robot
        init_qpos = None
        if robot._args.get("ik_init_at_home", False):
            init_qpos = robot._init_qpos.unsqueeze(0).expand(pose.shape[0], -1)
        q_pos = robot._robot_entity.inverse_kinematics(
            link=robot._ee_link,
            pos=pose[:, :3],
            quat=pose[:, 3:7],
            init_qpos=init_qpos,
            max_samples=robot._args.get("ik_max_samples", 50),
            max_solver_iters=robot._args.get("ik_max_solver_iters", 20),
            damping=robot._args.get("ik_damping", 0.01),
            dofs_idx_local=robot._arm_dof_idx,
        )
        arm = np.asarray(q_pos[:, robot._arm_dof_idx].cpu(), dtype=np.float32).reshape(-1)[:7]
        return np.concatenate([arm, [1.0 if open_gripper else 0.0]]).astype(np.float32)

    def _cube_held(self) -> bool:
        cube = np.asarray(self.env.cube_pos(), dtype=np.float64)
        if float(cube[2]) < self.env.cfg.table.top_z + 0.05:
            return False
        tcp = np.asarray(self.env.robot.ee_pose.cpu(), dtype=np.float64).reshape(-1)[:3]
        return float(np.linalg.norm(cube - tcp)) < self.held_cube_dist \
            and self.env.gripper_norm() < 0.6
