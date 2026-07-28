"""CPU-only tests for Milestone A's explicit environment-batch state contract."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from xsim.task_env import TaskEnv, TaskEnvCfg, _as_batch_np


class _Entity:
    def __init__(self):
        self.qpos = torch.arange(13, dtype=torch.float64).reshape(1, 13)

    def get_dofs_position(self):
        return self.qpos

    def get_dofs_velocity(self):
        return self.qpos + 20

    def get_dofs_force(self):
        return self.qpos + 40


class _Cube:
    def get_pos(self):
        return torch.tensor([[0.3, -0.1, 0.02]], dtype=torch.float64)


class BatchContractTest(unittest.TestCase):
    def _env(self):
        env = TaskEnv.__new__(TaskEnv)
        env.n_envs = 1
        env.robot_cfg = {"gripper_close_dof": 0.85}
        env.robot = SimpleNamespace(
            _robot_entity=_Entity(),
            _arm_dof_dim=7,
            ee_pose=torch.arange(7, dtype=torch.float64).reshape(1, 7),
        )
        env.cube = _Cube()
        env.cube2 = _Cube()
        env._cube_yaw = 0.2
        env._green_yaw = -0.1
        env.current_drop_xy = (0.35, 0.0)
        env.episode_extrinsics = {"low": np.eye(4)}
        return env

    def test_config_defaults_to_one_environment(self):
        self.assertEqual(TaskEnvCfg().n_envs, 1)

    def test_single_vector_gains_environment_axis(self):
        self.assertEqual(_as_batch_np(np.arange(7), width=7).shape, (1, 7))

    def test_state_accessors_keep_environment_axis(self):
        env = self._env()
        self.assertEqual(env.cube_pos_batch().shape, (1, 3))
        self.assertEqual(env.green_pos_batch().shape, (1, 3))
        self.assertEqual(env.gripper_norm_batch().shape, (1,))
        self.assertEqual(env.cube_yaw_batch().shape, (1,))
        self.assertEqual(env.green_yaw_batch().shape, (1,))
        self.assertEqual(env.drop_xy_batch().shape, (1, 2))
        for value in env.proprio_batch():
            self.assertEqual(value.shape, (1, 7))
        self.assertEqual(env.extrinsics_batch()["low"].shape, (1, 4, 4))

    def test_batch_helper_preserves_existing_batch(self):
        value = np.arange(14).reshape(2, 7)
        np.testing.assert_array_equal(_as_batch_np(value, width=7), value)


if __name__ == "__main__":
    unittest.main()
