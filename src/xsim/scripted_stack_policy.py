"""Scripted waypoint policy for the stack task (red cube onto the green cube).

Same skeleton as the lift protocol through the grasp — home -> above the red cube
(yaw-aligned, straight down) -> vertical plunge -> close -> weld -> lift — then instead
of releasing at transport height it lowers the red cube onto the green one and opens
with the cube's bottom a few millimetres above the green top, so the block sets down
instead of free-falling (the human stack_right demos release right at placement).
Recording ends shortly after the open, matching the lift protocol.
"""

from __future__ import annotations

import math

import torch

from xsim.task_env import BLOCK_SIZE
from xsim.scripted_lift_policy import ScriptedLiftPolicy, _nearest_side_grasp_quat, _yawed_top_down_quat

# Per-segment duration weights (x steps_per_segment): approach, plunge, close, lift,
# transport, lower-to-place, settled hold before release. Approach/plunge/close/lift
# match the lift protocol; the lower is deliberately slower than the lift so the
# placement is controlled, and the hold lets the controller settle so the recorded
# opening tail is stationary.
STACK_SEGMENT_WEIGHTS = (2.6, 0.8, 0.8, 0.4, 1.0, 0.8, 0.6)


class ScriptedStackPolicy(ScriptedLiftPolicy):
    def __init__(self, env, steps_per_segment: int = 40, grasp_tcp_offset: float = 0.018,
                 approach_height: float = 0.12, lift_height: float = 0.09,
                 segment_weights: tuple[float, ...] = STACK_SEGMENT_WEIGHTS):
        super().__init__(env, steps_per_segment=steps_per_segment,
                         grasp_tcp_offset=grasp_tcp_offset, approach_height=approach_height,
                         lift_height=lift_height, segment_weights=segment_weights)

    def reset(self) -> None:
        env = self.env
        device = env.device
        ee = env.robot.ee_pose.clone().reshape(-1)          # [7]
        home_quat = ee[3:7].clone()
        cube = torch.as_tensor(env.cube_pos_batch()[0], device=device, dtype=ee.dtype)   # red [3]
        green = torch.as_tensor(env.green_pos_batch()[0], device=device, dtype=ee.dtype)  # target [3]
        top_z = env.cfg.table.top_z

        grasp_quat = torch.as_tensor(
            _nearest_side_grasp_quat(float(env.cube_yaw_batch()[0]), home_quat),
            device=device,
            dtype=ee.dtype,
        )
        if torch.dot(home_quat, grasp_quat) < 0:  # avoid lerping across the antipode
            grasp_quat = -grasp_quat

        def pose(xyz, quat):
            return torch.cat([torch.as_tensor(xyz, device=device, dtype=ee.dtype), quat]).reshape(1, 7)

        grasp_z = top_z + self.grasp_tcp_offset
        lift_z = grasp_z + self.lift_height
        # place: the red cube's bottom ends up place_clearance above the green top, so
        # the release is a set-down, not a drop. The TCP sits grasp_tcp_offset above
        # where the cube bottom meets the surface, exactly as it did at the grasp.
        place_z = top_z + BLOCK_SIZE + env.cfg.stack.place_clearance + self.grasp_tcp_offset

        # Face alignment of the stacked pair: the red cube is welded to the TCP, so
        # twisting the wrist by the nearest 90-degree-equivalent yaw difference between
        # the two cubes squares red's faces with green's. The twist happens over the
        # transport segment (the lerp renormalizes quats), never exceeding 45 degrees.
        delta = (
            float(env.green_yaw_batch()[0]) - float(env.cube_yaw_batch()[0])
        ) % (math.pi / 2.0)
        if delta >= math.pi / 4.0:
            delta -= math.pi / 2.0
        # recover the chosen grasp yaw from the (0, cos(h), sin(h), 0) top-down quat
        grasp_yaw = 2.0 * math.atan2(float(grasp_quat[2]), float(grasp_quat[1]))
        place_quat = torch.as_tensor(
            _yawed_top_down_quat(grasp_yaw + delta), device=device, dtype=ee.dtype
        )
        if torch.dot(grasp_quat, place_quat) < 0:  # avoid lerping across the antipode
            place_quat = -place_quat

        above = pose([cube[0], cube[1], grasp_z + self.approach_height], grasp_quat)
        at = pose([cube[0], cube[1], grasp_z], grasp_quat)
        lift = pose([cube[0], cube[1], lift_z], grasp_quat)
        over_green = pose([green[0], green[1], lift_z], place_quat)
        place = pose([green[0], green[1], place_z], place_quat)

        self.waypoint_names = [
            "home",
            "above_block",
            "at_block_open",
            "at_block_closed",
            "lift",
            "over_green",
            "place",
            "place_settled",
        ]

        # (target_pose, open_gripper) waypoints
        self._waypoints = [
            (ee.reshape(1, 7), True),   # home
            (above, True),              # high above the red cube, yaw-aligned
            (at, True),                 # vertical plunge
            (at, False),                # close (grasp)
            (lift, False),              # lift quickly
            (over_green, False),        # transport over the green cube
            (place, False),             # lower onto the green cube
            (place, False),             # settle at the placement before release
        ]
        n_seg = len(self._waypoints) - 1
        weights = self.segment_weights if len(self.segment_weights) == n_seg else (1.0,) * n_seg
        self._segment_steps = [max(2, round(w * self.steps_per_segment)) for w in weights]
        self._commands = self._make_generator()
        # step 0 is the initial hold; segments follow in order (close is segment 3)
        self.grasp_lock_step = 1 + sum(self._segment_steps[:3])
        self.release_step = 1 + sum(self._segment_steps)
        self.n_steps = self.release_step
