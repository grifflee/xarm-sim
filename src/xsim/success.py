"""Single shared definition of task success for both the data generator and eval harness.

``episode_result`` is the one place that decides whether a lift/stack episode succeeded, so
the offline data generator (``scripts/generate_task_dataset.py``) and the upcoming eval
harness score episodes identically.

``cfg`` is duck-typed: it only needs the attributes ``task``, ``lift_threshold``,
``deliver_radius``, ``stack_xy_tol`` and ``stack_z_tol`` (the generator supplies them via its
``Config`` dataclass; any object with those fields works).
"""

from __future__ import annotations

import math

import numpy as np

from xsim.task_env import BLOCK_SIZE, TaskEnv


def _yaw_from_quat_wxyz(quat) -> float:
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(-1)[:4]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def episode_result(env: TaskEnv, cfg, max_rise: float) -> dict:
    """Task-specific success stats, computed after the unrecorded settle."""
    cube_end = env.cube_pos_batch()[0]
    lifted = max_rise >= cfg.lift_threshold
    if cfg.task == "stack":
        green = env.green_pos_batch()[0]
        stack_z = env.cfg.table.top_z + 1.5 * BLOCK_SIZE
        xy_err = float(np.linalg.norm(cube_end[:2] - green[:2]))
        z_err = float(cube_end[2] - stack_z)
        # face alignment of the settled pair, wrapped to the cube's 90-degree symmetry
        yaw_err = (_yaw_from_quat_wxyz(env.cube.get_quat().cpu())
                   - _yaw_from_quat_wxyz(env.cube2.get_quat().cpu())) % (math.pi / 2.0)
        if yaw_err >= math.pi / 4.0:
            yaw_err -= math.pi / 2.0
        stacked = xy_err <= cfg.stack_xy_tol and abs(z_err) <= cfg.stack_z_tol
        return {
            "max_rise": max_rise, "lifted": lifted, "stack_xy_err": xy_err,
            "stack_z_err": z_err, "stack_yaw_err_deg": abs(math.degrees(yaw_err)),
            "stacked": stacked, "success": lifted and stacked,
            "green_pos": [float(v) for v in green],
        }
    drop = env.drop_xy_batch()[0]
    deliver_dist = float(np.linalg.norm(cube_end[:2] - drop))
    delivered = deliver_dist <= cfg.deliver_radius and float(cube_end[2]) < 0.05
    return {
        "max_rise": max_rise, "lifted": lifted, "deliver_dist": deliver_dist,
        "delivered": delivered, "success": lifted and delivered,
        "drop_target": [float(drop[0]), float(drop[1])],
    }
