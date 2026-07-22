"""Reactive scripted lift expert: phase from live state, never from a clock.

Ported from ``upstream-main:src/xsim/suite/policies/lift_expert.py`` into this
fork's flat single-env layout. The waypoint teacher (``ScriptedTeacherPolicy``
/ ``ScriptedLiftPolicy``) plans at reset and paces segments on a shared tick
schedule, so its label at a given instant depends on hidden schedule state --
unfittable for a student that only sees the instantaneous observation. This
expert keeps the same grasp geometry (above -> at -> close -> lift -> release,
face-aligned yaw) but derives the phase every tick from the measured world
state (TCP pose, cube pose, gripper opening). Labels are a function of state by
construction: the plan waits for the arm, and a fumbled cube demotes the phase
so the aggregate collects recovery corrections. After a completed lift the
expert releases at height; the drop demotes the env back to APPROACH, so
fixed-horizon rollouts cycle grasp -> lift -> drop -> re-grasp.

The only hidden state is the phase itself plus a per-episode close-attempt tick
counter (abort a grasp that isn't seating).

This fork executes ABSOLUTE joints, so only the default joint-space mode is
kept -- upstream's ``cartesian`` / ``delta`` / orientation variants are dropped.
``act()`` returns one ``np.ndarray`` of shape ``(8,)`` = absolute
``[j0..j6, gripper_norm]`` float32. Both ``reset()`` and ``act()`` ignore the
observation; the expert reads privileged env state directly.

Gripper polarity (correctness gate -- traced end-to-end in this fork)
=====================================================================
Three conventions coexist; the expert lives entirely in NORM space and matches
upstream unchanged (open = 1.0, closed = 0.0). The dof inversion is handled for
us on both the execute and the record side, so no polarity flip is needed.

1. Finger-joint dof (physics):   0.0 = OPEN (fingers apart), 0.85 = hard closed,
   0.58 = the task grasp setpoint. See task_env.py:196-201.

2. Action gripper component ``action[7]`` (NORM, what this expert emits):
   1.0 = OPEN, 0.0 = CLOSED. The env's execution path thresholds it and maps to
   the dof (inverting the polarity for us):
     - genesis_gym.py:121  ``is_open = gripper > 0.5``
     - genesis_gym.py:127  fingers <- gripper_open_dof (0.0) if is_open
                                       else gripper_grasp_dof (0.58)
   This is the SAME path the waypoint teacher uses: teacher.py:143 emits
   ``[.., 1.0 if open_gripper else 0.0]``. So GRIPPER_OPEN=1.0 / GRIPPER_CLOSED=0.0
   drive the fingers correctly with no inversion.

3. Recorded MCAP ``norm`` (NORM): 1.0 = OPEN, 0.0 = CLOSED. Crucially the value
   written to the MCAP is NOT the commanded action -- it is the MEASURED finger
   dof read back after the physics step:
     - mcap_record.py:60  ``gripper_norm = env.gripper_norm()``
     - task_env.py:970-974 ``gripper_norm() = clip(1 - dof/0.85, 0, 1)`` -> open
                            dof 0.0 reads 1.0; grasp dof 0.58 reads ~0.318.
     - mcap_writer.py:157-159 writes ``norm`` as-is and ``rad = norm * 0.85``.
   So the recorded 1=open convention matches the existing training batches as
   long as the expert drives the fingers to the right dof -- which it does by
   emitting norm 1.0=open / 0.0=closed (item 2).

Because the expert already agrees with upstream's GRIPPER_OPEN/GRIPPER_CLOSED,
its ``held`` gate reads ``env.gripper_norm()`` (1=open, 0=closed) directly: a
seated 31.75 mm cube plateaus the finger dof mid-travel, so the measured norm
sits in the (grip_lo, grip_hi) band; closed-on-air drives it below grip_lo and
a fully open gripper sits above grip_hi.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from xsim.scripted_lift_policy import _nearest_side_grasp_quat

# Gripper NORM convention (matches the waypoint teacher / MCAP writer): see the
# module docstring. 1.0 drives the fingers open, 0.0 drives them to the grasp
# setpoint, and both read back through env.gripper_norm() with the same polarity.
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0

# Grasp geometry, ported verbatim from upstream lift.py (identical to this fork's
# ScriptedLiftPolicy defaults). The cube is 31.75 mm tall (task_env.BLOCK_SIZE);
# keep the TCP near its upper half so the fingers don't clip through the block.
GRASP_TCP_OFFSET = 0.018
APPROACH_HEIGHT = 0.12
LIFT_HEIGHT = 0.09

APPROACH, DESCEND, CLOSE, LIFT, RELEASE = range(5)


def _slerp(qa: np.ndarray, qb: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc quaternion slerp (wxyz), nlerp fallback when near-parallel.

    Single-quat numpy port of upstream waypoint._slerp (which was batched torch);
    same semantics: normalize, take the short way around the antipode, and blend.
    """
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:  # q and -q are one rotation; take the short way
        qb = -qb
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:  # nearly parallel: nlerp avoids the sin(theta)~0 blowup
        q = (1.0 - t) * qa + t * qb
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    s = math.sin(theta)
    wa = math.sin((1.0 - t) * theta) / s
    wb = math.sin(t * theta) / s
    q = wa * qa + wb * qb
    return q / np.linalg.norm(q)


class LiftExpertPolicy:
    """Single-env reactive lift expert over this fork's public TaskEnv surface.

    ``act() -> (8,) float32`` = ``[j0..j6, gripper_norm]``: a ``max_step``-capped
    Cartesian step from the measured EE toward the active phase's target, with
    orientation slerped ``rot_frac`` toward the face-aligned grasp quat, then
    solved to absolute joints by IK seeded at the LIVE qpos (label continuity).
    Phase transitions read only measured state; the only memory is the phase
    plus a close-attempt tick counter.
    """

    def __init__(
        self,
        env,
        # 0.025/0.15 swept upstream on the v8 spawn/init distribution; larger
        # steps destabilize per-tick IK near the base column.
        max_step: float = 0.025,   # commanded EE translation per tick, m
        rot_frac: float = 0.15,    # slerp fraction toward the grasp quat per tick
        tol_xy: float = 0.02,      # xy alignment gate, m
        tol_z: float = 0.015,      # z arrival gate, m
        grasp_r: float = 0.035,    # cube-to-TCP distance that counts as held, m
        # gripping band on env.gripper_norm() (1=open, 0=closed): closed-on-air
        # drives norm below grip_lo, a fully open gripper sits above grip_hi, and
        # seated on the 31.75 mm cube the finger dof plateaus so norm lands between.
        grip_lo: float = 0.20,
        grip_hi: float = 0.85,
        close_ticks_min: int = 12, # finger travel time before a grasp can count
        close_ticks_max: int = 30, # abort a close attempt that never seats
    ):
        self.env = env
        self.robot = env.robot
        self.max_step = max_step
        self.rot_frac = rot_frac
        self.tol_xy = tol_xy
        self.tol_z = tol_z
        self.grasp_r = grasp_r
        self.grip_lo = grip_lo
        self.grip_hi = grip_hi
        self.close_ticks_min = close_ticks_min
        self.close_ticks_max = close_ticks_max
        # xy clamp for chase targets: stay over the table even if the cube leaves
        # it (never labels a dive off the edge). Upstream reads env.arena.center_xy
        # / size_xy; this fork keeps the same geometry on cfg.table.
        cx, cy = env.cfg.table.center_xy
        sx, sy = env.cfg.table.size_xy
        m = 0.03
        self._xy_lo = np.array([cx - sx / 2 + m, cy - sy / 2 + m])
        self._xy_hi = np.array([cx + sx / 2 - m, cy + sy / 2 - m])
        self.reset()

    def reset(self, obs=None) -> None:
        self.phase = APPROACH
        self._close_ticks = 0

    def act(self, obs=None) -> np.ndarray:
        ee_pose = np.asarray(self.robot.ee_pose.cpu(), dtype=np.float64).reshape(-1)
        ee = ee_pose[:3]
        ee_quat = ee_pose[3:7]
        gnorm = float(self.env.gripper_norm())
        cube = np.asarray(self.env.cube_pos(), dtype=np.float64).reshape(-1)[:3]
        q = np.asarray(self.env.cube.get_quat().cpu(), dtype=np.float64).reshape(-1)[:4]

        top_z = self.env.cfg.table.top_z
        grasp_z = top_z + GRASP_TCP_OFFSET
        lift_z = grasp_z + LIFT_HEIGHT

        cube_xy = np.clip(cube[:2], self._xy_lo, self._xy_hi)
        xy_err = float(np.linalg.norm(ee[:2] - cube_xy))
        # gripping = cube at the TCP with the fingers seated on it (norm in the
        # cube-width band; below it they closed on air, above they're still open)
        held = (
            float(np.linalg.norm(cube - ee)) < self.grasp_r
            and gnorm > self.grip_lo
            and gnorm < self.grip_hi
        )
        near_at = xy_err < self.tol_xy and abs(ee[2] - grasp_z) < self.tol_z
        near_above = xy_err < self.tol_xy and abs(
            ee[2] - (grasp_z + APPROACH_HEIGHT)
        ) < 2 * self.tol_z

        p = self.phase
        # demotions first: lost the cube, or drifted off it while descending
        if p >= LIFT and not held:
            p = APPROACH
        if p == DESCEND and xy_err > 2 * self.tol_xy:
            p = APPROACH
        abort = p == CLOSE and not held and self._close_ticks >= self.close_ticks_max
        if abort:
            p = APPROACH
        # promotions
        if p == APPROACH and near_above:
            p = DESCEND
        starting_close = p == DESCEND and near_at
        if starting_close:
            p = CLOSE
        # fingers need travel time: a grasp only counts once the dwell has run and
        # the norm sits in the gripping band (not still sweeping through it)
        if p == CLOSE and held and self._close_ticks >= self.close_ticks_min:
            p = LIFT
        if p == LIFT and held and ee[2] > lift_z - self.tol_z:
            p = RELEASE
        if starting_close or abort:
            self._close_ticks = 0
        if p == CLOSE:
            self._close_ticks += 1
        self.phase = p

        z = (grasp_z + APPROACH_HEIGHT, grasp_z, grasp_z, lift_z, lift_z)[p]
        target = np.array([cube_xy[0], cube_xy[1], z])
        # RELEASE opens at height: the dropped cube's bounce diversifies re-grasp
        # poses, and the ~held demotion recycles the env back to APPROACH
        grip = GRIPPER_CLOSED if (p >= CLOSE and p != RELEASE) else GRIPPER_OPEN

        delta = target - ee
        dist = float(np.linalg.norm(delta))
        pos_cmd = ee + delta * min(1.0, self.max_step / max(dist, 1e-9))

        cube_yaw = 2.0 * math.atan2(q[3], q[0])
        grasp_quat = np.asarray(_nearest_side_grasp_quat(cube_yaw, ee_quat), dtype=np.float64)
        quat_cmd = _slerp(ee_quat, grasp_quat, self.rot_frac)

        # Cartesian command that produced this label, stashed for teacher-channel
        # recording (see mcap_writer /teacher/robot_states): pos in metres (xyz),
        # quat wxyz. float64 to match the measured ee_pose the writer scales.
        self.last_pos_cmd = np.asarray(pos_cmd, dtype=np.float64)
        self.last_quat_cmd = np.asarray(quat_cmd, dtype=np.float64)

        joints = self._ik_joints(pos_cmd, quat_cmd)
        return np.concatenate([joints, [grip]]).astype(np.float32)

    def _ik_joints(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Absolute arm joints for a commanded EE pose, via the same Genesis IK
        call the teacher uses (see teacher.py:126-143) -- but seeded at the LIVE
        qpos (upstream ``ik_from_current=True``) instead of home. Home seeding
        returns far-branch joint targets from randomized starts, and the reactive
        expert's actions double as regression labels, so they must be continuous
        in state.
        """
        robot = self.robot
        ent = robot._robot_entity
        init_qpos = ent.get_qpos()
        init_qpos = init_qpos.unsqueeze(0) if init_qpos.ndim == 1 else init_qpos
        pos_t = torch.as_tensor(pos, device=init_qpos.device, dtype=init_qpos.dtype).reshape(1, 3)
        quat_t = torch.as_tensor(quat, device=init_qpos.device, dtype=init_qpos.dtype).reshape(1, 4)
        q_pos = ent.inverse_kinematics(
            link=robot._ee_link,
            pos=pos_t,
            quat=quat_t,
            init_qpos=init_qpos,
            max_samples=robot._args.get("ik_max_samples", 50),
            max_solver_iters=robot._args.get("ik_max_solver_iters", 20),
            damping=robot._args.get("ik_damping", 0.01),
            dofs_idx_local=robot._arm_dof_idx,
        )
        return np.asarray(q_pos[:, robot._arm_dof_idx].cpu(), dtype=np.float32).reshape(-1)[:7]
