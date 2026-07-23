# Grasp weld audit against current mhyatt upstream

Date: 2026-07-23

Audited remote: `upstream` (`mhyatt000/xarm-sim`)

Audited tip: `788e6543e07b5b33635d47eb3d062204c3221a6f` (2026-07-22)

## Finding

Current upstream does **not** use a weld for lift. Commit `ae511c6` retired the legacy
generator containing `grasp_lock()` and the fixed-tick weld. The current suite uses a
physical, friction/contact grasp with Genesis `noslip_iterations=10` in the production
configuration.

This solves the original carry-creep problem but does not claim to solve every visual
contact artifact. Upstream's `docs/GRASP_TELEPORT_INVESTIGATION.md` separates:

1. carry slip after acquisition, controlled by `noslip_iterations=10`; and
2. a grab-time contact-seating pop while the fingers acquire the cube, which remained
   visible in the last dedicated investigation.

## How upstream prevents fabricated grasps

`src/xsim/suite/policies/lift_expert.py` is reactive and state-derived rather than
fixed-tick:

- TCP-to-cube distance must be below 0.035 m;
- XY error must be below 0.020 m and Z error below 0.015 m before close;
- gripper norm must be in the seated-on-cube band `(0.20, 0.85)`;
- close must dwell for at least 12 control ticks;
- a close that has not acquired the cube after 30 ticks aborts and retries;
- loss of the cube demotes the state back to approach; and
- IK is seeded from live qpos to avoid discontinuous home-seeded branches.

`src/xsim/suite/environments/manipulation/lift.py` independently makes success require
the cube to be lifted, slow relative to the end effector, near it in XY, physically in
contact with the robot, and held there for consecutive control ticks.

## Reconciliation in this branch

The legacy MCAP generator remains useful because it contains the calibrated cameras,
air-drop protocol, and data contract that the suite generator does not replace. It now
offers two explicit modes:

- `proximity_weld` (current checkpoint default): the weld may fire only after the same
  upstream proximity, gripper-band, and 12-control-tick dwell conditions. The old fixed
  tick is only a timeout. An unreachable target therefore fails without moving the cube.
- `physical`: no weld at any point; use `noslip_iterations=10`, matching upstream's
  physical-grasp approach.

Both modes record minimum TCP/cube distance, close XY/Z error, acquisition state, close
dwell, weld-fired state/step/distance, and gripper norm. Lifted and maximum-rise fields
are therefore no longer accepted alone as evidence of a valid grasp.

Verified seed `100000` at the production 120 Hz physics rate:

| mode | weld | acquisition | lift | delivery |
|---|---:|---:|---:|---:|
| proximity-gated weld | yes, after 48 physics ticks (12 control ticks) | physical first | 89.4 mm | 1.1 mm error |
| physical + no-slip 10 | never | physical | 87.8 mm | 2.18 mm error |

The labeled side-by-side video is packaged at
`grasp_ab/seed100000_weld_vs_physical_side_by_side.mp4` in the 10k checkpoint directory.
This is a human decision gate: review the acquisition pop and choose the grasp mode before
starting the 10,000 episodes.
