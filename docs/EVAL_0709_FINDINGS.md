# Eval findings, 2026-07-09: 0707@50000 through the grainlike serve

Audience: mhyatt. TL;DR — served through YOUR serving stack (`scripts/serve/bela.py`,
dev a3d2956+4bdd549: ModelPolicy → ActionDenormWrapper → GrainlikeWrapper, so the exact
training transforms run server-side and none of our client-side packing is in the loop),
the `0707_iconic-spaceship-1191` step-50000 checkpoint is **insensitive to cube position**
in closed-loop sim. This removes the "sim client packed the obs wrong" hypothesis that
yesterday's v1-contract eval left open: the failure is training-side.

## Setup

- Server: your `bela.py` serve on mayo (`--path .../0707_iconic-spaceship-1191/params/
  --step 50000 --dataset-name xarm_sim`), webpolicy websocket, port 8001 that day.
- Client: `scripts/eval_grid.py --policy remote` with the raw/grainlike payload
  (`ObsSpec(wire_format="grainlike")`, xarm-sim commit d809d11): native-res stacked
  [low, side, wrist] views, raw proprio, `info.step`; actions consumed from the
  denormalized `actions.{joints,gripper}`, first chunk row, re-queried at 30 Hz
  (`joint_abs` absolute position control, matching action = next-step proprio).
- Env: the nyx (splat) rendering domain the training data was generated in;
  weld-free grasping, noslip_iterations=10, close_setpoint 0.58; scripted baseline on
  this exact harness passes 9/9.
- Wire-level sanity (no sim): repeated identical obs → |Δaction| ≈ 0.01 rad (flow
  sampling noise); strongly perturbed obs (joints +0.6 rad, shifted images) → |Δ| ≈ 0.28
  rad. The pipe transmits; the model is not literally constant.

## Result 1: 3×3 position grid — 0/9, position-invariant trajectory

`outputs/eval/grid3x3_grainlike_0709` (branch eval-suite). 3×3 cube grid spanning the
training spawn range (x ∈ [0.35, 0.58], y ∈ [−0.15, 0.15]), 1 rep, 20 s per trial,
3-view video for every trial in `videos/`.

- 0/9 success, all `never_lifted`, `max_rise` = 0.000 in every trial.
- The trajectory is **near-identical for all 9 cube positions**: rise from home, hover
  over the same mid-table spot, dip slightly around t≈10–17 s, retract. See
  `montage_low_16s.png` (same arm silhouette in all 9 panels while the cube moves across
  the whole range) and `montage_wrist_16s.png` (the cube only enters the wrist view when
  it happens to sit under the fixed hover spot).
- The gripper never closes and the TCP never descends the last ~10 cm to grasp height.
- Rapid dithering around the hover pose = per-query flow sampling noise (no ensembling
  in the serve stack); consistent with the model emitting a marginal mean pose + noise.

## Result 2: warm-start probe — 0/3, actively leaves the pre-grasp state

Same serve, but the TCP is first driven (arm-joint IK only, gripper untouched) to a
top-down hover 9 cm above the cube before the policy takes over — a state the demos
visit mid-episode. Three cube positions (x=0.465, y ∈ {−0.15, 0, +0.15}), videos in
`outputs/eval/warmstart_grainlike_0709/videos/` (`--warm-start` on eval_grid).

- 0/3, all `never_lifted`. `min_tcp_cube` ≈ 0.085 m at ALL three positions — exactly the
  hand-over distance. The policy never got closer to the cube than where we placed it.
- Within ~2 s of handover it retracts from the pre-grasp hover back to the same marginal
  mid-table hover as Result 1 and stays there (`timeline_warmstart_g001.png`).
- Matches yesterday's v1-contract warm-start finding, now reproduced through the
  training-exact serve: no local grasp competence — the model pulls toward one
  attractor pose from anywhere, rather than continuing a grasp it is already lined up
  for. Consistent with a policy dominated by the dataset-marginal pose distribution.

## Result 3: the predicted 50-step horizon itself is cube-invariant

One query per cube position (x=0.465, y ∈ {−0.15, 0, +0.15}), inspecting the full
`actions` chunk rather than the executed row. Plot: `outputs/eval/horizon_plot.png`.

The chunk is well-formed — j1/j3/j5/j6 ramp smoothly and monotonically over the 50 steps,
FK'ing to a TCP path that rises 0.08 m → 0.12 m while drifting to table center. It is
also the SAME trajectory at every cube position. Base yaw (j0) is the joint that aims the
arm left/right:

| | cube y=−0.15 | cube y=0.00 | cube y=+0.15 | spread |
|---|---|---|---|---|
| j0, horizon row 0  | −0.0334 | −0.0373 | −0.0334 | 0.004 |
| j0, horizon row 49 | −0.0768 | −0.0611 | −0.0407 | 0.036 |
| **j0 needed (atan2(y,x))** | **−0.3120** | **0.0000** | **+0.3120** | **0.624** |

The model spans 8% of the required range, never changes sign, and for the y=+0.15 cube
(needs j0 POSITIVE) it drives j0 more negative — the wrong way. The 0.036 rad of
cross-position variation is the same magnitude as the flow head's per-query sampling
noise (0.012 rad on identical inputs), i.e. indistinguishable from zero signal.

The gripper channel never leaves 1.008–1.039 (1=open) across all 150 predicted steps at
all three positions; the close threshold is 0.5. The model does not plan a grasp at any
point in its own horizon.

Also checked and NEGATIVE: the `kp3dc_robot` action channels are equally position-locked
(max 0.030 m deviation across positions for a 0.30 m cube displacement). So this is not
"the keypoint head learned the task while the joint head is broken" — every output channel
is blind to the cube at serve time.

**The one lead this leaves.** grifflee reports that during TRAINING, the model's keypoint
predictions visually tracked the cube correctly. If that holds, the model did learn
obs-conditioned behavior and something about serve-time inputs is off-distribution. The
most conspicuous train/serve input difference: the 140-slot state input carries FK robot
keypoints (kp3dc, 126 of 140 slots) at train time, dropped only with `input_drop_prob`
= 0.25 — while our serve payload masks out all 126 on EVERY step, an input pattern
training essentially never produced. Next test on our side: compute FK keypoints
client-side (`robot_keypoints_in_cameras` + URDF FK; sim has GT calibration) and re-probe.

## The one ask

Run `scripts/debug/server_compare.py` (or any predict-on-training-batch) for this
checkpoint against a handful of `xarm_sim` training episodes — the arec data lives on
your machine, not mayo. It splits the remaining ambiguity in one shot:

- predictions match the GT actions on training inputs → the model learned the data and
  the gap is train/eval input mismatch (we then diff our eval obs against a training
  sample field by field);
- predictions do NOT match → the 0707 run itself (500 eps? objective? label alignment?)
  never fit the data, and no sim-side change will move the needle.

Side note from yesterday still stands: the demos open with a long static home hold and
action = next-step proprio, so the at-home expert action is ambiguous without a timestep
input; worth checking whether the learned policy is dominated by that static segment.
