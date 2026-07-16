# Agent Instructions — xarm-sim sim-data pipeline (handoff 2026-07-02)

You are continuing a verified pipeline that generates photoreal xArm7 block-lift
episodes as Foxglove MCAP for training (crossformer). Everything through pilot
verification is DONE and committed on branch `synthetic-lift-mcap`. Your job is the
scale-up and its verification. Read this whole file before running anything.

## 2026-07-15: remote CrossFormer lift evaluation findings (`dagger`)

Use `dagger` commit `43632c3` or newer for remote policy evaluation. In
`src/xsim/eval_policy.py`, `RemotePolicy.act()` now buffers the model's returned
action chunk and executes one row per control tick. The default open-loop horizon is
50, so server queries occur at episode steps 0, 50, 100, and so on. The older behavior
re-queried every tick and executed only row 0, producing severe jitter and invalidating
the earlier 0-success runs. `--action.open-loop-h 1` intentionally restores that
legacy behavior for an A/B test; do not use it for the normal evaluation.

Two checkpoints were evaluated on the identical 10 x 10 lift grid with one repetition,
20-second episodes, Nyx rendering, and an MP4 for every trial. The learned policy was
served remotely; these were not scripted-policy runs.

- `/home/grifflee/checkpoints/0715_still-star-1213/params`, step 100000:
  **40/100 success**; 59 `never_lifted`, 1 `dropped_in_transit`.
  Results are in
  `outputs/eval/0715_still-star-1213_step100000_100trials/`.
- `/data/store/weights/0707_iconic-spaceship-1191/params`, step 50000:
  **3/100 success**; 96 `never_lifted`, 1 `dropped_in_transit`.
  Results are in
  `outputs/eval/0707_iconic-spaceship-1191_step50000_100trials/`.
- The explicitly labeled side-by-side and delta plot is
  `outputs/eval/model_comparison_0715_vs_0707.png`. Each run has 100 JSONL records
  and 100 MP4s under its `videos/` directory.

Heatmap/table alignment: the robot is at x=0 and faces toward increasing x. The
heatmap's left edge (x=0.35) is near the robot and its right edge (x=0.58) is far.
Standing behind the robot and looking toward +x, the bottom edge (y=-0.15) is the
robot's right and the top edge (y=+0.15) is its left. Thus bottom-left means
near-robot/robot-right; top-left means near-robot/robot-left.

Both servers repeatedly warned that `kp3dc_robot` and `kp3dw_robot` proprio inputs
were absent and masked. This remains a train/serve mismatch, so the model comparison
is fair under the same serving contract, but neither result establishes the ceiling
with training-time keypoints supplied. Investigate client-side FK keypoints or use
`server_compare.py` before blaming weights alone.

For this evaluation workflow, grifflee requested **9xxx ports only**. Port 9001 was
used on loopback with `--host 127.0.0.1 --port 9001 --no-viser`; all evaluation and
server processes were stopped afterward. Always verify and close listeners when done.

Seed ranges: DAgger generation runs use base seed 61000 with the per-scene formula
`base + rep*10007 + grid_idx` (rep 0: 61000-61099, rep 1: 71007-71106, rep 2:
81014-81113, ...), all clear of training (9k/20k/30k+) and eval (51k) ranges. Corrected episodes are kept only when their release-trimmed
length is at most `--max-training-frames` (default 450 = 2x the 227-frame demo max,
grifflee's decision 2026-07-16); longer corrected hybrids are counted as
`corrected_too_long` and never written as training MCAP.

## 0. IN PROGRESS (2026-07-02 evening) -- new demonstration protocol verified through pilot

Commit `660d33b` introduced grifflee's new protocol; this follow-up made the validation
and timing fixes and ran the gates through a 10-episode Nyx pilot. Do not start batch_v3
until grifflee reviews the checkpoint artifacts below.

### 2026-07-03: batch_v3 GENERATED and gated (100 episodes)

grifflee approved the v5 pilot checkpoint; `outputs/sim_mcap/batch_v3` is the
current training batch. 4 shards x 25 episodes (seeds 9000-9099, per-episode
seed = shard seed + idx), merged flat with `scripts/merge_shards.py`
(hard links; shard dirs kept). Recipe = batch_v2 camera jitter
(`--env.cam-jitter-deg 15 --env.cam-jitter-cm 5`) + all v4/v5 changes
(table z=-0.01 3x2 ft, arm start jitter 3 deg, mounting plate).
Gates vs the REAL robot mcaps: `compare_batches.py` FORMAT PASS
(report_batch_v3.png), `validate_mcap.py` layout PASS on episodes 0/42/99
vs the May reference. 100/100 success, frames 154-227, 30.00 Hz,
100 unique per-episode camera extrinsics recorded in the manifest.

### 2026-07-02 late: table corrected + per-episode start-joint jitter (v4)

Two changes on top of the v3 protocol, from grifflee's remaining-adjustments list:

- **Table geometry corrected to his measurements**: the robot base sits on a 1 cm
  mounting plate, so `TableCfg.top_z = -0.01` (was 0.0), and the slab footprint is the
  real 3 ft x 2 ft top: `size_xy = (0.9144, 0.6096)` (was the IPM-estimated 0.93 x 0.62).
  Cube spawn/rest heights and all policy z-waypoints derive from `top_z`, so they
  follow automatically.
- **Random-ish start joints**: `TaskEnvCfg.arm_start_jitter_deg = 3.0` (new field; 0
  disables). Each arm joint gets a uniform +-3 deg offset from the IK-solved home at
  reset, drawn AFTER the cube/camera/drop draws so those streams stay per-seed stable.
  The scripted policy already reads the live TCP at reset, so trajectories adapt. TCP
  start spread is ~1-3 cm. `Manipulator.reset` in `grasp_env.py` gained an
  `arm_qpos_offset` argument for this.

- **Base decor (visual-only)**: `BaseDecorCfg` adds the flat light-metal mounting plate
  under the base: 13 x 18 cm rectangle centered on the base origin, top flush with it,
  filling the 1 cm table-top gap. Per grifflee: along x it is only as long as the
  base's outer ring; along y it sticks out ~1 inch past the ring on each side for the
  blue clamps. APPROVED by grifflee against the cap photos
  (`outputs/sim_preview/base_plate_compare_v6.png`, blink_test blend).
  `collision=False`; no rng draws added, so seed streams are unchanged.
  IMPORTANT: May-episode frames CANNOT be used to check the low/side cameras — those
  Logitechs were moved and recalibrated after May (only the wrist mount is unchanged).
  Use the cap.npz photos / blink_test for any real-vs-sim comparison.
  TRAP, do not repeat: an earlier revision modeled a red "E-stop" next to the base.
  That red blob in the real frames was the CUBE sitting near the plate — there is no
  E-stop on the table. grifflee caught it.
  NEGATIVE RESULT, do not retry: restoring the base area from the raw splat
  (keep-region in clean_splat.py) looks like smoke/smudges — ~80% of the scanned
  gaussians around the base have opacity <0.4 (dark reflective table = mush), and even
  opaque-only looks like dirt. Panels: `outputs/sim_preview/base_fill_compare.png`.
  Decor smoke: `raster_v4_decor_smoke_7400` 3/3, both gates PASS.

Verified: 3-episode raster smoke (`outputs/sim_mcap/raster_v4_jitter_smoke_7300`,
seed 7300): 3/3 success, `compare_batches.py` FORMAT PASS, `validate_mcap.py` PASS,
first-frame joint_states show distinct jittered starts, cube rest z = top_z + half cube.
A fresh 10-episode Nyx pilot for grifflee's visual checkpoint is the next gate before
any batch; the older `nyx_pilot_v3_sides_8100` artifacts predate these changes.

Current protocol: above cube (high, straight-down, side-grasp yaw chosen as the nearest
90-degree-equivalent cube-face alignment) -> vertical plunge -> close -> weld cube to
`link_tcp` -> fast lift -> transport to sampled drop target (x~U[0.30,0.40], y=0) ->
brief closed hold at the target -> open + delete weld -> cube drops -> recording ends
after the ~0.3 s opening tail. Unrecorded settle steps still run for the success check.
MCAPs carry ground-truth calibration: `foxglove.CameraCalibration`
on `/cam/{low,side}/camera_info` + `/camera/camera/color/camera_info`, and
`foxglove.FrameTransform` on `/tf`.

### Verified/fixed after `660d33b`

- `scripts/compare_batches.py` and `scripts/validate_mcap.py` now gate the six core real
  topics/schemas, allowlist/require the three `CameraCalibration` topics plus `/tf`, and
  use the new core frame-count gate 115-240. The default design envelope is 152-227
  frames after the added pre-release hold.
- `src/xsim/scripted_lift_policy.py` now has `SEGMENT_WEIGHTS=(2.6,.8,.8,.4,1.0,.6)`.
  The final `.6` segment is a closed hold at `over_drop`; it was added because MCAP tail
  decoding showed 10-20 mm TCP drift during the recorded opening tail without it.
- Gripper yaw now chooses the nearest equivalent side grasp (`cube_yaw + k*pi/2`) to the
  current wrist orientation. The old world-zero fold caused avoidable ~90-degree spins;
  diagnostic seeds 8000-8009 now select face-aligned yaws with ~15-41 degree wrist
  reorientation instead of ~95-129 degrees.
- Seed 6542 raster slip trace: welded TCP-z/cube-z drift was 0.013 mm while welded; cube
  dropped cleanly after release; `reset()` cleared the weld.
- Seed 6542 raster video/preview generated:
  - `outputs/sim_preview/protocol_v3_seed6542_raster_hold.mp4`
  - `outputs/sim_preview/protocol_v3_seed6542_preview_hold/`
  - preview milestones include `over_drop_settled` before `release`.
- 3-episode raster smoke after the side-grasp yaw fix
  (`outputs/sim_mcap/raster_v3_sides_smoke_7200`, seed 7200): 3/3 success;
  `validate_mcap.py` PASS; `compare_batches.py` FORMAT PASS; report at
  `outputs/batch_report/report_raster_v3_sides_smoke_7200.png`. MCAP tail decode: final
  10-frame TCP motion 0.05-0.17 mm, gripper norm opens ~0.318 -> 0.96+.
- 10-episode Nyx pilot after the side-grasp yaw fix
  (`outputs/sim_mcap/nyx_pilot_v3_sides_8100`, seeds 8100-8109): 10/10 success; frame
  counts 152-221; `validate_mcap.py` PASS; `compare_batches.py` FORMAT PASS; rate
  30.00 Hz; duration 6.16+/-0.72 s; report at
  `outputs/batch_report/report_nyx_pilot_v3_sides_8100.png`.
- Nyx checkpoint video generated at `outputs/sim_preview/nyx_pilot_v3_sides_seed8100.mp4`.

### Next step

Show grifflee the Nyx checkpoint video and report before generating batch_v3:

```bash
cd ~/repo/xarm-sim
ls outputs/sim_preview/nyx_pilot_v3_sides_seed8100.mp4 \
   outputs/batch_report/report_nyx_pilot_v3_sides_8100.png
```

If grifflee approves, generate sharded `batch_v3` (same pattern as batch_v2: shard dirs,
staggered launch, merge manifests). If he rejects the checkpoint, do not batch; adjust the
policy/visuals and rerun the raster smoke + 10-episode Nyx pilot first.

### Open items still needing grifflee's read

- Release is at lift height; there is no lowering before open.
- The dataset includes the ~0.3 s recorded opening tail.
- Segment timing is now `(2.6,.8,.8,.4,1.0,.6)`, giving ~5-7 s episodes.

## 1. Non-negotiable constraints (from grifflee)

1. **Table rendering: `table_mode="slab"` (the default). NEVER pass
   `--env.table-mode plane`** for testing or generation.
2. **Never move, jitter, or "improve" the camera poses.** `low`/`side` extrinsics in
   `src/xsim/task_env.py` (`LOW_C2W_CV`, `SIDE_C2W_CV`, `LOGITECH_FOV_DEG`) come from
   the real calibration (`/data/store/opencv_calibrated`). Label correctness depends on
   them exactly as they are.
3. **Do not re-tune the splat alignment by eye.** `DEFAULT_SPLAT_POS/QUAT/SCALE` were
   solved numerically and human-verified. If the room is rescanned or cameras
   recalibrated, re-solve with `scripts/align_ransac.py` (stage 1 → grifflee confirms
   landmarks → solve), then regenerate `assets/lab_clean.ply` via `scripts/clean_splat.py`.
4. **Do not remove the two conversion calls in `_make_light_field`**
   (`float3_z_up_to_y_up_a`, `quaternion_z_up_to_y_up_a`). They work around a gs-nyx bug:
   the scene exporter converts meshes/cameras from Genesis z-up to Nyx y-up but passes
   light fields through raw. Removing them silently misplaces the splat background.
5. **Keep `gripper_grasp_dof: 0.58`.** 0.53 matches the real recorded close depth
   (norm 0.37) but rigid sim fingers drop the cube at that width — tested, 0/3.
6. **Human checkpoints**: before building on any new visual/scene interpretation, and
   after any batch, show grifflee a labeled artifact (image/report) and wait for his
   read. He catches visual errors reliably; this protocol caught every real bug so far.

## 2. Current state (what is already verified)

- **Geometry**: sim robot rendered through the calibrated cameras matches real photos at
  **median 1.9 px** edge error over all 24 calibration configs (`scripts/blink_test.py`;
  artifacts in `outputs/blink_test/`, incl. sim↔real flip GIFs).
- **Wrist camera**: uncalibrated by nature (robot never sees itself); mount is a guess
  validated against real May episode frames at matched joints (~40 px mid-grasp).
- **Old pilot v2 protocol (verified, now superseded)**: low IK-solved ready pose,
  lift 0.09 m, release near 85%, tempo jitter 0.85–1.30× → 6.6–9.8 s episodes.
  `outputs/sim_mcap/pilot_v2/` was 10/10 success and format PASS; its report showed
  TCP/gripper profiles inside the real 10–90% band.
- **New protocol after commit `660d33b` (pilot-verified, awaiting grifflee review)**:
  straight-down approach with nearest side-grasp yaw, vertical plunge, close, weld to
  `link_tcp`, fast lift, transport to sampled drop target, brief closed hold,
  open/unweld, and end recording after the short opening tail. Section 0 has the current
  checkpoint artifacts.
- **Docs**: `docs/TRAINING_HANDOFF.md` = consumer-facing (calibrated-vs-guessed
  inventory, eval caveats). Read it once.

## 3. Repo navigation and daily operations

Prefer the source over the README when in doubt; the README still describes the older
generic grasp demo in places. The current lift-data path is:

- `src/xsim/task_env.py` -- the active xArm7 lift environment: robot config, calibrated
  `low`/`side` cameras, guessed wrist camera, table/slab rendering, Nyx splat setup,
  spawn/drop ranges, and camera-jitter toggles.
- `src/xsim/scripted_lift_policy.py` -- scripted waypoint policy used for these demos.
- `src/xsim/mcap_writer.py` -- Foxglove MCAP topic/schema writer matching the real lift
  logs.
- `scripts/generate_task_dataset.py` -- main entrypoint for simulation. `generate`
  writes MCAP; `preview` writes milestone PNGs; `video` writes an MP4 contact sheet.
- `scripts/compare_batches.py` -- batch format gate plus label-distribution report
  against `/data/store/mcaps/single/lift`.
- `scripts/validate_mcap.py` -- inspect one MCAP and optionally compare its topic/schema
  set against a reference real episode.
- `scripts/blink_test.py` and `scripts/verify_splat_alignment.py` -- calibration and
  alignment diagnostics. Use these before trusting visual/camera changes.
- `scripts/align_ransac.py` and `scripts/clean_splat.py` -- only for rescan/recalibration
  workflows.
- `scripts/scripted_grasp_policy_room.py`, `src/xsim/grasp_env.py`, and
  `src/xsim/scripted_grasp_policy.py` -- older/generic grasp environment, not the current
  lift MCAP production path.
- `outputs/` -- generated artifacts, gitignored. Do not commit MCAPs, videos, reports, or
  large assets.

Common starting point:

```bash
cd ~/repo/xarm-sim
uv sync
```

Fast local sanity checks:

```bash
# Fast raster video, no MCAP, useful for policy/physics checks.
uv run python scripts/generate_task_dataset.py \
    --mode video --backend gpu \
    --video-path outputs/sim_preview/raster_check.mp4 --seed 0

# Photoreal Nyx video, no MCAP, useful before/after a batch.
uv run python scripts/generate_task_dataset.py \
    --mode video --backend gpu --env.render-backend nyx \
    --video-path outputs/sim_preview/nyx_check.mp4 --seed 0

# Milestone PNGs at reset/waypoints/settled.
uv run python scripts/generate_task_dataset.py \
    --mode preview --backend gpu \
    --preview-dir outputs/sim_preview/task_env --seed 0
```

Small MCAP smoke test:

```bash
uv run python scripts/generate_task_dataset.py \
    --n-episodes 3 --backend gpu --env.render-backend nyx \
    --out-dir /data/store/griffen_sim_mcaps/smoke --seed 300

uv run python scripts/compare_batches.py --sim-dir /data/store/griffen_sim_mcaps/smoke
uv run python scripts/validate_mcap.py \
    /data/store/griffen_sim_mcaps/smoke/episode_000000.mcap \
    --reference /data/store/mcaps/single/lift/2026-05-25_1457_episode_000002.mcap
```

Diagnostic commands:

```bash
# Robot/camera calibration blink checks against cap.npz.
uv run python scripts/blink_test.py --mode cap --configs 0 5 10

# Wrist sanity panels against a real May episode.
uv run python scripts/blink_test.py --mode wrist

# Splat-room alignment panels: real | sim | blend plus an oblique overview.
uv run python scripts/verify_splat_alignment.py --tag check
```

Simulation and toggle notes:

- `--mode generate|preview|video`: `generate` writes one `episode_*.mcap` per rollout and
  `manifest.json`; `preview` writes PNGs; `video` writes an MP4 contact sheet. Preview
  and video do not write MCAP.
- `--backend gpu|cpu`: Genesis backend. Use `gpu` on mayo/RTX for normal work.
- `--env.render-backend raster|nyx`: `raster` is fast and has no photoreal splat room;
  `nyx` is the production photoreal path and reads the splat. Deliverable MCAPs should
  use `nyx` unless grifflee explicitly asks for a raster-only physics pilot.
- `--env.nyx-spp N`: Nyx samples per pixel. Default is 8; increasing it costs time.
- `--env.splat-uri PATH`: defaults to `assets/lab_clean.ply` if present, else
  `/data/store/lab.ply`. Do not swap splats for production without rerunning alignment
  checks.
- `--env.splat-pos`, `--env.splat-quat`, `--env.splat-rot-rpy-deg`, `--env.splat-scale`:
  experiment-only alignment overrides. Do not bake changes without the RANSAC/human
  checkpoint workflow in section 1.
- `--env.rectangle-x LO HI` / `--env.rectangle-y LO HI`: cube spawn rectangle. Any
  widening needs a raster pilot, a Nyx pilot, `compare_batches.py`, and grifflee's read.
- `--env.drop-x-range LO HI` / `--env.drop-y Y`: sampled release target. Changing
  these changes label dynamics and success evaluation.
- `--steps-per-segment`, `--hold-steps`, `--grasp-tcp-offset`: policy/tempo/contact
  controls. These affect the real-vs-sim distribution report.
- `--save-failures`: keeps failed MCAP rollouts instead of deleting them. Use only for
  debugging; production batches should be success-gated.
- `--env.cam-jitter-deg`, `--env.cam-jitter-cm`: per-episode jitter of the two
  STATIC cams (low/side) around their calibrated nominals, recorded in
  `manifest.json` and the MCAP calibration topics. Config defaults are 0, but the
  PRODUCTION RECIPE for batches is `--env.cam-jitter-deg 15 --env.cam-jitter-cm 5`
  — used by every approved batch since batch_v2 (batch_v3, the 5k batch). Pass it
  explicitly on generation runs; leave it off only for calibration/alignment
  diagnostics that need the exact nominal poses (blink_test-style checks).
- `--env.wrist-jitter-deg`, `--env.wrist-jitter-cm`: same idea for the wrist-cam
  mount offset. NOT part of the production recipe — every approved batch keeps
  these at 0 (the wrist mount is a verified guess; jittering it has never been
  reviewed).
- Stack appearance randomization: use the dim ceiling-panel recipe (grifflee reviewed
  per-change panels on 2026-07-07), not the earlier over-bright smoke. Pass
  `--env.nyx-light-type ceiling_panel`, `--env.nyx-light-intensity 6`, and
  `--env.nyx-light-intensity-jitter 0.25`; the ceiling light samples x/y on the
  overhead panel plane and aims down at the table. Pair it with
  `--env.robot-roughness-jitter 0.35`, `--env.cube-hue-jitter-deg 4`,
  `--env.cube-value-jitter 0.2`, and `--env.stack.free-placement`. Hue jitter must
  stay at 4: at -5 deg and below the red cube reads pink (grifflee rejected that).
  Do not use intensity 120; that Nyx spot-light scale washed out the scene.
- Free-placement visibility guarantee: both cubes must be visible to BOTH static
  cams at spawn under the production 15/5 camera jitter. `_sample_free_stack_xy`
  enforces it (side-cam keep-out wedge x<0.38 & y<-0.10, y ceiling 0.14) — derived
  from a 7x7 render audit on 2026-07-07. Don't widen the free ranges or drop the
  wedge rejection without re-running that audit.
- `batch_v1_33300` (stack_ceiling_appearance_dim_batch_v1_33300) is TAINTED: its
  appearance seed equaled the reset seed, so lighting correlates with cube placement
  (r=0.85). Regenerate; don't train on it.
- `--env.table-transparent`: hides the visual table slab but keeps table collision.
  Debugging only.
- `--env.table-mode plane`: the CLI exposes it, but section 1 prohibits it for this
  pipeline. Keep the default `slab`.
- `--env.show-viewer`: opens the Genesis viewer and requires a GUI-capable session.
- `--env.res WIDTH HEIGHT`, `--env.physics-dt`, `--env.record-every`: low-level capture
  shape/rate knobs. The baseline is 640x480, 1/120 s physics, `record_every=4` -> 30 Hz.

Tyro maps dataclass fields to CLI flags with hyphens and nested dataclasses with dots,
for example `TaskEnvCfg.render_backend` becomes `--env.render-backend` and
`cam_jitter_cm` becomes `--env.cam-jitter-cm`.

## 4. Your tasks, in order

### Task A — Scale-up batch

Ask grifflee for N if not already given (100 ≈ 2 h, 200 ≈ 4 h, 400 ≈ 8 h at ~70 s/ep on
the RTX 5090; run in background / overnight):

```bash
cd ~/repo/xarm-sim
uv run python scripts/generate_task_dataset.py \
    --n-episodes N --env.render-backend nyx \
    --out-dir /data/store/griffen_sim_mcaps/<name> --seed 1000
```

- Seed 1000 avoids overlap with pilot seeds (100–109, 200–209). Any fresh range works;
  never reuse a range already in a kept batch (manifest.json lists them).
- Success-gated: failed grasps are dropped automatically. Expect ≥ 90% kept (pilots ran
  10/10). If the success rate drops noticeably below that, STOP and investigate before
  burning GPU time — nothing about a bigger batch should change the physics.
- Episodes are ~100 MB each. Generate under /data/store/griffen_sim_mcaps so the small root disk is not the limiter.

### Task B — Verify the batch

```bash
uv run python scripts/compare_batches.py --sim-dir /data/store/griffen_sim_mcaps/<name>
```

Pass criteria:
- prints `FORMAT: PASS` (six core topics/schemas match a real reference; the three
  `CameraCalibration` topics plus `/tf` are present; core frame counts 115–240;
  exits non-zero on failure — treat any failure as a hard stop).
- rate ≈ 30.00 Hz, durations spread roughly 5–7 s for the new protocol.
- In `outputs/batch_report/report_<name>.png`: sim TCP/gripper median inside the real
  band (like report_pilot_v2); joint histograms overlapping the real support.
- Also eyeball ~3 episodes visually: `--mode video` on the generator config renders an
  mp4 contact sheet without writing MCAP. Confirm the cube reads RED (not salmon — the
  albedo was just darkened and hasn't been reviewed in a full batch), robot on the
  table, background room correct.

**Show grifflee the report PNG + one video before calling it done.**

### Task C (optional, flagged improvement) — widen cube spawn coverage

The reports show sim joint1 narrower than real (cube spawn area is small/symmetric).
If grifflee wants better coverage:
1. Edit `TaskEnvCfg.rectangle_x`/`rectangle_y` in `src/xsim/task_env.py` modestly
   (e.g., y ±0.15 → ±0.22; keep everything on the real table: world x ∈ [−0.09, 0.84],
   y ∈ [−0.30, 0.32] minus margins for the gripper).
2. Run a 10-episode RASTER pilot first (fast): success rate must stay high — spawns too
   close to the table edge or robot base will fail IK/grasps.
3. Then a 10-episode nyx pilot + `compare_batches.py` + grifflee's read, THEN a batch.

### Task D — Publish

After grifflee approves the batch: copy per `docs/TRAINING_HANDOFF.md` §"Where the data
is" (shared location `/data/store/mcaps/sim/lift` pending mhyatt's confirmation).
Commit any code changes with clear messages on `synthetic-lift-mcap` and push to origin
(grifflee's fork). Do not commit `outputs/` (gitignored) or the `.ply` assets.

## 5. Tool map

| Tool | Purpose |
| --- | --- |
| `scripts/generate_task_dataset.py` | episodes → MCAP (`generate`), preview PNGs, video mp4; writes manifest.json |
| `scripts/compare_batches.py` | batch format gate + label-distribution report vs real MCAPs |
| `scripts/blink_test.py` | accuracy check: sim vs real photos at calibration configs (cap mode) and vs May wrist frames (wrist mode) |
| `scripts/verify_splat_alignment.py` | nyx render vs calibration photos at the solved splat transform |
| `scripts/align_ransac.py` | re-solve scene alignment from the ZED fused cloud (only after rescan/recalibration) |
| `scripts/clean_splat.py` | regenerate `assets/lab_clean.ply` (crop baked robot/table, cull floaters, flatten SH) |
| `scripts/validate_mcap.py` | single-file MCAP inspector/differ |

## 6. Environment gotchas (will bite you otherwise)

- Renders only refresh on `scene.step()` — `set_qpos` + render without stepping shows
  stale frames. (See `pose_robot` in blink_test.py for the correct pattern.)
- Genesis `add_camera` default `near=0.1` clips the wrist camera's own gripper; the env
  passes `near=0.02`.
- Everything runs headless via EGL on mayo; view images via scp or `mpv --vo=x11`.
- Real MCAPs: `/xarm/robot_states` TCP position is in **millimetres**; gripper norm
  1=open, 0=closed; wrist topic name says "compressed" but payloads are raw rgb8.
- The real episodes (409 files) are from the OLD camera rig — never pixel-compare the
  static cameras against them; `cap.npz` is the current-rig image ground truth.
