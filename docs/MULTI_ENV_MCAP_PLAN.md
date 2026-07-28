# Multi-environment MCAP generation plan

Status: design draft, 2026-07-26  
Production branch: `new-dagger-crossformer`  
Upstream reference: `origin/upstream-main` at `788e654`

## Milestone status

### Milestone A — implemented, awaiting approval (2026-07-28)

- Added `TaskEnvCfg.n_envs=1`; `Manipulator` and `scene.build()` now use that value.
- Added explicit batch-shaped state surfaces for images, cube positions/yaws, drop
  targets, proprioception, gripper state, green-cube state, and camera extrinsics.
- Migrated the production generator, scripted policies, integrity observations, and
  success scoring to select slot zero explicitly from those batch surfaces.
- Values above one fail loudly until Milestone B supplies independent reset/policy state.
- Added CPU-only contract tests.

Validation on this host:

- 4/4 batch-contract tests pass;
- compile/import checks pass;
- CPU/raster seed 6542 completed successfully with 219 frames;
- generated MCAP passed topic, schema, image-payload, calibration, and transform checks;
- parent commit and Milestone A produced the same success verdict, frame count, MCAP
  size, and task statistics. Separate CPU runs differed by about `9e-10 m` in maximum
  rise, so byte identity is not claimed;
- GPU/Madrona validation is pending because this host cannot access an NVIDIA driver.

## Outcome

Run one `TaskEnv` process per GPU with a configurable batch of Genesis/Madrona
environments, while continuing to emit the same independent MCAP files and manifest
records consumed by CrossFormer.

CrossFormer and the MCAP schema do not need to change. Batching ends at the generator:
each environment slot still produces an ordinary single-episode MCAP.

The first production version should use synchronous batches:

1. reset all `B` slots with `B` independent episode seeds;
2. simulate and render all live slots together;
3. close each slot's MCAP as that episode finishes;
4. hold finished slots harmlessly until every slot is done;
5. reset the next batch.

This captures the main Madrona/Genesis batching gain without making partial reset and
slot refill part of the first correctness boundary.

## Why the code is currently scalar

The scalar build dates to the initial lift-task implementation:

- `TaskEnv` constructs `Manipulator(num_envs=1)`;
- `TaskEnv` calls `scene.build(n_envs=1)`;
- the production policy, integrity gate, MCAP writer, and manifest loop each own one
  episode at a time.

This was deliberate scope reduction, not a CrossFormer constraint.
`docs/HANDOFF_10K_DATASET.md` explicitly instructed the implementation to adapt to the
single-env `Manipulator` and "drop all the n_envs bookkeeping."

Selecting `render_backend="batch"` only selects Madrona's batch-capable implementation.
With `scene.build(n_envs=1)`, it currently renders a batch of one.

## Upstream code to port

Do not merge the entire upstream suite. It contains training algorithms, networks,
wrappers, and task behavior that are unrelated to MCAP generation. Port the batch
contracts and renderer mechanics into the validated production task.

### Required reference files

| Upstream path | What to reuse |
| --- | --- |
| `src/xsim/suite/environments/base.py` | Always-batched state, `(B, ...)` API, `envs_idx`, per-env timestep |
| `src/xsim/suite/environments/robot_env.py` | Batched camera poses, attached-camera synchronization, all-env rendering, per-env splat buffers |
| `src/xsim/suite/robots/robot.py` | Batched robot state/control/reset conventions |
| `src/xsim/suite/models/cameras.py` | Batch-camera construction and renderer option selection |
| `src/xsim/suite/models/cam_space.py` | Per-env camera sampling contracts |
| `src/xsim/suite/renderers/batch.py` | Madrona renderer/light configuration |
| `src/xsim/suite/renderers/splat_bg.py` | Chunked multi-camera gsplat background rendering |
| `src/xsim/suite/wrappers/image_obs.py` | `(B,V,3,H,W)` frame assembly and background compositing |
| `src/xsim/suite/policies/waypoint.py` | Batched poses, quaternion interpolation, and batched IK actions |
| `src/xsim/suite/policies/lift.py` | Batched cube-dependent waypoint construction |
| `src/xsim/algo/dagger.py` | `live`, `done`, success, and episode-length masks over a synchronous rollout |

The relevant upstream history begins with:

- `4c9df55`: suite environment foundation;
- `b724eb3`: `(n_envs, ...)` everywhere and partial reset;
- `1103ded`: Madrona batch rendering for image rollouts;
- `01e926e`: live gsplat backgrounds and per-env camera jitter;
- `7795a48`: arena-owned camera sampling;
- `d4b09df`: randomized attached-camera mounts;
- `07dc498`: batched randomized starts and branch-continuous IK;
- `07434c4`: collector extraction, including rollout masks.

The updated upstream ref must remain the source of truth during the port. Copying these
files verbatim is not sufficient because `TaskEnv` contains the calibrated production
scene, physical grasp protocol, shadow catcher, lighting controls, and MCAP metadata that
the suite does not.

## Production code that must become batched

### 1. Establish the batch contract

Add `TaskEnvCfg.n_envs`, defaulting to `1`.

Inside `TaskEnv`, keep the leading environment dimension even at `B=1`:

- cube position/yaw: `(B,3)` / `(B,)`;
- drop target: `(B,2)`;
- robot proprioception: `(B,7)` and EE pose `(B,7)`;
- camera extrinsics: `(B,4,4)` per camera;
- spawn, arm-start, appearance, light, and integrity metadata: one record per slot;
- lifecycle state: `live`, step count, released, success, and abort reason `(B,)`.

Change both:

```python
Manipulator(num_envs=B, ...)
scene.build(n_envs=B)
```

Avoid compatibility methods that silently flatten slot zero. Scalar convenience methods
may exist only when `B == 1`; batched generation should use explicitly batched methods.

### 2. Vectorize deterministic reset

Implement `reset(seeds: Sequence[int], envs_idx=None)`.

Each slot must use its own `np.random.default_rng(seed)`. Preserve the existing per-seed
draw order independently:

1. cube;
2. cameras;
3. drop target;
4. arm start;
5. lighting.

Vectorize placement and robot reset through Genesis `envs_idx`. Port upstream's batched
camera-pose buffers and attached-camera transforms.

Initial delivery only needs full-batch reset. Retain the `envs_idx` shape in internal
APIs so asynchronous refill can be added without another rewrite.

Acceptance: for a fixed seed list, slot `i` must produce the same sampled task metadata
as scalar generation of `seed[i]`, subject only to documented numerical differences from
batched physics.

### 3. Vectorize the scripted production policy

Do not replace the production protocol with upstream `LiftPolicy`; its segment list and
termination behavior differ.

Port the mechanics from upstream `WaypointPolicy`, but preserve this branch's:

- segment weights and release tail;
- per-start approach scaling;
- nearest face-aligned grasp yaw;
- physical/no-weld lift behavior;
- drop-target sampling;
- exact grasp offset and control rate.

Policy state must be per slot:

- current segment;
- remaining segment ticks;
- release and close boundaries;
- active/finished state;
- waypoint poses and gripper command.

Because approach duration can differ per slot, a single shared segment counter is not
correct. `step()` must return batched robot commands and a per-slot policy-state view.

Finished slots should receive a stable hold action until the batch completes.

### 4. Render all slots in one Madrona pass

Port from upstream `RobotEnv`:

- full arrays passed to batch camera `set_pose`;
- batched wrist-camera synchronization;
- `render_views(all_envs=True)`;
- `(B,H,W,3)` outputs per camera;
- per-camera, per-env splat background buffers;
- chunked gsplat rendering to cap peak memory;
- forced render after reset when simulation time has not advanced.

Preserve production-only rendering:

- calibrated low/side camera definitions;
- the verified wrist mount;
- aligned splat transform;
- shadow-catcher segmentation and compositing;
- 96x72 output;
- approved materials and light rig.

The installed Madrona Python adapter does **not** currently expose distinct light
parameters per environment. Its `get_lights_properties_torch()` repeats one scene light
rig across `num_worlds`. The native manager already consumes flattened
`num_worlds * num_lights` buffers, so independent lights are feasible, but require an
adapter/API extension.

Until that extension is implemented, the initial batch implementation must either:

- require lighting jitters to be zero; or
- sample one light condition per batch and explicitly record the induced correlation.

Do not silently claim independent per-episode lighting variation.

### 5. Add a batched integrity gate

Convert `GraspIntegrity` fields and measurements to `(B, ...)` arrays:

- close-time TCP/cube error;
- acquisition pass/fail;
- maximum rise;
- slip telemetry;
- abort reason.

The gate must mask each slot independently. One failed grasp must not stop or delete the
other slots' episodes.

### 6. Add an episodic MCAP writer pool

Keep `EpisodeMcapWriter` as the single-file primitive. Add a coordinator owning up to
`B` writer instances.

At batch reset:

- allocate the global episode number and path for each slot;
- open one writer per slot;
- write that slot's calibration and transforms.

At each recorded tick:

- render once for the batch;
- slice images, proprioception, gripper state, and transforms by slot;
- write only slots that are still recording.

At slot completion:

- close its writer immediately;
- apply the existing success gate;
- delete only that slot's failed MCAP unless `--save-failures`;
- append one manifest entry.

Writer pooling is deliberately independent of CrossFormer. The output remains one MCAP
per episode with the existing topics, encodings, timestamps, and frame cadence.

### 7. Integrate the generator and shard supervisor

Add `--n-envs`, default `1`, to `generate_task_dataset.py`.

Replace the scalar episode loop with batches while preserving:

- `--n-episodes` as attempted episodes, not batches;
- episode offsets and seed mapping;
- resume behavior;
- failure gaps in filenames;
- progress lines and manifest schema.

Handle the final partial batch either by building at full `B` and masking unused slots,
or by requiring an attempt count divisible by `B` initially. Prefer masking so resume and
extensions remain straightforward.

Change `run_shards.py` defaults only after single-process batching is validated. The
likely operating point is one process per GPU with `B=8..64`, not two scalar processes
per GPU. The optimum must be measured.

## Delivery sequence

### Milestone A: batch-shaped `B=1`

Refactor all public/internal state to retain a leading batch dimension, but keep
`n_envs=1`.

Gates:

- existing preview/video modes work;
- existing unit/smoke checks pass;
- decoded MCAP topic/schema/count/rate parity;
- fixed-seed spawn, camera, arm-start, light, and policy metadata parity;
- no throughput regression larger than 5%.

This is the most important correctness checkpoint.

### Milestone B: `B=2` physics and raster

Enable two independently seeded slots with raster rendering and no MCAP first.

Gates:

- independent cube/arm/camera state;
- no cross-environment contacts or pose leakage;
- independent policy and grasp-gate outcomes;
- scalar-vs-batch trajectory comparison for the same seeds.

### Milestone C: `B=2` Madrona and MCAP

Enable batch rendering, splat compositing, and writer pooling.

Gates:

- all three cameras differ correctly by slot;
- wrist transforms follow the matching robot;
- no frame is written to the wrong episode;
- `validate_mcap.py` passes every output;
- `compare_batches.py` reports `FORMAT: PASS`;
- failed-slot deletion does not affect successful siblings.

### Milestone D: scaling benchmark

Benchmark `B = 1, 2, 4, 8, 16, 32, 64` on one L40S using production resolution and
episode length. Record:

- scene build time;
- attempts and kept episodes/hour;
- physics, render, composite, MCAP, and reset time;
- peak VRAM and host memory;
- GPU utilization;
- success rate and frame-count distribution.

Stop increasing `B` when throughput plateaus, memory safety margin drops below the shard
supervisor threshold, or MCAP writing becomes dominant.

### Milestone E: production supervisor

Select the measured batch size, then change the multi-GPU launch layout. Run:

1. fixed-seed scalar/batch equivalence pilot;
2. 10-episode visual and MCAP checkpoint;
3. 100-episode throughput/correctness pilot;
4. operator approval;
5. production batch.

## Tests required before production

- Batch shape tests at `B=1`, `2`, and a non-power-of-two size.
- Deterministic independent RNG tests.
- Per-slot camera/extrinsic association test.
- Per-slot policy schedule test with different approach lengths.
- Mixed success/failure writer-pool test.
- Final partial-batch and resume tests.
- Scalar-vs-batch decoded MCAP comparison for identical seeds.
- Long-run descriptor/file-handle test.
- VRAM recovery test across repeated batches.
- Existing format and training-conversion gates.

## Explicit non-goals for the first implementation

- No change to CrossFormer.
- No change to MCAP schemas or topic names.
- No asynchronous slot refill.
- No attempt to run 2048 simultaneous MCAP writers.
- No replacement of the approved production task with upstream `Lift`/`LiftEZ`.
- No change to camera calibration, splat alignment, grasp protocol, or success criteria.

## Expected performance

The gain must be measured, not inferred from upstream's renderer-only frame rate.
Physics, policy IK, splat background refresh, YUYV conversion, and many open MCAP writers
remain in the production path.

Nevertheless, even a modest `B=8` removes seven redundant scene builds, CUDA contexts,
splat loads, and renderer dispatch streams relative to eight scalar processes. It is the
correct next throughput experiment once Milestones A-C establish parity.
