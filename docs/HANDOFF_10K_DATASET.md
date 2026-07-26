# Handoff — full-table spawns, randomized arm starts, 10,000-episode lift batch

Written 2026-07-23 for a fresh agent picking this up. Branch `new-dagger-crossformer`.
Read `AGENTS.md` first for the standing rules; this file is the delta.

> Historical incoming handoff: implementation through the required visual/MCAP checkpoint
> is now complete and pushed in commits `634a29f`, `9313268`, `1b0ba60`, and `6a3f514`.
> See `docs/10K_CHECKPOINT.md` and `docs/GRASP_WELD_UPSTREAM_AUDIT.md` for current state.
> Physical/no-weld grasp was approved on 2026-07-23 and the lift weld is retired. The
> batch shadow catcher was also approved after a labeled before/after review. The complete
> ten-case checkpoint was regenerated with both decisions under
> `/data/store/griffen_sim_mcaps/lift_10k_checkpoint/visual_batch`. No generation process
> remains active and the 10,000-episode run has not started.

---

## Objective

Generate a **10,000-episode lift dataset** — the largest CrossFormer training set so far.
Before that, the demonstration distribution has to match the deployment story.

**Deployment is a continuous "run mode" loop on the real robot:** a human shoves the cube
to a new spot, the robot finds it, picks it up, carries it to the middle of the table,
drops it from a few inches up — then the cube is moved again **without the arm ever
returning to home**. The lift → transport → air-drop protocol is correct and **stays
exactly as it is**. What must change is (a) where the cube can spawn and (b) where the arm
can start.

---

## State of the working tree — NOTHING IS COMMITTED

```
A  .gitmodules                  <- new submodule, staged
A  extra/gs-madrona             <- gitlink @ 4326a21, staged
 M pyproject.toml               <- 4 uv stanzas for the madrona source build
 M uv.lock
?? scripts/fetch_assets.sh      <- new, working
?? scripts/spawn_feasibility.py <- new, working, already produced its results
```

`src/` is **untouched**. None of the env/policy work below has been started.

Backups of the pre-change `pyproject.toml` / `uv.lock` are at
`/tmp/claude-1002/-home-grifflee/85e1d7ae-b9b7-4cdc-a136-306aeeb953a8/scratchpad/*.bak`
(session-scoped — copy them somewhere durable if you want a rollback path).

---

## DONE and verified — do not redo

### 1. Splat asset provenance ✅

`assets/lab_aligned.ply` is **byte-identical** (`cmp` clean) to the sole asset of GitHub
release **`assets-v1`** on `mhyatt000/xarm-sim`: 328,002,116 bytes, md5
`13a21b6e3df686d2cc7169f52f0879a3`. It is wired as `DEFAULT_SPLAT_PATH`
(`src/xsim/task_env.py:38`) with an **identity** runtime transform (`splat_pos (0,0,0)`,
`splat_quat (0,0,0,1)`, `splat_scale 1.0`) — the alignment is baked into the ply, the old
two-layer scan-frame + runtime-pose arrangement is gone.

`scripts/fetch_assets.sh` (new) re-fetches it idempotently. `.ply` files are gitignored and
never LFS'd; a missing file is a hard `FileNotFoundError` at env construction
(`task_env.py:864-865`), not a silent fallback.

**Still to do:** `AGENTS.md` §3 still says the splat "defaults to `assets/lab_clean.ply` if
present, else `/data/store/lab.ply`" — stale, needs replacing with the above. Ready-to-paste
replacement text is in the git history of this handoff's companion work, or just write it
fresh from the facts here.

### 2. Madrona batch renderer — FIXED ✅

It was aborting with SIGABRT/exit 134. **It now renders.** Verified today:

```
raytracer  (use_rasterizer=False): RENDER OK shape=(1, 64, 64, 3) uint8 mean=7.824  exit 0
rasterizer (use_rasterizer=True):  RENDER OK shape=(1, 64, 64, 3) uint8 mean=7.834  exit 0
```

**Root cause (for the record):** madrona ships no precompiled device code — it JIT-compiles
its megakernel and BVH kernels with NVRTC and links with nvJitLink. The PyPI wheel
`gs-madrona 0.0.7.post2` (which `genesis-world 1.2.0` hard-pins with `==`) probed
`libnvrtc.so.13` first (CUDA 13.0 here) and fed the result to the **nvJitLink 12.4.127 it
bundled** → `ERROR 4 in nvvmAddNVVMContainerToProgram` → abort. Separately, nvJitLink 12.4's
arch table stops at `sm_90a`, so it could never target the 5090's `sm_120` anyway.

**The fix:** mirror upstream — vendor `mhyatt000/gs-madrona` as the `extra/gs-madrona`
submodule (pinned at `4326a21`, "Undef CUDA 13's cuMemAdvise→cuMemAdvise_v2 rename before
dynamic loading") and build from source, escaping the genesis pin with
`[tool.uv] override-dependencies`. Installed result is an **editable source build, version
0.0.8**, and its bundled `libnvJitLink.so.12` is now **release 12.8** (sm_120-capable).

`mean≈7.8` is near-black, and that is **expected**: madrona takes no lights from the Genesis
scene. See the trap below.

### 3. Reach boundary — measured ✅

`scripts/spawn_feasibility.py` (new), 483 cells at 3 cm pitch, one scripted-lift episode per
cell. Results are committed under `docs/spawn_feasibility/` (`feasibility.png` heatmap,
`results.jsonl` per-cell, `summary.json`, `sweep.log`) so they travel with the repo; they
were generated into `/data/store/griffen_sim_mcaps/spawn_feasibility/`.

**The feasible spawn region is an ANNULUS in radius from the base, not a rectangle.**

| r (m) | r (in) | clean rate |
|---|---|---|
| < 0.150 | < 5.9" | **0/44 — total failure** |
| 0.175 | 6.9" | 0.75 |
| 0.250 | 9.8" | 0.95 |
| **0.275 – 0.700** | **10.8" – 27.6"** | **1.00 (269/269)** |
| 0.725 | 28.5" | 0.67 |
| >= 0.750 | >= 29.5" | **0/10 — total failure** |

Measured full extension ≈ **0.72–0.74 m**. The sweep was deliberately stopped at x=0.74 —
grifflee does not want spawns near the limit (see his constraint below).

---

## Remaining work

### Task A — cube spawn annulus (approved, spec'd, NOT implemented)

File: `src/xsim/task_env.py`.

```python
# in TaskEnvCfg
# Cube spawn is an ANNULUS about the base, not a box: measured 2026-07-23
# (scripts/spawn_feasibility.py, 483 cells) the top-down grasp is 100% clean for
# r in 0.275-0.700 and fails entirely below 0.15 / above 0.75. Capped well inside
# that (0.445 = 60% of the 0.74 m measured extension) per grifflee: stay at
# 70-80% of full extension, do not push the limit. None = legacy rectangle-only.
spawn_radius: tuple[float, float] | None = (0.25, 0.445)
spawn_max_tries: int = 100

rectangle_x: tuple[float, float] = (0.0, 0.445)      # was (0.20, 0.40) — now the
rectangle_y: tuple[float, float] = (-0.288, 0.288)   # proposal box; radius rejects
```

Add `_sample_lift_xy(rng)` — rejection-sample x,y from the rectangle until
`spawn_radius[0] <= hypot(x,y) <= spawn_radius[1]`. **Mirror the existing
`_sample_free_stack_xy` helper** for structure and style. Call it from the lift branch of
`reset()` (~`task_env.py:1001-1004`). `spawn_radius=None` must reproduce today's plain
uniform-rectangle behaviour exactly.

**Why these numbers:** area +20.7% over the current box (grifflee asked for ~20%, an earlier
+60% proposal was rejected as too much). Validates at **165/165 clean** against the measured
sweep, including **12/12 at x < 0.10** (beside the robot, not in front) — which was the main
risk of a radial constraint, since the robot has a front and the annulus doesn't. The real
gain is angular, not areal: today's box spans ~±36° of approach direction, the annulus spans
the full ±49° the table allows.

**Two known costs, both accepted:**
- Seed compatibility breaks (rejection sampling consumes a variable number of draws, so the
  camera/drop/joint streams shift per seed). Unavoidable when changing the spawn region.
  Existing batches stay valid; seed *N* just won't reproduce its old scene.
- The camera lookat box widens as a side effect — it is derived from `rectangle_x`/
  `rectangle_y` (`_camera_lookat_bounds`, `task_env.py:1059-1063`). Moot once Task C lands.

### Task B — randomized arm start pose (approved, spec'd, NOT implemented)

Files: `src/xsim/task_env.py`, `src/xsim/grasp_env.py`.

**Port upstream's implementation, don't invent one.** See
`git show upstream-main:src/xsim/suite/environments/robot_env.py` → `_randomize_init_tcp`
(~434-448) and the `init_tcp_box` field (~47-58). Upstream samples a TCP position uniformly
in a box, **keeps the home EE orientation**, IK-solves, and seats the arm via
`set_arm_qpos`. Their production value is `((0.10, 0.40), (-0.3048, 0.3048), (-0.01, 0.30))`.
Adapt to our single-env `Manipulator` (drop all the n_envs bookkeeping).

`Manipulator.reset` (`grasp_env.py:649`) currently takes only `arm_qpos_offset` — add an
absolute `arm_qpos=` path, keeping the offset behaviour. Solve IK with the existing
`self._robot_entity.inverse_kinematics(...)` call already used by `go_to_goal`
(`grasp_env.py:719`).

**Mixture, not a uniform box** — the deployment loop is not uniform. Defaults:

| bucket | weight | sample |
|---|---|---|
| `post_drop` | 0.40 | TCP over the drop zone (x ~ U(0.30,0.40), y≈0 + jitter) at transport height (`top_z + 0.018 + 0.09` ≈ 0.098), gripper **open** — literally the state the arm is in when the human moves the cube again |
| `far` | 0.25 | xy radius ~ U(0.50, 0.58), heading ~ U(-40°, +40°), z ~ U(0.05, 0.25) — forces reaching **backwards** toward the base |
| `broad` | 0.25 | xy radius ~ U(0.20, 0.58), heading ~ U(-49°, +49°), z ~ U(0.02, 0.35) |
| `home` | 0.10 | today's behaviour (home qpos + `arm_start_jitter_deg`) |

**Safety cap: never sample a start TCP beyond xy radius 0.58** (~78% of the 0.74 m measured
extension). grifflee's explicit constraint is 70–80% of extension, do not probe the limit.
Note this cap is deliberately *wider* than the 0.445 cube-spawn cap — a start pose is
free-space positioning at height, not a reach-down-and-grasp, so it is far less demanding.

Keep the home EE orientation for all sampled starts (as upstream does); do not randomize
wrist orientation in this pass. On IK failure or a large achieved-vs-requested TCP error,
redraw up to N times then fall back to home — **count and expose the fallbacks**.

Add `arm_start_mode: Literal["home", "mixture"]` (or an `init_tcp_box=None` sentinel).
Reproducing today's behaviour exactly must stay reachable by config — approved batches
depend on it.

### Task C — grasp integrity (REQUIRED before B ships; NOT implemented)

Files: `src/xsim/scripted_lift_policy.py`, `scripts/generate_task_dataset.py`.

**⚠ THE MOST IMPORTANT ITEM IN THIS HANDOFF. `lifted=True` is currently not evidence of a
grasp.**

The sweep found 37 near-base cells reporting `lifted=True` with the cube rising 0.22–0.32 m
while `min_ee_cube` (closest the gripper ever got) was **0.06–0.23 m**. The gripper never
touched it. Cause: `env.grasp_lock()` fires on a **fixed tick** —
`ScriptedLiftPolicy.reset()` computes `grasp_lock_step = 1 + sum(self._segment_steps[:3])`
(`scripted_lift_policy.py:145`) and `run_episode` (`generate_task_dataset.py:498-501`) welds
when the step index hits it, regardless of whether the fingers arrived. The weld then
teleports the cube to `link_tcp`. See `docs/GRASP_TELEPORT_INVESTIGATION.md`.

This is survivable today only because the approach distance is always ~3 cm. **Task B makes
approach distance vary from ~3 cm to ~0.6 m, at which point a fixed-tick weld fabricates
grasps across the whole batch.** Do not ship B without C.

1. **Gate the weld on measured proximity.** Reuse `lift_expert.py:127-138`'s thresholds
   (`grasp_r = 0.035`, `tol_xy = 0.02`, `tol_z = 0.015`) rather than inventing new ones.
   Demote the tick index to a **timeout**: if proximity is never met, do not weld and let the
   episode fail honestly.
2. **Record `min_ee_cube` and `close_xy_err` in the manifest, permanently.** These are the
   only reason the bug was visible. Also record whether the weld fired, at which step, and
   the TCP-to-cube distance at that moment.
3. **Seed approach IK from live qpos, not home.** `XARM7_ROBOT_CFG` sets
   `"ik_init_at_home": True` (`task_env.py:196`), which returns far-branch/elbow-flip
   solutions from poses far from home. `lift_expert.py:243-247` documents this and seeds
   from live qpos instead.
4. **Scale the approach segment by distance.** `SEGMENT_WEIGHTS = (2.6,.8,.8,.4,1.0,.6)` ×
   `steps_per_segment = 108` is distance-independent, so a 0.6 m approach is commanded in the
   same 2.33 s as a 3 cm one. Scale **only** the first segment, floored at today's value
   (short approaches unchanged) and capped. Leave the other five alone — they are all
   distance-bounded already.

**Consequence to handle:** episode length becomes variable, which breaks the 115–240 frame
gate in `scripts/validate_mcap.py:55-58` (`_recorded_frames`, `FRAME_MARGIN_LOW=37`,
`FRAME_MARGIN_HIGH=13`). That gate is **derived** from the fixed weights, so re-derive it
from the new floor/cap — do not just loosen it.

### Task D — camera jitter as a toggle (approved, NOT implemented)

File: `src/xsim/task_env.py`.

Commit `cce4093` deleted `cam_jitter_deg` / `cam_jitter_cm` / `wrist_jitter_deg` /
`wrist_jitter_cm` and replaced them with `camera_mode: Literal["fixed","ball","shell"]`,
default `"shell"`. grifflee wants the old ±15°/±5 cm behaviour back **as a toggle**.

Current options and why none of them is "slight movement":

| mode | position | aim |
|---|---|---|
| `fixed` | exactly calibrated | exactly calibrated |
| `ball` | solid ball r=0.10 m about each calibrated position | random point in the lookat box |
| `shell` (default) | chopped spherical shell about the **base**: outer 1.348 m, inner 0.674 m, clipped x ∈ [-0.305, 1.039], z ∈ [-0.01, 0.922] | random point in the lookat box |

`ball` is the middle option for *position* but not for *orientation*: because the aim point
is drawn from the whole lookat box, viewing direction already swings **38.7°** (low) /
**32.4°** (side) end-to-end — and since the lookat box is derived from the spawn rectangle,
Task A grows that to **~59° / ~49°**. So `ball` gets *more* aggressive as a side effect of
widening spawns, which is backwards.

**Restore `camera_mode="jitter"`** from `old-dagger-soft:src/xsim/task_env.py:847-884`. It is
self-contained: re-add the four config fields, the `_rot_from_rpy_deg` helper
(`old-dagger-soft:54-57`, deleted here), and the jitter branch of `_randomize_cameras`. It
applies an rpy delta in the camera frame and an xyz delta in world around
`self._nominal_c2w_gl` (still present, `task_env.py:781`), and never touches the lookat box.
Production values: 15 deg / 5 cm, wrist jitter 0.

Keep `shell`/`ball`/`fixed` available; only the production default changes. Fix the
`AGENTS.md` §3 bullet that still documents the deleted flags as "the PRODUCTION RECIPE" —
passing them today is a tyro hard error.

### Task E — port the `"batch"` render path (NOT implemented)

Files: `src/xsim/task_env.py` (+ a new small config module if you prefer).

The wheel works now, but **`TaskEnv` has no `"batch"` code path** — `render_backend` is only
`Literal["raster","nyx"]` (`task_env.py:623`) and `grep -r madrona src/` returns nothing.
Upstream's lives behind `render_backend="batch"`. Four contained pieces:

1. Pass `gs.options.renderers.BatchRenderer(use_rasterizer=cfg.use_rasterizer)` into
   `gs.Scene(renderer=...)` — see `upstream-main:.../robot_env.py:85-89` (`_scene_renderer`).
2. Drop `env_idx=0` on camera creation for batch (upstream ~`robot_env.py:146-151`).
3. **Add the light rig.** Copy `BatchConfig` from
   `git show upstream-main:src/xsim/suite/renderers/batch.py` — key/fill directional pair,
   intensities 1.7 / 0.85. **Madrona takes no lights from the Genesis scene; without this,
   frames are near-black** (that is the `mean≈7.8` in the probe above).
4. The splat composite carries over **unchanged** — upstream uses the identical
   `np.where((seg == 0)[..., None], bg, rgb[..., :3])` trick we already have
   (`upstream-main:.../robot_env.py:347-351` vs our `task_env.py:1196`).

Most of upstream's apparent complexity is per-env buffer bookkeeping for `n_envs=2048` that
is unnecessary at `n_envs=1`.

**Open decision for grifflee:** which render path the 10k batch actually uses. The standing
rule has been "never run model-facing sim without `render_backend=nyx`", but `c82de3e`
("dagger: make composite gsplat the model-facing default") moved the default the other way,
and grifflee has said he wants "the new gsplat and madrona config". Produce a 3-seed ×
3-camera parity panel across nyx / raster-composite / batch — in the style of
`/data/store/griffen_sim_mcaps/dagger_runs/render_parity_check/` — and let him pick.

### Task F — the visual checkpoint (blocking, grifflee asked for this explicitly)

**At least 10 renders for him to inspect before any big run**, covering the *borders* of the
new distributions rather than random draws:

- edge spawns: nearest-to-base (r≈0.25), farthest (r≈0.445), and the ±y extremes
- a **backwards-reach** episode: arm starts extended at the far edge, cube near the base
- a **post-drop start**: arm above table center, gripper open, cube somewhere new
- a labeled spawn/start coverage figure (feasibility heatmap with the chosen annulus drawn
  on it) so the borders are legible as numbers
- the render-path parity panel from Task E

MP4s + a contact-sheet PNG, all under `/data/store/griffen_sim_mcaps/`.
Then `compare_batches.py` → `FORMAT: PASS` and `validate_mcap.py` against the real reference.

**Nothing generates until he has read these.**

### Task G — the 10k batch

- **Seeds `100000–109999`.** Clear of every reserved range (9000–9099 batch_v3, 20000–24999
  the 5k lift batch, 30000+ stack, 51k eval, 61k+ DAgger).
- **Output `/data/store/griffen_sim_mcaps/lift_10k_100000`.** Never under
  `~/repo/xarm-sim/outputs` — that is the small root disk.
- **Sharding is manual.** Without appearance randomization the generator is a strictly
  sequential single-process loop and `--pool-workers` is silently ignored
  (`generate_task_dataset.py:557-573`). Follow the batch_v3 pattern: N concurrent shard
  processes with per-shard `--seed`/`--episode-offset`, then `scripts/merge_shards.py`.
  **30 GB RAM with ~16 GB free — not the GPU — is the binding constraint on shard count.**
- **Budget:** ~880 GB (88 MB/ep measured) and ~9 h at 2 shards / ~19 h single-process, from
  the observed 6.7 s/ep per shard on `batch_2500_nyx_20000`. `/data/store` has 4.7 TB free.
- **Verify the effective config from the written `manifest.json`, never from the command
  line.**
- **Never kill the generator to trim a batch** — `manifest.json` is only written at a natural
  run end.

**Two provenance bugs — FIXED in `1b0ba60`, no action needed (verified 2026-07-25):**
- `_write_manifest` used to compute `splat_md5` only when `render_backend == "nyx"`, so on
  raster/composite/batch it recorded `None` and `merge_shards.py`'s `(splat_file,
  splat_md5)` agreement check degenerated to a no-op. The guard is gone; the md5 is now
  computed whenever the splat file exists, on every render path.
- `_write_manifest` used to write `success_rate` as the string `"n/N"` while
  `merge_shards.py` overwrote it with a float ratio. It is now a float in both places.

Confirmed empirically against a `render_backend=batch` run: the shard manifest carries
`splat_md5: 13a21b6e3df686d2cc7169f52f0879a3` and `success_rate: 1.0` (float). Do not
re-fix these; the line numbers cited in earlier revisions of this file no longer exist.

---

## Traps

1. **`lifted` / `max_rise` are not trustworthy** until Task C lands. `min_ee_cube` and
   `close_xy_err` are. See Task C.
2. **Madrona renders black without an explicit light rig.** It ignores Genesis scene lights.
3. **Multi-CUDA host.** `/usr/bin/nvcc` is **12.0**; 12.8 / 13.0 / 13.3 live under
   `/usr/local/`; `CUDA_HOME` is unset. A naive rebuild picks the one toolkit that cannot
   target `sm_120`. Export `CUDA_HOME=/usr/local/cuda-12.8`,
   `CUDACXX=$CUDA_HOME/bin/nvcc`, put its bin first on PATH, and pass the
   `CUDA_nvJitLink_LIBRARY` / `CUDA_NVJITLINK_LIBRARY` / `CUDA_cudart_static_LIBRARY`
   CMake pins (already documented in the `[tool.uv]` comment in `pyproject.toml`).
4. **The camera lookat box is derived from the spawn rectangle.** Widening spawns widens
   camera aim. Task D removes the coupling.
5. **RNG draw order in `reset()` is load-bearing** and documented at
   `task_env.py:1008-1009` (cube → cameras → drop → start joints). New draws go **last**.
6. **`task_env.DEFAULT_SPLAT_POS` (~:483) and `clean_splat.RAW_SCAN_SOLVE` (~:52) differ by
   ~2 cm.** `verify_splat_alignment.py` / `icp_splat.py` are seeded from a solve that is
   stale relative to the bake in use. Only matters on a rescan. Do not "fix" by eye.
7. **`--env.cam-jitter-deg` / `--env.cam-jitter-cm` no longer exist** — they are a tyro hard
   error today, despite AGENTS.md calling them the production recipe.
8. Standing repo rules that still apply: `table_mode="slab"` always; never move the
   calibrated camera constants; `gripper_grasp_dof: 0.58` is load-bearing (0.53 drops the
   cube); show grifflee a labeled artifact before building on any new visual interpretation.

---

## Suggested order

```
C (grasp integrity)  ─┐
A (spawn annulus)    ─┼─> F (visual checkpoint) ──> grifflee reads ──> G (10k batch)
B (arm starts)       ─┤
D (camera jitter)    ─┤
E (batch render path)─┘
```

C gates B. A, D, E are independent. F needs all of them and is blocking on a human read.
