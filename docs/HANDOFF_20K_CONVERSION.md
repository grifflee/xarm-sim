# Handoff — 21k lift dataset, arec conversion, and the state of the pipeline

Written 2026-07-26 evening for the agent picking this up. `AGENTS.md` holds the standing
rules and `docs/GENERATION_RUNBOOK.md` the batch procedure; this file is the current state
and the immediate next step.

**Everything is on branch `new-dagger-crossformer`, pushed through `54a15e8`.**

---

## 1. What exists

`/nas/glee10/sim_mcaps/lift_10k_100000` — one batch directory, two generation waves:

| | shards | episodes | seeds | lighting |
|---|---|---|---|---|
| wave 1 | `shard_00`–`07` | 10,500 | 100000–110499 | **fixed** (none) |
| wave 2 | `shard_08`–`15` | 10,500 | 110500–120999 | **jittered** |

~21,000 episodes, ~14 GB, 96x72, `render_backend=batch` (Madrona + gsplat), physical
no-weld grasp, camera jitter 15deg/5cm, front-only spawn annulus r=0.25–0.55.

**Success rate 99.98%** — 3 policy failures in ~19,000 episodes. Wave 1 was a clean
10,500/10,500 and passed `compare_batches.py` with `FORMAT: PASS` against the 21 real
robot episodes in `~/ghome/mcaps/lift_valid`.

**The two waves are NOT identical**, deliberately:
- wave 2 has per-episode lighting jitter (direction ±12deg, intensity ±25%, shadow
  strength 0.30–0.60);
- wave 2 also has the shadow-blur resolution fix, wave 1 does not — so wave 1's table
  shadow is markedly flatter. If you ever compare the halves, shadow appearance is a
  confound, not a clean lighting A/B.

---

## 2. arec conversion — **DONE 2026-07-26 23:02**

`xarm_sim_96` v0.0.1 is built and verified at `/nas/glee10/arrayrecords/xarm_sim_96`,
symlinked to `~/.cache/arrayrecords/xarm_sim_96`.

| | |
|---|---|
| episodes | 20,995 (all present, contiguous idx 0–20994) |
| records | 3,882,631 (184.93/episode vs 185.04 baseline) |
| size | 94 GB — image 92 GB / proprio 2.1 GB, 777 shards each |
| build | 2,634 s = **44 min** at 1,474 rec/s |
| peak RSS | 11.1 GB (flat after warm-up; cap was 200 GB) |
| VRAM | one 34.5 GB grab on GPU 0 only |

**Two things in the original plan below were wrong, and are left in place with corrections
so the reasoning is auditable:**

1. **`--mp` tuning is not a lever** (§2d suggested it might halve the run). Measured
   4/8/16 = 130.5/130.2/130.9 s, identical. The bottleneck was the single main process
   downstream of `mp_prefetch`; the workers idled at ~6% CPU.
2. **The 8.4 h estimate (and the 129 rec/s behind it) measured a real defect, not a floor.**
   `robot_keypoints_in_cameras` rebuilt the whole robot model — full URDF mesh load plus a
   fresh `jax.jit` compile — once per episode, ~7/8 of main-process time. Fixed in
   crossformer `e881a76`. The run took 44 minutes. See `AGENTS.md` for the full diagnosis
   and the `ps` → `py-spy` method that found it.

Also note: the §2d write-pass timing excluded the counting pass in `main()`, which fully
decodes every image in every MCAP a second time just to sum a tqdm total. `--est-steps`
skips it; `total` feeds only `cfg.progress()`, so it is provably cosmetic.

### The command actually used

```bash
cd ~/ghome/crossformer
systemd-run --user --scope -p MemoryMax=200G -p MemorySwapMax=0 -- \
  uv run --no-sync python scripts/data/make/mcap_robot_sim.py \
    --mode build --no-recursive --mp 4 \
    --name xarm_sim_96 --version 0.0.1 \
    --path /nas/glee10/sim_mcaps/lift_21k_flat \
    --root /nas/glee10/arrayrecords \
    --est-steps 3884915 \
    --urdf /home/glee10/ghome/xarm-sim/xarm7_standalone.urdf \
    --mesh-dir /home/glee10/ghome/xarm-sim/assets
```

The flat input dir `/nas/glee10/sim_mcaps/lift_21k_flat` (20,995 hard links, every file
head+tail magic-validated, 0 rejected) is **kept** — it is the provenance record of exactly
what was converted, and hard links cost no disk.

---

## 2 (original). The plan as written before the run

### 2a. `merge_shards.py` will silently drop ~4,300 episodes. Do not use it here.

Wave 2 was interrupted mid-run (see §4) and shards 10–15 were **resumed**. A resumed shard
writes a manifest covering only the episodes *that process* generated (~500–600), not the
~700 that preceded it. `merge_shards.py` iterates manifest entries, so it would link the
resumed portion and **silently drop the rest — no error, no warning**.

Route around it: build a flat directory of hard links to every valid `.mcap` across all 16
shard dirs, and convert from there.

```python
# sketch: link every VALID episode into one flat dir
import pathlib
MAGIC = bytes.fromhex("894d434150300d0a")
src  = pathlib.Path("/nas/glee10/sim_mcaps/lift_10k_100000")
dst  = pathlib.Path("/nas/glee10/sim_mcaps/lift_21k_flat"); dst.mkdir(exist_ok=True)
n = 0
for f in sorted(src.glob("shard_*/episode_*.mcap")):
    with f.open("rb") as fh:
        head = fh.read(8); fh.seek(-8, 2); tail = fh.read(8)
    if head == MAGIC and tail == MAGIC:            # skip any truncated file
        (dst / f"episode_{n:06d}.mcap").hardlink_to(f); n += 1
print(n)
```
Hard links, so this costs no extra disk. Verify the count matches the sum of per-shard
`ls | wc -l` before proceeding.

### 2b. The converter is NOT CPU-only. Never run it during generation.

`mcap_robot_sim.py` imports JAX transitively, and **JAX preallocates ~75% of a GPU on
import** even though the script contains no CUDA code. On 2026-07-26 this tripped
`run_shards.py`'s total-VRAM guard at 94% and the supervisor killed six live shards. Wait
until `nvidia-smi` shows 0 MiB on all four GPUs.

(Untested idea: `XLA_PYTHON_CLIENT_PREALLOCATE=false` may make concurrent conversion safe.
Do not find out on a production run.)

### 2c. The command

```bash
cd ~/ghome/crossformer
systemd-run --user --scope -p MemoryMax=200G -p MemorySwapMax=0 -- \
  uv run --no-sync python scripts/data/make/mcap_robot_sim.py \
    --mode build --no-recursive --mp 4 \
    --name xarm_sim_96 --version 0.0.1 \
    --path /nas/glee10/sim_mcaps/lift_21k_flat \
    --root /nas/glee10/arrayrecords \
    --urdf /home/glee10/ghome/xarm-sim/xarm7_standalone.urdf \
    --mesh-dir /home/glee10/ghome/xarm-sim/assets
```

Every flag is load-bearing:
- **`--no-recursive`** — it defaults to `recursive=True` globbing `**/*.mcap`. Pointed at a
  merged batch root it finds both the flat hard links and the shard-dir originals and
  ingests **every episode twice**, silently.
- **absolute `--urdf` / `--mesh-dir`** — the defaults are relative and only resolve inside
  xarm-sim, not crossformer.
- **`--root /nas/...`** — 21 TB free. `/home` is a shared NFS at 91%.
- **`systemd-run` memory cap** — this box was crashed once by a leak in
  `compute_dataset_statistics` (since fixed upstream in `3c492c0`, which replaced a
  `list()` of every record with streaming Welford accumulators).
- **new `--name`** — 96x72 gives image shape `[3,72,96,3]`, a different schema fingerprint
  from the 640x480 `xarm_sim` build, so `--append` to that dataset is refused by design.

Afterwards symlink where the loader looks:
`~/.cache/arrayrecords/xarm_sim_96 -> /nas/glee10/arrayrecords/xarm_sim_96`
(matching the existing `xarm_sim` symlink convention).

### 2d. Timing — measured, not guessed

**129 records/sec, 185 records/episode.** One record = one 1/30 s timestep (all 3 camera
views + proprio), after `filter_noops` drops ~5% of steps where the arm did not move.

```
21,000 episodes x 185 = 3.9M records  ->  ~8.4 hours
10,500 episodes       = 1.9M records  ->  ~4.2 hours
```

`--mp` worker scaling is **unmeasured** — 4 is the default and more may be much faster. A
3-minute test on ~100 episodes would settle it and could halve the run. Do that first.

---

## 3. What changed in the pipeline today (all committed)

Read `AGENTS.md`'s 2026-07-26 section for the full list. The load-bearing ones:

- **The model only ever sees 64x64.** Training (`loader.py:imresize`) and serving
  (`GrainlikeWrapper._resize_images` -> `augmax.Resize(64)`) both **squash** the whole 4:3
  frame. There is **no crop** anywhere in crossformer `dev` — `center_crop` is defined and
  never called, and three separate comments claiming otherwise are stale. This is why
  render res dropped 640x480 -> 96x72 (~44x fewer pixels, none of which the model saw).
- **Grasp acquisition is a gate now, not a log line.** Thresholds 12/6/6 mm (were
  35/20/15); `GraspIntegrity` aborts the episode at the close rather than carrying a grasp
  it has already measured as bad. `abort_reason` is in the manifest.
- **Spawn region**: front-only (`x >= 0.115`), `+y <= 0.20`, radius out to 0.55 (was 0.445,
  which was only 60% of measured reach). The `+y` cap comes from
  `scripts/spawn_visibility.py` — lift previously had **no** camera-visibility constraint
  at all, unlike stack.
- **Arm start** is constrained to the table and kept clear of the cube (3D check, so the
  `post_drop` bucket is not distorted).
- **Lighting/shadow jitter** with physically-enforced bounds: lights cannot go below 30deg
  elevation, shadow strength capped at 0.65. Off by default.
- **Three resolution-dependent constants** were found and fixed (`STATIC_CAM_MARGIN_PX`,
  the visibility-audit margin, `batch_shadow_blur_px`). All were absolute pixel counts
  tuned at 640x480 that silently changed meaning at 96x72. **There may be more — a
  deliberate sweep is worthwhile.**

New tools: `run_shards.py` (sharded supervisor with guards + `--resume`),
`spawn_visibility.py`, `bench_render.py`, `bench_shards.py`, `res_sweep.py`,
`render_lighting_range.py`.

---

## 4. Known damage and quirks in this dataset

- **3 permanently missing episodes**: `011365`, `012644`, `013613`. Eight files were
  deleted in total (6 truncated by the guard kill, 2 destroyed by a `--dry-run` that
  mutated the filesystem, since fixed), but `--resume` regenerated five of them because
  those were each shard's *last* write and resuming at `max_valid + 1` landed on them.
  The three above were not: two were live writes in shards that were never resumed, and
  `013613` was a hole *mid-sequence*, which `--resume` steps over rather than fills.
  **Known limitation of `--resume`: it recovers a truncated tail, not an interior hole.**
  Immaterial at 21,000 episodes, and the flat-link script in §2a skips holes anyway since
  it renumbers contiguously.
- **Shards 10–15 have incomplete manifests** (resumed portion only). The episodes are fine;
  the per-episode diagnostics (spawn, `close_xy_err`, light draws) for their first ~700
  episodes each are permanently lost. Nothing downstream needs them — the converter reads
  MCAPs, not manifests.
- **Shard stdout never reaches the shard logs.** Something in the Genesis stack swallows it
  (`flush=True` and `PYTHONUNBUFFERED=1` both applied, neither fixed it). Consequence: the
  supervisor's success-rate guard is **inert**. Use the filename-gap method instead —
  failures are deleted, files are `episode_{attempt_index}.mcap`, so a failure is a gap.
  `run_shards.py` also has a file-based stall guard that does not depend on stdout.

---

## 4b. BIGGEST KNOWN PERFORMANCE ISSUE: `n_envs=1`

`TaskEnv` hardcodes `self.scene.build(n_envs=1)` (`task_env.py:985`). Every generation
shard runs **one** environment and steps episodes serially; parallelism comes from running
8 separate OS processes.

**This is off Madrona's design point by three orders of magnitude.** Madrona exists to
render thousands of environments in one process. Upstream does exactly that —
`scripts/simpledagger.py` runs "one rank per GPU, each owning `n_envs` envs AND a DDP
replica", with tuning notes about VRAM "peaking ~30GB and OOM'ing 44GB L40S ranks at
`n_envs >= 2048`" — i.e. mhyatt is batching on this same hardware.

**Why we are not, and it was never a decision.** `n_envs=1` was inherited from the original
single-environment `GraspEnv` demo and carried forward by `670fd85`, a *rename* commit.
Nobody weighed batched against serial. It was then actively cemented: this repo's own
`docs/HANDOFF_10K_DATASET.md` instructs future porting work to strip batching out —

> "Adapt to our single-env `Manipulator` (**drop all the n_envs bookkeeping**)."
> "Most of upstream's apparent complexity is per-env buffer bookkeeping for `n_envs=2048`
> that is **unnecessary at n_envs=1**."

That advice reads upstream's batching as incidental complexity rather than as its
performance architecture, and it is self-reinforcing: single-env, therefore strip
batching, therefore still single-env. **Do not follow it.**

**Measured cost** (`scripts/bench_render.py`, 2026-07-26, 96x72 batch renderer):
- one process = **4.90 s/episode**; physics 40% and IK 18% of that, both of which vectorise
  across environments — that is 58% of the work sitting on the exact axis batching helps;
- **scene build 10.6 s and process startup 14.8 s are paid PER PROCESS**, and we pay them
  8x per batch;
- 8 shards deliver ~3,200 ep/h, versus 735 ep/h for one process — a 4.4x return on 8x the
  processes, because they contend for 4 GPUs instead of sharing one batched context.

**The foundation already exists**: `GraspEnv` takes `num_envs`
(`grasp_env.py:256`, `scene.build(n_envs=env_cfg["num_envs"], ...)`). What blocks it is
that `TaskEnv` is scalar throughout — `cube_pos()`, `reset(seed=...)`, the scripted policy,
`GraspIntegrity` and the MCAP writer all assume one environment. Vectorising those is real
work, not a config flag.

**Bounded, not free:** upstream's own comment records OOM at `n_envs >= 2048` on 44 GB
L40S, so there is a ceiling on this hardware. But anything above 1 is likely a large win,
and this is a far bigger lever than shard count or render resolution — both of which were
measured on 2026-07-26 and found close to exhausted.

## 5. Open items

- **Shard count**: 8 stays. `bench_shards.py` found 4/6/8 all within noise (~3,000–3,200
  ep/h) and none beating the real batch's 3,211 ep/h. An earlier "use 6" claim came from a
  transient and did not reproduce.
- **`docs/to-do-variation.md`** is the sim2real variation backlog: physics randomization
  (does not exist at all — one friction, one mass, for a contact-rich grasp), actuation
  noise (does not exist), and the fact that the policy has never seen a human hand or a
  cube in motion, both of which occur in the real deployment loop.
- **The `xarm_sim` 640x480 dataset predates all of today's fixes** — it contains
  beside-base spawns and ~3.6% of episodes where the cube is in neither static camera. If a
  checkpoint trained on it underperforms near the base, suspect the data first.
- The white xArm clips at the nominal key light (27% of lit pixels >=250), so intensity
  jitter is asymmetric — widen downward, not symmetrically.
