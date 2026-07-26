# Batch generation — operator runbook

How to launch, supervise, and finish a large lift batch on a multi-GPU host. Written so a
fresh session can run one end to end with no prior context. `AGENTS.md` holds the standing
rules; this is the procedure.

Companion to `docs/DAGGER_ROUNDS.md` (that one is for DAgger rounds; this is bulk
generation).

## 0. Host setup (once per machine)

`AGENTS.md` §3 has the full fresh-clone sequence. The two that bite hardest:

- **`luc_env.sh` is gitignored** (per-host). Recreate it: `CUDA_HOME`, `PATH`, and the
  `LD_PRELOAD` of the toolkit's `libnvJitLink.so.13`. Without the preload every shard dies
  ~40 s in with "Madrona batch renderer is only supported on Linux x86-64", which is a lie
  — see the RUNTIME TRAP section of `AGENTS.md`. `run_shards.py`'s preflight checks for it.
- **`scripts/fetch_assets.sh`** pulls the gitignored 328 MB splat. A missing splat is a hard
  error at env construction, and preflight verifies its md5.

## 1. Launch

Always inside tmux — a batch outlives an ssh session.

```bash
cd ~/ghome/xarm-sim
tmux new-session -d -s lift10k \
  "source ./luc_env.sh && uv run --no-sync python scripts/run_shards.py; exec bash"
```

Dry-run first if anything about the config changed:

```bash
source ./luc_env.sh && uv run --no-sync python scripts/run_shards.py --dry-run
```

Defaults: 10,500 attempts, seeds 100000–110499, 8 shards (2/GPU),
`/nas/glee10/sim_mcaps/lift_10k_100000`. `--n-episodes` counts **attempts**, not keeps.

`run_shards.py` owns the fan-out and the guards; its module docstring explains the safety
posture. Do not launch `generate_task_dataset.py` directly for a batch — you lose every
guard.

## 2. Supervise — check in every ~10 minutes

**The one command that tells you the truth.** Success/failure is NOT readable from the
logs (see traps). It is readable from the filenames: failed episodes are deleted by the
keep-gate, and each file is `episode_{attempt_index}.mcap`, so **a failure is a gap in the
numbering**.

```bash
uv run --no-sync python -c "
import pathlib, re
root=pathlib.Path('/nas/glee10/sim_mcaps/lift_10k_100000')
tot_att=tot_ok=0
for d in sorted(root.glob('shard_*')):
    ids=sorted(int(re.search(r'(\d+)',f.name).group(1)) for f in d.glob('episode_*.mcap'))
    if not ids: continue
    att=ids[-1]-ids[0]+1; ok=len(ids); tot_att+=att; tot_ok+=ok
    missing=sorted(set(range(ids[0],ids[-1]+1))-set(ids))
    print(f'{d.name} kept={ok} attempted={att} failed={att-ok} {missing[:6]}')
print(f'TOTAL {tot_ok}/{tot_att} = {tot_ok/tot_att:.1%}')
"
```

Pace and ETA (per-shard gap x shard count is the machine rate):

```bash
uv run --no-sync python -c "
import pathlib, statistics
root=pathlib.Path('/nas/glee10/sim_mcaps/lift_10k_100000'); gaps=[]; tot=0
for d in sorted(root.glob('shard_*')):
    ts=sorted(f.stat().st_mtime for f in d.glob('episode_*.mcap')); tot+=len(ts)
    gaps+=[b-a for a,b in zip(ts[-30:],ts[-29:])]   # recent only: distinguishes
med=statistics.median(gaps); thru=8/med             # plateau from degradation
print(f'kept={tot} gap={med:.1f}s thru={thru*3600:.0f}/h eta={(10500-tot)/thru/3600:.2f}h')
"
```

Machine state:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
free -g | sed -n '2p'
du -sh /nas/glee10/sim_mcaps/lift_10k_100000 ; df -h /nas | tail -1
tail -3 ~/ghome/logs/lift_10k_100000_supervisor.log
```

Report per check-in: episodes/total, success rate, s/episode, throughput, ETA, GPU VRAM +
util, system MemAvailable, disk. Break cadence immediately for a guard trip, a dead shard,
or a success-rate drop.

**Reading the numbers.** GPU *utilization* at 90% is healthy — it is compute, which is
time-shared, and it means the hardware is busy. The dangerous axes are GPU **memory** (not
overcommittable: exhaustion kills processes, possibly other users') and **system RAM**
(exhaustion either thrashes swap and takes the box down, or invokes the OOM-killer on an
arbitrary victim). Guards abort on totals, not our share, because the machine dying is
equally fatal whoever caused it. CPU saturation is deliberately NOT guarded: it only makes
things slower and never fails.

## 3. Traps — signals that lie

- **The supervisor log is append-mode.** It carries entries from every previous run and
  dry-run in the same file. The live run is always at the **tail**; reading the head shows
  stale config from hours ago.
- **`eps=0/10500 rate=100.0%` on the poll line is meaningless.** That count parses the
  shard logs, and shard stdout does not reach them (next trap). Trust `kept=` (files on
  disk) and the filename-gap command above.
- **Shard stdout is swallowed.** Observed 2026-07-26: shard logs sat at 523 bytes while
  250 episodes *per shard* were on disk — ~18 KB of output, far past any 8 KB block buffer,
  and `/proc/<pid>/fdinfo/1` confirmed the offset never moved. Something in the Genesis
  stack captures or suppresses it. `flush=True` and `PYTHONUNBUFFERED=1` are both set and
  did not fix it. Consequence: the **success-rate guard is inert**; the stall guard
  (`stall_abort_s`, counts files) is the one that works.
- **`du` appears to double after the merge.** `merge_shards.py` hard-links the flat
  `episode_NNNNNN.mcap` names to the shard-dir inodes. Same bytes, two names -- `du` counts
  them once and is telling the truth; `ls | wc -l` reads ~2x. Hard links, not symlinks:
  neither name is "the original", and deleting one leaves the other fully intact.
- **The arec converter will DOUBLE the dataset if you let it recurse.** `mcap_robot_sim.py`
  defaults to `recursive=True`, globbing `**/*.mcap`, so pointed at a merged batch root it
  finds the flat files AND the shard-dir files and ingests every episode twice. It does not
  error -- you get a silently duplicated dataset. **Always pass `--no-recursive`** when
  converting a merged batch, so it globs only `*.mcap` at the root.
- **The shard dirs are redundant after merging** and can be deleted (`rm -rf shard_*`)
  without losing data or space: the flat hard links keep the bytes, and the merged
  manifest records every episode's `seed` and `shard`. Keeping them is still recommended --
  they cost nothing and each carries the per-shard manifest needed to regenerate that
  block.

## 4. Extending a batch in place

A later wave writes new shard dirs into the SAME `--out-dir`; `merge_shards.py` globs
`shard_*` so every wave folds into one flat batch and therefore one arec build.

```bash
uv run --no-sync python scripts/run_shards.py \
  --seed 110500 --shard-index-offset 8 --episode-index-offset 10500 \
  --out-dir /nas/glee10/sim_mcaps/lift_10k_100000
```

Seeds must not overlap a previous wave or scenes repeat. Preflight guards only the dirs the
new wave writes and reports prior ones.

**Shard count**: 8 is about the ceiling on 4x L40S at 96x72. Measured 6 shards = 1.22 ep/s,
8 shards = 1.10 ep/s — the extra parallelism costs more in contention than it returns, with
GPUs already at ~90% utilization. Try 6 for a second wave.

## 5. Finish

```bash
# 1. merge every wave into one flat batch (hard links; shard dirs stay intact)
uv run --no-sync python scripts/merge_shards.py --batch-dir /nas/glee10/sim_mcaps/lift_10k_100000

# 2. format gate against the REAL robot episodes -- must print FORMAT: PASS
uv run --no-sync python scripts/compare_batches.py \
    --sim-dir /nas/glee10/sim_mcaps/lift_10k_100000 \
    --real-dir ~/ghome/mcaps/lift_valid \
    --reference ~/ghome/mcaps/lift_valid/2026-05-25_1457_episode_000002.mcap

# 3. provenance: read the MERGED manifest, never the command line
uv run --no-sync python -c "
import json; m=json.load(open('/nas/glee10/sim_mcaps/lift_10k_100000/manifest.json'))
print(m['attempted'], len(m['episodes']), m['success_rate'], m['splat_md5'])
print({k:m['config']['env'][k] for k in ('res','render_backend','noslip_iterations','camera_mode','spawn_radius')})"
```

Confirm in the manifest: `render_backend=batch`, `noslip_iterations=10`, `res=[96,72]`,
`camera_mode=jitter`, `splat_md5` present, and `weld_fired=false` on every episode.

`~/ghome/mcaps/lift_valid` is a symlink farm of the *valid* real episodes — one file in
`~/ghome/mcaps/lift` is truncated. Check head AND tail magic (`894d434150300d0a`) before
adding more.

Then convert to arec (crossformer repo, its own venv). Note this batch's `res=[96,72]` gives
image shape `[3,72,96,3]`, a different schema fingerprint from the 640x480 `xarm_sim` build
— it is a NEW dataset, not an `--append`.

## 6. Recovery

- **A shard dies**: its `manifest.json` is lost (only written at a natural end), but the
  other shards are unaffected. The supervisor logs the exact command to regenerate it;
  rerun that shard's seed/offset block into the same `shard_NN` dir, then merge.
- **Never kill the generator to trim a batch** — you lose that shard's provenance. Trim at
  merge time instead.
- **Aborting**: `tmux kill-session -t lift10k` stops the supervisor but NOT the shards
  (deliberate: the supervisor dying should not destroy the run). Kill shards explicitly by
  process group, then verify `nvidia-smi` shows 0 MiB before relaunching.
