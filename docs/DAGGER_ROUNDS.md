# DAgger rounds — operator runbook

Paper-DAgger (Ross et al., arXiv:1011.0686) for the served crossformer student. Rounds
are **manual, one at a time**: each crossformer retrain is hours-long, so there is no
automated multi-round loop. This is the checklist an operator repeats per round — every
numbered step is a separate command, and you may stop after any of them.

The no-regret core: round 0 collects pure-teacher BC data; each later round mixes teacher
and student control per replan window, labels **every** visited state with the reactive
expert, and appends those labels into one growing dataset that the next model trains on.

## 0. One-time setup (read before round 0)

- **Dataset name is `xarm_sim_dagger`** — one arec dataset, aggregated across all rounds.
  Never mix dagger data into `xarm_sim` (separate stats; the action distribution differs).
- **Seed range 61000+** is reserved for dagger collection (`--seed`, default `61000`),
  clear of training and of the eval base seed (51000). Bump the base if a round needs
  >1000 episodes; do not overlap eval seeds.
- **Rendering stays `nyx`** (splat background, colored robot). Raster frames are
  out-of-domain for the student and silently invalidate the data. This is the default in
  both scripts and must never be relied on via a bare CLI flag. HARD RULE: before any real
  collection, verify the *effective* config against the **last valid run's summary.json**
  (`env.render_backend`, `noslip_iterations`, `close_setpoint`) — not the script defaults.
- Data lands under `/data/store/griffen_sim_mcaps/` (dagger MCAPs in `dagger_mcaps/`,
  diagnostics in `dagger_runs/`, evals in `evals/`). Never write to repo-local `outputs/`.

## Per-round loop

### 1. Eval the current checkpoint → `summary.json`

Serve the checkpoint, then grid-eval it. Produces `summary.json` with the
`overall_success_rate` that feeds the next collection's β. (Round 0 has no student yet —
skip this step; treat success ≐ 0.)

```bash
uv run python scripts/eval_grid.py --task lift --policy remote \
    --host localhost --port 8001 --env.render-backend nyx --backend gpu --video-every 10
# -> /data/store/griffen_sim_mcaps/evals/lift/summary.json
```

### 2. Collect with `dagger_rounds.py`

**Round 0** — pure teacher, `--beta 1.0`. Short-circuits every student query, so **no
model server is needed**:

```bash
uv run python scripts/dagger_rounds.py --round 0 --beta 1.0 \
    --model-name <student> --grid-nx 3 --grid-ny 3
```

**Round N (N≥1)** — β derived from the prior eval, student served and queried:

```bash
uv run python scripts/dagger_rounds.py --round N \
    --beta-from /data/store/griffen_sim_mcaps/evals/lift/summary.json \
    --model-name <student> --host localhost --port 9001
```

Pass exactly one of `--beta` / `--beta-from`. Every episode is written (failures
included — those are the states DAgger exists to label); no success or length gate. Add
`--cells-from <prior results.jsonl> --cells failed` to concentrate a round on the cells
the last eval failed.

### 3. Convert **only the new round's** MCAPs → append into the one dataset

Point `--root` at just this round's MCAP directory. `--append` continues the existing
shard numbering, offsets episode ids past the current count, and recomputes proprio stats
over the union — prior rounds' shards are untouched and image conversion is not repeated.

```bash
# First round only (dataset does not exist yet): a plain build, NO --append
uv run python scripts/data/make/mcap_robot_sim.py --mode build \
    --name xarm_sim_dagger --version 0.0.1 \
    --root /data/store/griffen_sim_mcaps/dagger_mcaps/<student>/paper-v1/<round0-run>

# Every later round: --append the new round's dir into the same dataset
uv run python scripts/data/make/mcap_robot_sim.py --mode build --append \
    --name xarm_sim_dagger --version 0.0.1 \
    --root /data/store/griffen_sim_mcaps/dagger_mcaps/<student>/paper-v1/<roundN-run>
```

### 4. Retrain, grab weights, stop

Fine-tune crossformer on the single aggregated `xarm_sim_dagger` dataset (the union of all
appended rounds). Collect the new checkpoint. **Stop here** until the operator restarts the
loop at step 1 for round N+1.

## The β formula

β is the per-window probability the **teacher** drives control (student drives with
`1 − β`). Computed once per round from the previous eval, not inside a loop:

```
beta = clip(1 - 1.2 * overall_success_rate, beta_floor=0.2, 1.0)
```

Round 0 uses β = 1.0 (pure teacher). Two safety terms, each a scar from a real failure:

- **`beta_floor = 0.2`** — β never reaches 0. A pure-student collection once poisoned an
  aggregate upstream (success collapsed to 0); keeping ≥20% teacher windows guarantees
  correct labels keep entering the data even at a strong student.
- **Success-adaptive** — β tracks `1 − 1.2·success`, so a weak student is collected under
  mostly-teacher control and β only relaxes as evals improve. Open-loop β schedules that
  ignore measured success outran slow learners; tying β to the last eval keeps the schedule
  from getting ahead of the model.

## What NOT to do

- **No keep-gates, no trimming.** Every visited state is a label, failures included. Do not
  filter by success, length, or "corrected-only". Only the hard `max_control_steps` cap
  applies.
- **Do not delete or rebuild prior rounds' shards.** Aggregation is append-only; a full
  rebuild reprocesses every image for a few percent of new data and risks renumbering.
- **Do not mix into `xarm_sim`.** Dagger data is its own dataset (`xarm_sim_dagger`) with
  its own stats.
- **Do not swap the render backend to raster** for speed — it invalidates the round.

## Troubleshooting

- **`--append` refuses with a "mismatch" error.** By design: append aborts (never silently
  rebuilds) if the schema fingerprint, writers spec, writer_options, or `shard_size` differ
  from the existing `meta.json`. Keep `--name`, `--version`, `--writer`, and any
  `--shard-size` identical to the round-0 build. If you truly changed the schema, that is a
  new dataset name, not an append.
- **`--beta-from ... no 'overall_success_rate' key`.** You pointed at a non-eval JSON; use
  the `summary.json` from step 1 (`eval_grid.py`), not a collection summary.
- **"provide exactly one of --beta / --beta-from".** Pass one, not both and not neither.
- **Round 0 tries to contact a server.** You passed `--beta` < 1.0; round 0 must be
  `--beta 1.0` so no student client is constructed.
