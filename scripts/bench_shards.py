"""Find the best shard count for batch generation on this host.

`scripts/bench_render.py` measures ONE process. This measures the fan-out: how aggregate
throughput actually scales as concurrent shards are added, which is the number that decides
how long a batch takes.

The two are not interchangeable. A single process costs ~4.90 s/episode, so 8 shards
"should" deliver 1.63 ep/s -- but the 10,500-episode batch on 2026-07-26 ran at 0.90 ep/s,
an 82% contention penalty. Whether 6 shards avoids that was never measured cleanly: the only
6-shard figure came from a transient during staggered launch, while two more shards were
still building scenes. This settles it.

Measures STEADY STATE only: the window begins when the last shard writes its first episode
(so startup and stagger are excluded) and ends at the earliest shard completion (so the
tail, where shards drop out and contention falls, is excluded too). Both ends matter --
including either would flatter the high-shard configs.

    uv run python scripts/bench_shards.py --counts 4 6 8
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import time

import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Cfg:
    counts: tuple[int, ...] = (4, 6, 8)
    """Shard counts to compare."""
    episodes_per_shard: int = 20
    """Enough to get a stable steady-state window after stagger; 20 gives ~10 episodes
    inside the window even at the slowest config."""
    seed: int = 700_000
    """Clear of every reserved range (training 9k/20k/30k, eval 51k, dagger 61k+,
    the 10k batch 100000-110499)."""
    stagger_s: float = 15.0
    """Process startup is ~14.8 s, so this staggers scene builds without wasting time."""
    out_root: Path = Path("/nas/glee10/sim_mcaps/bench_shards")
    keep_data: bool = False


def steady_state_rate(shard_dirs: list[Path]) -> tuple[float, int, float]:
    """(episodes/sec, episodes counted, window seconds) over the all-shards-busy window."""
    per_shard = []
    for d in shard_dirs:
        ts = sorted(f.stat().st_mtime for f in d.glob("episode_*.mcap"))
        if len(ts) < 3:
            return 0.0, 0, 0.0
        per_shard.append(ts)
    # window: last shard's first write -> earliest shard's last write
    t0 = max(ts[0] for ts in per_shard)
    t1 = min(ts[-1] for ts in per_shard)
    if t1 <= t0:
        return 0.0, 0, 0.0
    n = sum(sum(1 for t in ts if t0 < t <= t1) for ts in per_shard)
    return n / (t1 - t0), n, t1 - t0


def main(cfg: Cfg) -> None:
    results = []
    for n in cfg.counts:
        out = cfg.out_root / f"n{n}"
        if out.exists():
            shutil.rmtree(out)
        total = n * cfg.episodes_per_shard
        print(f"\n=== {n} shards x {cfg.episodes_per_shard} episodes ===", flush=True)
        t_start = time.time()
        proc = subprocess.run(
            ["uv", "run", "--no-sync", "python", "scripts/run_shards.py",
             "--n-episodes", str(total), "--n-shards", str(n),
             "--seed", str(cfg.seed), "--out-dir", str(out),
             "--stagger-s", str(cfg.stagger_s), "--poll-s", "30"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        wall = time.time() - t_start
        if proc.returncode != 0:
            print(f"  FAILED rc={proc.returncode}\n{proc.stdout[-800:]}\n{proc.stderr[-400:]}")
            continue
        dirs = sorted(out.glob("shard_*"))
        rate, counted, window = steady_state_rate(dirs)
        kept = sum(len(list(d.glob("episode_*.mcap"))) for d in dirs)
        results.append({
            "shards": n, "kept": kept, "wall_s": wall,
            "steady_ep_s": rate, "counted": counted, "window_s": window,
            "per_shard_s": n / rate if rate else 0.0,
        })
        print(f"  kept={kept} wall={wall:.0f}s  steady={rate:.3f} ep/s "
              f"({counted} eps over {window:.0f}s)  per-shard={n/rate if rate else 0:.2f} s/ep",
              flush=True)
        if not cfg.keep_data:
            shutil.rmtree(out, ignore_errors=True)

    print("\n" + "=" * 66)
    print(f"{'shards':>7} {'ep/s':>8} {'ep/hour':>9} {'s/ep/shard':>11} {'vs best':>9}")
    best = max((r["steady_ep_s"] for r in results), default=0.0)
    for r in results:
        print(f"{r['shards']:>7} {r['steady_ep_s']:>8.3f} {r['steady_ep_s']*3600:>9.0f} "
              f"{r['per_shard_s']:>11.2f} {r['steady_ep_s']/best*100 if best else 0:>8.0f}%")
    if results:
        top = max(results, key=lambda r: r["steady_ep_s"])
        print(f"\nbest: {top['shards']} shards at {top['steady_ep_s']*3600:.0f} ep/h "
              f"-> 10,500 episodes in {10500/top['steady_ep_s']/3600:.2f} h")
    (cfg.out_root / "bench_shards.json").parent.mkdir(parents=True, exist_ok=True)
    (cfg.out_root / "bench_shards.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Cfg))
