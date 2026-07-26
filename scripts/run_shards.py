"""Launch and supervise sharded dataset generation across multiple GPUs.

``generate_task_dataset.py`` is a strictly sequential single-process loop
(``--pool-workers`` is ignored), so a large batch is N concurrent shard processes with
disjoint ``--seed`` / ``--episode-offset`` blocks, merged afterwards with
``merge_shards.py``.  This script owns that fan-out and, more importantly, watches it.

Safety posture on the luc host -- a SHARED box (other users have sessions; ``/nas`` is a
37 TB volume shared with ~12 people) with a prior incident where a runaway job consumed
~1 TB of RAM and crashed it:

* every shard runs under ``systemd-run --user --scope`` with an explicit ``MemoryMax``
  and ``MemorySwapMax=0``, so the *kernel* kills a runaway shard long before the poll loop
  would notice -- and ``MemorySwapMax=0`` is what prevents the swap-thrash that took the
  machine down before,
* shards start staggered, because startup (NVRTC megakernel compile + splat load) is the
  spike, not steady state,
* guards are polled on *total machine* pressure, not just our own: system MemAvailable,
  total per-GPU memory, free disk, our aggregate RSS, and the running success rate.

The total-vs-ours distinction is deliberate.  Guards trip on TOTALS, because a machine
running out of RAM or VRAM is equally fatal whoever caused it -- and when another user is
the cause we are the ones holding 8 cheaply-sheddable processes, so backing off is *more*
important, not less.  Our own share is measured too, but only to tell you which it is:
a leak on our side, or a neighbour we should yield to.

Guards abort by killing the *newest* shard first and re-checking, rather than nuking the
whole run: ``generate_task_dataset.py`` only writes ``manifest.json`` at a natural end, so
every kill destroys that shard's provenance and its share of the batch.

``--dry-run`` prints the plan and the preflight result without launching anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time
from typing import Literal, TextIO

import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Measured on luc: render_backend=batch, 3 cameras, arm-start mixture.
# VRAM/RSS were measured at 640x480 (2026-07-25) and are kept as CONSERVATIVE upper bounds
# -- 96x72 uses less, and over-reserving is the safe direction for guards on a shared box.
# MB/episode is measured at the current 96x72 (2026-07-26) because it drives the disk
# preflight, where an 82 MB/ep figure would overstate the batch by ~100x.
MEASURED_VRAM_GB = 8.7
MEASURED_RSS_GB = 2.6
MEASURED_MB_PER_EPISODE = 0.83
EXPECTED_SPLAT_MD5 = "13a21b6e3df686d2cc7169f52f0879a3"


@dataclass
class Cfg:
    n_episodes: int = 10_500
    """ATTEMPTS, not keeps. Generation is success-gated (failed episodes are deleted), so
    the delivered dataset is attempts x success rate. The pilot ran 98.8%, and the
    tightened acquisition gate can only lower that, so 10,500 attempts is 10,000 keeps
    plus a 500-episode margin. Surplus keeps are harmless -- merge_shards renumbers
    contiguously -- so overshoot rather than come up short after a 7-hour run."""

    n_shards: int = 8
    out_dir: Path = Path("/nas/glee10/sim_mcaps/lift_10k_100000")
    seed: int = 100_000
    """Base seed. Shard k covers [seed + k*per, seed + (k+1)*per).

    10,500 attempts claims 100000-110499, past the 100000-109999 block reserved in
    docs/HANDOFF_10K_DATASET.md. 110000+ is unclaimed. Note DAgger's
    `61000 + rep*10007` formula reaches 101028 at round 4, which already overlapped the
    reserved block before this change; renumber one of the two if DAgger gets that far."""

    gpus: tuple[int, ...] = (0, 1, 2, 3)
    render_backend: str = "batch"
    task: str = "lift"

    memory_max: str = "32G"
    """Per-shard systemd MemoryMax. A shard measures ~2.6 GB, so this is >10x headroom
    while keeping N shards well inside physical RAM (8 x 32 = 256 GB of 1 TB)."""

    stagger_s: float = 60.0
    poll_s: float = 30.0

    # --- guards -------------------------------------------------------------------
    # These trip on TOTAL machine pressure, not just our share: the box crashing is
    # equally fatal whoever caused it, and we are the ones who can cheaply back off.
    vram_abort_frac: float = 0.85
    """Abort if TOTAL memory on any GPU we use exceeds this fraction of its capacity."""

    min_sys_avail_gb: float = 100.0
    """Abort if system-wide MemAvailable drops below this (host has ~1 TB, we expect to
    leave >950 GB free). Catches other users' pressure as well as our own."""

    rss_abort_gb: float = 400.0
    """Abort if OUR aggregate shard RSS exceeds this (we expect ~21 GB). This one is
    scoped to us on purpose -- it is the 'we are leaking' signal."""

    disk_min_free_tb: float = 2.0
    """Never let free space on the output volume fall below this. /nas is shared."""

    min_success_rate: float = 0.85
    """Abort if the success rate falls below this (AGENTS.md wants >=90%); a collapsed
    success rate means hours of GPU time producing garbage."""

    success_grace_episodes: int = 200
    """Don't apply the success guard until this many episodes have been attempted."""

    abort_mode: Literal["newest", "all"] = "newest"

    shard_index_offset: int = 0
    """Name this wave's shard dirs from shard_{offset:02d} upward.

    EXTENDING A BATCH: a second wave writes new shard dirs into the SAME --out-dir, and
    merge_shards.py globs shard_* so it folds every wave into one flat batch (and one arec
    build). Continue the seed range and set both offsets past the first wave, e.g. after
    10,500 attempts from seed 100000 in 8 shards:

        --seed 110500 --shard-index-offset 8 --episode-index-offset 10500 \\
        --out-dir <same as wave 1>

    Seeds must not overlap a previous wave, or scenes repeat."""

    episode_index_offset: int = 0
    """Added to every shard's --episode-offset, so a later wave's episode ids continue
    rather than restart. merge_shards renumbers anyway, but this keeps the raw shard files
    unambiguous."""

    extra_args: str = ""
    """Extra flags forwarded to generate_task_dataset.py, as ONE shell-quoted string.
    Must use the = form or tyro parses the inner flags as its own:
        --extra-args="--slip-abort-mm 1e9 --slip-abort-deg 1e9\""""

    log_dir: Path = Path.home() / "ghome" / "logs"
    dry_run: bool = False


@dataclass
class Shard:
    idx: int
    gpu: int
    seed: int
    offset: int
    n: int
    out: Path
    log: Path
    proc: subprocess.Popen | None = None
    killed: bool = False

    @property
    def seed_hi(self) -> int:
        return self.seed + self.n - 1

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def counts(self) -> tuple[int, int]:
        """(attempted, succeeded) parsed from the shard log."""
        try:
            text = self.log.read_text(errors="replace")
        except OSError:
            return 0, 0
        attempted = succeeded = 0
        for line in text.splitlines():
            if line.startswith("[OK ]"):
                attempted += 1
                succeeded += 1
            elif line.startswith("[kept]") or line.startswith("[drop]"):
                attempted += 1
        return attempted, succeeded


def plan(cfg: Cfg) -> list[Shard]:
    base, rem = divmod(cfg.n_episodes, cfg.n_shards)
    shards, offset = [], 0
    for k in range(cfg.n_shards):
        n = base + (1 if k < rem else 0)  # spread the remainder over the first shards
        name_idx = k + cfg.shard_index_offset
        shards.append(
            Shard(
                idx=name_idx,
                gpu=cfg.gpus[k % len(cfg.gpus)],
                seed=cfg.seed + offset,
                offset=offset + cfg.episode_index_offset,
                n=n,
                out=cfg.out_dir / f"shard_{name_idx:02d}",
                log=cfg.log_dir / f"{cfg.out_dir.name}_shard{name_idx:02d}.log",
            )
        )
        offset += n
    return shards


def command(cfg: Cfg, s: Shard) -> list[str]:
    return [
        "systemd-run", "--user", "--scope",
        "-p", f"MemoryMax={cfg.memory_max}", "-p", "MemorySwapMax=0",
        "--",
        "uv", "run", "--no-sync", "python", "scripts/generate_task_dataset.py",
        "--task", cfg.task,
        "--n-episodes", str(s.n),
        "--backend", "gpu",
        "--env.render-backend", cfg.render_backend,
        "--out-dir", str(s.out),
        "--seed", str(s.seed),
        "--episode-offset", str(s.offset),
        *shlex.split(cfg.extra_args),
    ]


# --- measurement ------------------------------------------------------------------


def disk_free_tb(path: Path) -> float:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e12


def shard_pids(s: Shard) -> list[int]:
    if s.proc is None or s.proc.poll() is not None:
        return []
    try:
        out = subprocess.run(
            ["pgrep", "-g", str(os.getpgid(s.proc.pid))],
            capture_output=True, text=True,
        ).stdout
    except (ProcessLookupError, OSError):
        return []
    return [int(p) for p in out.split() if p.isdigit()]


def rss_gb(pids: list[int]) -> float:
    total_kb = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total_kb += int(line.split()[1])
                    break
        except (FileNotFoundError, ProcessLookupError, ValueError):
            continue
    return total_kb / 1024 / 1024


def sys_avail_gb() -> float:
    """System-wide MemAvailable -- the honest 'how close is this box to trouble' number."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return float("inf")


def gpu_memory(our_pids: set[int]) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """(ours_gb, total_gb, capacity_gb) per GPU index.

    We abort on TOTAL, since a full GPU is fatal regardless of whose job filled it.
    Our own share is tracked alongside so the log says which of the two it is.
    """
    totals: dict[int, float] = {}
    caps: dict[int, float] = {}
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        i, used, cap = (x.strip() for x in line.split(","))
        totals[int(i)] = float(used) / 1024
        caps[int(i)] = float(cap) / 1024

    ours: dict[int, float] = {i: 0.0 for i in totals}
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    uuid_to_idx = {}
    uu = subprocess.run(["nvidia-smi", "--query-gpu=index,gpu_uuid",
                         "--format=csv,noheader"], capture_output=True, text=True).stdout
    for line in uu.strip().splitlines():
        i, uuid = (x.strip() for x in line.split(","))
        uuid_to_idx[uuid] = int(i)
    for line in apps.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, mem = (x.strip() for x in line.split(","))
        if int(pid) in our_pids and uuid in uuid_to_idx:
            ours[uuid_to_idx[uuid]] += float(mem) / 1024

    return ours, totals, caps


# --- preflight --------------------------------------------------------------------


def preflight(cfg: Cfg, shards: list[Shard]) -> list[str]:
    """Cheap checks that each prevent a class of multi-hour failure."""
    problems: list[str] = []

    # LD_PRELOAD: without it every shard dies ~40 s in, with a misleading message.
    preload = os.environ.get("LD_PRELOAD", "")
    if "libnvJitLink" not in preload:
        problems.append(
            "LD_PRELOAD lacks libnvJitLink -- gs_madrona will fail to import and genesis "
            "will report the misleading 'only supported on Linux x86-64'. "
            "Run: source luc_env.sh")

    # Splat identity: a wrong/missing splat silently changes the visual domain.
    splat = PROJECT_ROOT / "assets" / "lab_aligned.ply"
    if not splat.exists():
        problems.append(f"splat missing: {splat} (run scripts/fetch_assets.sh)")
    else:
        h = hashlib.md5()
        with splat.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != EXPECTED_SPLAT_MD5:
            problems.append(f"splat md5 {h.hexdigest()} != expected {EXPECTED_SPLAT_MD5}")

    # Never silently overwrite, but DO allow extending: only the shard dirs this wave
    # writes must be empty. Earlier waves' shard_* dirs are left alone and merge_shards
    # folds them all together.
    clash = [s.out.name for s in shards if list(s.out.glob("episode_*.mcap"))]
    if clash:
        problems.append(f"these shard dirs already contain episodes: {clash}; "
                        "raise --shard-index-offset to extend, or use a fresh --out-dir")
    prior = sorted({p.parent.name for p in cfg.out_dir.glob("shard_*/episode_*.mcap")}) \
        if cfg.out_dir.exists() else []
    if prior:
        print(f"note: extending an existing batch -- {len(prior)} prior shard dir(s) "
              f"present ({prior[0]}..{prior[-1]}), they will be left untouched and merged")

    # Disk: projected need against free space, keeping the shared-volume floor intact.
    need_tb = cfg.n_episodes * MEASURED_MB_PER_EPISODE / 1e6
    free_tb = disk_free_tb(cfg.out_dir)
    if free_tb - need_tb < cfg.disk_min_free_tb:
        problems.append(
            f"insufficient disk: need ~{need_tb:.2f} TB, free {free_tb:.2f} TB, "
            f"floor {cfg.disk_min_free_tb:.2f} TB")

    if cfg.n_shards % len(cfg.gpus):
        problems.append(f"n_shards {cfg.n_shards} not divisible by {len(cfg.gpus)} GPUs "
                        "-- load will be uneven")
    return problems


# --- main -------------------------------------------------------------------------


def main(cfg: Cfg) -> None:
    shards = plan(cfg)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    sup_log = cfg.log_dir / f"{cfg.out_dir.name}_supervisor.log"

    per_gpu: dict[int, int] = {}
    for s in shards:
        per_gpu[s.gpu] = per_gpu.get(s.gpu, 0) + 1
    max_per_gpu = max(per_gpu.values())
    need_tb = cfg.n_episodes * MEASURED_MB_PER_EPISODE / 1e6
    hours = cfg.n_episodes / cfg.n_shards * 20.9 / 3600

    log: TextIO = open(sup_log, "a", buffering=1)  # line-buffered, never behind a pipe

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        log.write(msg + "\n")

    emit(f"batch      : {cfg.n_episodes} episodes -> {cfg.out_dir}")
    emit(f"render     : {cfg.render_backend}  task={cfg.task}")
    emit(f"shards     : {cfg.n_shards} across GPUs {cfg.gpus} ({max_per_gpu} per GPU)")
    emit(f"projected  : ~{max_per_gpu * MEASURED_VRAM_GB:.1f} GB VRAM on the busiest GPU, "
         f"~{len(shards) * MEASURED_RSS_GB:.1f} GB RSS, ~{need_tb:.2f} TB disk, ~{hours:.1f} h")
    emit(f"caps       : MemoryMax={cfg.memory_max}/shard | abort at "
         f"{cfg.vram_abort_frac:.0%} TOTAL VRAM, <{cfg.min_sys_avail_gb:.0f} GB system "
         f"MemAvailable, {cfg.rss_abort_gb:.0f} GB our RSS, "
         f"<{cfg.disk_min_free_tb:.1f} TB free, <{cfg.min_success_rate:.0%} success")
    emit(f"supervisor log: {sup_log}")
    emit()
    for s in shards:
        emit(f"  shard {s.idx:02d}  gpu{s.gpu}  seeds {s.seed}-{s.seed_hi}  "
             f"episodes {s.offset}-{s.offset + s.n - 1}  -> {s.out.name}")

    emit()
    problems = preflight(cfg, shards)
    if problems:
        emit("PREFLIGHT FAILED:")
        for p in problems:
            emit(f"  - {p}")
        raise SystemExit(1)
    emit(f"preflight  : OK (free {disk_free_tb(cfg.out_dir):.2f} TB)")

    if cfg.dry_run:
        emit("\n--dry-run: nothing launched. Shard 0 command:")
        emit("  " + " ".join(shlex.quote(c) for c in command(cfg, shards[0])))
        return

    emit()
    for s in shards:
        s.out.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(s.gpu),
            OMP_NUM_THREADS="8",
            MKL_NUM_THREADS="8",
        )
        with s.log.open("w") as fh:
            s.proc = subprocess.Popen(
                command(cfg, s), cwd=PROJECT_ROOT, env=env,
                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
            )
        emit(f"launched shard {s.idx:02d} on gpu{s.gpu} (pid {s.proc.pid}) -> {s.log}")
        if s is not shards[-1]:
            time.sleep(cfg.stagger_s)

    emit(f"\nall {len(shards)} shards up; polling every {cfg.poll_s:.0f}s\n")
    start = time.time()

    def kill(s: Shard, why: str) -> None:
        if not s.alive():
            return
        emit(f"!! killing shard {s.idx:02d} ({why}). Its manifest is lost; recover with:")
        emit("   " + " ".join(shlex.quote(c) for c in command(cfg, s)))
        try:
            os.killpg(os.getpgid(s.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        s.killed = True

    try:
        while any(s.alive() for s in shards):
            time.sleep(cfg.poll_s)

            pids_by_shard = {s.idx: shard_pids(s) for s in shards}
            all_pids = {p for v in pids_by_shard.values() for p in v}
            our_rss = sum(rss_gb(v) for v in pids_by_shard.values())
            avail = sys_avail_gb()
            ours, totals, caps = gpu_memory(all_pids)
            free_tb = disk_free_tb(cfg.out_dir)
            attempted = sum(s.counts()[0] for s in shards)
            succeeded = sum(s.counts()[1] for s in shards)
            rate = succeeded / attempted if attempted else 1.0
            alive = [s for s in shards if s.alive()]

            tot_frac, tot_gpu = max((totals[i] / caps[i], i) for i in cfg.gpus)
            our_gpu_gb = sum(ours.get(i, 0.0) for i in cfg.gpus)
            foreign_gpu_gb = sum(max(0.0, totals[i] - ours.get(i, 0.0)) for i in cfg.gpus)
            emit(f"[{time.time() - start:7.0f}s] alive={len(alive)}/{len(shards)} "
                 f"eps={succeeded}/{cfg.n_episodes} rate={rate:.1%} "
                 f"rss={our_rss:.1f}GB avail={avail:.0f}GB "
                 f"gpu_tot={tot_frac:.0%}(ours {our_gpu_gb:.1f}GB/others "
                 f"{foreign_gpu_gb:.1f}GB) free={free_tb:.2f}TB")

            # Trip on TOTAL machine pressure regardless of cause; the ours/others split
            # in the message only tells you whether to debug us or yield to a neighbour.
            reason = None
            if avail < cfg.min_sys_avail_gb:
                reason = (f"system MemAvailable {avail:.0f} GB below "
                          f"{cfg.min_sys_avail_gb:.0f} GB (ours: {our_rss:.1f} GB)")
            elif tot_frac > cfg.vram_abort_frac:
                reason = (f"gpu{tot_gpu} total VRAM {tot_frac:.0%} "
                          f"(ours {our_gpu_gb:.1f} GB, others {foreign_gpu_gb:.1f} GB)")
            elif our_rss > cfg.rss_abort_gb:
                reason = f"our aggregate RSS {our_rss:.1f} GB -- we are leaking"
            elif free_tb < cfg.disk_min_free_tb:
                reason = f"free disk {free_tb:.2f} TB below floor"
            elif attempted >= cfg.success_grace_episodes and rate < cfg.min_success_rate:
                reason = f"success rate {rate:.1%} below {cfg.min_success_rate:.0%}"

            if reason:
                emit(f"!! GUARD TRIPPED: {reason}")
                if cfg.abort_mode == "all":
                    for s in alive:
                        kill(s, reason)
                    raise SystemExit(2)
                # step down one shard at a time; re-measure next cycle
                kill(alive[-1], reason)
                if len(alive) == 1:
                    emit("!! last shard killed -- aborting")
                    raise SystemExit(2)
    except KeyboardInterrupt:
        emit("\ninterrupted -- shards left running. Kill them yourself if that was intended.")
        raise

    codes = {s.idx: (s.proc.returncode if s.proc else None) for s in shards}
    kept = sum(len(list(s.out.glob("episode_*.mcap"))) for s in shards)
    bad = {k: v for k, v in codes.items() if v != 0}
    emit(f"\ndone in {(time.time() - start) / 3600:.2f} h; kept {kept} episodes")
    emit(f"exit codes: {codes}")
    if bad:
        emit(f"FAILED shards: {bad}")
        raise SystemExit(1)
    emit(f"\nnext: uv run python scripts/merge_shards.py --batch-dir {cfg.out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
