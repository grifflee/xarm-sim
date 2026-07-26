"""Measure per-episode generation cost, so the price of a randomization feature is a number.

Environment randomization gets added by feel: someone proposes a knob, it looks cheap, and
the only way to find out what it actually costs has been to launch a multi-hour batch and
watch the ETA. This makes it a measurement. Each arm runs N episodes of exactly the work
``generate_task_dataset.py`` does -- same ``TaskEnv``, same scripted policy, same
``EpisodeMcapWriter``, via its own ``run_episode`` -- with a wall clock around each phase.
Turn the feature on in a second arm and the table prints the delta.

The breakdown matters as much as the total, because the phases scale differently: scene
build is a one-off *unless* the feature forces a per-episode rebuild (Nyx appearance
randomization does exactly that -- one subprocess per episode, so the whole startup cost
lands on every episode), render scales with pixels x cameras, physics with episode length,
MCAP with frames x pixels. A feature costing 10% of render is cheap; one costing a scene
rebuild is not, and only the split tells them apart.

Arms share the seed range, so every arm simulates the same episodes: the comparison is
paired, and a delta survives the +-10% episode-to-episode spread the tempo jitter creates.
An arm that changes the trajectory itself (spawn region, tempo) breaks the pairing -- for
those, read ms/frame rather than seconds/episode.

Phase splits are wall clock at the Python call boundary, exclusive of nested phases. GPU
work is queued asynchronously, so whichever call forces the next device sync absorbs work
its predecessor started -- read physics/splat/render as a group, not as independent line
items. The per-episode TOTAL is exact regardless.

**ONE PROCESS.** This is per-process cost, not a batch projection. The 10,500-episode
batch on this host ran 8 concurrent shards over 4 L40S at ~8.9 s/episode/shard, and
aggregate throughput was already saturating (6 shards measured 1.22 ep/s, 8 shards 1.10).
A single unconcurrent shard is faster per episode than any of that. Never multiply this
script's ep/h by the shard count; measure the fan-out with ``run_shards.py`` instead.

Genesis is a process-global singleton -- a second ``gs.Scene`` in one process raises -- so
each arm runs in its own subprocess through the ``--single`` self-invocation, the same
pattern ``scripts/res_sweep.py`` uses.

Every command below needs ``source ./luc_env.sh`` first, in the same shell: without its
LD_PRELOAD, ``gs_madrona`` fails to import and Genesis blames Linux x86-64 (AGENTS.md).

    source ./luc_env.sh

    # baseline only, current default config
    CUDA_VISIBLE_DEVICES=1 systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=0 -- \
        uv run --no-sync python scripts/bench_render.py --episodes 20

    # A/B one feature: camera jitter off vs the production 15 deg / 5 cm
    ... uv run --no-sync python scripts/bench_render.py \
        --arms "fixed: --env.camera-mode fixed" \
               "jitter: --env.camera-mode jitter --env.cam-jitter-deg 15 --env.cam-jitter-cm 5"
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from typing import Literal

import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

_T_IMPORT = time.perf_counter()

import genesis as gs  # noqa: E402

GENESIS_IMPORT_S = time.perf_counter() - _T_IMPORT

from xsim.task_env import BaseDecorCfg, StackCfg, TableCfg, TaskEnv, TaskEnvCfg  # noqa: E402

import generate_task_dataset as gen  # noqa: E402

# Reference points from the 2026-07-26 10,500-episode batch (96x72, batch renderer, 3
# cameras, 8 shards on 4x L40S). Printed with the results so a single-process number is
# never mistaken for machine throughput.
REF_SHARDS = 8
REF_S_PER_EP_PER_SHARD = 8.9
REF_BATCH_EP_PER_HOUR = 3250

# Healthy 96x72 batch frames measure ~75-130 mean pixel value; the failure mode is a
# near-black composite at ~7.8 that renders and times perfectly. 15 sits well below any
# legitimately dark frame and well above the broken one.
DARK_FRAME_PX = 15.0

PHASES = ("reset", "physics", "camsync", "splat", "render", "comp", "control", "mcap", "other")
PHASE_LABELS = {
    "reset": "reset",        # env.reset, minus the splat render and IK inside it
    "physics": "physics",    # scene.step alone, minus everything nested below
    "camsync": "camsync",    # re-pose the wrist camera on its link, every physics step
    "splat": "splat_bg",     # gsplat background, re-rendered per moving camera during step
    "render": "render",      # camera render (Madrona/raster/Nyx) + device->host copy
    "comp": "comp",          # splat compositing + shadow-catcher transfer (CPU/numpy)
    "control": "ctrl/IK",    # go_to_goal: IK solve and gripper command
    "mcap": "mcap",          # writer open/log/close
    "other": "other",        # policy, grasp integrity probes, success eval, bookkeeping
}


@dataclass
class Cfg:
    episodes: int = 20
    """Timed episodes per arm. Deliberately low -- a benchmark nobody runs because it takes
    an hour is worth nothing. Raise it when a delta lands inside the episode-to-episode
    spread the summary prints."""

    warmup: int = 1
    """Episodes run and DISCARDED before timing starts. The first episode after a scene
    build pays one-time GPU costs no later episode repeats (megakernel/graph capture, first
    allocations); including it inflates a 20-episode mean by a visible margin."""

    arms: tuple[str, ...] = ("baseline:",)
    """A/B arms, one shell-quoted string each, formatted ``label: --env.flag value ...``.
    Flags are applied on top of this script's own ``--env.*`` settings, so each arm carries
    only what differs. Every arm runs in its own subprocess (Genesis singleton).

    Keep the ``label:`` prefix. argparse classifies any bare token starting with ``-`` as a
    flag rather than a value unless it contains a space, so ``--arms "--env.camera-mode
    fixed"`` happens to work while a single-token override would not; the prefix guarantees
    the space and names the table row."""

    task: Literal["lift", "stack"] = "lift"

    seed: int = 950_000
    """Episode seeds are ``seed + i``. 950000+ is clear of training (9k/20k/30k), eval
    (51k), DAgger (61k+), the 10k batch (100000-110499) and spawn_visibility
    (900000-900199). Nothing is kept, but reusing a live range makes logs ambiguous."""

    backend: Literal["gpu", "cpu"] = "gpu"

    out_dir: Path = Path("/nas/glee10/sim_mcaps/bench_render")
    """Scratch for the MCAPs and the result JSON. Leave it on the volume a real batch
    writes to: MCAP write time is one of the measured phases and /nas is not a local disk."""

    keep_mcap: bool = False
    """Keep the benchmark's MCAPs instead of deleting each one after it is measured. Off by
    default -- 20 episodes x 2 arms is ~33 MB of files nobody will ever train on."""

    render_backend_note: bool = True
    """Print the batch-vs-single-process reference block. Turn off for machine parsing."""

    json_out: Path | None = None
    """Where to write this arm's raw result (single mode). The parent sets it per arm."""

    label: str = "single"
    """Row name for this arm. The parent sets it from the ``label:`` prefix."""

    single: bool = False
    """Internal: run ONE arm in this process instead of forking one subprocess per arm.
    Genesis is a process-global singleton -- the second ``gs.Scene`` in a process raises --
    so an A/B cannot happen in one process. Same self-invocation pattern as res_sweep.py."""

    env: TaskEnvCfg = field(
        default_factory=lambda: TaskEnvCfg(noslip_iterations=10, render_backend="batch")
    )
    """Shared baseline config for every arm. ``batch`` (Madrona) is what production
    generation runs, so it is the default here even though TaskEnvCfg defaults to raster --
    benchmarking raster would measure a path no batch uses. noslip 10 matches the lift
    generation requirement enforced in generate_task_dataset.main."""


# ---------------------------------------------------------------------------- timing --


class Timers:
    """Wall-clock accumulator for one episode's phases, EXCLUSIVE of nested phases.

    Nesting is not academic here: ``TaskEnv.step`` re-renders the gsplat background for the
    moving wrist camera every ``splat_resplat_every`` steps, and ``reset`` renders it for
    every camera. Timing the outer call alone would file all of that under "physics" and
    hide a cost that scales with camera count, not with simulation. A phase is charged only
    the time not already charged to a phase inside it, so the columns stay additive."""

    def __init__(self) -> None:
        self.total: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self._open: list[list[float]] = []  # per open frame: [time spent in children]

    def clear(self) -> None:
        self.total.clear()
        self.calls.clear()
        self._open.clear()

    @contextmanager
    def phase(self, name: str):
        frame = [0.0]
        self._open.append(frame)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._open.pop()
            self.total[name] += elapsed - frame[0]
            self.calls[name] += 1
            if self._open:
                self._open[-1][0] += elapsed

    def wrap(self, obj, attr: str, name: str) -> None:
        """Time every call to ``obj.attr``. Instance attribute, so the class is untouched --
        ``task_env.py``/``batch_renderer.py`` are read-only here (another agent owns them)."""
        inner = getattr(obj, attr)

        def timed(*args, **kwargs):
            with self.phase(name):
                return inner(*args, **kwargs)

        setattr(obj, attr, timed)


class FrameHealth:
    """Mean pixel value over every rendered frame.

    A throughput number can look perfect while the frames are black: the batch path
    composites a near-black image (mean ~7.8) when the light rig is broken and never
    raises. Cheap enough to run on every frame at 96x72 -- and it is computed outside the
    render timer so it does not pay itself into the result."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.sum = 0.0
        self.n = 0
        self.lo = math.inf
        self.hi = -math.inf

    def observe(self, images: dict) -> None:
        for img in images.values():
            m = float(img.mean())
            self.sum += m
            self.n += 1
            self.lo = min(self.lo, m)
            self.hi = max(self.hi, m)

    @property
    def mean(self) -> float:
        return self.sum / self.n if self.n else float("nan")


def _instrument_env(env: TaskEnv, timers: Timers, health: FrameHealth) -> None:
    timers.wrap(env, "reset", "reset")
    timers.wrap(env, "step", "physics")
    timers.wrap(env.robot, "go_to_goal", "control")  # IK + gripper command
    # Private, but this is where a third of the per-step cost lives and the alternative is
    # editing task_env.py, which is off limits here. Missing attrs are tolerated so the
    # script survives a refactor of that file rather than crashing a benchmark run.
    for attr, name in (("_render_splat_bg", "splat"),
                       ("_composite_splat", "comp"),
                       ("_sync_attached_cams", "camsync")):
        if hasattr(env, attr):
            timers.wrap(env, attr, name)

    inner_render = env.render

    def render():
        with timers.phase("render"):
            images = inner_render()
        health.observe(images)
        return images

    env.render = render


def _install_timed_writer(timers: Timers) -> None:
    """Time the MCAP writer without touching the writer module.

    ``run_episode`` resolves ``EpisodeMcapWriter`` from generate_task_dataset's globals, so
    rebinding the name there is enough. Open and close are counted too: the close writes
    the chunk index and summary, which is a real per-episode cost."""
    base = gen.EpisodeMcapWriter

    class TimedEpisodeMcapWriter(base):  # type: ignore[valid-type,misc]
        def __enter__(self):
            with timers.phase("mcap"):
                return super().__enter__()

        def __exit__(self, *exc):
            with timers.phase("mcap"):
                return super().__exit__(*exc)

        def log_calibration(self, *args, **kwargs):
            with timers.phase("mcap"):
                return super().log_calibration(*args, **kwargs)

        def log_step(self, *args, **kwargs):
            with timers.phase("mcap"):
                return super().log_step(*args, **kwargs)

    gen.EpisodeMcapWriter = TimedEpisodeMcapWriter


# ------------------------------------------------------------------------ single arm --


def run_arm(cfg: Cfg) -> dict:
    """Build one env and time ``cfg.warmup + cfg.episodes`` generation episodes."""
    if cfg.task == "lift" and cfg.env.noslip_iterations != 10:
        raise ValueError("lift generation is physical/no-weld and requires --env.noslip-iterations 10")

    out_dir = cfg.out_dir / _safe(cfg.label)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    gs.init(backend=gs.gpu if cfg.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    init_s = time.perf_counter() - t0

    env_cfg = replace(cfg.env, task=cfg.task)
    t0 = time.perf_counter()
    env = TaskEnv(env_cfg)
    build_s = time.perf_counter() - t0

    gen_cfg = gen.Config(
        task=cfg.task,
        mode="generate",
        out_dir=out_dir,
        n_episodes=cfg.episodes,
        seed=cfg.seed,
        backend=cfg.backend,
        env=env_cfg,
    )

    timers = Timers()
    health = FrameHealth()
    _instrument_env(env, timers, health)
    _install_timed_writer(timers)

    print(f"[{cfg.label}] genesis import {GENESIS_IMPORT_S:.1f} s | gs.init {init_s:.1f} s | "
          f"scene build {build_s:.1f} s | {cfg.warmup} warmup + {cfg.episodes} timed episodes",
          flush=True)

    rows: list[dict] = []
    for i in range(cfg.warmup + cfg.episodes):
        timers.clear()
        health.clear()  # the render closure holds this object; rebinding it would not stick
        path = out_dir / f"bench_{i:04d}.mcap"
        t0 = time.perf_counter()
        stats = gen.run_episode(env, gen_cfg, i, path)
        wall = time.perf_counter() - t0
        mb = path.stat().st_size / 1e6 if path.exists() else 0.0
        if not cfg.keep_mcap:
            path.unlink(missing_ok=True)

        phases = {name: timers.total.get(name, 0.0) for name in PHASES if name != "other"}
        phases["other"] = wall - sum(phases.values())
        rows.append({
            "episode": i,
            "warmup": i < cfg.warmup,
            "wall_s": wall,
            "phases": phases,
            "frames": stats["frames"],
            "mcap_mb": mb,
            "success": bool(stats["success"]),
            "abort_reason": stats.get("abort_reason"),
            "frame_mean": health.mean,
            "frame_min": health.lo,  # darkest single frame: a mean can hide a black camera
        })
        tag = "warmup" if i < cfg.warmup else f"ep{i - cfg.warmup}"
        print(f"[{cfg.label}] {tag:>7}: {wall:6.2f}s "
              + " ".join(f"{PHASE_LABELS[k]}={phases[k]:.2f}" for k in PHASES)
              + f" frames={stats['frames']:3d} px={health.mean:5.1f} "
                f"ok={stats['success']}", flush=True)

    result = summarize(cfg, rows, {"genesis_import_s": GENESIS_IMPORT_S,
                                   "gs_init_s": init_s, "scene_build_s": build_s})
    if cfg.json_out is not None:
        cfg.json_out.parent.mkdir(parents=True, exist_ok=True)
        cfg.json_out.write_text(json.dumps(result, indent=2))
    return result


def summarize(cfg: Cfg, rows: list[dict], one_off: dict) -> dict:
    timed = [r for r in rows if not r["warmup"]]
    if not timed:
        raise RuntimeError("no timed episodes (all warmup?)")
    walls = [r["wall_s"] for r in timed]
    mean_wall = statistics.fmean(walls)
    phase_mean = {k: statistics.fmean([r["phases"][k] for r in timed]) for k in PHASES}
    return {
        "label": cfg.label,
        "episodes": len(timed),
        "warmup": cfg.warmup,
        "one_off": one_off,
        "one_off_total_s": sum(one_off.values()),
        "s_per_episode": mean_wall,
        "s_per_episode_sd": statistics.stdev(walls) if len(walls) > 1 else 0.0,
        "s_per_episode_min": min(walls),
        "s_per_episode_max": max(walls),
        "episodes_per_s": 1.0 / mean_wall,
        "episodes_per_h": 3600.0 / mean_wall,
        "phases_s": phase_mean,
        # Length-normalized: the tempo jitter makes episodes 152-227 frames, so this is the
        # metric to use when an arm changes trajectory length as well as per-frame cost.
        "ms_per_frame": statistics.fmean([1000.0 * r["wall_s"] / max(r["frames"], 1) for r in timed]),
        "frames_mean": statistics.fmean([r["frames"] for r in timed]),
        "mcap_mb_mean": statistics.fmean([r["mcap_mb"] for r in timed]),
        "frame_px_mean": statistics.fmean([r["frame_mean"] for r in timed]),
        "frame_px_min": min(r["frame_min"] for r in timed),
        "successes": sum(int(r["success"]) for r in timed),
        "warmup_s": [r["wall_s"] for r in rows if r["warmup"]],
        "episodes_raw": timed,
    }


# --------------------------------------------------------------------- arm fan-out ---


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name) or "arm"


def parse_arm(spec: str) -> tuple[str, list[str]]:
    """``"label: --env.flag v"`` -> ``("label", ["--env.flag", "v"])``.

    A spec with no label (or whose text before the first colon is itself a flag, e.g. a
    path value containing a colon) keeps its flags and gets an auto label."""
    label, sep, flags = spec.partition(":")
    if not sep or label.strip().startswith("-"):
        argv = shlex.split(spec)
        return (" ".join(argv).replace("--env.", "")[:24] or "default"), argv
    return (label.strip() or "arm"), shlex.split(flags)


def _env_to_jsonable(env: TaskEnvCfg) -> dict:
    return gen._config_to_jsonable(env)


def _env_from_jsonable(raw: dict) -> TaskEnvCfg:
    raw = {k: v for k, v in raw.items() if k in {f.name for f in fields(TaskEnvCfg)}}
    raw["stack"] = StackCfg(**raw["stack"])
    raw["table"] = TableCfg(**raw["table"])
    raw["base_decor"] = BaseDecorCfg(**raw["base_decor"])
    if raw.get("splat_uri") is not None:
        raw["splat_uri"] = Path(raw["splat_uri"])
    # JSON has no tuples; every tuple-typed field would silently become a list.
    return TaskEnvCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


def run_arms(cfg: Cfg) -> list[dict]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    base_env = cfg.out_dir / "base_env.json"
    base_env.write_text(json.dumps(_env_to_jsonable(cfg.env)))

    results: list[dict] = []
    for spec in cfg.arms:
        label, extra = parse_arm(spec)
        json_out = cfg.out_dir / f"result_{_safe(label)}.json"
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "--single",
            "--label", label,
            "--episodes", str(cfg.episodes),
            "--warmup", str(cfg.warmup),
            "--task", cfg.task,
            "--seed", str(cfg.seed),
            "--backend", cfg.backend,
            "--out-dir", str(cfg.out_dir),
            "--json-out", str(json_out),
            "--base-env-json", str(base_env),
            *(["--keep-mcap"] if cfg.keep_mcap else []),
            *extra,
        ]
        print(f"\n=== arm '{label}': {' '.join(shlex.quote(a) for a in extra) or '(no overrides)'}",
              flush=True)
        t0 = time.perf_counter()
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        wall = time.perf_counter() - t0
        result = json.loads(json_out.read_text())
        result["subprocess_wall_s"] = wall
        result["overrides"] = extra
        results.append(result)
    return results


# ------------------------------------------------------------------------- reporting --


def _fmt_pct(delta: float) -> str:
    return f"{delta:+.1f}%" if math.isfinite(delta) else "   n/a"


def report(cfg: Cfg, results: list[dict]) -> None:
    width = max(12, *(len(r["label"]) for r in results))
    head = f"{'arm':<{width}}" + "".join(f"{PHASE_LABELS[k]:>9}" for k in PHASES)

    print("\n== per-episode wall clock, mean over "
          f"{results[0]['episodes']} episodes (seconds) ==")
    print(head + f" |{'TOTAL':>8}{'sd':>7}{'ep/s':>8}{'ep/h':>8}")
    for r in results:
        print(f"{r['label']:<{width}}"
              + "".join(f"{r['phases_s'][k]:9.2f}" for k in PHASES)
              + f" |{r['s_per_episode']:8.2f}{r['s_per_episode_sd']:7.2f}"
                f"{r['episodes_per_s']:8.3f}{r['episodes_per_h']:8.0f}")
    print("  Phases are EXCLUSIVE and sum to TOTAL. reset = env.reset minus the splat render"
          " and IK nested\n  inside it; physics = scene.step alone; camsync = re-posing the"
          " wrist camera on its link every\n  step; splat_bg = gsplat background re-render;"
          " comp = splat compositing + shadow catcher;\n  ctrl/IK = go_to_goal; other ="
          " policy, grasp-integrity probes, success eval.")

    if len(results) > 1:
        ref = results[0]
        print(f"\n== delta vs '{ref['label']}' (per-episode seconds) ==")
        print(head + f" |{'TOTAL':>8}")
        for r in results[1:]:
            def d(a: float, b: float) -> str:
                return _fmt_pct(100.0 * (a - b) / b) if b > 1e-9 else "n/a"
            print(f"{r['label']:<{width}}"
                  + "".join(f"{d(r['phases_s'][k], ref['phases_s'][k]):>9}" for k in PHASES)
                  + f" |{d(r['s_per_episode'], ref['s_per_episode']):>8}")
        print(f"  Episode wall time varies {100.0 * ref['s_per_episode_sd'] / ref['s_per_episode']:.0f}%"
              " episode to episode (sampled tempo -> 152-227 frames), but arms share the seed"
              "\n  range, so the comparison is PAIRED: unless the arm changes the trajectory"
              " itself (spawn region,\n  policy tempo), both arms simulate the same episodes"
              " and the delta is real. If it does, compare\n  ms/frame in the health table"
              " instead of TOTAL.")

    print("\n== one-off startup, per PROCESS (seconds) ==")
    print(f"{'arm':<{width}}{'import':>9}{'gs.init':>9}{'build':>9} |{'total':>9}"
          f"{'per-ep if rebuilt':>20}")
    for r in results:
        o = r["one_off"]
        print(f"{r['label']:<{width}}{o['genesis_import_s']:9.1f}{o['gs_init_s']:9.1f}"
              f"{o['scene_build_s']:9.1f} |{r['one_off_total_s']:9.1f}"
              f"{r['one_off_total_s']:20.1f}")
    print("  'per-ep if rebuilt' is what a feature costs EVERY episode if it forces a fresh")
    print("  process/scene (Nyx appearance randomization does: one subprocess per episode).")

    print("\n== health / output ==")
    print(f"{'arm':<{width}}{'px mean':>9}{'frames':>8}{'ms/frame':>10}{'MB/ep':>8}"
          f"{'success':>10}{'warmup s':>10}")
    for r in results:
        warm = ",".join(f"{w:.1f}" for w in r["warmup_s"]) or "-"
        print(f"{r['label']:<{width}}{r['frame_px_mean']:9.1f}{r['frames_mean']:8.0f}"
              f"{r['ms_per_frame']:10.1f}{r['mcap_mb_mean']:8.2f}"
              f"{r['successes']:6d}/{r['episodes']:<3d}{warm:>10}")
    print("  px mean is the mean pixel value over every rendered frame: ~7.8 means the batch")
    print("  path produced near-black frames (broken light rig) while timing looked fine.")
    for r in results:
        if r["frame_px_min"] < DARK_FRAME_PX:
            print(f"  WARNING '{r['label']}': darkest frame mean {r['frame_px_min']:.1f} "
                  f"(< {DARK_FRAME_PX}). A camera rendered near-black -- check the lights/splat "
                  f"before trusting this arm's timing.")

    if cfg.render_backend_note:
        best = min(results, key=lambda r: r["s_per_episode"])
        print(f"\n== ONE PROCESS ==\nThis is {best['s_per_episode']:.1f} s/episode in a single "
              f"unconcurrent process = {best['episodes_per_h']:.0f} ep/h for that process.\n"
              f"Reference (2026-07-26, this host): the 10,500-episode batch ran {REF_SHARDS} "
              f"concurrent shards over 4x L40S\nat ~{REF_S_PER_EP_PER_SHARD} s/episode/shard "
              f"= ~{REF_BATCH_EP_PER_HOUR} ep/h machine-wide. Per-episode cost RISES with "
              f"concurrency\n(GPU contention) and aggregate throughput saturates around 6-8 "
              f"shards, so do not multiply\nthe number above by shard count -- use "
              f"run_shards.py to measure the fan-out itself.")


def main(cfg: Cfg) -> None:
    if cfg.single:
        run_arm(cfg)
        return
    results = run_arms(cfg)
    report(cfg, results)
    summary = cfg.out_dir / "bench_summary.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {summary}")


def _cli() -> Cfg:
    """Parse argv, honouring the parent's serialized base env config.

    ``--base-env-json`` is pulled out before tyro so the child's defaults become the
    PARENT's ``--env.*`` settings; the arm's own flags then override individual fields on
    top. Without this a shared baseline set on the parent would be silently dropped in
    every child."""
    argv = sys.argv[1:]
    default = Cfg()
    if "--base-env-json" in argv:
        i = argv.index("--base-env-json")
        default = Cfg(env=_env_from_jsonable(json.loads(Path(argv[i + 1]).read_text())))
        del argv[i:i + 2]
    return tyro.cli(Cfg, args=argv, default=default)


if __name__ == "__main__":
    main(_cli())
