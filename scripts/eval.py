"""Grid-evaluation harness driven through a Gymnasium-like protocol.

Same job as ``scripts/eval_grid.py`` — sweep a lift/stack policy over a table-position
grid, give it one try per position, repeat the grid ``reps`` times (fresh appearance/
lighting/camera/arm jitter each rep), and report an overall + per-position success
heatmap — but the trial loop here talks to the env and policy through the plain
Gymnasium contract from ``src/tmp.py`` instead of the eval-specific adapters:

    obs = env.reset(seed=..., options={grid point})   # options place the cube(s)
    while not done:
        action = policy.step(obs)["action"]
        obs, reward, done, info = env.step(action)     # env owns scoring

Because the env now reports ``reward``/``done``/``info`` itself, the harness no longer
tracks cube rise, pokes proprio, runs a warm-start, or handles welds/grippers — all of
that moves inside the env. The env is expected to place the cube(s) from the ``options``
dict passed to ``reset`` and to put ``success`` (and any task metrics) into the terminal
``info``. Wiring the Genesis ``TaskEnv`` to this contract is a follow-up.

    uv run python scripts/eval.py --task lift --grid-nx 3 --grid-ny 3 --reps 1
"""

from __future__ import annotations
from rich import print

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Literal, Protocol

import numpy as np
import tyro
import jax

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xsim.task_env import TaskEnvCfg  # noqa: E402


# ---------------------------------------------------------------------------------------
# Gymnasium-like protocol (see src/tmp.py). The env/policy passed to this harness only
# need these methods; anything else is their own business.
# ---------------------------------------------------------------------------------------


class GymEnv(Protocol):
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Any: ...
    def step(self, action: Any) -> tuple[Any, float, bool, dict]: ...
    def render(self) -> dict[str, np.ndarray]: ...


class GymPolicy(Protocol):
    def reset(self) -> None: ...
    def step(self, obs: Any) -> np.ndarray: ...


class ChunkClient:
    """Wraps ``webpolicy.Client`` to return a plain ``(H, A)`` action array.

    The served crossformer returns a dict of named action parts with leading batch/window
    singleton dims, e.g.::

        {"actions": {"joints": (1,1,H,7), "gripper": (1,1,H,1), "kp3dc_robot": (1,1,H,3,14,3)}}

    ``ActionChunkWrapper`` iterates the *leading* axis, so it needs the horizon ``H`` to be
    leading. This concatenates ``[joints, gripper]`` on the last axis (``A = 7 + 1 = 8``),
    discards everything else, and reshapes to ``(H, A)`` — collapsing the batch/window dims.
    """

    def __init__(self, client: Any):
        self._client = client

    def reset(self, *args, **kwargs) -> None:
        self._client.reset(*args, **kwargs)

    def step(self, obs: Any) -> np.ndarray:
        act = self._client.step(obs)
        parts = act["actions"] if isinstance(act, dict) and "actions" in act else act
        joints = np.asarray(parts["joints"], dtype=np.float32)
        gripper = np.asarray(parts["gripper"], dtype=np.float32)
        action_dim = joints.shape[-1] + gripper.shape[-1]
        return np.concatenate([joints, gripper], axis=-1).reshape(-1, action_dim)


def spec(tree: Any) -> Any:
    summary = jax.tree.map(
        lambda x: (tuple(x.shape), x.dtype) if hasattr(x, "shape") else type(x), tree)
    return summary


@dataclass
class Config:
    task: Literal["lift", "stack"] = "lift"
    grid_nx: int = 10
    grid_ny: int = 10
    reps: int = 10
    seed: int = 51000                 # eval base seed, clear of training ranges
    time_limit: bool = True           # cap episode length (env sets `done` on timeout)
    max_control_steps: int = 600      # time limit in CONTROL steps (30 Hz -> 600 = 20 s)
    out: Path | None = None           # default: PROJECT_ROOT/outputs/eval/<task>
    backend: Literal["gpu", "cpu"] = "gpu"
    cube_yaw: float = 0.0             # deterministic cube (and green) yaw at spawn
    video_every: int = 0             # save an mp4 of every Nth trial (0 = off, 1 = all)
    resume: bool = True
    host: str = "localhost"          # webpolicy inference server
    port: int = 8001
    chunk_h: int = 50                # actions executed open-loop per policy inference
    # physics rate (sim fidelity) and action rate (how fast the trained trajectory plays).
    # control_every = sim_hz / control_hz physics steps are held per model action. Defaults
    # (120/30 -> 4) reproduce the dataset's cadence; lower control_hz gives the arm more
    # sim-time to reach each waypoint (at the cost of playing the trajectory slower).
    sim_hz: int = 120
    control_hz: int = 30
    env: TaskEnvCfg = field(default_factory=lambda: TaskEnvCfg(noslip_iterations=10))


def build_env(cfg: Config) -> GymEnv:
    """Construct the Gymnasium-like env: TaskEnv under the video + action-chunk wrappers.

    TODO: adapt TaskEnv itself to the reset(options=...) / step(action) contract.
    """
    import genesis as gs

    from xsim.task_env import TaskEnv
    from xsim.wrappers import ActionChunkWrapper, GenesisGymAdapter, VideoRecordWrapper

    gs.init(backend=gs.gpu if cfg.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    cfg.env.task = cfg.task

    # sim rate feeds the env's integrator dt; control rate sets how many physics steps the
    # adapter holds each action. control_every falls out of the two (rounded to an int).
    cfg.env.physics_dt = 1.0 / cfg.sim_hz
    control_every = max(1, round(cfg.sim_hz / cfg.control_hz))
    eff_control_hz = cfg.sim_hz / control_every
    print(f"[cadence] sim_hz={cfg.sim_hz} control_hz={cfg.control_hz} "
          f"-> control_every={control_every} (effective control_hz={eff_control_hz:.1f})")

    # GenesisGymAdapter is a temporary shim making TaskEnv gym-conformant (see its docstring)
    env: GymEnv = GenesisGymAdapter(
        TaskEnv(cfg.env), control_every=control_every,
        max_control_steps=cfg.max_control_steps if cfg.time_limit else None)
    if cfg.video_every > 0:
        out = cfg.out if cfg.out is not None else PROJECT_ROOT / "outputs" / "eval" / cfg.task
        # video wraps the raw env so it captures every physics step, then chunking wraps that
        env = VideoRecordWrapper(
            env, out / "videos",
            episode_trigger=lambda ep: ep % cfg.video_every == 0)
    return ActionChunkWrapper(env, h=cfg.chunk_h)


def build_policy(cfg: Config, env: GymEnv) -> GymPolicy:
    """The served crossformer over the websocket, wrapped to emit a ``(H, A)`` action array."""
    from webpolicy.client import Client

    return ChunkClient(Client(cfg.host, cfg.port))


def main(cfg: Config) -> None:
    env = build_env(cfg)
    policy = build_policy(cfg, env)

    for ep in range(50):  # Run 5 episodes
        obs = env.reset()
        policy.reset()
        done = False
        while not done:
            action = policy.step(obs)              # (H, A) array; ActionChunkWrapper unrolls it
            obs, reward, done, info = env.step(action)
            print(f"ep{ep} reward={reward} done={done} success={info.get('success')}")


if __name__ == "__main__":
    main(tyro.cli(Config))
