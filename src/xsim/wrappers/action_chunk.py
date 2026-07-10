"""Action-chunking wrapper.

Policies that emit an action *chunk* (a horizon of ``h`` actions per inference, e.g. a
served crossformer) drive the env open-loop for the whole chunk before observing again.
This wrapper hides that: ``step(chunk)`` runs up to ``h`` child steps in sequence,
accumulates reward, and returns the *final* obs (plus ``done`` as soon as any child step
ends the episode). One outer ``step`` == one policy inference.

The chunk is iterated along its leading axis:

- array-like ``chunk`` of shape ``(H, action_dim)`` -> ``chunk[i]`` per child step;
- ``dict`` chunk (named action parts, ``{part: (H, ...)}``) -> ``{part: v[i]}`` per step.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from xsim.wrappers.base import Wrapper


class ActionChunkWrapper(Wrapper):
    def __init__(self, env: Any, h: int = 50):
        super().__init__(env)
        self.h = h

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        obs = reward = done = info = None
        total_reward = 0.0
        executed = 0
        for a in _iter_chunk(action, self.h):
            obs, reward, done, info = self.env.step(a)
            total_reward += float(reward)
            executed += 1
            if done:
                break
        if executed == 0:
            raise ValueError("ActionChunkWrapper received an empty action chunk")
        return obs, total_reward, done, info


def _iter_chunk(action: Any, h: int) -> Iterator[Any]:
    """Yield up to ``h`` per-step actions from an array-like or dict chunk."""
    if isinstance(action, dict):
        length = min(len(v) for v in action.values())
        for i in range(min(h, length)):
            yield {k: v[i] for k, v in action.items()}
    else:
        for i in range(min(h, len(action))):
            yield action[i]
