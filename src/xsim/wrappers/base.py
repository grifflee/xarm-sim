"""Minimal Gymnasium-style wrapper base.

Follows the same lightweight contract as ``src/tmp.py`` (``reset() -> obs``,
``step(action) -> (obs, reward, done, info)``, ``render() -> dict``) rather than pulling
in a hard ``gymnasium`` dependency, so wrappers compose over the repo's env without
committing to the 5-tuple API. Unknown attributes forward to the wrapped env, so
``wrapper.cfg`` / ``wrapper.robot`` keep working through any wrapper stack.
"""

from __future__ import annotations

from typing import Any


class Wrapper:
    def __init__(self, env: Any):
        self.env = env

    @property
    def unwrapped(self) -> Any:
        """The innermost env underneath any stack of wrappers."""
        return self.env.unwrapped if isinstance(self.env, Wrapper) else self.env

    def reset(self, **kwargs) -> Any:
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        return self.env.step(action)

    def render(self) -> dict:
        return self.env.render()

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    def __getattr__(self, name: str) -> Any:
        # only reached when normal lookup misses; forward to the wrapped env. Guard `env`
        # so an access before __init__ sets it raises cleanly instead of recursing.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)
