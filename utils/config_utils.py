from __future__ import annotations

import os
import re
from typing import Any


_ENV_PATTERN = re.compile(r"^\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))$")


def _required_env_name(value: str) -> str | None:
    match = _ENV_PATTERN.match(value)
    if not match:
        return None
    return match.group("braced") or match.group("plain")


def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in strings, lists, and dicts."""

    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]

    if isinstance(value, tuple):
        return tuple(expand_env_vars(v) for v in value)

    if isinstance(value, str):
        env_name = _required_env_name(value)
        expanded = os.path.expandvars(value)
        if env_name is not None and expanded == value:
            raise KeyError(f"Environment variable '{env_name}' is required by this config.")
        return os.path.expanduser(expanded)

    return value
