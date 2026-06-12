from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment YAML config.

    Args:
        path: YAML file path.

    Returns:
        The top-level config mapping.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must contain a YAML mapping.")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write an experiment config to YAML.

    Args:
        config: Config mapping to serialize.
        path: Destination YAML path.

    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
