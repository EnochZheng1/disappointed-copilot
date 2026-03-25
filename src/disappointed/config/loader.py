"""YAML config loader with layered overrides and env var support."""

import os
from pathlib import Path

import yaml

from .schema import AppConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict. Modifies base in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _apply_env_overrides(data: dict, prefix: str = "DCOP_") -> None:
    """Apply environment variable overrides.

    Format: DCOP_SECTION__KEY=value (double underscore separates nesting).
    Example: DCOP_CAMERA__BACKEND=webcam
    """
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("__")
        target = data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = env_value


def load_config(*config_paths: Path) -> AppConfig:
    """Load and merge YAML configs. Later files override earlier ones.

    Args:
        *config_paths: Paths to YAML config files. Later files take precedence.

    Returns:
        Validated AppConfig instance.
    """
    merged: dict = {}
    for path in config_paths:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            _deep_merge(merged, data)

    _apply_env_overrides(merged)
    return AppConfig(**merged)
