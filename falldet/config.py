"""YAML config loading with dotted-key command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _deep_update(base: dict, other: dict) -> dict:
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply overrides such as ``train.epochs=50``."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        value = yaml.safe_load(raw)
        node: dict = {}
        leaf = node
        *parents, last = key.split(".")
        for part in parents:
            leaf = leaf.setdefault(part, {})
        leaf[last] = value
        _deep_update(cfg, node)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False, allow_unicode=True)
