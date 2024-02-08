"""Hierarchical config management with CLI overrides."""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: str | Path = "configs/base.yaml",
    overrides: list[str] | None = None,
) -> DictConfig:
    base = OmegaConf.load(config_path)
    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        base = OmegaConf.merge(base, cli)
    OmegaConf.resolve(base)
    return base


def parse_args_to_config() -> DictConfig:
    args = sys.argv[1:]
    config_path = "configs/base.yaml"
    overrides = []
    for arg in args:
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
        elif "=" in arg:
            overrides.append(arg)
    return load_config(config_path, overrides)
