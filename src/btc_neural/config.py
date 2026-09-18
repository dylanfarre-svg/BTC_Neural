"""Configuration loading: config/default.yaml, overridden by config/local.yaml and .env."""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

load_dotenv(ROOT / ".env")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    cfg = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text())
    local = CONFIG_DIR / "local.yaml"
    if local.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(local.read_text()) or {})
    return cfg


def path(key: str) -> Path:
    p = Path(load_config()[key])
    return p if p.is_absolute() else ROOT / p


def halvings() -> list[pd.Timestamp]:
    return [pd.Timestamp(d) for d in load_config()["halvings"]]


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)
