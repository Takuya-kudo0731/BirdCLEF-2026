"""
Shared configuration loader for BirdCLEF 2026.

Both train.py and inference.py import from here to avoid duplication.
"""

import types

import yaml


def load_config(path: str) -> types.SimpleNamespace:
    """Load a YAML config file and return a ``SimpleNamespace`` object."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _dict_to_namespace(raw)


def _dict_to_namespace(d: dict) -> types.SimpleNamespace:
    ns = types.SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns
