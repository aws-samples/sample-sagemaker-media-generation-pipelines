"""Known model identifiers registry.

Loads models.yaml and exposes KNOWN_MODELS — a tuple of model suffixes
used in filenames and DynamoDB sort keys. Order matters: longer variants
come first so regex matching picks the most specific match.

Usage:
    from schema.models import KNOWN_MODELS
"""

from __future__ import annotations

from pathlib import Path

import yaml

_MODELS_PATH = Path(__file__).parent / "models.yaml"


def _load() -> tuple[str, ...]:
    """Load models.yaml into an ordered tuple."""
    with open(_MODELS_PATH) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise TypeError(f"models.yaml must be a list, got {type(raw).__name__}")
    return tuple(raw)


KNOWN_MODELS = _load()
