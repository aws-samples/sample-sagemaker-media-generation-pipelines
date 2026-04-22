"""DynamoDB column name registry.

Loads columns.yaml and exposes COL — a namespace of constant-to-column
mappings. Steps, Lambdas, and CDK code all import COL to reference
DynamoDB attribute names.

Usage:
    from schema.columns import COL

    record = {
        COL.ID: "tokyo-rain-alley",
        COL.STEP: "t2v#wan22#g0",
        COL.PROMPT: "A cinematic shot...",
    }
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

_SCHEMA_PATH = Path(__file__).parent / "columns.yaml"


def _load() -> SimpleNamespace:
    """Load columns.yaml into a namespace and validate no duplicate values."""
    with open(_SCHEMA_PATH) as f:
        raw = yaml.safe_load(f)

    seen: dict[str, str] = {}
    for const, col in raw.items():
        if col in seen:
            raise ValueError(
                f"Duplicate column name '{col}' mapped by both '{seen[col]}' and '{const}' in columns.yaml"
            )
        seen[col] = const

    return SimpleNamespace(**raw)


COL = _load()
