> **Navigation:** [← Main README](../README.md) | [Extending Guide — Schema Package](../docs/EXTENDING.md#schema-package)

# schema/

Shared data registries used across all containers, Lambdas, and CDK code. These registries ensure consistent DynamoDB attribute names and model identifiers throughout the entire framework — changing a value in one place propagates everywhere automatically.

## Why This Exists

Without a centralized registry, DynamoDB column names and model identifiers would be hardcoded as string literals across dozens of files (containers, Lambdas, CDK stacks, tests). A typo in any one of them would cause silent data mismatches. The `schema/` package eliminates this by providing a single source of truth loaded at runtime.

## Contents

### `columns.yaml` + `columns.py`

The column registry. `columns.yaml` maps constant names to DynamoDB attribute names. `columns.py` loads the YAML and exposes the `COL` namespace:

```python
from schema.columns import COL

record = {
    COL.ID: "tokyo-rain-alley",
    COL.STEP: "t2v#wan22#g0",
    COL.PROMPT: "A cinematic shot of rain in Tokyo",
}
```

To add or rename a column, edit `columns.yaml` — all code that imports `COL` picks it up automatically.

### `models.yaml` + `models.py`

The model registry. `models.yaml` lists known model suffixes (e.g. `wan22`, `z_image_turbo`). `models.py` loads the YAML and exposes `KNOWN_MODELS` as a tuple. Used for filename parsing, DynamoDB sort key construction, and A2I asset labeling.

## Synth-Time Copy

At CDK synth time, `app.py` copies this directory into `processing_job/schema/` so container builds (rooted at `processing_job/`) can access both registries. The copy is gitignored and recreated on every synth.
