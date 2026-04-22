> **Navigation:** [← Main README](../README.md) | [Extending Guide — Adding Tests](../docs/EXTENDING.md#adding-tests)

# tests/

Test suite for the modular media generation framework. All tests live under `tests/unit/` and are organized by framework domain.

## Directory Structure

```
tests/unit/
├── __init__.py
├── conftest.py          # Root: shared mocks, Hypothesis profile, step_names re-export
├── step_names.py        # Hardcoded test constants (no YAML reads)
├── inputs.json
├── core/                # Shared infrastructure, constructs, config models, security properties
│   ├── conftest.py      # SecurityStack factory, _default_cfg(), template cache fixtures
│   └── test_*.py
├── cicd/                # CI/CD pipeline stack, config, deploy script
│   ├── conftest.py      # CiCdPipelineStack factory, mock_read_config_prefix
│   └── test_*.py
├── retrieval/           # Retrieval stack, config, container, ingest Lambda
│   ├── conftest.py      # _valid_retrieval_config() helper
│   └── test_*.py
├── processing/          # Processing job construct, model, Dockerfiles
│   ├── conftest.py      # _valid_step(), _default_cfg() helpers
│   └── test_*.py
├── model_validation/    # Model validation agent, captioning, t2a, t2i, vbench
│   └── test_*.py
├── steps/               # Per-step tests (t2v, captioning, vrag, loadtest, etc.)
│   └── test_*.py
└── integration/         # Full stack chain integration tests
    └── test_*.py
```

## Pytest Markers

Defined in `pyproject.toml` under `[tool.pytest.ini_options]`. Every test file sets its marker at module level:

```python
pytestmark = pytest.mark.core  # or cicd, retrieval, processing, etc.
```

| Marker | Directory | Description |
|---|---|---|
| `core` | `core/` | Shared infrastructure, constructs, config models, security properties |
| `cicd` | `cicd/` | CI/CD pipeline stack, config, deploy script |
| `retrieval` | `retrieval/` | Retrieval stack, config, container, ingest Lambda |
| `processing` | `processing/` | Processing job construct, model, Dockerfiles |
| `model_validation` | `model_validation/` | Model validation agent, captioning, t2a, t2i, vbench |
| `steps_vrag` | `steps/` | V-RAG LLM step tests |
| `steps_t2v` | `steps/` | Text-to-video step tests |
| `steps_captioning` | `steps/` | Captioning step tests |
| `steps_t2a` | `steps/` | Text-to-audio step tests |
| `steps_t2i` | `steps/` | Text-to-image step tests |
| `steps_i2v` | `steps/` | Image-to-video step tests |
| `steps_flf2v` | `steps/` | First-last-frame-to-video step tests |
| `steps_loadtest` | `steps/` | Load test Lambda tests |
| `steps_setup` | `steps/` | Setup container tests (dataset_ingest) |
| `integration` | `integration/` | Full stack chain integration tests |

## Conftest Hierarchy

Fixtures are organized in a hierarchy — import helpers from the appropriate level.

| File | Provides |
|---|---|
| `tests/unit/conftest.py` | `_mock_from_asset()`, `_mock_s3_asset()`, Hypothesis CI profile, per-config `{construct_id}_downloads.json` creation, `step_names` re-export (`STEP_0`, `STEP_1`, etc.) |
| `core/conftest.py` | `security_stack` (class-scoped), `_default_cfg()`, `_default_pipeline_config()` |
| `cicd/conftest.py` | `_create_cicd_pipeline_stack()`, `mock_read_config_prefix` fixture |
| `retrieval/conftest.py` | `_valid_retrieval_config()` |
| `processing/conftest.py` | `_valid_step()`, `_default_cfg()` |

Import shared helpers from root conftest:

```python
from tests.unit.conftest import _mock_from_asset, _mock_s3_asset
from tests.unit.conftest import STEP_0, STEP_1  # re-exported from step_names
```

## Hypothesis Configuration

Registered in root `conftest.py`:

```python
settings.register_profile(
    "ci",
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
settings.load_profile("ci")
```

- CI profile: `max_examples=50`, `deadline=None`, `suppress_health_check=[too_slow]`
- Profile is loaded by default for all test runs
- Individual property tests can override with `@settings(max_examples=100)` for deeper fuzzing

## Template Caching Pattern

CDK stack synthesis is expensive. Use class-scoped fixtures to avoid re-synthesizing per test method:

```python
@pytest.fixture(scope="class")
def security_stack():
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
        return SecurityStack(app, "SecStack", prefix="dev", env=env)

class TestSecurityStackResources:
    def test_vpc_created(self, security_stack):
        template = assertions.Template.from_stack(security_stack)
        template.resource_count_is("AWS::EC2::VPC", 1)

    def test_kms_key_created(self, security_stack):
        template = assertions.Template.from_stack(security_stack)
        template.resource_count_is("AWS::KMS::Key", 1)
```

Always mock `lambda_.Code.from_asset` and `s3_assets.Asset` when synthesizing stacks.

## Adding a New Test File

1. Determine which subdirectory the test belongs to (see Markers table above)
2. Create `tests/unit/<subdir>/test_<feature>.py`
3. Add `pytestmark = pytest.mark.<category>` at module level
4. Import shared fixtures from the category's `conftest.py`
5. If a new marker is needed: add it to `pyproject.toml` `[tool.pytest.ini_options]` markers list
6. Update `test_commands` in `config/cicd/cicd.yaml` for every pipeline that should run the new marker

## Running Tests

```bash
# Local — all tests
uv run pytest tests/unit/ -x --no-header -q

# Local — by marker
uv run pytest tests/unit/ -m core -x --no-header -q
uv run pytest tests/unit/ -m 'core or cicd' -x --no-header -q

# Local — parallel
uv run pytest tests/unit/ -x --no-header -q -n auto

# CI — per-pipeline (resolved from test_commands in cicd.yaml)
uv run pytest tests/unit/ -x --no-header -q -n auto -m 'core or cicd or retrieval or processing or model_validation or steps_vrag or steps_t2v or steps_i2v or integration'
```

## Per-Pipeline Test Scoping

Each CI/CD pipeline config runs only relevant markers. See the [`test_commands` mapping in config/README.md](../config/README.md#test_commands-mapping) for the full table.
