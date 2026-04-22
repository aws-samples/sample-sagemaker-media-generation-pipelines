# Config Authoring Guide

> **Navigation:** [← Main README](../README.md) | [Extending Guide](EXTENDING.md) | [config/README.md](../config/README.md)

This guide teaches you how to create your own pipeline configuration from scratch. Every pipeline in the framework is defined entirely by a YAML config file — no CDK code changes required.

---

## Config-Driven Philosophy

The framework follows a strict config-driven design: **YAML config + container directory = new pipeline**. All infrastructure (ECR repos, CodeBuild projects, S3 buckets, processing steps, Lambda functions, A2I flows) is created automatically from the config at CDK synth time.

To add a new pipeline:
1. Create a YAML file in `config/pipeline/`
2. Create container directories under `processing_job/` for any new steps
3. Deploy — CDK reads the config and provisions everything

Zero CDK code changes. Zero CloudFormation templates to write. The config is the single source of truth.

---

## Top-Level Keys

Every pipeline config YAML supports the following top-level keys:

| Key | Type | Required | Description |
|---|---|---|---|
| `construct_id` | `str` | Yes | Short kebab-case identifier (e.g. `t2i`, `vr`, `ma`). Combined with `shared_prefix` to form the full resource prefix. Must match `^[a-z][a-z0-9-]*$`. |
| `steps` | `dict[str, ContainerConfig]` | Yes | Per-step container configurations keyed by step name. Each key becomes a SageMaker Processing Job with its own ECR repo and S3 buckets. |
| `pipeline_graph` | `dict[str, list[str]]` | No | DAG definition: `step_name -> [dependency_step_names]`. An empty list `[]` means the step has no dependencies (root step). Controls execution order and S3 wiring between steps. |
| `s3_downloads` | `list[S3Download]` | No | Model weights to download. Each entry has `url` (HTTP/HTTPS), `path` (destination in models bucket), and optional `extract: true` for zip archives. |
| `a2i` | `dict[str, A2IConfig]` | No | A2I human review flow definitions. Each key creates a flow definition with a Cognito-backed workforce portal for reviewing generated assets. |
| `lambda_steps` | `dict[str, LambdaStepConfig]` | No | Lambda functions invoked as pipeline steps. Each entry specifies `lambda_path` (directory under `lambdas/`), optional `a2i_name` (must reference a key in `a2i`), and `media_type`. |
| `setup` | `dict[str, SetupConfig]` | No | Standalone setup processing jobs (e.g. dataset download). Each entry creates a `ReusableProcessingJob` with a trigger Lambda. |
| `retrieval` | `str \| null` | No | Filename of a retrieval config in `config/retrieval/` (e.g. `"retrieval.yaml"`). When set, activates the RetrievalStack with OpenSearch Serverless, SQS ingestion, and Bedrock embedding. |
| `dynamodb` | `DynamoDBConfig` | No | Key schema for the DynamoDB results table. Defaults to `partition_key: id`, `sort_key: step`. |

---

## ContainerConfig Fields

Each entry under `steps` is validated against the `ContainerConfig` Pydantic model. All fields:

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `InstanceType` | `Literal` | *(required)* | One of: `ml.c5.xlarge`, `ml.g4dn.2xlarge`, `ml.g5.xlarge`, `ml.g5.8xlarge`, `ml.m5.xlarge`, `ml.m5.2xlarge`, `ml.m5.4xlarge` | AWS instance type for the processing job. Use `ml.g5.*` for GPU workloads, `ml.m5.*` or `ml.c5.*` for CPU-only. |
| `InstanceCount` | `int` | *(required)* | 1–40 | Number of instances. Work is sharded across instances automatically. |
| `VolumeSizeInGB` | `int` | *(required)* | 50–125 | EBS volume size attached to each instance. |
| `ContainerEntrypoint` | `list[str]` | `[]` | — | Container entrypoint command (e.g. `["python3", "main.py"]`). |
| `ContainerArguments` | `list[str]` | *(required)* | min length 1 | Arguments passed to the container (e.g. `["--generate"]`). SageMaker requires at least one argument. |
| `models_prefix` | `list[str]` | `[]` | — | S3 key prefixes inside the models bucket to mount for this step (e.g. `["wan_22_t2v"]`). |
| `ecr_image` | `str` | `""` | — | ECR image name if different from the step name. Lets multiple steps share one Docker image (e.g. `vbench_t2v` and `vbench_i2v` both use `"vbench"`). |
| `input_channel` | `str` | `"input"` | — | SageMaker input channel name. Mounted at `/opt/ml/processing/input/<channel>/`. |
| `include_data_input` | `bool` | `false` | — | If `true`, also mounts the DataStack input bucket as an additional input channel (for accessing `inputs.json`). |
| `num_assets_per_prompt` | `int` | `1` | 1–100 | Number of assets to generate per prompt. Injected as the `NUM_ASSETS_PER_PROMPT` environment variable. |
| `Environment` | `dict[str, str]` | `{}` | — | Custom environment variables passed to the container (e.g. `VRAG_LLM_WORKERS: "10"`). |


---

## Annotated Minimal Example

Below is a minimal pipeline config that defines one generation step with A2I human review. Based on `config_t2i.yaml`, simplified for clarity:

```yaml
# config/pipeline/config_my_pipeline.yaml

# Unique identifier — combined with shared_prefix to form resource names.
# Must be lowercase alphanumeric + hyphens, starting with a letter.
construct_id: myp

# Model weights to download to the S3 models bucket (one-time).
s3_downloads:
  - url: "https://huggingface.co/my-org/my-model/resolve/main/model.safetensors"
    path: "my_model/model.safetensors"

# DynamoDB key schema for the results table (defaults shown).
dynamodb:
  partition_key: id
  sort_key: step

# Processing steps — each key becomes a SageMaker Processing Job.
steps:
  generate:
    InstanceCount: 1                              # Single instance
    InstanceType: ml.g5.xlarge                     # GPU instance
    VolumeSizeInGB: 50                             # 50 GB EBS volume
    ContainerEntrypoint: ["python3", "main.py"]    # Container entrypoint
    ContainerArguments: ["--generate"]              # At least one argument required
    models_prefix: ["my_model"]                    # Mount model weights from S3
    num_assets_per_prompt: 3                       # Generate 3 assets per prompt

# A2I human review flow definitions.
a2i:
  my_review:
    media_type: image                              # image | video | audio
    task_title: "Review generated image"
    task_description: "Is this image good?"
    task_count: 1                                  # Number of reviewers per asset
    task_timeout_seconds: 3600                     # 1 hour timeout

# Lambda steps invoked as SageMaker Pipeline steps.
lambda_steps:
  submit_a2i_generate:
    lambda_path: submit_a2i_review                 # Directory under lambdas/
    a2i_name: my_review                            # Must match a key in a2i above
    media_type: image

# Pipeline DAG: step -> [dependencies]. Empty list = root step.
pipeline_graph:
  generate: []                                     # No dependencies — runs first
  submit_a2i_generate: ["generate"]                # Runs after generate completes
```

This config creates:
- An ECR repository and CodeBuild project for the `generate` step
- S3 input/output buckets
- A DynamoDB results table
- A model download processing job
- A SageMaker Pipeline with two steps: `generate` → `submit_a2i_generate`
- An A2I flow definition with a Cognito-backed workforce portal
- A Lambda function for submitting assets to A2I review

All from ~40 lines of YAML.

---

## Prefix System

Resource naming uses a two-level prefix system to support multiple pipelines in the same AWS account:

- `shared_prefix` — defined in `config/cicd/cicd.yaml` (default: `dev`). Must be exactly 3 lowercase alphanumeric characters (e.g. `dev`, `prd`, `abc`). This constraint ensures AOSS policy names stay within the 32-character limit. Used for shared stacks deployed once across all pipeline configs (SecurityStack, CodeBuildStack, CiCdPipelineStack, ContainerPipelineStack).
- `construct_id` — defined per pipeline config in `config/pipeline/*.yaml`.
- **Full prefix** = `{shared_prefix}{construct_id}` (e.g. `devt2i`, `devvr`).

Per-config stacks use the full prefix: `{prefix}-DataStack`, `{prefix}-A2IStack`, `{prefix}-PipelineStack`.

### Current Prefix Table

| Config File | `construct_id` | Full Prefix |
|---|---|---|
| `config_vrag.yaml` | `vr` | `devvr` |
| `config_i2v.yaml` | `i2v` | `devi2v` |
| `config_motionart.yaml` | `ma` | `devma` |
| `config_t2a.yaml` | `t2a` | `devt2a` |
| `config_t2i.yaml` | `t2i` | `devt2i` |
| `config_t2v.yaml` | `t2v` | `devt2v` |

---

## Environment Variable Expansion

Pipeline config YAMLs support `${ENV_VAR}` syntax. Before YAML parsing, the framework runs `os.path.expandvars()` on the raw file content, resolving any `${VAR}` or `$VAR` references from the current environment.

Environment variables are loaded from the `.env` file at the project root (via `python-dotenv`). The `.env.example` template shows the required variables:

```bash
AWS_ACCOUNT_ID=0123456789
REGION=us-east-1
```

This means you can use environment variables anywhere in a pipeline config:

```yaml
steps:
  my_step:
    Environment:
      MY_BUCKET: "${AWS_ACCOUNT_ID}-${REGION}-my-bucket"
```

At synth time, `${AWS_ACCOUNT_ID}` and `${REGION}` are replaced with the actual values from `.env`.

---

## Registering in CI/CD

To include a new pipeline config in the CI/CD system, update `config/cicd/cicd.yaml` with three entries:

### 1. `pipeline_configs`

Add the filename to the list. This creates a dedicated CodePipeline for the config:

```yaml
pipeline_configs:
  - "config_vrag.yaml"
  - "config_i2v.yaml"
  - "config_my_pipeline.yaml"   # ← add here
```

### 2. `test_commands`

Map the config filename to a pytest command with the relevant markers. Only tests matching these markers run in CI for this pipeline:

```yaml
test_commands:
  config_my_pipeline.yaml: "uv run pytest tests/unit/ -x --no-header -q -n auto -m 'core or cicd or processing or model_validation or steps_my_step or integration'"
```

A `model_validator` on `CicdConfig` ensures every entry in `pipeline_configs` has a corresponding key in `test_commands` — missing entries cause a validation error at synth time.

### 3. `input_data`

Map the config filename to sample input files under `sample_input_data/`. These are uploaded during the CI/CD UploadInput stage:

```yaml
input_data:
  config_my_pipeline.yaml:
    - "inputs_my_pipeline.json"
    - "images/my_pipeline/"      # optional: reference images
```

---

## Pydantic Validation

All config models use strict validation:

```python
model_config = ConfigDict(
    extra="forbid",       # Unknown fields are rejected
    strict=True,          # No type coercion (int stays int, str stays str)
)
```

### What This Means

- **`extra='forbid'`** — Any field not defined in the model raises a `ValidationError`. Catches typos like `Instanccount` instead of `InstanceCount`.
- **`strict=True`** — No implicit type conversion. A string `"2"` is rejected where an `int` is expected. Booleans must be `true`/`false`, not `"true"`.

### Cross-Field Validators

- `PipelineConfig._check_a2i_name_references` — every `lambda_steps[*].a2i_name` must reference a key in the `a2i` dict.
- `CicdConfig._check_test_commands_coverage` — every entry in `pipeline_configs` must have a corresponding key in `test_commands`.
- `SetupConfig._check_prompts_le_images` — `num_prompts` cannot exceed `test_image_count`.

---

## Navigation

- [← Main README](../README.md) — project overview and getting started
- [Extending the Framework](EXTENDING.md) — add new steps, pipelines, and containers
- [Pipeline Use Cases](USECASES.md) — per-pipeline use cases, DAGs, and models
- [config/README.md](../config/README.md) — Pydantic model reference and directory layout
