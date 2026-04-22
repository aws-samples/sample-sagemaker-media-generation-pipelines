> **Navigation:** [← Main README](../README.md) | [Config Guide](../docs/CONFIG_GUIDE.md) | [Use Cases](../docs/USECASES.md)

# config/

Configuration layer for the modular media generation framework. All pipeline behavior, CI/CD orchestration, and retrieval subsystem settings are driven by Pydantic-validated YAML files in this directory. Adding a new pipeline or modifying compute resources requires only YAML changes — zero CDK code modifications.

## Directory Layout

```
config/
├── config.py                          # Pydantic models and loaders for all config files
├── pipeline/
│   ├── config_vrag.yaml               # V-RAG pipeline (I2V with retrieval and VBench)
│   ├── config_i2v.yaml                 # Image-to-video only with VBench evaluation
│   ├── config_motionart.yaml          # MotionArt first-last-frame-to-video with A2I
│   ├── config_t2a.yaml               # Text-to-audio with A2I review
│   ├── config_t2i.yaml               # Text-to-image with A2I review
│   └── config_t2v.yaml               # Text-to-video with VBench evaluation and A2I review
├── retrieval/
│   └── retrieval.yaml                 # Retrieval subsystem config (OpenSearch, SQS, Lambda)
└── cicd/
    └── cicd.yaml                      # CI/CD pipeline configuration (stages, per-config tests)
```

## Pydantic Models (`config.py`)

All config models use `extra='forbid'` and `strict=True` — unknown fields are rejected, and type coercion is disabled.

| Model | Description |
|---|---|
| `PipelineConfig` | Top-level pipeline config: `construct_id`, `steps`, `pipeline_graph`, `s3_downloads`, `a2i`, `lambda_steps`, `setup`, `retrieval` |
| `ContainerConfig` | Per-step container settings: instance type/count, volume, entrypoint, arguments, `models_prefix`, `ecr_image`, `input_channel`, `num_assets_per_prompt`, `Environment` |
| `ModelDownloadConfig` | Extends `ContainerConfig` with `MaxRuntimeInSeconds` for the one-time model download job |
| `S3Download` | Single model download entry: `url`, `path`, optional `extract` flag |
| `DynamoDBConfig` | Key schema for the results table: `partition_key` (default `id`), `sort_key` (default `step`) |
| `A2IConfig` | A2I human review flow: `workteam_name`, `media_type`, `task_title`, `task_description`, `task_count`, `task_timeout_seconds`, `max_concurrent_tasks` |
| `LambdaStepConfig` | Lambda pipeline step: `lambda_path`, `a2i_name`, `media_type` |
| `SetupConfig` | Extends `ContainerConfig` for setup jobs: `dataset_url`, `dataset_script`, `num_prompts`, `test_image_count` |
| `RetrievalConfig` | Retrieval subsystem: AOSS collection, embedding model, SQS tuning, Lambda resource limits |
| `CicdConfig` | CI/CD pipeline: stages, per-config test commands, deploy settings, shared prefix |

Loaders: `get_pipeline_config()`, `get_cicd_config()`, `get_retrieval_config()`. Pipeline configs support `${ENV_VAR}` expansion via `os.path.expandvars`.

---

## Pipeline Configs

Each YAML in `config/pipeline/` defines a self-contained SageMaker Pipeline. See [Pipeline Use Cases](../docs/USECASES.md) for per-pipeline DAGs, steps, models, and when to use each config. See [Config Authoring Guide](../docs/CONFIG_GUIDE.md) for how to create your own.

---

## `retrieval.yaml` — RetrievalConfig

Located at `config/retrieval/retrieval.yaml`. Configures the image retrieval subsystem (OpenSearch Serverless + SQS + Bedrock Titan embedding). Activated per-pipeline by setting `retrieval: "retrieval.yaml"` in a pipeline config. Currently used by `config_vrag.yaml` (V-RAG).

| Field | Type | Default | Description |
|---|---|---|---|
| `collection_name` | str (3-32 chars, `[a-z0-9-]`) | — | AOSS collection base name (deployed as `{prefix}-{collection_name}` to avoid account-wide collisions) |
| `embedding_model_id` | str | `amazon.titan-embed-image-v1` | Bedrock embedding model ID (e.g. `amazon.titan-embed-image-v1` or `amazon.nova-2-multimodal-embeddings-v1:0`) |
| `embedding_dimension` | int | `1024` | Embedding vector dimension (Titan: 1024 only; Nova: 256, 384, 1024, or 3072) |
| `index_name` | str | — | OpenSearch index name for kNN queries |
| `query_k` | int (1-100) | `5` | Number of nearest neighbours to retrieve |
| `sqs_visibility_timeout_seconds` | int (30-43200) | — | SQS visibility timeout in seconds |
| `sqs_max_receive_count` | int (1-10) | — | Max SQS receive count before routing to DLQ |
| `ingest_lambda_timeout_seconds` | int (1-900) | — | Ingest Lambda timeout in seconds |
| `ingest_lambda_memory_mb` | int (128-10240) | — | Ingest Lambda memory in MB |
| `ingest_lambda_reserved_concurrency` | int \| null (1-1000) | `null` | Ingest Lambda reserved concurrency (null = unreserved) |

Current values in `retrieval.yaml`:

```yaml
collection_name: retrieval-images
embedding_model_id: amazon.nova-2-multimodal-embeddings-v1:0
embedding_dimension: 1024
index_name: image-vectors
query_k: 1
sqs_visibility_timeout_seconds: 960
sqs_max_receive_count: 3
ingest_lambda_timeout_seconds: 300
ingest_lambda_memory_mb: 2048
ingest_lambda_reserved_concurrency: 40
```

---

## `cicd.yaml` — CicdConfig

Located at `config/cicd/cicd.yaml`. Controls the CI/CD CodePipeline stack. One independent CodePipeline is created per config file listed in `pipeline_configs`. All pipelines share a single SecurityStack deployed locally. Select an alternate CI/CD config with `-c cicd_config_file=cicd_prod.yaml` (or `make deploy CICD_CONFIG=cicd_prod.yaml`).

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Whether to create the CiCdPipelineStack |
| `notification_email` | str \| null | `null` | Email for SNS failure notifications |
| `source_excludes` | list[str] | `.venv/`, `cdk.out/`, `.git/`, ... | Glob patterns excluded from the S3 source asset |
| `compute_type` | `SMALL` \| `MEDIUM` \| `LARGE` \| `X2_LARGE` | `X2_LARGE` | CodeBuild compute type for all stages |
| `timeout_minutes` | int (1-480) | `60` | CodeBuild timeout per stage |
| `test_commands` | dict[str, str] | see below | Maps each pipeline config filename to its pytest command string |
| `rollback` | bool | `true` | Whether to pass `--rollback` or `--no-rollback` to CDK deploy |
| `test_a2i` | bool | `false` | Enable the optional A2ISmokeTest stage |
| `shared_prefix` | str (exactly 3 chars, `[a-z][a-z0-9]{2}`) | `dev` | 3-letter prefix for shared stacks (SecurityStack). Kept short to stay within AOSS 32-char policy name limits. All pipeline configs share this single SecurityStack. |
| `pipeline_configs` | list[str] (min 1) | `["config_vrag.yaml"]` | Config file names (relative to `config/pipeline/`) to deploy |
| `input_data` | dict[str, list[str]] | `{}` | Maps each pipeline config filename to sample_input_data/ paths for the UploadInput stage |

Validated with `extra='forbid'` and `strict=True`. A `model_validator` ensures every entry in `pipeline_configs` has a corresponding key in `test_commands`.

### `test_commands` Mapping

Each pipeline config runs only the pytest markers relevant to its steps:

| Config File | Test Command (markers) |
|---|---|
| `config_vrag.yaml` | `core or cicd or retrieval or processing or model_validation or steps_vrag or steps_i2v or steps_setup or integration` |
| `config_i2v.yaml` | `core or cicd or processing or model_validation or steps_i2v or integration` |
| `config_motionart.yaml` | `core or cicd or processing or model_validation or steps_captioning or steps_flf2v or integration` |
| `config_t2a.yaml` | `core or cicd or processing or model_validation or steps_t2a or integration` |
| `config_t2i.yaml` | `core or cicd or processing or model_validation or steps_t2i or integration` |
| `config_t2v.yaml` | `core or cicd or processing or model_validation or steps_t2v or integration` |
| `config_t2v.yaml` | `core or cicd or processing or model_validation or steps_t2v or integration` |

### `input_data` Mapping

Maps each config to sample input files uploaded during the CI/CD UploadInput stage:

| Config File | Input Data Paths |
|---|---|
| `config_vrag.yaml` | `inputs_t2v.json` |
| `config_i2v.yaml` | `inputs_i2v.json`, `images/i2v/` |
| `config_motionart.yaml` | `images/captioning/` |
| `config_t2a.yaml` | `inputs_audio.json` |
| `config_t2i.yaml` | `inputs_t2v.json` |
| `config_t2v.yaml` | `inputs_t2v.json` |
| `config_t2v.yaml` | `inputs_t2v.json` |
