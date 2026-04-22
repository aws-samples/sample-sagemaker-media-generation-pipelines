> **Navigation:** [← Main README](../README.md) | [Extending Guide — Container Integration](../docs/EXTENDING.md#container-integration) | [Extending Guide — Creating a New Container](../docs/EXTENDING.md#creating-a-new-container)

# processing_job/

SageMaker Processing Job code. Each subdirectory is a pipeline step with its own `Dockerfile` and `main.py`.

## Shared Code

- `common/` — Shared modules copied into t2v/i2v containers at build time:
  - `wan22.py` — Wan 2.2 ComfyUI workflows (run_i2v, run_t2v)
  - `is_queue_empty.py` — ComfyUI queue polling helper (used by all ComfyUI-based steps: t2v, i2v, t2i, t2a, flf2v)
  - `log_outputs.py` — Scans output dir and writes results to DynamoDB
  - `dynamodb.py` — DynamoDB operations helper
  - `utils.py` — File scanning and video metadata extraction

## Synth-Time Copies

At CDK synth time, `app.py` copies the top-level [`schema/`](../schema/README.md) directory into `processing_job/schema/` so container builds can access the DynamoDB column registry and model identifiers. This copy is gitignored and recreated on every synth.

## Docker Build Flow

Container images are built by the dedicated container build pipeline. See [infrastructure/cicd_pipeline/README.md](../infrastructure/cicd_pipeline/README.md#container-build-pipeline) for details on content-hash caching and the build process.
