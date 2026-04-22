# codebuild_trigger

CloudFormation custom resource handler that starts CodeBuild projects during CDK deploys.

## How It Works

Invoked by CDK's `cr.Provider` framework on stack Create/Update. Reads `PROJECT_NAMES` (comma-separated) from environment variables and calls `start_build` on each project. Fire-and-forget — does not wait for builds to complete.

On Delete events, it no-ops and returns the existing physical resource ID.

## Packaging

Zip-based Lambda bundled directly by CDK (no Docker/CodeBuild). Uses `aws-lambda-powertools` for structured logging.

## Environment Variables

- `PROJECT_NAMES` — Comma-separated list of CodeBuild project names to trigger
