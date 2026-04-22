"""Buildspec factories for each CI/CD CodeBuild project.

Each function returns a buildspec dict ready for
``codebuild.BuildSpec.from_object()``.
"""

UV_INSTALL = (
    "curl -LsSf https://astral.sh/uv/install.sh | sh && "
    'export PATH="$HOME/.local/bin:$PATH" && '
    "npm install -g aws-cdk && "
    "uv venv && uv sync --extra dev"
)


def lint_and_synth(cfg_file: str) -> dict:
    """Buildspec for the LintAndSynth action in QualityGate stage."""
    return {
        "version": "0.2",
        "phases": {
            "install": {"commands": [UV_INSTALL]},
            "build": {
                "commands": [
                    "set -e",
                    "git init -q && git add -A && git commit -q -m init --allow-empty",
                    "uv run pre-commit run --all-files",
                    f"uv run cdk synth -c config_file={cfg_file}",
                ]
            },
        },
    }


def unit_test(test_command: str) -> dict:
    """Buildspec for the Test action in QualityGate stage."""
    return {
        "version": "0.2",
        "phases": {
            "install": {"commands": [UV_INSTALL]},
            "build": {"commands": ["set -e", test_command]},
        },
    }


def cdk_synth(cfg_file: str) -> dict:
    """Buildspec for the Synth action in QualityGate stage."""
    return {
        "version": "0.2",
        "phases": {
            "install": {"commands": [UV_INSTALL]},
            "build": {"commands": ["set -e", f"uv run cdk synth -c config_file={cfg_file}"]},
        },
    }


def deploy(deploy_script: str) -> dict:
    """Buildspec for the Deploy action in Deploy stage."""
    return {
        "version": "0.2",
        "phases": {
            "install": {"commands": [UV_INSTALL]},
            "build": {"commands": [deploy_script]},
        },
    }


def model_download() -> dict:
    """Buildspec for the ModelDownload action."""
    return {
        "version": "0.2",
        "phases": {
            "install": {
                "runtime-versions": {"python": "3.13"},
                "commands": ["pip install boto3 loguru"],
            },
            "build": {
                "commands": [
                    "set -e",
                    "echo 'Running model download script...'",
                    "python processing_job/model_download/main.py",
                ],
            },
        },
    }


def upload_input(sync_commands: list[str]) -> dict:
    """Buildspec for the UploadInput action."""
    return {
        "version": "0.2",
        "phases": {
            "build": {"commands": sync_commands},
        },
    }


def trigger_pipeline() -> dict:
    """Buildspec for the TriggerPipeline action."""
    return {
        "version": "0.2",
        "phases": {
            "build": {
                "commands": [
                    "set -e",
                    "echo 'Looking up pipeline trigger Lambda from stack outputs...'",
                    (
                        "LAMBDA_NAME=$(aws cloudformation describe-stacks "
                        "--stack-name ${CONFIG_PREFIX}-PipelineStack "
                        "--query \"Stacks[0].Outputs[?contains(OutputKey,'TriggerLambdaName') "
                        "&& !contains(OutputKey,'ModelDownload')].OutputValue\" "
                        "--output text)"
                    ),
                    'if [ -z "$LAMBDA_NAME" ]; then echo "ERROR: Could not find TriggerLambdaName output"; exit 1; fi',
                    'echo "Lambda: $LAMBDA_NAME"',
                    (
                        "aws lambda invoke --function-name $LAMBDA_NAME "
                        "--payload '{}' --cli-binary-format raw-in-base64-out "
                        "/tmp/response.json"
                    ),
                    "cat /tmp/response.json",
                    "echo 'Pipeline triggered successfully'",
                ]
            },
        },
    }


def container_step_build() -> dict:
    """Buildspec for building a single container from the project root.

    Wraps processing_job/buildspec.yml by cd'ing into processing_job/ first.
    Expects STEP_NAME and ECR_REPO_URI environment variables.
    """
    return {
        "version": "0.2",
        "env": {"variables": {"SKIP_BUILD": "false"}},
        "phases": {
            "pre_build": {
                "commands": [
                    "cd processing_job",
                    "echo Logging in to Amazon ECR...",
                    "aws ecr get-login-password --region $AWS_DEFAULT_REGION "
                    "| docker login --username AWS --password-stdin $ECR_REPO_URI",
                    "echo Logging in to SageMaker DLC ECR for base images...",
                    "aws ecr get-login-password --region $AWS_DEFAULT_REGION "
                    "| docker login --username AWS --password-stdin "
                    "763104351884.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com",
                    'echo "Computing content hash for $STEP_NAME..."',
                    "cp -r common/ $STEP_NAME/common/",
                    "cp -r schema/ $STEP_NAME/schema/",
                    "export CONTENT_HASH=$(find $STEP_NAME -type f | sort "
                    "| xargs sha256sum | sha256sum | cut -d' ' -f1 | head -c 12)",
                    'echo "Content hash = $CONTENT_HASH"',
                    "export ECR_REPO_NAME=$(echo $ECR_REPO_URI | sed 's|.*amazonaws.com/||')",
                    "if aws ecr describe-images --repository-name $ECR_REPO_NAME "
                    "--image-ids imageTag=$CONTENT_HASH 2>/dev/null; then "
                    'echo "Cached — skipping build"; '
                    "MANIFEST=$(aws ecr batch-get-image --repository-name $ECR_REPO_NAME "
                    "--image-ids imageTag=$CONTENT_HASH "
                    "--query 'images[0].imageManifest' --output text); "
                    "aws ecr put-image --repository-name $ECR_REPO_NAME "
                    '--image-tag latest --image-manifest "$MANIFEST" 2>/dev/null || true; '
                    "export SKIP_BUILD=true; "
                    'else echo "No cache — will build"; fi',
                ]
            },
            "build": {
                "commands": [
                    'if [ "$SKIP_BUILD" = "true" ]; then echo "Skipping (cached)"; else '
                    "docker build -t $ECR_REPO_URI:latest -f $STEP_NAME/Dockerfile $STEP_NAME && "
                    "docker tag $ECR_REPO_URI:latest $ECR_REPO_URI:$CONTENT_HASH && "
                    "BUILD_TAG=$(echo $CODEBUILD_BUILD_ID | sed 's/.*://') && "
                    "docker tag $ECR_REPO_URI:latest $ECR_REPO_URI:$BUILD_TAG; fi",
                ]
            },
            "post_build": {
                "commands": [
                    'if [ "$SKIP_BUILD" = "true" ]; then echo "Skipping push"; else '
                    "docker push $ECR_REPO_URI:latest && "
                    "docker push $ECR_REPO_URI:$CONTENT_HASH && "
                    "docker push $ECR_REPO_URI:$BUILD_TAG; fi",
                ]
            },
        },
    }
