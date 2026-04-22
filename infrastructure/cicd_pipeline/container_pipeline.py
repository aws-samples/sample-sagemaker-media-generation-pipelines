"""Dedicated container build pipeline.

Creates a single CodePipeline that builds ALL container images used
across all pipeline configs. Each container gets its own CodeBuild
action in the ContainerBuild stage, running in parallel.

Stages: Source → QualityGate → ContainerBuild

Only containers referenced by at least one config are included.
Containers with ``ecr_image`` overrides (e.g. vbench_t2v → vbench)
are deduplicated so the shared image is built once.
"""

from aws_cdk import Stack
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_codepipeline as codepipeline
from aws_cdk import aws_codepipeline_actions as codepipeline_actions
from aws_cdk import aws_s3_assets as s3_assets
from constructs import Construct
from loguru import logger

from config.config import CicdConfig
from infrastructure.cicd_pipeline import buildspecs, policies
from infrastructure.cicd_pipeline.helpers import collect_unique_containers, create_codebuild_project
from infrastructure.security import SecurityStack
from project_constructs.codepipeline import CodePipelineTemplate


def create_container_pipeline(
    scope: Construct,
    prefix: str,
    cicd_config: CicdConfig,
    security_stack: SecurityStack,
    source_asset: s3_assets.Asset,
    artifact_bucket,
    env: dict,
) -> CodePipelineTemplate:
    """Create the dedicated container build pipeline."""
    tag = f"{prefix}-containers-cicd"
    account = Stack.of(scope).account
    region = Stack.of(scope).region

    src_out = codepipeline.Artifact("Source-containers")

    # Source
    source_stage = codepipeline.StageProps(
        stage_name="Source",
        actions=[
            codepipeline_actions.S3SourceAction(
                action_name="S3Source",
                bucket=source_asset.bucket,
                bucket_key=source_asset.s3_object_key,
                output=src_out,
            )
        ],
    )

    # QualityGate — parallel: lint+synth and container-relevant unit tests
    test_command = "uv run pytest tests/unit/ -x --no-header -q -n auto -m 'processing or model_validation'"

    lint_proj = create_codebuild_project(
        scope,
        tag,
        "lint-synth",
        buildspecs.lint_and_synth("config_vrag.yaml"),
        security_stack,
        codebuild.ComputeType.SMALL,
        cicd_config.timeout_minutes,
        env,
    )
    source_asset.grant_read(lint_proj.role)  # type: ignore[arg-type]

    test_proj = create_codebuild_project(
        scope,
        tag,
        "test",
        buildspecs.unit_test(test_command),
        security_stack,
        codebuild.ComputeType.X2_LARGE,
        cicd_config.timeout_minutes,
        env,
    )
    source_asset.grant_read(test_proj.role)  # type: ignore[arg-type]

    qa_stage = codepipeline.StageProps(
        stage_name="QualityGate",
        actions=[
            codepipeline_actions.CodeBuildAction(
                action_name="LintAndSynth",
                project=lint_proj,
                input=src_out,
                run_order=1,
            ),
            codepipeline_actions.CodeBuildAction(
                action_name="Test",
                project=test_proj,
                input=src_out,
                run_order=1,
            ),
        ],
    )

    # ContainerBuild — one action per unique container, all parallel
    containers = collect_unique_containers(cicd_config.pipeline_configs, cicd_config.shared_prefix)
    logger.info("Container pipeline: {} unique containers: {}", len(containers), [c["container"] for c in containers])

    build_actions: list[codepipeline_actions.CodeBuildAction] = []
    for entry in containers:
        container = entry["container"]
        ecr_prefix = entry["prefix"]
        container_clean = container.replace("_", "-")
        ecr_repo_name = f"{ecr_prefix}/processing/{container_clean}"
        ecr_repo_arn = f"arn:aws:ecr:{region}:{account}:repository/{ecr_repo_name}"

        step_proj = create_codebuild_project(
            scope,
            tag,
            f"build-{container_clean}",
            buildspecs.container_step_build(),
            security_stack,
            codebuild.ComputeType.LARGE,
            cicd_config.timeout_minutes,
            {
                **env,
                "STEP_NAME": codebuild.BuildEnvironmentVariable(value=container),
                "ECR_REPO_URI": codebuild.BuildEnvironmentVariable(
                    value=f"{account}.dkr.ecr.{region}.amazonaws.com/{ecr_repo_name}"
                ),
            },
            privileged=True,
        )
        step_proj.role.add_managed_policy(  # type: ignore[union-attr]
            policies.container_step_build_policy(scope, f"{tag}-{container_clean}", ecr_repo_arn)
        )
        build_actions.append(
            codepipeline_actions.CodeBuildAction(
                action_name=f"Build-{container_clean}",
                project=step_proj,
                input=src_out,
                run_order=1,
            )
        )

    build_stage = codepipeline.StageProps(
        stage_name="ContainerBuild",
        actions=build_actions,
    )

    # Assemble pipeline
    pipeline_name = f"{prefix}-container-build-pipeline"
    pipeline = CodePipelineTemplate(
        scope,
        f"{tag}-Pipeline",
        pipeline_name=pipeline_name,
        artifact_bucket=artifact_bucket.bucket,
        kms_key=security_stack.kms_key,
        stages=[source_stage, qa_stage, build_stage],
    )
    logger.info("Created container build pipeline: {} ({} containers)", pipeline_name, len(containers))
    return pipeline
