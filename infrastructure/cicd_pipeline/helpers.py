"""Shared helpers for CI/CD pipeline construction."""

from aws_cdk import Duration, RemovalPolicy
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_logs as logs
from constructs import Construct

from infrastructure.security import SecurityStack


def create_codebuild_project(
    scope: Construct,
    prefix: str,
    name: str,
    build_spec: dict | codebuild.BuildSpec,
    security_stack: SecurityStack,
    compute_type: codebuild.ComputeType,
    timeout_minutes: int,
    environment_variables: dict,
    privileged: bool = False,
) -> codebuild.PipelineProject:
    """Create a CodeBuild PipelineProject with VPC, logging, and KMS."""
    log_group = logs.LogGroup(
        scope,
        f"{prefix}-{name}-LogGroup",
        log_group_name=f"/aws/codebuild/{prefix}-{name}",
        retention=logs.RetentionDays.ONE_WEEK,
        removal_policy=RemovalPolicy.DESTROY,
        encryption_key=security_stack.kms_key,
    )
    bs = build_spec if isinstance(build_spec, codebuild.BuildSpec) else codebuild.BuildSpec.from_object(build_spec)
    return codebuild.PipelineProject(
        scope,
        f"{prefix}-{name}-Project",
        project_name=f"{prefix}-{name}",
        environment=codebuild.BuildEnvironment(
            build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
            compute_type=compute_type,
            environment_variables=environment_variables,
            privileged=privileged,
        ),
        build_spec=bs,
        timeout=Duration.minutes(timeout_minutes),
        vpc=security_stack.vpc,
        subnet_selection=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
        security_groups=[security_stack.security_group],
        logging=codebuild.LoggingOptions(
            cloud_watch=codebuild.CloudWatchLoggingOptions(log_group=log_group, enabled=True),
        ),
    )


def collect_unique_containers(pipeline_configs: list[str], shared_prefix: str) -> list[dict]:
    """Collect deduplicated containers across all pipeline configs.

    Returns a list of dicts with container name and ECR prefix.
    Steps with ``ecr_image`` override use that name (e.g. vbench_t2v → vbench).
    Setup jobs from each config's ``setup`` field are also included.
    Every entry's ``prefix`` equals ``shared_prefix``.
    """
    from config.config import get_pipeline_config

    seen: set[str] = set()
    result: list[dict] = []
    for cfg_file in pipeline_configs:
        cfg = get_pipeline_config(cfg_file)
        for step_name, step_cfg in cfg.steps.items():
            container = step_cfg.ecr_image or step_name
            if container not in seen:
                seen.add(container)
                result.append({"container": container, "prefix": shared_prefix})
        for setup_name in cfg.setup:
            if setup_name not in seen:
                seen.add(setup_name)
                result.append({"container": setup_name, "prefix": shared_prefix})
    return result


def collect_unique_step_names(pipeline_configs: list[str], shared_prefix: str) -> list[str]:
    """Return deduplicated container name strings across all pipeline configs."""
    return [entry["container"] for entry in collect_unique_containers(pipeline_configs, shared_prefix)]
