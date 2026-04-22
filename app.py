#!/usr/bin/env python3
"""
ComfyUI Application

This module serves as the main entry point for the ComfyUI video upscaling CDK application.
It orchestrates the deployment of three main stacks:
1. SecurityStack - Provides VPC, KMS keys, and security configurations
2. DataStack - Creates S3 buckets for input/output storage
3. ComfyUiStack - Deploys SageMaker processing jobs and Lambda triggers

The application uses environment variables for AWS account configuration and
supports custom configuration files for processing job parameters.
"""

import json
import os
import shutil
import tomllib
from pathlib import Path

import aws_cdk as cdk
from aws_cdk import Aspects, Tag
from cdk_nag import AwsSolutionsChecks
from dotenv import load_dotenv
from loguru import logger

from config.config import get_cicd_config, get_pipeline_config, get_retrieval_config
from infrastructure.a2i_stack import A2IStack
from infrastructure.cicd_pipeline.codebuild_stack import CodeBuildStack
from infrastructure.cicd_pipeline.container_stack import ContainerPipelineStack
from infrastructure.cicd_pipeline.helpers import collect_unique_step_names
from infrastructure.cicd_pipeline.stack import CiCdPipelineStack
from infrastructure.data import DataStack
from infrastructure.pipeline import PipelineStack
from infrastructure.security import SecurityStack

# Load environment variables from .env file
load_dotenv()

# Read project name from pyproject.toml
with open(Path(__file__).parent / "pyproject.toml", "rb") as f:
    PROJECT_NAME = tomllib.load(f)["project"]["name"]

# Create CDK environment from environment variables
env = cdk.Environment(account=os.environ["AWS_ACCOUNT_ID"], region=os.environ["REGION"])

# Initialize CDK application
app = cdk.App()

# Get configuration file path from context or use default
config_fp = app.node.try_get_context("config_file") or "config_vrag.yaml"
pipeline_config = get_pipeline_config(config_fp)
logger.info("Loaded pipeline config from: {}", config_fp)

# Write per-config downloads manifest into the model_download container
# directory so it gets baked into the Docker image at build time.  Each
# pipeline config gets its own file ({construct_id}_downloads.json) to
# avoid clobbering when the CI/CD stack synths multiple configs.
downloads_path = Path(f"processing_job/model_download/{pipeline_config.construct_id}_downloads.json")
downloads_path.write_text(json.dumps([dl.model_dump() for dl in pipeline_config.s3_downloads], indent=2) + "\n")
logger.info("Wrote {} downloads to {}", len(pipeline_config.s3_downloads), downloads_path)

# Copy schema/ into processing_job/ so container builds can access it.
# The S3 asset for container CodeBuild projects is rooted at processing_job/,
# so schema/ needs to be inside it at synth time.
_schema_dest = Path("processing_job/schema")
if _schema_dest.exists():
    shutil.rmtree(_schema_dest)
shutil.copytree("schema", str(_schema_dest))
logger.info("Copied schema/ into processing_job/ for container builds")

# Extract prefix and step names from config
cicd_fp = app.node.try_get_context("cicd_config_file") or "cicd.yaml"
cicd_config = get_cicd_config(cicd_fp)

# Write downloads manifests for ALL pipeline configs in the CI/CD config,
# not just the one being synth'd.  The source artifact bundles the entire
# processing_job/ directory, so every pipeline's model download step needs
# its manifest present.
for other_cfg_file in cicd_config.pipeline_configs:
    if other_cfg_file == config_fp:
        continue  # already written above
    try:
        other_config = get_pipeline_config(other_cfg_file)
        other_path = Path(f"processing_job/model_download/{other_config.construct_id}_downloads.json")
        other_path.write_text(json.dumps([dl.model_dump() for dl in other_config.s3_downloads], indent=2) + "\n")
        logger.info("Wrote {} downloads to {}", len(other_config.s3_downloads), other_path)
    except Exception as exc:
        logger.warning("Could not write downloads for {}: {}", other_cfg_file, exc)
shared_prefix = cicd_config.shared_prefix
prefix = f"{shared_prefix}{pipeline_config.construct_id}"
step_names = list(pipeline_config.steps.keys())
logger.info(
    "construct_id: {}, shared_prefix: {}, prefix: {}, steps: {}",
    pipeline_config.construct_id,
    shared_prefix,
    prefix,
    step_names,
)

# Deduplicate ECR image names across ALL pipeline configs (steps + setup jobs)
all_step_names = collect_unique_step_names(cicd_config.pipeline_configs, shared_prefix)

# Create security stack with VPC, KMS, and security groups
security_stack = SecurityStack(app, f"{shared_prefix}-SecurityStack", prefix=shared_prefix, env=env)
logger.info("Instantiated SecurityStack")

# Lambda function directory names for CodeBuild ECR builds.
# Auto-discovered from lambdas/ subdirectories that contain a Dockerfile.
lambda_names = sorted(d.name for d in Path("lambdas").iterdir() if d.is_dir() and (d / "Dockerfile").exists())

# Create single shared CodeBuild stack with ECR repos and build projects
codebuild_stack = CodeBuildStack(
    app,
    f"{shared_prefix}-CodeBuildStack",
    security_stack=security_stack,
    step_names=all_step_names,
    lambda_names=lambda_names,
    prefix=shared_prefix,
    env=env,
)
codebuild_stack.add_dependency(security_stack)
logger.info("Instantiated CodeBuildStack")

# Load retrieval config if referenced by pipeline config
retrieval_config = None
if pipeline_config.retrieval:
    retrieval_config = get_retrieval_config(pipeline_config.retrieval)
    logger.info("Loaded retrieval config: {}", pipeline_config.retrieval)

# Create data stack with S3 buckets for input/output (+ optional retrieval construct)
data_stack = DataStack(
    app,
    f"{prefix}-DataStack",
    security_stack=security_stack,
    dynamodb_config=pipeline_config.dynamodb,
    pipeline_config=pipeline_config,
    prefix=prefix,
    retrieval_config=retrieval_config,
    ecr_prefix=shared_prefix,
    env=env,
)
# data_stack.add_dependency(security_stack)
logger.info("Instantiated DataStack")

# Create Pipeline stack with config-driven processing steps (includes model download job)
# Determine if A2I stack is needed (both a2i config and lambda_steps referencing it)
referenced_a2i_names = {ls_cfg.a2i_name for ls_cfg in pipeline_config.lambda_steps.values() if ls_cfg.a2i_name}
active_a2i_names = referenced_a2i_names & set(pipeline_config.a2i.keys())

a2i_stack = None
submit_lambdas = {}
a2i_constructs = {}
if active_a2i_names:
    a2i_stack = A2IStack(
        app,
        f"{prefix}-A2IStack",
        security_stack=security_stack,
        data_stack=data_stack,
        pipeline_config=pipeline_config,
        prefix=prefix,
        env=env,
    )
    a2i_stack.add_dependency(security_stack)
    a2i_stack.add_dependency(data_stack)
    submit_lambdas = a2i_stack.submit_lambdas
    a2i_constructs = a2i_stack.a2i_constructs
    logger.info("Instantiated A2IStack")

pipeline_stack = PipelineStack(
    app,
    f"{prefix}-PipelineStack",
    security_stack=security_stack,
    data_stack=data_stack,
    pipeline_config=pipeline_config,
    submit_lambdas=submit_lambdas,
    a2i_constructs=a2i_constructs,
    retrieval_construct=data_stack.retrieval,
    prefix=prefix,
    ecr_prefix=shared_prefix,
    env=env,
)
pipeline_stack.add_dependency(security_stack)
pipeline_stack.add_dependency(data_stack)
if a2i_stack:
    pipeline_stack.add_dependency(a2i_stack)
logger.info("Instantiated PipelineStack")

# Conditionally create CI/CD pipeline stack
if cicd_config.enabled:
    cicd_stack = CiCdPipelineStack(
        app,
        f"{shared_prefix}-CiCdPipelineStack",
        security_stack=security_stack,
        cicd_config=cicd_config,
        prefix=shared_prefix,
        env=env,
    )
    cicd_stack.add_dependency(security_stack)
    logger.info("Instantiated CiCdPipelineStack")

    container_stack = ContainerPipelineStack(
        app,
        f"{shared_prefix}-ContainerPipelineStack",
        security_stack=security_stack,
        cicd_config=cicd_config,
        artifact_bucket=cicd_stack.artifact_bucket,
        prefix=shared_prefix,
        env=env,
    )
    container_stack.add_dependency(security_stack)
    container_stack.add_dependency(cicd_stack)
    logger.info("Instantiated ContainerPipelineStack")

# Add CDK Nag for security and best practice compliance checking
Aspects.of(app).add(AwsSolutionsChecks())
Aspects.of(app).add(
    Tag(
        key="application",
        value=PROJECT_NAME,
    ),
)
# Synthesize the CDK application
app.synth()
