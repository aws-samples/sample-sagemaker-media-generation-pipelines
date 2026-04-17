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

import os
from dotenv import load_dotenv

import aws_cdk as cdk
from aws_cdk import Aspects, Tag
from cdk_nag import AwsSolutionsChecks

from config.config import get_configs
from infrastructure.data import DataStack
from infrastructure.security import SecurityStack
from infrastructure.comfyui import ComfyUiSmStack

# Load environment variables from .env file
load_dotenv()

# Create CDK environment from environment variables
env = cdk.Environment(account=os.environ["AWS_ACCOUNT_ID"], region=os.environ["REGION"])

# Initialize CDK application
app = cdk.App()

# Get configuration file path from context or use default
config_fp = app.node.try_get_context("config_file") or "config.yaml"
cfg = get_configs(config_fp)

# Create security stack with VPC, KMS, and security groups
security_stack = SecurityStack(app, "SecurityStack", env=env)

# Create data stack with S3 buckets for input/output
data_stack = DataStack(app, "DataStack", security_stack=security_stack, env=env)
data_stack.add_dependency(security_stack)

# Create main ComfyUI stack with processing jobs and Lambda triggers
comfyui_sm_stack = ComfyUiSmStack(app, "ComfyUiSmStack", security_stack=security_stack, data_stack=data_stack, env=env, cfg=cfg)
comfyui_sm_stack.add_dependency(security_stack)

# Add CDK Nag for security and best practice compliance checking
Aspects.of(app).add(AwsSolutionsChecks())
Aspects.of(app).add(
    Tag(
        key="application",
        value="ComfyUI-blog",
    )
)
# Synthesize the CDK application
app.synth()
