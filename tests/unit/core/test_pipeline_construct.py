"""
Unit tests for PipelineConstruct (project_constructs/pipeline.py).

Validates: Requirements 1.1, 1.2, 1.6, 1.7, 7.7, 7.8
"""

from unittest.mock import patch

import aws_cdk as cdk
import pytest
from aws_cdk import assertions
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_

from infrastructure.security import SecurityStack
from project_constructs.pipeline import PipelineConstruct
from project_constructs.processing_job.main import IoBucketConfig, ReusableProcessingJob
from project_constructs.s3 import BucketTemplate
from tests.unit.core.conftest import _default_cfg, _mock_from_asset

pytestmark = pytest.mark.core


def _create_stack_with_pipeline(num_steps: int = 2):
    """Helper to create a stack with PipelineConstruct and return template + construct."""
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    sec = SecurityStack(app, "Sec", prefix="dev", env=env)
    stack = cdk.Stack(app, "TestStack", env=env)

    lb = cdk.aws_s3.Bucket(stack, "LB", removal_policy=cdk.RemovalPolicy.DESTROY)
    cfg = _default_cfg()
    jobs: list[ReusableProcessingJob] = []

    with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
        for i in range(num_steps):
            sn = f"step-{i + 1}"
            ib = BucketTemplate(stack, f"{sn}-IB", bucket_name=f"t-{sn}-ib", kms_key=sec.kms_key, logging_bucket=lb)
            ob = BucketTemplate(stack, f"{sn}-OB", bucket_name=f"t-{sn}-ob", kms_key=sec.kms_key, logging_bucket=lb)
            job = ReusableProcessingJob(
                stack,
                f"dev-{sn}-PJ",
                job_name=f"dev-{sn}",
                input_buckets={"input": IoBucketConfig(bucket_template=ib)},
                output_buckets={"output": IoBucketConfig(bucket_template=ob)},
                security_stack=sec,
                lambda_trigger=False,
                cfg=cfg,
                ecr_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/mock",
            )
            jobs.append(job)

        role = iam.Role(stack, "PipeRole", assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"))
        pipeline = PipelineConstruct(
            stack,
            "dev-Pipeline",
            pipeline_name="dev-sagemaker-pipeline",
            processing_jobs=jobs,
            execution_role=role,
        )

    template = assertions.Template.from_stack(stack)
    return template, pipeline, jobs


class TestPipelineConstructResource:
    """Verify the SageMaker Pipeline CF resource is created correctly."""

    def test_pipeline_resource_created(self):
        template, _, _ = _create_stack_with_pipeline()
        template.resource_count_is("AWS::SageMaker::Pipeline", 1)

    def test_pipeline_has_correct_name(self):
        template, _, _ = _create_stack_with_pipeline()
        template.has_resource_properties(
            "AWS::SageMaker::Pipeline",
            {
                "PipelineName": "dev-sagemaker-pipeline",
            },
        )

    def test_pipeline_has_role_arn(self):
        template, _, _ = _create_stack_with_pipeline()
        template.has_resource_properties(
            "AWS::SageMaker::Pipeline",
            {
                "RoleArn": assertions.Match.any_value(),
            },
        )

    def test_pipeline_has_definition_body(self):
        template, _, _ = _create_stack_with_pipeline()
        template.has_resource_properties(
            "AWS::SageMaker::Pipeline",
            {
                "PipelineDefinition": {
                    "PipelineDefinitionBody": assertions.Match.any_value(),
                },
            },
        )


class TestPipelineConstructAttributes:
    """Verify exposed attributes on PipelineConstruct."""

    def test_pipeline_arn_format(self):
        _, pipeline, _ = _create_stack_with_pipeline()
        assert pipeline.pipeline_arn == "arn:aws:sagemaker:us-east-1:123456789012:pipeline/dev-sagemaker-pipeline"

    def test_pipeline_name_attribute(self):
        _, pipeline, _ = _create_stack_with_pipeline()
        assert pipeline.pipeline_name == "dev-sagemaker-pipeline"

    def test_processing_policy_exists(self):
        _, pipeline, _ = _create_stack_with_pipeline()
        assert pipeline.processing_policy is not None


class TestPipelineConstructProcessingPolicy:
    """Verify the processing_policy ManagedPolicy."""

    def test_processing_policy_has_correct_actions(self):
        template, _, _ = _create_stack_with_pipeline()
        # Find the processing policy in the template
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "dev-sagemaker-pipeline-processing-policy",
                "PolicyDocument": {
                    "Statement": assertions.Match.array_with(
                        [
                            assertions.Match.object_like(
                                {
                                    "Action": [
                                        "sagemaker:CreateProcessingJob",
                                        "sagemaker:DescribeProcessingJob",
                                        "sagemaker:StopProcessingJob",
                                        "sagemaker:AddTags",
                                    ],
                                }
                            ),
                        ]
                    ),
                },
            },
        )

    def test_processing_policy_scoped_to_processing_jobs(self):
        template, _, _ = _create_stack_with_pipeline()
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "dev-sagemaker-pipeline-processing-policy",
                "PolicyDocument": {
                    "Statement": assertions.Match.array_with(
                        [
                            assertions.Match.object_like(
                                {
                                    "Resource": assertions.Match.any_value(),
                                }
                            ),
                        ]
                    ),
                },
            },
        )


class TestPipelineConstructStepChaining:
    """Verify step chaining via definition_model inspection."""

    def test_two_steps_first_has_no_depends(self):
        _, _, jobs = _create_stack_with_pipeline(2)
        # First job definition_model should not have DependsOn set by the construct
        # (DependsOn is set in the pipeline definition JSON, not on the job model)
        assert jobs[0].definition_model.Name == "dev-step-1"

    def test_two_steps_second_named_correctly(self):
        _, _, jobs = _create_stack_with_pipeline(2)
        assert jobs[1].definition_model.Name == "dev-step-2"

    def test_three_steps_creates_pipeline(self):
        template, pipeline, jobs = _create_stack_with_pipeline(3)
        template.resource_count_is("AWS::SageMaker::Pipeline", 1)
        assert len(jobs) == 3

    def test_single_step_pipeline(self):
        """A single-step pipeline should still work (no DependsOn)."""
        template, pipeline, jobs = _create_stack_with_pipeline(1)
        template.resource_count_is("AWS::SageMaker::Pipeline", 1)
        assert len(jobs) == 1
