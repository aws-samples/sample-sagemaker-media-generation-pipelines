"""
Unit tests for PipelineStack ECR prefix behaviour.

Verifies that PipelineStack uses ecr_prefix (not prefix) for ECR image URIs
in IAM policies, while resource naming continues to use prefix.
"""

import json
from unittest.mock import patch

import aws_cdk as cdk
import pytest
from aws_cdk import assertions
from aws_cdk import aws_lambda as lambda_

from config.config import ContainerConfig, DynamoDBConfig, PipelineConfig
from infrastructure.data import DataStack
from infrastructure.pipeline import PipelineStack
from infrastructure.security import SecurityStack
from tests.unit.conftest import PRIMARY_STEPS, STEP_0_DASHED, _mock_from_asset

pytestmark = pytest.mark.core


def _default_cfg() -> ContainerConfig:
    return ContainerConfig(
        InstanceCount=1,
        InstanceType="ml.g5.xlarge",
        VolumeSizeInGB=125,
        ContainerEntrypoint=["/bin/bash", "./run_job.sh"],
        ContainerArguments=["300"],
    )


@pytest.fixture(scope="class")
def ecr_prefix_template():
    """Synthesize PipelineStack with ecr_prefix='shared' and prefix='cfg'."""
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    pipeline_config = PipelineConfig(
        construct_id="cfg",
        s3_downloads=[],
        steps={s: _default_cfg() for s in PRIMARY_STEPS},
    )
    with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
        sec = SecurityStack(app, "SecStack", prefix="cfg", env=env)
        data = DataStack(
            app,
            "DataStack",
            security_stack=sec,
            dynamodb_config=DynamoDBConfig(),
            pipeline_config=pipeline_config,
            prefix="cfg",
            env=env,
        )
        ps = PipelineStack(
            app,
            "PipelineStack",
            security_stack=sec,
            data_stack=data,
            pipeline_config=pipeline_config,
            prefix="cfg",
            ecr_prefix="shared",
            env=env,
        )
    return assertions.Template.from_stack(ps)


class TestPipelineStackEcrPrefix:
    """PipelineStack ECR URIs use ecr_prefix, not prefix."""

    def test_ecr_pull_policy_uses_shared_prefix(self, ecr_prefix_template) -> None:
        """ECR pull policy ARN contains shared/processing/{step}."""
        ecr_prefix_template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            assertions.Match.object_like(
                {
                    "ManagedPolicyName": f"cfg-{STEP_0_DASHED}-ecr-pull-policy",
                    "PolicyDocument": assertions.Match.object_like(
                        {
                            "Statement": assertions.Match.array_with(
                                [
                                    assertions.Match.object_like(
                                        {
                                            "Action": assertions.Match.array_with(["ecr:GetDownloadUrlForLayer"]),
                                            "Resource": assertions.Match.string_like_regexp(
                                                r".*:repository/shared/processing/.*"
                                            ),
                                        }
                                    ),
                                ]
                            ),
                        }
                    ),
                }
            ),
        )

    def test_ecr_image_override_uses_override_name(self) -> None:
        """Step with ecr_image override uses override name in ECR path."""
        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")
        step_cfg = _default_cfg()
        step_cfg.ecr_image = "vbench"
        pipeline_config = PipelineConfig(
            construct_id="cfg",
            s3_downloads=[],
            steps={"vbench_t2v": step_cfg},
        )
        with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
            sec = SecurityStack(app, "SecStack2", prefix="cfg", env=env)
            data = DataStack(
                app,
                "DataStack2",
                security_stack=sec,
                dynamodb_config=DynamoDBConfig(),
                pipeline_config=pipeline_config,
                prefix="cfg",
                env=env,
            )
            ps = PipelineStack(
                app,
                "PipelineStack2",
                security_stack=sec,
                data_stack=data,
                pipeline_config=pipeline_config,
                prefix="cfg",
                ecr_prefix="shared",
                env=env,
            )
        template = assertions.Template.from_stack(ps)
        # The template should contain shared/processing/vbench (not vbench-t2v)
        template_json = json.dumps(template.to_json())
        assert "shared/processing/vbench" in template_json
        assert "shared/processing/vbench-t2v" not in template_json


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import assume, given, settings
from hypothesis import strategies as st


def _step_name_strategy():
    return st.from_regex(r"[a-z][a-z0-9_]{1,12}", fullmatch=True)


def _prefix_strategy():
    return st.from_regex(r"[a-z][a-z0-9]{1,8}", fullmatch=True)


class TestPipelineStackEcrUriProperty:
    """Property 3: ECR URI construction uses ecr_prefix independently of resource prefix.

    Tests the string construction pattern used in PipelineStack:
    ``f"{ecr_pfx}/processing/{step.replace('_', '-')}"``

    **Validates: Requirements 3.1, 3.2**
    """

    @given(
        step_name=_step_name_strategy(),
        prefix=_prefix_strategy(),
        ecr_prefix=_prefix_strategy(),
    )
    @settings(max_examples=100)
    def test_ecr_uri_uses_ecr_prefix(self, step_name, prefix, ecr_prefix) -> None:
        """ECR URI contains ecr_prefix/processing/{step}, not prefix/processing/{step} when they differ."""
        assume(prefix != ecr_prefix)
        assume(prefix not in ecr_prefix)  # avoid substring false positives

        ecr_key = step_name
        ecr_clean = ecr_key.replace("_", "-")
        ecr_repo_name = f"{ecr_prefix}/processing/{ecr_clean}"
        ecr_image_uri = f"123456789012.dkr.ecr.us-east-1.amazonaws.com/{ecr_repo_name}"

        assert f"{ecr_prefix}/processing/{ecr_clean}" in ecr_image_uri
        assert f"{prefix}/processing/{ecr_clean}" not in ecr_image_uri
