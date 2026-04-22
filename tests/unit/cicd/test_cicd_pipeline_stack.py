"""
Unit tests for the CiCdPipelineStack.

Tests verify per-config pipeline creation, security configuration,
environment variable injection, and optional stage toggling based
on CicdConfig flags.

The ``s3_assets.Asset`` is mocked to avoid copying the entire project
directory (including heavy .venv/) into a temp staging folder during
``cdk synth``, which would consume multiple GB of disk space per test.
"""

import pytest
from aws_cdk import assertions

from config.config import CicdConfig
from infrastructure.cicd_pipeline.stack import CiCdPipelineStack
from tests.unit.cicd.conftest import _create_cicd_pipeline_stack

pytestmark = pytest.mark.cicd


@pytest.fixture(scope="module")
def single_config_template():
    """Synthesize the default single-config CiCdPipelineStack once for the module."""
    _, template = _create_cicd_pipeline_stack()
    return template


class TestCiCdPipelineStackResources:
    """Tests for per-config pipeline resource creation."""

    def test_creates_one_pipeline_per_config(self, single_config_template):
        """Single config → 1 config pipeline (container pipeline in its own stack)."""
        single_config_template.resource_count_is("AWS::CodePipeline::Pipeline", 1)

    def test_creates_multiple_pipelines(self):
        """Two configs → 2 config pipelines."""
        config = CicdConfig(
            pipeline_configs=["config_vrag.yaml", "config_t2a.yaml"],
            test_commands={
                "config_vrag.yaml": "uv run pytest tests/unit/ -x --no-header -q",
                "config_t2a.yaml": "uv run pytest tests/unit/ -x --no-header -q",
            },
        )
        _, template = _create_cicd_pipeline_stack(config)
        template.resource_count_is("AWS::CodePipeline::Pipeline", 2)

    def test_creates_artifact_bucket(self, single_config_template):
        resources = single_config_template.find_resources("AWS::S3::Bucket")
        assert len(resources) >= 2

    def test_creates_sns_topic(self, single_config_template):
        single_config_template.resource_count_is("AWS::SNS::Topic", 1)

    def test_creates_codebuild_projects_per_config(self, single_config_template):
        """Single config creates CodeBuild projects: lint, test, deploy, per-step builds, model-download, upload-input, trigger."""
        projects = single_config_template.find_resources("AWS::CodeBuild::Project")
        # At minimum: quality-gate + deploy + model-download + upload-input + trigger + at least 1 build step
        assert len(projects) >= 6

    def test_pipeline_has_five_stages(self, single_config_template):
        """Default config: Source, QualityGate, Deploy, ModelDownloadAndUpload, TriggerPipeline."""
        single_config_template.has_resource_properties(
            "AWS::CodePipeline::Pipeline",
            {
                "Stages": assertions.Match.array_with(
                    [
                        assertions.Match.object_like({"Name": "Source"}),
                        assertions.Match.object_like({"Name": "QualityGate"}),
                        assertions.Match.object_like({"Name": "Deploy"}),
                        assertions.Match.object_like({"Name": "ModelDownloadAndUpload"}),
                        assertions.Match.object_like({"Name": "TriggerPipeline"}),
                    ]
                )
            },
        )

    def test_deploy_stage_has_deploy_only(self, single_config_template):
        """Deploy stage has only Deploy action (containers built by dedicated pipeline)."""
        pipelines = single_config_template.find_resources("AWS::CodePipeline::Pipeline")
        for resource in pipelines.values():
            stages = resource["Properties"]["Stages"]
            deploy_stages = [s for s in stages if s["Name"] == "Deploy"]
            if not deploy_stages:
                continue
            actions = deploy_stages[0]["Actions"]
            assert len(actions) == 1
            assert actions[0]["Name"] == "Deploy"

    def test_model_download_and_upload_has_parallel_actions(self, single_config_template):
        """ModelDownloadAndUpload stage has both actions at run_order=1 (parallel)."""
        single_config_template.has_resource_properties(
            "AWS::CodePipeline::Pipeline",
            {
                "Stages": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "Name": "ModelDownloadAndUpload",
                                "Actions": assertions.Match.array_with(
                                    [
                                        assertions.Match.object_like(
                                            {
                                                "Name": "TriggerModelDownload",
                                                "RunOrder": 1,
                                            }
                                        ),
                                        assertions.Match.object_like(
                                            {
                                                "Name": "UploadInputData",
                                                "RunOrder": 1,
                                            }
                                        ),
                                    ]
                                ),
                            }
                        ),
                    ]
                )
            },
        )

    def test_creates_events_rule_per_pipeline(self, single_config_template):
        single_config_template.resource_count_is("AWS::Events::Rule", 1)

    def test_two_configs_create_two_events_rules(self):
        config = CicdConfig(
            pipeline_configs=["config_vrag.yaml", "config_t2a.yaml"],
            test_commands={
                "config_vrag.yaml": "uv run pytest tests/unit/ -x --no-header -q",
                "config_t2a.yaml": "uv run pytest tests/unit/ -x --no-header -q",
            },
        )
        _, template = _create_cicd_pipeline_stack(config)
        template.resource_count_is("AWS::Events::Rule", 2)

    def test_events_rule_targets_sns(self, single_config_template):
        single_config_template.has_resource_properties(
            "AWS::Events::Rule",
            {
                "EventPattern": assertions.Match.object_like(
                    {
                        "source": ["aws.codepipeline"],
                        "detail": assertions.Match.object_like({"state": ["FAILED"]}),
                    }
                ),
            },
        )

    def test_sns_email_subscription_when_configured(self):
        config = CicdConfig(
            notification_email="test@example.com",
            pipeline_configs=["config_vrag.yaml"],
        )
        _, template = _create_cicd_pipeline_stack(config)
        template.resource_count_is("AWS::SNS::Subscription", 1)
        template.has_resource_properties(
            "AWS::SNS::Subscription",
            {"Protocol": "email", "Endpoint": "test@example.com"},
        )

    def test_no_sns_subscription_when_email_is_none(self, single_config_template):
        single_config_template.resource_count_is("AWS::SNS::Subscription", 0)


class TestQualityGateStage:
    """Tests for the QualityGate pipeline stage."""

    def test_quality_gate_stage_present(self, single_config_template):
        """Pipeline has a QualityGate stage."""
        single_config_template.has_resource_properties(
            "AWS::CodePipeline::Pipeline",
            {"Stages": assertions.Match.array_with([assertions.Match.object_like({"Name": "QualityGate"})])},
        )

    def test_quality_gate_between_source_and_deploy(self, single_config_template):
        """QualityGate is at stage index 1 (between Source and Deploy)."""
        pipelines = single_config_template.find_resources("AWS::CodePipeline::Pipeline")
        for _lid, resource in pipelines.items():
            stages = resource["Properties"]["Stages"]
            stage_names = [s["Name"] for s in stages]
            if "Deploy" not in stage_names:
                continue  # container build pipeline
            assert stage_names.index("QualityGate") == 1
            assert stage_names.index("Source") == 0
            assert stage_names.index("Deploy") == 2

    def test_quality_gate_has_lint_and_test_projects(self, single_config_template):
        """QualityGate stage has separate CodeBuild projects for lint and test."""
        projects = single_config_template.find_resources("AWS::CodeBuild::Project")
        has_lint = any(
            "pre-commit run" in res.get("Properties", {}).get("Source", {}).get("BuildSpec", "")
            for res in projects.values()
        )
        has_test = any(
            "pytest" in res.get("Properties", {}).get("Source", {}).get("BuildSpec", "")
            and "pre-commit" not in res.get("Properties", {}).get("Source", {}).get("BuildSpec", "")
            for res in projects.values()
        )
        assert has_lint, "No CodeBuild project found with pre-commit in buildspec"
        assert has_test, "No CodeBuild project found with pytest in buildspec"


class TestCiCdPipelineStackSecurity:
    """Tests for security configuration."""

    def test_artifact_bucket_kms_encrypted(self, single_config_template):
        single_config_template.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "BucketEncryption": assertions.Match.object_like(
                    {
                        "ServerSideEncryptionConfiguration": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "ServerSideEncryptionByDefault": assertions.Match.object_like(
                                            {"SSEAlgorithm": "aws:kms"}
                                        )
                                    }
                                )
                            ]
                        )
                    }
                )
            },
        )

    def test_artifact_bucket_ssl_enforced(self, single_config_template):
        single_config_template.has_resource_properties(
            "AWS::S3::BucketPolicy",
            {
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Condition": assertions.Match.object_like(
                                            {"Bool": {"aws:SecureTransport": "false"}}
                                        ),
                                        "Effect": "Deny",
                                    }
                                )
                            ]
                        )
                    }
                )
            },
        )

    def test_codebuild_projects_in_vpc(self, single_config_template):
        single_config_template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {"VpcConfig": assertions.Match.object_like({"VpcId": assertions.Match.any_value()})},
        )

    def test_codebuild_log_groups_kms_encrypted(self, single_config_template):
        single_config_template.has_resource_properties(
            "AWS::Logs::LogGroup",
            {"KmsKeyId": assertions.Match.any_value()},
        )


class TestCiCdPipelineStackEnvVars:
    """Tests for CodeBuild environment variable injection."""

    def test_codebuild_has_aws_account_id_env_var(self, single_config_template):
        single_config_template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {
                "Environment": assertions.Match.object_like(
                    {
                        "EnvironmentVariables": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Name": "AWS_ACCOUNT_ID",
                                        "Type": "PLAINTEXT",
                                        "Value": "123456789012",
                                    }
                                )
                            ]
                        )
                    }
                )
            },
        )

    def test_codebuild_has_region_env_var(self, single_config_template):
        single_config_template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {
                "Environment": assertions.Match.object_like(
                    {
                        "EnvironmentVariables": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Name": "REGION",
                                        "Type": "PLAINTEXT",
                                        "Value": "us-east-1",
                                    }
                                )
                            ]
                        )
                    }
                )
            },
        )


class TestCiCdPipelineStackTriggerStage:
    """Tests for the TriggerPipeline stage with manual approval."""

    def test_trigger_stage_always_present(self, single_config_template):
        single_config_template.has_resource_properties(
            "AWS::CodePipeline::Pipeline",
            {"Stages": assertions.Match.array_with([assertions.Match.object_like({"Name": "TriggerPipeline"})])},
        )

    def test_trigger_stage_has_approval_and_codebuild_actions(self, single_config_template):
        single_config_template.has_resource_properties(
            "AWS::CodePipeline::Pipeline",
            {
                "Stages": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "Name": "TriggerPipeline",
                                "Actions": assertions.Match.array_with(
                                    [
                                        assertions.Match.object_like(
                                            {
                                                "Name": "ApproveExecution",
                                                "ActionTypeId": assertions.Match.object_like(
                                                    {
                                                        "Category": "Approval",
                                                    }
                                                ),
                                                "RunOrder": 1,
                                            }
                                        ),
                                        assertions.Match.object_like(
                                            {
                                                "Name": "TriggerSageMakerPipeline",
                                                "RunOrder": 2,
                                            }
                                        ),
                                    ]
                                ),
                            }
                        ),
                    ]
                )
            },
        )


class TestCiCdPipelineStackHelpers:
    """Tests for static helper methods."""

    def test_cfg_label_default_config(self):
        assert CiCdPipelineStack._cfg_label("config_vrag.yaml") == "vrag"

    def test_cfg_label_t2a(self):
        assert CiCdPipelineStack._cfg_label("config_t2a.yaml") == "t2a"

    def test_cfg_label_motionart(self):
        assert CiCdPipelineStack._cfg_label("config_motionart.yaml") == "motionart"

    def test_cfg_label_i2v(self):
        assert CiCdPipelineStack._cfg_label("config_i2v.yaml") == "i2v"


class TestCiCdModelDownloadAction:
    """Tests for TriggerModelDownload running download script directly via CodeBuild (Req 1.2)."""

    def test_trigger_model_download_is_codebuild_action(self, single_config_template):
        """TriggerModelDownload action type is CodeBuild, not Lambda invoke."""
        single_config_template.has_resource_properties(
            "AWS::CodePipeline::Pipeline",
            {
                "Stages": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "Name": "ModelDownloadAndUpload",
                                "Actions": assertions.Match.array_with(
                                    [
                                        assertions.Match.object_like(
                                            {
                                                "Name": "TriggerModelDownload",
                                                "ActionTypeId": assertions.Match.object_like(
                                                    {
                                                        "Category": "Build",
                                                        "Provider": "CodeBuild",
                                                    }
                                                ),
                                            }
                                        ),
                                    ]
                                ),
                            }
                        ),
                    ]
                )
            },
        )

    def test_model_download_project_uses_x2_large_compute(self, single_config_template):
        """CI/CD model download CodeBuild project uses X2_LARGE compute."""
        single_config_template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {
                "Environment": assertions.Match.object_like(
                    {
                        "ComputeType": "BUILD_GENERAL1_2XLARGE",
                        "EnvironmentVariables": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Name": "MODELS_BUCKET",
                                    }
                                ),
                            ]
                        ),
                    }
                ),
            },
        )

    def test_model_download_project_has_models_bucket_env_var(self, single_config_template):
        """Model download project has MODELS_BUCKET env var."""
        single_config_template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {
                "Environment": assertions.Match.object_like(
                    {
                        "EnvironmentVariables": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Name": "MODELS_BUCKET",
                                        "Type": "PLAINTEXT",
                                        "Value": assertions.Match.string_like_regexp(".*models-bucket"),
                                    }
                                ),
                            ]
                        ),
                    }
                ),
            },
        )

    def test_model_download_project_timeout_120_minutes(self, single_config_template):
        """Model download project has 120-minute timeout."""
        single_config_template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {
                "TimeoutInMinutes": 120,
                "Environment": assertions.Match.object_like(
                    {
                        "ComputeType": "BUILD_GENERAL1_2XLARGE",
                    }
                ),
            },
        )


class TestCiCdModelDownloadIamPermissions:
    """Tests for CI/CD model download IAM permissions."""

    def test_model_download_has_s3_permissions(self, single_config_template):
        """Model download policy includes S3 read/write actions on models bucket."""
        single_config_template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Effect": "Allow",
                                        "Action": assertions.Match.array_with(
                                            [
                                                "s3:GetObject",
                                                "s3:PutObject",
                                                "s3:ListBucket",
                                            ]
                                        ),
                                        "Resource": assertions.Match.array_with(
                                            [
                                                assertions.Match.string_like_regexp("arn:aws:s3:::.*models-bucket"),
                                                assertions.Match.string_like_regexp("arn:aws:s3:::.*models-bucket/\\*"),
                                            ]
                                        ),
                                    }
                                ),
                            ]
                        ),
                    }
                ),
            },
        )

    def test_model_download_has_kms_permissions(self, single_config_template):
        """Model download policy includes KMS permissions."""
        single_config_template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Effect": "Allow",
                                        "Action": assertions.Match.array_with(
                                            [
                                                "kms:GenerateDataKey",
                                                "kms:Decrypt",
                                            ]
                                        ),
                                    }
                                ),
                            ]
                        ),
                    }
                ),
            },
        )

    def test_no_lambda_invoke_permission_for_model_download(self, single_config_template):
        """Model download policy does NOT include lambda:InvokeFunction."""
        policies = single_config_template.find_resources("AWS::IAM::ManagedPolicy")
        for logical_id, resource in policies.items():
            policy_doc = resource.get("Properties", {}).get("PolicyDocument", {})
            statements = policy_doc.get("Statement", [])
            for stmt in statements:
                actions = stmt.get("Action", [])
                if isinstance(actions, str):
                    actions = [actions]
                resources = stmt.get("Resource", [])
                if isinstance(resources, str):
                    resources = [resources]
                is_model_download_policy = any("models-bucket" in str(r) for r in resources)
                if is_model_download_policy:
                    assert "lambda:InvokeFunction" not in actions, (
                        f"Model download policy {logical_id} should not have lambda:InvokeFunction"
                    )


class TestCiCdDeployScriptFlags:
    """Tests that the deploy script uses CI/CD-safe CDK flags."""

    def test_deploy_script_uses_method_direct(self, single_config_template):
        """Deploy buildspec must include --method=direct."""
        projects = single_config_template.find_resources("AWS::CodeBuild::Project")
        deploy_buildspecs = []
        for _lid, res in projects.items():
            spec = res.get("Properties", {}).get("Source", {}).get("BuildSpec", "")
            if "cdk deploy" in spec and "resolve_stacks" in spec:
                deploy_buildspecs.append(spec)
        assert deploy_buildspecs, "No deploy CodeBuild project found"
        for spec in deploy_buildspecs:
            assert "--method=direct" in spec, (
                "Deploy script must use --method=direct to avoid TTY prompts for resource replacements in CodeBuild"
            )

    def test_deploy_script_uses_require_approval_never(self, single_config_template):
        """Deploy buildspec must include --require-approval never."""
        projects = single_config_template.find_resources("AWS::CodeBuild::Project")
        for _lid, res in projects.items():
            spec = res.get("Properties", {}).get("Source", {}).get("BuildSpec", "")
            if "cdk deploy" in spec and "resolve_stacks" in spec:
                assert "--require-approval never" in spec
