"""
CI/CD Pipeline infrastructure stack.

Creates one CodePipeline per pipeline config file. Each pipeline runs:

    Source → QualityGate → Deploy → ModelDownloadAndUpload → TriggerPipeline
"""

import hashlib

from aws_cdk import (
    AssetHashType,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_codebuild as codebuild,
)
from aws_cdk import (
    aws_codepipeline as codepipeline,
)
from aws_cdk import (
    aws_codepipeline_actions as codepipeline_actions,
)
from aws_cdk import (
    aws_events as events,
)
from aws_cdk import (
    aws_events_targets as events_targets,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_s3 as s3,
)
from aws_cdk import (
    aws_s3_assets as s3_assets,
)
from aws_cdk import (
    aws_sns_subscriptions as sns_subscriptions,
)
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger

from config.config import CicdConfig
from infrastructure.cicd_pipeline import buildspecs, policies
from infrastructure.cicd_pipeline.deploy_script import generate_deploy_script
from infrastructure.cicd_pipeline.helpers import create_codebuild_project
from infrastructure.security import SecurityStack
from project_constructs.codepipeline import CodePipelineTemplate
from project_constructs.s3 import BucketTemplate
from project_constructs.sns import SnsTopicTemplate


class CiCdPipelineStack(Stack):
    """CDK Stack that creates one CodePipeline per pipeline config file."""

    @staticmethod
    def _dir_content_hash(paths: list[str], excludes: list[str] | None = None) -> str:
        """Compute a SHA-256 hash over the contents of the given paths.

        Accepts both files and directories. Only regular files are included.
        Paths matching any pattern in *excludes* are skipped.
        The hash is deterministic (sorted file order).
        """
        from pathlib import Path

        if excludes is None:
            excludes = []
        h = hashlib.sha256()
        for p in sorted(paths):
            root = Path(p)
            if not root.exists():
                continue
            if root.is_file():
                h.update(str(root).encode())
                h.update(root.read_bytes())
            else:
                for fp in sorted(root.rglob("*")):
                    if not fp.is_file():
                        continue
                    rel = str(fp)
                    if any(pat.rstrip("/") in rel for pat in excludes):
                        continue
                    h.update(rel.encode())
                    h.update(fp.read_bytes())
        return h.hexdigest()[:16]

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        security_stack: SecurityStack,
        cicd_config: CicdConfig,
        prefix: str = "dev",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        logger.info("Creating CiCdPipelineStack with prefix: {}", prefix)

        account = Stack.of(self).account
        region = Stack.of(self).region
        rollback_flag = "--rollback" if cicd_config.rollback else "--no-rollback"

        # ── Shared infrastructure ────────────────────────────────────
        logging_bucket = s3.Bucket(
            self,
            f"{prefix}-cicd-logs-bucket",
            bucket_name=f"{account}-{region}-{prefix}-cicd-logs-bucket",
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            object_ownership=s3.ObjectOwnership.OBJECT_WRITER,
        )
        self.artifact_bucket = BucketTemplate(
            self,
            f"{prefix}-cicd-artifact",
            bucket_name=f"{account}-{region}-{prefix}-cicd-artifact-bucket",
            kms_key=security_stack.kms_key,
            logging_bucket=logging_bucket,
        )
        self.notification_topic = SnsTopicTemplate(
            self,
            f"{prefix}-cicd-notification",
            topic_name=f"{prefix}-cicd-notification",
            kms_key=security_stack.kms_key,
            service_principals=["events.amazonaws.com"],
        )
        if cicd_config.notification_email:
            self.notification_topic.topic.add_subscription(
                sns_subscriptions.EmailSubscription(cicd_config.notification_email)
            )
        # Shared directories that affect all pipelines
        shared_hash_paths = [
            "infrastructure",
            "project_constructs",
            "app.py",
            "lambdas",
            "tests",
            "schema",
            "config/config.py",
            "config/cicd",
            "config/retrieval",
            ".pre-commit-config.yaml",
            "pyproject.toml",
            "Makefile",
        ]

        cb_compute = {
            "SMALL": codebuild.ComputeType.SMALL,
            "MEDIUM": codebuild.ComputeType.MEDIUM,
            "LARGE": codebuild.ComputeType.LARGE,
            "X2_LARGE": codebuild.ComputeType.X2_LARGE,
        }[cicd_config.compute_type]

        env = {
            "AWS_ACCOUNT_ID": codebuild.BuildEnvironmentVariable(value=account),
            "REGION": codebuild.BuildEnvironmentVariable(value=region),
        }

        # ── Per-config pipelines ─────────────────────────────────────
        # Each pipeline gets its own source asset whose hash covers the
        # shared code directories plus that pipeline's own config file.
        # This way, changing one config only triggers that pipeline.
        self.pipelines: dict[str, CodePipelineTemplate] = {}
        for cfg_file in cicd_config.pipeline_configs:
            per_cfg_asset = s3_assets.Asset(
                self,
                f"{prefix}-cicd-source-{self._cfg_label(cfg_file)}",
                path=".",
                exclude=cicd_config.source_excludes,
                asset_hash=self._dir_content_hash(
                    shared_hash_paths + [f"config/pipeline/{cfg_file}"],
                    excludes=cicd_config.source_excludes,
                ),
                asset_hash_type=AssetHashType.CUSTOM,
            )
            self._create_pipeline(
                cfg_file=cfg_file,
                cicd_config=cicd_config,
                security_stack=security_stack,
                source_asset=per_cfg_asset,
                cb_compute=cb_compute,
                env=env,
                rollback_flag=rollback_flag,
            )

        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "CodeBuild roles use AWS managed policies for admin deploy access",
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CodeBuild roles need wildcard for VPC, S3, CodeBuild, Lambda, KMS access",
                },
                {"id": "AwsSolutions-CB4", "reason": "CodeBuild projects use KMS-encrypted pipeline artifact bucket"},
            ],
            apply_to_children=True,
        )

    # ── Pipeline factory ─────────────────────────────────────────

    def _create_pipeline(
        self,
        cfg_file: str,
        cicd_config: CicdConfig,
        security_stack: SecurityStack,
        source_asset: s3_assets.Asset,
        cb_compute: codebuild.ComputeType,
        env: dict,
        rollback_flag: str,
    ) -> None:
        cfg_label = self._cfg_label(cfg_file)
        cfg_prefix = f"{cicd_config.shared_prefix}{self._read_config_prefix(cfg_file)}"
        tag = f"{cfg_prefix}-cicd"
        account = Stack.of(self).account
        region = Stack.of(self).region

        src_out = codepipeline.Artifact(f"Source-{cfg_label}")

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

        # QualityGate — parallel: lint+synth and unit tests
        test_command = cicd_config.test_commands[cfg_file]

        lint_proj = create_codebuild_project(
            self,
            tag,
            "lint-synth",
            buildspecs.lint_and_synth(cfg_file),
            security_stack,
            codebuild.ComputeType.SMALL,
            cicd_config.timeout_minutes,
            env,
        )
        source_asset.grant_read(lint_proj.role)  # type: ignore[arg-type]

        test_proj = create_codebuild_project(
            self,
            tag,
            "test",
            buildspecs.unit_test(test_command),
            security_stack,
            cb_compute,
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

        # Deploy
        deploy_proj = create_codebuild_project(
            self,
            tag,
            "deploy",
            buildspecs.deploy(generate_deploy_script(cfg_prefix, cfg_file, rollback_flag)),
            security_stack,
            cb_compute,
            cicd_config.timeout_minutes,
            env,
        )
        source_asset.grant_read(deploy_proj.role)  # type: ignore[arg-type]
        deploy_proj.role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AdministratorAccess"))  # type: ignore[union-attr]

        deploy_stage = codepipeline.StageProps(
            stage_name="Deploy",
            actions=[
                codepipeline_actions.CodeBuildAction(
                    action_name="Deploy", project=deploy_proj, input=src_out, run_order=1
                ),
            ],
        )

        # ModelDownloadAndUpload
        models_bucket = f"{account}-{region}-{cfg_prefix}-models-bucket"
        dl_proj = create_codebuild_project(
            self,
            tag,
            "model-download",
            buildspecs.model_download(),
            security_stack,
            codebuild.ComputeType.X2_LARGE,
            120,
            {
                **env,
                "MODELS_BUCKET": codebuild.BuildEnvironmentVariable(value=models_bucket),
                "CONFIG_PREFIX": codebuild.BuildEnvironmentVariable(value=cfg_prefix),
                "SHARED_PREFIX": codebuild.BuildEnvironmentVariable(value=cicd_config.shared_prefix),
            },
        )
        dl_proj.role.add_managed_policy(policies.model_download_policy(self, tag, f"arn:aws:s3:::{models_bucket}"))  # type: ignore[union-attr]

        input_bucket_name = f"{account}-{region}-{cfg_prefix}-input-bucket"
        sync_cmds = self._build_sync_commands(cicd_config.input_data.get(cfg_file, []), input_bucket_name)
        ul_proj = create_codebuild_project(
            self,
            tag,
            "upload-input",
            buildspecs.upload_input(sync_cmds),
            security_stack,
            codebuild.ComputeType.SMALL,
            30,
            env,
        )
        ul_proj.role.add_managed_policy(policies.upload_input_policy(self, tag, f"arn:aws:s3:::{input_bucket_name}"))  # type: ignore[union-attr]

        model_upload_stage = codepipeline.StageProps(
            stage_name="ModelDownloadAndUpload",
            actions=[
                codepipeline_actions.CodeBuildAction(
                    action_name="TriggerModelDownload", project=dl_proj, input=src_out, run_order=1
                ),
                codepipeline_actions.CodeBuildAction(
                    action_name="UploadInputData", project=ul_proj, input=src_out, run_order=1
                ),
            ],
        )

        # TriggerPipeline
        trigger_proj = create_codebuild_project(
            self,
            tag,
            "trigger-pipeline",
            buildspecs.trigger_pipeline(),
            security_stack,
            codebuild.ComputeType.SMALL,
            30,
            {**env, "CONFIG_PREFIX": codebuild.BuildEnvironmentVariable(value=cfg_prefix)},
        )
        trigger_proj.role.add_managed_policy(policies.trigger_pipeline_policy(self, tag, region, account, cfg_prefix))  # type: ignore[union-attr]

        trigger_stage = codepipeline.StageProps(
            stage_name="TriggerPipeline",
            actions=[
                codepipeline_actions.ManualApprovalAction(
                    action_name="ApproveExecution",
                    notification_topic=self.notification_topic.topic,
                    additional_information=f"Approve to trigger the SageMaker pipeline for '{cfg_file}' (prefix: {cfg_prefix}).",
                    run_order=1,
                ),
                codepipeline_actions.CodeBuildAction(
                    action_name="TriggerSageMakerPipeline", project=trigger_proj, input=src_out, run_order=2
                ),
            ],
        )

        # Assemble
        pipeline_name = f"{cfg_prefix}-pipeline"
        self.pipelines[cfg_label] = CodePipelineTemplate(
            self,
            f"{tag}-Pipeline",
            pipeline_name=pipeline_name,
            artifact_bucket=self.artifact_bucket.bucket,
            kms_key=security_stack.kms_key,
            stages=[source_stage, qa_stage, deploy_stage, model_upload_stage, trigger_stage],
        )
        logger.info("Created pipeline: {}", pipeline_name)

        events.Rule(
            self,
            f"{tag}-FailureRule",
            rule_name=f"{tag}-pipeline-failure",
            event_pattern=events.EventPattern(
                source=["aws.codepipeline"],
                detail_type=["CodePipeline Pipeline Execution State Change"],
                detail={"state": ["FAILED"], "pipeline": [pipeline_name]},
            ),
            targets=[events_targets.SnsTopic(self.notification_topic.topic)],
        )

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _build_sync_commands(input_paths: list[str], input_bucket_name: str) -> list[str]:
        """Build the S3 sync/cp commands for the UploadInput action."""
        bucket_uri = f"s3://{input_bucket_name}"
        cmds = ["set -e"]
        for path in input_paths:
            src = f"sample_input_data/{path}"
            if path.endswith("/"):
                cmds.append(f"aws s3 sync {src} {bucket_uri}/ --exact-timestamps")
            else:
                cmds.append(f"aws s3 cp {src} {bucket_uri}/{path}")
        if not input_paths:
            cmds.append("echo 'No input data configured for this pipeline'")
        return cmds

    @staticmethod
    def _cfg_label(cfg_file: str) -> str:
        """``config_vrag.yaml`` → ``vrag``, ``config_t2a.yaml`` → ``t2a``."""
        stem = cfg_file.removesuffix(".yaml").removeprefix("config_").removeprefix("config")
        return stem.replace("_", "-") if stem else "config"

    @staticmethod
    def _read_config_prefix(cfg_file: str) -> str:
        """Read ``construct_id`` from a pipeline config YAML."""
        import yaml

        with open(f"./config/pipeline/{cfg_file}", encoding="utf-8") as f:
            return str(yaml.safe_load(f).get("construct_id", "dev"))
