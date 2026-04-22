"""
Reusable SageMaker Pipeline construct.

This module provides a PipelineConstruct that creates a SageMaker Pipeline
with chained processing steps and optional Lambda steps. Each processing step
is backed by a ReusableProcessingJob instance, reusing its definition_model
to build the pipeline definition JSON. Lambda steps invoke AWS Lambda functions.
"""

import json
import re
from dataclasses import dataclass

from aws_cdk import (
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_sagemaker as sagemaker,
)
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger

from project_constructs.processing_job.main import ReusableProcessingJob

# Regex to detect CDK Token placeholders in S3 URIs (e.g. ${Token[TOKEN.123]})
_CDK_TOKEN_RE = re.compile(r"\$\{Token\[TOKEN\.\d+\]\}")


@dataclass
class LambdaStepInfo:
    """Metadata for a Lambda step in the pipeline."""

    name: str
    function_arn: str


def _extract_bucket_token(uri: str) -> str | None:
    """Return the CDK Token substring from an S3 URI, or None if not found."""
    m = _CDK_TOKEN_RE.search(uri)
    return m.group(0) if m else None


def _create_s3_uri_join(base_uri: str, exec_id_expr: dict, trailing_slash: bool = True) -> dict:
    """Build a Std:Join expression that appends the execution ID to a base S3 URI.

    Result: s3://<bucket>/<optional-path>/<execution_id>/
    """
    base = base_uri.rstrip("/")
    values = [base, exec_id_expr]
    if trailing_slash:
        values.append("")
    return {"Std:Join": {"On": "/", "Values": values}}


class PipelineConstruct(Construct):
    """
    Reusable construct that creates a SageMaker Pipeline with chained steps.

    Attributes:
        pipeline_arn: The SageMaker Pipeline ARN.
        pipeline_name: The SageMaker Pipeline name.
        processing_policy: ManagedPolicy for CreateProcessingJob/DescribeProcessingJob.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        pipeline_name: str,
        processing_jobs: list[ReusableProcessingJob],
        execution_role: iam.Role,
        pipeline_graph: dict[str, list[str]] | None = None,
        lambda_steps: list[LambdaStepInfo] | None = None,
    ) -> None:
        super().__init__(scope, construct_id)

        logger.info(
            "Creating PipelineConstruct: name={}, processing_steps={}, lambda_steps={}",
            pipeline_name,
            len(processing_jobs),
            len(lambda_steps or []),
        )

        self.pipeline_name = pipeline_name
        region = Stack.of(self).region
        account = Stack.of(self).account

        # Build a name->job lookup for graph-based dependency resolution
        _job_by_name: dict[str, ReusableProcessingJob] = {job.definition_model.Name: job for job in processing_jobs}

        # SageMaker Pipeline execution variable — resolved at runtime
        self._exec_id_expr = {"Get": "Execution.PipelineExecutionId"}

        # Build pipeline steps from processing job definitions
        steps = self._build_steps(processing_jobs, pipeline_graph)

        # Build Lambda steps
        lambda_step_dicts = self._build_lambda_steps(lambda_steps or [], pipeline_graph)

        # Inject EXECUTION_ID env var and rewrite S3 paths with execution ID
        self._add_environment_variable(steps, "EXECUTION_ID", self._exec_id_expr)
        self._add_metadata_to_paths(steps, processing_jobs)
        self._add_io_paths_to_environment(steps, processing_jobs)

        # Inject EXECUTION_ID into Lambda step Arguments so the Lambda
        # receives the pipeline execution ID at runtime.

        all_steps = steps + lambda_step_dicts

        pipeline_definition = json.dumps({"Version": "2020-12-01", "Steps": all_steps})

        # Create the SageMaker Pipeline resource
        cfn_pipeline = sagemaker.CfnPipeline(
            self,
            f"{construct_id}-CfnPipeline",
            pipeline_name=pipeline_name,
            pipeline_definition={
                "PipelineDefinitionBody": pipeline_definition,
            },
            role_arn=execution_role.role_arn,
        )
        cfn_pipeline.apply_removal_policy(RemovalPolicy.DESTROY)

        self.pipeline_arn = f"arn:aws:sagemaker:{region}:{account}:pipeline/{pipeline_name}"

        # ManagedPolicy for CreateProcessingJob / DescribeProcessingJob + Lambda invoke
        lambda_arns = [ls.function_arn for ls in (lambda_steps or [])]
        statements = [
            iam.PolicyStatement(
                actions=[
                    "sagemaker:CreateProcessingJob",
                    "sagemaker:DescribeProcessingJob",
                    "sagemaker:StopProcessingJob",
                    "sagemaker:AddTags",
                ],
                resources=[f"arn:aws:sagemaker:{region}:{account}:processing-job/*"],
            )
        ]
        if lambda_arns:
            statements.append(
                iam.PolicyStatement(
                    actions=["lambda:InvokeFunction"],
                    resources=lambda_arns,
                )
            )

        self.processing_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-ProcessingPolicy",
            managed_policy_name=f"{pipeline_name}-processing-policy",
            statements=statements,
        )

        NagSuppressions.add_resource_suppressions(
            self.processing_policy,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Processing job names are dynamic; wildcard required",
                    "appliesTo": [f"Resource::arn:aws:sagemaker:{region}:{account}:processing-job/*"],
                }
            ],
        )

        logger.info("Created SageMaker Pipeline: {}", pipeline_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_steps(
        self,
        processing_jobs: list[ReusableProcessingJob],
        pipeline_graph: dict[str, list[str]] | None,
    ) -> list[dict]:
        """Convert ReusableProcessingJob definitions into pipeline step dicts."""
        steps: list[dict] = []
        for idx, job in enumerate(processing_jobs):
            defn = job.definition_model
            step: dict = {
                "Name": defn.Name,
                "Type": "Processing",
                "Arguments": {
                    "AppSpecification": defn.Arguments.AppSpecification.model_dump(),
                    "ProcessingResources": defn.Arguments.ProcessingResources.model_dump(),
                    "NetworkConfig": defn.Arguments.NetworkConfig.model_dump(),
                    "ProcessingInputs": [inp.model_dump() for inp in defn.Arguments.ProcessingInputs],
                    "ProcessingOutputConfig": defn.Arguments.ProcessingOutputConfig.model_dump(),
                    "RoleArn": defn.Arguments.RoleArn,
                    "Environment": dict(defn.Arguments.Environment),
                    "StoppingCondition": defn.Arguments.StoppingCondition.model_dump(),
                },
            }

            # Use explicit graph if provided, otherwise fall back to linear chaining
            if pipeline_graph and defn.Name in pipeline_graph:
                deps = pipeline_graph[defn.Name]
                if deps:
                    step["DependsOn"] = deps
                    logger.debug("Step '{}' depends on {}", defn.Name, deps)
            elif idx > 0:
                prev_name = processing_jobs[idx - 1].definition_model.Name
                step["DependsOn"] = [prev_name]
                logger.debug("Step '{}' depends on '{}'", defn.Name, prev_name)

            steps.append(step)
        return steps

    def _build_lambda_steps(
        self,
        lambda_steps: list[LambdaStepInfo],
        pipeline_graph: dict[str, list[str]] | None,
    ) -> list[dict]:
        """Convert LambdaStepInfo instances into pipeline Lambda step dicts."""
        steps: list[dict] = []
        for ls in lambda_steps:
            step: dict = {
                "Name": ls.name,
                "Type": "Lambda",
                "Arguments": {
                    "execution_id": self._exec_id_expr,
                },
                "FunctionArn": ls.function_arn,
                "OutputParameters": [],
            }
            if pipeline_graph and ls.name in pipeline_graph:
                deps = pipeline_graph[ls.name]
                if deps:
                    step["DependsOn"] = deps
                    logger.debug("Lambda step '{}' depends on {}", ls.name, deps)
            steps.append(step)
        return steps

    @staticmethod
    def _add_environment_variable(steps: list[dict], key: str, value: object) -> None:
        """Inject an environment variable into every step."""
        for step in steps:
            step["Arguments"]["Environment"][key] = value

    def _add_metadata_to_paths(
        self,
        steps: list[dict],
        processing_jobs: list[ReusableProcessingJob],
    ) -> None:
        """Rewrite inter-step S3 URIs to include the execution ID prefix.

        Uses the ``needs_exec_id`` flag on each IoBucketConfig to decide which
        input channels should be scoped to the current pipeline execution.
        Output URIs are rewritten when they contain a CDK Token (i.e. they
        reference a dynamically-named bucket).
        """
        for step, job in zip(steps, processing_jobs, strict=False):
            args = step["Arguments"]

            # Rewrite OUTPUT_S3_URI env var
            env = args["Environment"]
            if "OUTPUT_S3_URI" in env:
                base_uri = env["OUTPUT_S3_URI"]
                if _extract_bucket_token(str(base_uri)):
                    env["OUTPUT_S3_URI"] = _create_s3_uri_join(base_uri, self._exec_id_expr, trailing_slash=False)

            # Rewrite S3Output URIs
            for out in args["ProcessingOutputConfig"].get("Outputs", []):
                s3out = out.get("S3Output", {})
                uri = s3out.get("S3Uri", "")
                if _extract_bucket_token(str(uri)):
                    s3out["S3Uri"] = _create_s3_uri_join(uri, self._exec_id_expr)

            # Rewrite S3Input URIs — only for channels flagged with
            # needs_exec_id (inter-step channels reading from another
            # step's output bucket).
            exec_id_channels = {name for name, io_cfg in job.input_buckets.items() if io_cfg.needs_exec_id}
            for inp in args["ProcessingInputs"]:
                input_name = inp.get("InputName", "")
                s3in = inp.get("S3Input", {})
                uri = s3in.get("S3Uri", "")
                if input_name in exec_id_channels and uri:
                    s3in["S3Uri"] = _create_s3_uri_join(uri, self._exec_id_expr)

    def _add_io_paths_to_environment(
        self,
        steps: list[dict],
        processing_jobs: list[ReusableProcessingJob],
    ) -> None:
        """Add BUCKET_OUTPUT_<name> and BUCKET_INPUT_<name> env vars to each step.

        These contain the (possibly rewritten) S3 URIs so the container knows
        the actual S3 paths at runtime.
        """
        for step, _job in zip(steps, processing_jobs, strict=False):
            env = step["Arguments"]["Environment"]

            # Output paths
            for out in step["Arguments"]["ProcessingOutputConfig"].get("Outputs", []):
                s3out = out.get("S3Output", {})
                name = out.get("OutputName", "").upper()
                if name and "S3Uri" in s3out:
                    env[f"BUCKET_OUTPUT_{name}"] = s3out["S3Uri"]

            # Input paths
            for inp in step["Arguments"]["ProcessingInputs"]:
                s3in = inp.get("S3Input", {})
                name = inp.get("InputName", "").upper()
                if name and "S3Uri" in s3in:
                    env[f"BUCKET_INPUT_{name}"] = s3in["S3Uri"]
