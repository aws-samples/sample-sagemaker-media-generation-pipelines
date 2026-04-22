"""
Pipeline infrastructure stack.

This module defines the PipelineStack which composes the PipelineConstruct,
per-step S3 buckets, processing jobs, and a trigger Lambda. Fully config-driven:
adding a new processing step requires only a new YAML key and container directory.
"""

from aws_cdk import (
    CfnOutput,
    Stack,
)
from aws_cdk import (
    aws_iam as iam,
)
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger

from config.config import PipelineConfig
from infrastructure.data import DataStack
from infrastructure.security import SecurityStack
from project_constructs.lambda_function import LambdaTemplate
from project_constructs.pipeline import LambdaStepInfo, PipelineConstruct
from project_constructs.processing_job.main import IoBucketConfig, ReusableProcessingJob
from project_constructs.retrieval import RetrievalConstruct
from project_constructs.s3 import BucketTemplate

BEDROCK_HAIKU_MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"


class PipelineStack(Stack):
    """
    CDK Stack that composes the SageMaker Pipeline with config-driven steps.

    Attributes:
        pipeline_construct: The SageMaker Pipeline construct.
        trigger_lambda: The pipeline trigger Lambda.
        pipeline_execution_policy: Policy for StartPipelineExecution.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        security_stack: SecurityStack,
        data_stack: DataStack,
        pipeline_config: PipelineConfig,
        submit_lambdas: dict[str, LambdaTemplate] | None = None,
        a2i_constructs: dict | None = None,
        retrieval_construct: RetrievalConstruct | None = None,
        prefix: str = "dev",
        ecr_prefix: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        logger.info("Creating PipelineStack with prefix: {}", prefix)

        region = Stack.of(self).region
        account = Stack.of(self).account
        ecr_pfx = ecr_prefix or prefix
        step_names = list(pipeline_config.steps.keys())
        graph = pipeline_config.pipeline_graph

        # --- Per-step resources (graph-driven) ---
        processing_jobs: list[ReusableProcessingJob] = []
        output_buckets: dict[str, BucketTemplate] = {}

        # Build pipeline graph name mapping: step_name -> prefixed name for DependsOn
        pipeline_graph: dict[str, list[str]] = {}
        all_graph_names = set(step_names) | set(pipeline_config.lambda_steps.keys())
        for step_name in all_graph_names:
            deps = graph.get(step_name, [])
            pipeline_graph[f"{prefix}-{step_name}"] = [f"{prefix}-{dep}" for dep in deps]

        for step_name in step_names:
            step_cfg = pipeline_config.steps[step_name]
            bucket_safe_name = step_name.replace("_", "-")

            # Output bucket for this step
            output_bucket = BucketTemplate(
                self,
                f"{prefix}-{step_name}-OutputBucket",
                bucket_name=f"{account}-{region}-{prefix}-{bucket_safe_name}-output-bucket",
                kms_key=security_stack.kms_key,
                logging_bucket=data_stack.logs_bucket,
            )
            output_buckets[step_name] = output_bucket
            logger.info("Created output bucket for step: {}", step_name)

            # Determine input bucket from graph dependencies
            deps = graph.get(step_name, [])
            channel = step_cfg.input_channel
            shard_input = step_cfg.InstanceCount > 1
            if not deps:
                # Root step: reads from DataStack input bucket
                input_buckets = {channel: IoBucketConfig(bucket_template=data_stack.input_bucket)}
                input_source: str | list[str] = "DataStack"
            elif shard_input and step_cfg.include_data_input:
                # Multi-instance step needing DataStack: sharded from parent + FullyReplicated from DataStack
                parent = deps[0]
                input_buckets = {
                    "shards": IoBucketConfig(bucket_template=output_buckets[parent], sharded=True, needs_exec_id=True),
                    channel: IoBucketConfig(bucket_template=data_stack.input_bucket),
                }
                input_source = parent
            elif shard_input:
                # Multi-instance step (no DataStack): shard parent output directly on the input channel
                parent = deps[0]
                input_buckets = {
                    channel: IoBucketConfig(bucket_template=output_buckets[parent], sharded=True, needs_exec_id=True),
                }
                input_source = parent
            elif len(deps) == 1:
                # Single dependency: reads from parent's output bucket
                parent = deps[0]
                input_buckets = {channel: IoBucketConfig(bucket_template=output_buckets[parent], needs_exec_id=True)}
                input_source = parent
            else:
                # Multiple dependencies: use first dep as primary input
                parent = deps[0]
                input_buckets = {channel: IoBucketConfig(bucket_template=output_buckets[parent], needs_exec_id=True)}
                input_source = deps

            # Add DataStack input bucket if step needs inputs.json
            if step_cfg.include_data_input and channel != "input":
                input_buckets["input"] = IoBucketConfig(bucket_template=data_stack.input_bucket)
                logger.debug("Added DataStack input channel for step: {}", step_name)

            # Add models bucket inputs (one channel per prefix)
            for _i, prefix_path in enumerate(step_cfg.models_prefix):
                channel_name = (
                    "models" if len(step_cfg.models_prefix) == 1 else f"models_{prefix_path.replace('/', '_')}"
                )
                input_buckets[channel_name] = IoBucketConfig(
                    bucket_template=data_stack.models_bucket,
                    path=prefix_path,
                )
            logger.debug("S3 wiring for {}: input from {}", step_name, input_source)

            # ECR image: use ecr_image override if set, otherwise step_name
            # String-based ECR lookup to avoid cross-stack CloudFormation exports
            ecr_key = step_cfg.ecr_image or step_name
            ecr_clean = ecr_key.replace("_", "-")
            ecr_repo_name = f"{ecr_pfx}/processing/{ecr_clean}"
            ecr_image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{ecr_repo_name}"
            ecr_repo_arn = f"arn:aws:ecr:{region}:{account}:repository/{ecr_repo_name}"
            step_clean = step_name.replace("_", "-")

            ecr_pull_policy = iam.ManagedPolicy(
                self,
                f"{prefix}-{step_name}-EcrPullPolicy",
                managed_policy_name=f"{prefix}-{step_clean}-ecr-pull-policy",
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=[
                            "ecr:GetDownloadUrlForLayer",
                            "ecr:BatchGetImage",
                            "ecr:BatchCheckLayerAvailability",
                        ],
                        resources=[ecr_repo_arn],
                    )
                ],
            )
            ecr_auth_policy = iam.ManagedPolicy(
                self,
                f"{prefix}-{step_name}-EcrAuthPolicy",
                managed_policy_name=f"{prefix}-{step_clean}-ecr-auth-policy",
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=["ecr:GetAuthorizationToken"],
                        resources=["*"],
                    )
                ],
            )

            job = ReusableProcessingJob(
                self,
                f"{prefix}-{step_name}-ProcessingJob",
                job_name=f"{prefix}-{step_name}",
                input_buckets=input_buckets,
                output_buckets={"output": IoBucketConfig(bucket_template=output_bucket)},
                security_stack=security_stack,
                lambda_trigger=False,
                cfg=step_cfg,
                environment={
                    "DYNAMODB_TABLE_NAME": data_stack.dynamodb_table.table.table_name,
                    "STEP_NAME": step_name,
                    "LOCAL_OUTPUT_DIR": "/opt/ml/processing/output/output/",
                    "NUM_ASSETS_PER_PROMPT": str(step_cfg.num_assets_per_prompt),
                    **({"UPSTREAM_STEP": deps[0]} if deps else {}),
                    **step_cfg.Environment,
                },
                ecr_image_uri=ecr_image_uri,
                ecr_pull_policy=ecr_pull_policy,
                ecr_auth_policy=ecr_auth_policy,
            )
            logger.info("Created processing job for step: {}", step_name)

            # Attach DynamoDB read/write policy to each processing job role
            job.role.add_managed_policy(data_stack.dynamodb_table.read_write_policy)
            logger.debug("Attached DynamoDB read/write policy to step: {}", step_name)

            # Wire AOSS + Bedrock + S3 permissions for the retrieval step
            if step_name == "retrieval" and retrieval_construct is not None:
                # Use string-based ARNs to avoid cross-stack CloudFormation exports
                # (same pattern as ECR repos — deploy-order only, no data dependencies)
                aoss_collection_arn = f"arn:aws:aoss:{region}:{account}:collection/*"
                # Use string-based bucket ARN to avoid cross-stack CloudFormation
                # exports (same pattern as ECR repos in this stack).
                retrieval_bucket_name = f"{account}-{region}-{prefix}-retrieval-images-bucket"
                retrieval_bucket_arn = f"arn:aws:s3:::{retrieval_bucket_name}"

                # Consolidate all retrieval-specific permissions into a single
                # managed policy to stay within the IAM 10-policy-per-role limit.
                retrieval_policy = iam.ManagedPolicy(
                    self,
                    f"{prefix}-retrieval-StepPolicy",
                    managed_policy_name=f"{prefix}-retrieval-step-policy",
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=["aoss:APIAccessAll"],
                            resources=[aoss_collection_arn],
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=["bedrock:InvokeModel"],
                            resources=[
                                f"arn:aws:bedrock:{region}::foundation-model/{retrieval_construct.retrieval_config.embedding_model_id}"
                            ],
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=["s3:GetObject", "s3:ListBucket"],
                            resources=[
                                retrieval_bucket_arn,
                                f"{retrieval_bucket_arn}/*",
                            ],
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=["ssm:GetParameter"],
                            resources=[f"arn:aws:ssm:{region}:{account}:parameter/{prefix}/retrieval/*"],
                        ),
                    ],
                )
                job.role.add_managed_policy(retrieval_policy)

                # Use SSM parameter lookup at runtime instead of cross-stack CFn tokens
                job.definition_model.Arguments.Environment["AOSS_ENDPOINT_SSM"] = f"/{prefix}/retrieval/aoss-endpoint"
                job.definition_model.Arguments.Environment["AOSS_INDEX_NAME"] = (
                    retrieval_construct.retrieval_config.index_name
                )
                job.definition_model.Arguments.Environment["QUERY_K"] = str(
                    retrieval_construct.retrieval_config.query_k
                )
                job.definition_model.Arguments.Environment["EMBEDDING_MODEL_ID"] = (
                    retrieval_construct.retrieval_config.embedding_model_id
                )
                job.definition_model.Arguments.Environment["EMBEDDING_DIMENSION"] = str(
                    retrieval_construct.retrieval_config.embedding_dimension
                )
                job.update_definition()
                logger.info("Wired AOSS + Bedrock + S3 permissions for retrieval step")

            # Wire Bedrock InvokeModel permission for the vrag_llm step
            if step_name == "vrag_llm":
                # Derive Bedrock model ARN from the configured model ID
                vrag_model_id = step_cfg.Environment.get(
                    "VRAG_LLM_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0"
                )
                # Strip version suffix (e.g. ":0") for base model
                base_model = vrag_model_id.split(":")[
                    0
                ]  # "us.anthropic.claude-3-5-haiku-20241022-v1" or "qwen.qwen3-32b-v1"
                # For foundation-model ARN: strip region prefix (e.g. "us.") but keep provider prefix
                # Region prefixes are 2-letter codes like "us", "eu"; provider prefixes are longer like "anthropic", "qwen"
                parts = base_model.split(".", 1)
                foundation_model = parts[1] if len(parts) == 2 and len(parts[0]) <= 2 else base_model

                vrag_llm_policy = iam.ManagedPolicy(
                    self,
                    f"{prefix}-vrag-llm-StepPolicy",
                    managed_policy_name=f"{prefix}-vrag-llm-step-policy",
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "bedrock:InvokeModel",
                                "bedrock:InvokeModelWithResponseStream",
                            ],
                            resources=[
                                f"arn:aws:bedrock:*::foundation-model/{foundation_model}*",
                                f"arn:aws:bedrock:*:*:inference-profile/{base_model}*",
                            ],
                        ),
                    ],
                )
                job.role.add_managed_policy(vrag_llm_policy)
                logger.info("Wired Bedrock permissions for vrag_llm step (model: {})", vrag_model_id)

            # Wire S3 read permission on retrieval images bucket for the i2v step
            if step_name == "i2v" and retrieval_construct is not None:
                retrieval_bucket_name = f"{account}-{region}-{prefix}-retrieval-images-bucket"
                retrieval_bucket_arn = f"arn:aws:s3:::{retrieval_bucket_name}"
                i2v_retrieval_policy = iam.ManagedPolicy(
                    self,
                    f"{prefix}-i2v-RetrievalReadPolicy",
                    managed_policy_name=f"{prefix}-i2v-retrieval-read-policy",
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=["s3:GetObject", "s3:ListBucket"],
                            resources=[
                                retrieval_bucket_arn,
                                f"{retrieval_bucket_arn}/*",
                            ],
                        ),
                    ],
                )
                job.role.add_managed_policy(i2v_retrieval_policy)
                logger.info("Wired S3 read permission on retrieval images bucket for i2v step")

            processing_jobs.append(job)

        # --- Build LambdaStepInfo list from config-driven lambda_steps ---
        # submit_lambdas come from A2IStack (passed in); if None, no A2I steps
        if submit_lambdas is None:
            submit_lambdas = {}
        if a2i_constructs is None:
            a2i_constructs = {}
        lambda_step_infos: list[LambdaStepInfo] = []
        for ls_name, ls_cfg in pipeline_config.lambda_steps.items():
            # Reuse the submit Lambda from A2IStack
            if ls_cfg.a2i_name and ls_cfg.a2i_name in submit_lambdas:
                fn = submit_lambdas[ls_cfg.a2i_name]

                # Wire SOURCE_BUCKET to the generation step's output bucket.
                # We construct the bucket name as a plain string to avoid
                # cross-stack token references (A2IStack Lambda ← PipelineStack
                # bucket) which would create a cyclic dependency.
                deps = graph.get(ls_name, [])
                for dep in deps:
                    if dep in output_buckets:
                        dep_safe = dep.replace("_", "-")
                        bucket_name = f"{account}-{region}-{prefix}-{dep_safe}-output-bucket"
                        bucket_arn = f"arn:aws:s3:::{bucket_name}"
                        fn.lambda_function.add_environment(
                            "SOURCE_BUCKET",
                            bucket_name,
                        )
                        fn.lambda_function.add_environment(
                            "UPSTREAM_STEP",
                            dep,
                        )
                        fn.function_role.add_to_policy(
                            iam.PolicyStatement(
                                effect=iam.Effect.ALLOW,
                                actions=[
                                    "s3:GetObject",
                                    "s3:ListBucket",
                                ],
                                resources=[
                                    bucket_arn,
                                    f"{bucket_arn}/*",
                                ],
                            )
                        )
                        logger.info(
                            "Wired A2I submit Lambda '{}' SOURCE_BUCKET to {}'s output bucket ({})",
                            ls_name,
                            dep,
                            bucket_name,
                        )

                        # Grant the A2I flow definition role read access to
                        # the generation step's output bucket.  The
                        # grant_read_access Liquid filter in the task template
                        # generates presigned URLs using this role's credentials,
                        # so it must be able to read from the bucket where the
                        # generated media actually lives.
                        a2i_construct = a2i_constructs.get(ls_cfg.a2i_name)
                        if a2i_construct:
                            a2i_construct.role.add_to_policy(
                                iam.PolicyStatement(
                                    effect=iam.Effect.ALLOW,
                                    actions=[
                                        "s3:GetObject",
                                        "s3:ListBucket",
                                    ],
                                    resources=[
                                        bucket_arn,
                                        f"{bucket_arn}/*",
                                    ],
                                )
                            )
                            logger.info(
                                "Granted A2I flow definition role '{}' read access to output bucket {}",
                                ls_cfg.a2i_name,
                                bucket_name,
                            )
                            # Suppress cdk-nag for the wildcard S3 object read
                            # on the flow definition role's default policy.
                            # The role needs s3:GetObject on bucket/* to generate
                            # presigned URLs for the A2I task template.
                            if a2i_construct.role.node.try_find_child("DefaultPolicy"):
                                NagSuppressions.add_resource_suppressions(
                                    a2i_construct.role.node.find_child("DefaultPolicy"),
                                    suppressions=[
                                        {
                                            "id": "AwsSolutions-IAM5",
                                            "reason": (
                                                "A2I flow definition role requires s3:GetObject "
                                                "on output bucket objects to generate presigned "
                                                "URLs in the task template"
                                            ),
                                            "appliesTo": [
                                                f"Resource::arn:aws:s3:::{bucket_name}/*",
                                            ],
                                        },
                                    ],
                                )
                        break
            else:
                # Create a standalone Lambda for non-A2I lambda steps
                fn = LambdaTemplate(
                    self,
                    f"{prefix}-{ls_name}-LambdaStep",
                    function_name=f"{prefix}-{ls_name}",
                    lambda_path=ls_cfg.lambda_path,
                    description=f"Pipeline Lambda step: {ls_name}",
                    vpc=security_stack.vpc,
                    kms_key=security_stack.kms_key,
                    env_vars={},
                )
            lambda_step_infos.append(
                LambdaStepInfo(
                    name=f"{prefix}-{ls_name}",
                    function_arn=fn.lambda_function.function_arn,
                )
            )
            logger.info("Registered Lambda step: {} -> {}", ls_name, fn.lambda_function.function_name)

        # --- Pipeline execution role ---
        pipeline_role = iam.Role(
            self,
            f"{prefix}-PipelineExecutionRole",
            role_name=f"{prefix}-pipeline-execution-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[security_stack.kms_key_policy],
        )

        # --- PipelineConstruct ---
        self.pipeline_construct = PipelineConstruct(
            self,
            f"{prefix}-SageMakerPipeline",
            pipeline_name=f"{prefix}-sagemaker-pipeline",
            processing_jobs=processing_jobs,
            execution_role=pipeline_role,
            pipeline_graph=pipeline_graph,
            lambda_steps=lambda_step_infos,
        )

        # Attach processing policy to pipeline role
        pipeline_role.add_managed_policy(self.pipeline_construct.processing_policy)

        # --- Pass role policies for each processing job ---
        pass_role_statements = [
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["iam:PassRole"],
                resources=[job.role.role_arn],
            )
            for job in processing_jobs
        ]
        pass_role_policy = iam.ManagedPolicy(
            self,
            f"{prefix}-PipelinePassRolePolicy",
            managed_policy_name=f"{prefix}-pipeline-pass-role-policy",
            statements=pass_role_statements,
        )
        pipeline_role.add_managed_policy(pass_role_policy)

        # --- Pipeline Trigger Lambda ---
        self.trigger_lambda = LambdaTemplate(
            self,
            f"{prefix}-PipelineTriggerLambda",
            function_name=f"{prefix}-pipeline-trigger",
            lambda_path="trigger_pipeline",
            description="Triggers the SageMaker Pipeline execution",
            vpc=security_stack.vpc,
            kms_key=security_stack.kms_key,
            env_vars={
                "PIPELINE_NAME": f"{prefix}-sagemaker-pipeline",
            },
        )
        logger.info("Created pipeline trigger Lambda")

        # --- Pipeline execution policy ---
        self.pipeline_execution_policy = iam.ManagedPolicy(
            self,
            f"{prefix}-PipelineExecutionPolicy",
            managed_policy_name=f"{prefix}-pipeline-execution-policy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["sagemaker:StartPipelineExecution"],
                    resources=[self.pipeline_construct.pipeline_arn],
                )
            ],
        )
        self.trigger_lambda.function_role.add_managed_policy(self.pipeline_execution_policy)
        self.trigger_lambda.function_role.add_managed_policy(security_stack.kms_key_policy)
        logger.info("Created pipeline execution policy")

        # --- CfnOutputs ---
        CfnOutput(
            self,
            f"{prefix}-PipelineArn",
            value=self.pipeline_construct.pipeline_arn,
            description="SageMaker Pipeline ARN",
        )
        CfnOutput(
            self,
            f"{prefix}-TriggerLambdaName",
            value=self.trigger_lambda.lambda_function.function_name,
            description="Pipeline trigger Lambda function name",
        )

        # --- CDK Nag Suppressions ---
        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "Lambda and SageMaker roles use AWS managed policies",
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Processing job and pipeline roles require wildcard for dynamic resource names",
                },
            ],
            apply_to_children=True,
        )
