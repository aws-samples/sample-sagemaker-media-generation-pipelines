"""
Retrieval infrastructure construct.

This module defines the RetrievalConstruct which creates a serverless pipeline
for event-driven image ingestion from S3 → embeddings via Bedrock → indexed in
OpenSearch Serverless. Semantic search queries happen inside the SageMaker
Processing Job container (no query Lambda).

The construct is instantiated inside DataStack (not as a separate stack) to
avoid cyclic cross-stack dependencies between SecurityStack and a standalone
RetrievalStack. When a Lambda in a VPC accesses S3 via event notifications,
CDK auto-adds the bucket ARN to the VPC Gateway Endpoint policy in
SecurityStack, creating a back-reference that forms a cycle.

Architecture:
- S3 bucket (images/ prefix) → S3 event notification → SQS queue
- SQS queue → Lambda (ingest) → Bedrock Titan embeddings → OpenSearch Serverless

The construct exposes ``self.opensearch`` so PipelineStack can read the
collection ARN/endpoint and grant ``aoss:APIAccessAll`` to the retrieval
processing job role.
"""

from aws_cdk import (
    CfnOutput,
    Stack,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda_event_sources as lambda_event_sources,
)
from aws_cdk import (
    aws_s3 as s3,
)
from aws_cdk import (
    aws_s3_notifications as s3_notifications,
)
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger

from config.config import RetrievalConfig, SetupConfig
from infrastructure.security import SecurityStack
from project_constructs.lambda_function import LambdaTemplate
from project_constructs.opensearch import OpenSearchServerlessConstruct
from project_constructs.processing_job.main import IoBucketConfig, ReusableProcessingJob
from project_constructs.s3 import BucketTemplate
from project_constructs.sqs import SqsQueueTemplate


class RetrievalConstruct(Construct):
    """
    CDK Construct for the image retrieval ingestion pipeline.

    Creates an S3 ingestion bucket (via BucketTemplate), SQS queue, ingest
    Lambda, and OpenSearch Serverless collection. Designed to be instantiated
    inside an existing stack (e.g. DataStack) to avoid cross-stack cyclic
    dependencies.

    Attributes:
        ingestion_bucket: BucketTemplate for image uploads and base64 copies.
        sqs_queue: SqsQueueTemplate for ingestion event delivery.
        ingest_lambda: LambdaTemplate for the ingest function.
        opensearch: OpenSearchServerlessConstruct exposing collection ARN/endpoint.
        retrieval_config: The RetrievalConfig used to create this construct.
        setup_trigger_lambdas: Dict of setup job trigger Lambdas keyed by job name.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        logging_bucket: s3.Bucket,
        retrieval_config: RetrievalConfig,
        prefix: str = "dev",
        ecr_prefix: str | None = None,
        setup_configs: dict[str, SetupConfig] | None = None,
        output_bucket: BucketTemplate | None = None,
        security_stack: SecurityStack | None = None,
    ) -> None:
        """
        Initialize the RetrievalConstruct.

        Args:
            scope: The parent construct (typically DataStack).
            construct_id: Unique identifier for this construct.
            logging_bucket: S3 bucket for server access logging (from DataStack).
            retrieval_config: RetrievalConfig with pipeline configuration.
            prefix: Resource naming prefix (default: "dev").
            ecr_prefix: Optional ECR namespace prefix. Falls back to ``prefix``
                when not provided.
            setup_configs: Optional dict of setup job configs keyed by job name.
            output_bucket: Optional BucketTemplate for setup job SageMaker output
                (DataStack input bucket, for writing inputs_i2v.json).
            security_stack: Optional SecurityStack needed by ReusableProcessingJob
                for VPC/KMS.
        """
        super().__init__(scope, construct_id)

        self.retrieval_config = retrieval_config
        if setup_configs is None:
            setup_configs = {}
        stack = Stack.of(self)
        region = stack.region
        account = stack.account
        ecr_pfx = ecr_prefix or prefix

        logger.info("Creating RetrievalConstruct with prefix: {}", prefix)

        # 1. Ingestion bucket via BucketTemplate (S3-managed encryption to avoid
        #    cyclic cross-stack dependency with SecurityStack's KMS key — the S3
        #    event notification creates a back-reference from SecurityStack)
        ingestion_bucket_name = f"{account}-{region}-{prefix}-retrieval-images-bucket"
        self.ingestion_bucket = BucketTemplate(
            self,
            f"{prefix}-RetrievalImagesBucket",
            bucket_name=ingestion_bucket_name,
            logging_bucket=logging_bucket,
        )
        logger.info("Created retrieval ingestion bucket: {}", ingestion_bucket_name)

        # 2. SQS queue for ingestion events (SQS-managed encryption to avoid
        #    cross-stack KMS cycle; DLQ with 14-day retention)
        self.sqs_queue = SqsQueueTemplate(
            self,
            f"{prefix}-RetrievalIngestQueue",
            queue_name=f"{prefix}-retrieval-ingest-queue",
            visibility_timeout_seconds=retrieval_config.sqs_visibility_timeout_seconds,
            max_receive_count=retrieval_config.sqs_max_receive_count,
        )
        logger.info("Created retrieval SQS queue")

        # 3. Ingest Lambda (VPC-deployed, KMS-encrypted env vars)
        self.ingest_lambda = LambdaTemplate(
            self,
            f"{prefix}-RetrievalIngestLambda",
            function_name=f"{prefix}-retrieval-ingest",
            lambda_path="retrieval_ingest",
            description=(
                "Downloads images from S3, generates Bedrock Titan embeddings, indexes to OpenSearch Serverless"
            ),
            timeout=retrieval_config.ingest_lambda_timeout_seconds,
            memory_size=retrieval_config.ingest_lambda_memory_mb,
            reserved_concurrent_executions=retrieval_config.ingest_lambda_reserved_concurrency,
            env_vars={
                "RETRIEVAL_BUCKET_NAME": ingestion_bucket_name,
                "AOSS_INDEX_NAME": retrieval_config.index_name,
                "EMBEDDING_MODEL_ID": retrieval_config.embedding_model_id,
                "EMBEDDING_DIMENSION": str(retrieval_config.embedding_dimension),
            },
        )
        logger.info("Created retrieval ingest Lambda")

        # 3b. Load Test Lambda (no VPC — AOSS has public network policy)
        self.load_test_lambda = LambdaTemplate(
            self,
            f"{prefix}-RetrievalLoadTestLambda",
            function_name=f"{prefix}-retrieval-load-test",
            lambda_path="retrieval_load_test",
            description="Load test for retrieval ingestion pipeline",
            timeout=900,
            memory_size=1024,
            env_vars={
                "RETRIEVAL_BUCKET_NAME": ingestion_bucket_name,
                "INGEST_QUEUE_URL": self.sqs_queue.queue.queue_url,
                "INGEST_FUNCTION_NAME": f"{prefix}-retrieval-ingest-Lambda",
                "AOSS_INDEX_NAME": retrieval_config.index_name,
            },
        )
        logger.info("Created retrieval load test Lambda")

        # 3c. Setup processing jobs (created BEFORE OpenSearch so real role ARNs
        #     can be passed directly to principal_arns — no string-based ARNs needed)
        self.setup_jobs: dict[str, ReusableProcessingJob] = {}
        self.setup_trigger_lambdas: dict[str, LambdaTemplate] = {}
        for setup_name, setup_cfg in setup_configs.items():
            if security_stack is None:
                raise ValueError("security_stack is required when setup_configs is provided")

            setup_clean = setup_name.replace("_", "-")
            ecr_repo_name = f"{ecr_pfx}/processing/{setup_clean}"
            ecr_image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{ecr_repo_name}"
            ecr_repo_arn = f"arn:aws:ecr:{region}:{account}:repository/{ecr_repo_name}"

            ecr_pull_policy = iam.ManagedPolicy(
                self,
                f"{prefix}-{setup_name}-EcrPullPolicy",
                managed_policy_name=f"{prefix}-{setup_clean}-setup-ecr-pull-policy",
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
                f"{prefix}-{setup_name}-EcrAuthPolicy",
                managed_policy_name=f"{prefix}-{setup_clean}-setup-ecr-auth-policy",
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=["ecr:GetAuthorizationToken"],
                        resources=["*"],
                    )
                ],
            )

            # Build output buckets dict: only if output_bucket is provided
            job_output_buckets: dict[str, IoBucketConfig] = {}
            if output_bucket is not None:
                job_output_buckets["output"] = IoBucketConfig(bucket_template=output_bucket)

            job = ReusableProcessingJob(
                self,
                f"{prefix}-{setup_name}-SetupJob",
                job_name=f"{prefix}-{setup_name}",
                input_buckets={},
                output_buckets=job_output_buckets,
                security_stack=security_stack,
                lambda_trigger=True,
                cfg=setup_cfg,
                environment={
                    "RETRIEVAL_BUCKET_NAME": ingestion_bucket_name,
                    "AOSS_INDEX_NAME": retrieval_config.index_name,
                    "DATASET_URL": setup_cfg.dataset_url,
                    "DATASET_SCRIPT": setup_cfg.dataset_script,
                    "NUM_PROMPTS": str(setup_cfg.num_prompts),
                    "TEST_IMAGE_COUNT": str(setup_cfg.test_image_count),
                    **setup_cfg.Environment,
                },
                ecr_image_uri=ecr_image_uri,
                ecr_pull_policy=ecr_pull_policy,
                ecr_auth_policy=ecr_auth_policy,
            )

            # Attach existing managed policies for S3 and SSM access
            job.role.add_managed_policy(self.ingestion_bucket.read_write_policy)

            # Inline AOSS data plane access (same pattern as ingest Lambda)
            aoss_collection_arn = f"arn:aws:aoss:{region}:{account}:collection/*"
            aoss_policy = iam.ManagedPolicy(
                self,
                f"{prefix}-{setup_name}-AossPolicy",
                managed_policy_name=f"{prefix}-{setup_clean}-setup-aoss-policy",
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=["aoss:APIAccessAll"],
                        resources=[aoss_collection_arn],
                    ),
                ],
            )
            job.role.add_managed_policy(aoss_policy)

            # Supplemental Bedrock policy for Strands Agent
            # Derive ARN from model ID — same pattern as vrag_llm in pipeline.py:
            # strip version suffix, use wildcard, include both inference-profile
            # and foundation-model ARN types.
            model_id = "us.amazon.nova-2-lite-v1:0"
            base_model = model_id.split(":")[0]  # "us.amazon.nova-2-lite-v1"
            parts = base_model.split(".", 1)
            foundation_model = parts[1] if len(parts) == 2 and len(parts[0]) <= 2 else base_model
            bedrock_resources = [
                f"arn:aws:bedrock:*::foundation-model/{foundation_model}*",
                f"arn:aws:bedrock:*:*:inference-profile/{base_model}*",
            ]
            bedrock_policy = iam.ManagedPolicy(
                self,
                f"{prefix}-{setup_name}-BedrockPolicy",
                managed_policy_name=f"{prefix}-{setup_clean}-setup-bedrock-policy",
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=[
                            "bedrock:InvokeModel",
                            "bedrock:InvokeModelWithResponseStream",
                        ],
                        resources=bedrock_resources,
                    ),
                ],
            )
            job.role.add_managed_policy(bedrock_policy)

            self.setup_jobs[setup_name] = job
            self.setup_trigger_lambdas[setup_name] = job.trigger_lambda
            logger.info("Created setup processing job: {}", setup_name)

        # 4. OpenSearch Serverless collection (VECTORSEARCH)
        # Include both the ingest Lambda role and the SageMaker processing job
        # role (created later in PipelineStack) in the data access policy.
        # The SageMaker role name follows the convention: {prefix}-retrieval-ProcessingJob-sagemaker-role
        sagemaker_role_arn = f"arn:aws:iam::{account}:role/{prefix}-retrieval-ProcessingJob-sagemaker-role"
        self.opensearch = OpenSearchServerlessConstruct(
            self,
            f"{prefix}-RetrievalOss",
            collection_name=retrieval_config.collection_name,
            prefix=prefix,
            ssm_parameter_name=f"/{prefix}/retrieval/aoss-endpoint",
            principal_arns=[
                self.ingest_lambda.function_role.role_arn,
                self.load_test_lambda.function_role.role_arn,
                sagemaker_role_arn,
                *[job.role.role_arn for job in self.setup_jobs.values()],
            ],
        )
        logger.info("Created OpenSearch Serverless collection")

        # 5. Inject AOSS endpoint into Lambda environment
        self.ingest_lambda.lambda_function.add_environment("AOSS_ENDPOINT", self.opensearch.collection_endpoint)
        self.load_test_lambda.lambda_function.add_environment("AOSS_ENDPOINT", self.opensearch.collection_endpoint)

        # 5b. Inject AOSS endpoint SSM parameter name into setup jobs
        #     (same pattern as the retrieval step in PipelineStack — avoids
        #     cross-stack CFn token issues with processing job definitions)
        for job in self.setup_jobs.values():
            job.definition_model.Arguments.Environment["AOSS_ENDPOINT_SSM"] = f"/{prefix}/retrieval/aoss-endpoint"
            job.update_definition()
            # Attach SSM read policy for AOSS endpoint parameter
            job.role.add_managed_policy(self.opensearch.endpoint_parameter.read_policy)

        # 6. S3 bucket notification → SQS queue (images/ prefix only)
        self.ingestion_bucket.bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3_notifications.SqsDestination(self.sqs_queue.queue),
            s3.NotificationKeyFilter(prefix="images/"),
        )

        # 7. SQS → Lambda event source with batch item failure reporting
        self.ingest_lambda.lambda_function.add_event_source(
            lambda_event_sources.SqsEventSource(
                self.sqs_queue.queue,
                batch_size=10,
                report_batch_item_failures=True,
            )
        )

        # 8. IAM: Ingest Lambda permissions (least-privilege)
        # S3 read-write via BucketTemplate managed policy
        self.ingest_lambda.function_role.add_managed_policy(self.ingestion_bucket.read_write_policy)

        # SQS consume
        self.ingest_lambda.function_role.add_managed_policy(self.sqs_queue.consume_policy)

        # Bedrock InvokeModel on Titan embedding model
        self.ingest_lambda.function_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["bedrock:InvokeModel"],
                resources=[f"arn:aws:bedrock:{region}::foundation-model/{retrieval_config.embedding_model_id}"],
            )
        )

        # AOSS data plane access
        self.ingest_lambda.function_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["aoss:APIAccessAll"],
                resources=[self.opensearch.collection.attr_arn],
            )
        )

        # SSM parameter read (AOSS endpoint)
        self.ingest_lambda.function_role.add_managed_policy(self.opensearch.endpoint_parameter.read_policy)

        # 8b. IAM: Load Test Lambda permissions
        self.load_test_lambda.function_role.add_managed_policy(self.ingestion_bucket.read_write_policy)
        self.load_test_lambda.function_role.add_managed_policy(self.sqs_queue.consume_policy)
        self.load_test_lambda.function_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["cloudwatch:GetMetricData"],
                resources=["*"],
            )
        )
        self.load_test_lambda.function_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["aoss:APIAccessAll"],
                resources=[self.opensearch.collection.attr_arn],
            )
        )
        self.load_test_lambda.function_role.add_managed_policy(self.opensearch.endpoint_parameter.read_policy)

        # 9. CloudFormation Outputs
        CfnOutput(
            stack,
            f"{prefix}-RetrievalBucketOutput",
            value=self.ingestion_bucket.bucket.bucket_name,
            description="S3 bucket for retrieval images (images/ prefix triggers ingestion)",
        )

        CfnOutput(
            stack,
            f"{prefix}-RetrievalCollectionEndpointOutput",
            value=self.opensearch.collection_endpoint,
            description="OpenSearch Serverless collection endpoint",
        )

        logger.info("RetrievalConstruct initialized successfully")

        # 10. CDK Nag suppressions
        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "Lambda roles use AWS managed policies for basic execution and VPC access",
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Bedrock InvokeModel and AOSS APIAccessAll require wildcard on model/collection resource paths; S3 bucket policies need wildcard for object-level access",
                },
                {
                    "id": "AwsSolutions-SQS4",
                    "reason": "SQS queue is KMS-encrypted and used internally via Lambda event source; SSL is enforced at the bucket notification level",
                },
            ],
            apply_to_children=True,
        )
