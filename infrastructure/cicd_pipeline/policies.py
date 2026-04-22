"""IAM policy factories for CI/CD CodeBuild projects.

Each function creates and returns an ``iam.ManagedPolicy`` scoped to
the minimum permissions needed by its CodeBuild action.
"""

from aws_cdk import Stack
from aws_cdk import aws_iam as iam
from constructs import Construct


def container_build_policy(scope: Construct, tag: str) -> iam.ManagedPolicy:
    """Policy for the ContainerBuild action (start/poll CodeBuild builds)."""
    return iam.ManagedPolicy(
        scope,
        f"{tag}-ContainerBuildPolicy",
        statements=[
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "codebuild:ListProjects",
                    "codebuild:StartBuild",
                    "codebuild:BatchGetBuilds",
                ],
                resources=["*"],
            ),
        ],
    )


def model_download_policy(
    scope: Construct,
    tag: str,
    bucket_arn: str,
) -> iam.ManagedPolicy:
    """Policy for the ModelDownload action (S3 read/write + KMS)."""
    return iam.ManagedPolicy(
        scope,
        f"{tag}-ModelDownloadPolicy",
        statements=[
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                    "s3:DeleteObject",
                    "s3:GetBucketLocation",
                    "s3:AbortMultipartUpload",
                    "s3:ListMultipartUploadParts",
                ],
                resources=[bucket_arn, f"{bucket_arn}/*"],
            ),
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["kms:GenerateDataKey", "kms:Decrypt"],
                resources=["*"],
            ),
        ],
    )


def upload_input_policy(
    scope: Construct,
    tag: str,
    bucket_arn: str,
) -> iam.ManagedPolicy:
    """Policy for the UploadInput action (S3 put + KMS)."""
    return iam.ManagedPolicy(
        scope,
        f"{tag}-UploadInputPolicy",
        statements=[
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                resources=[bucket_arn, f"{bucket_arn}/*"],
            ),
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["kms:GenerateDataKey", "kms:Decrypt"],
                resources=["*"],
            ),
        ],
    )


def trigger_pipeline_policy(
    scope: Construct,
    tag: str,
    region: str,
    account: str,
    cfg_prefix: str,
) -> iam.ManagedPolicy:
    """Policy for the TriggerPipeline action (Lambda invoke + CFN describe)."""
    return iam.ManagedPolicy(
        scope,
        f"{tag}-TriggerPipelinePolicy",
        statements=[
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["lambda:InvokeFunction"],
                resources=[
                    f"arn:aws:lambda:{region}:{account}:function:{cfg_prefix}-*",
                ],
            ),
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["cloudformation:DescribeStacks"],
                resources=[
                    f"arn:aws:cloudformation:{region}:{account}:stack/{cfg_prefix}-PipelineStack/*",
                ],
            ),
        ],
    )


def container_step_build_policy(scope: Construct, tag: str, ecr_repo_arn: str) -> iam.ManagedPolicy:
    """Policy for an individual container build action (ECR push + SageMaker DLC pull)."""
    region = Stack.of(scope).region
    sagemaker_dlc_account = "763104351884"
    return iam.ManagedPolicy(
        scope,
        f"{tag}-StepBuildPolicy",
        statements=[
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:PutImage",
                    "ecr:InitiateLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload",
                    "ecr:DescribeImages",
                ],
                resources=[ecr_repo_arn],
            ),
            # Cross-account pull for SageMaker DLC base images (e.g. pytorch-inference)
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:BatchCheckLayerAvailability",
                ],
                resources=[
                    f"arn:aws:ecr:{region}:{sagemaker_dlc_account}:repository/*",
                ],
            ),
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],
            ),
        ],
    )
