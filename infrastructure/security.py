"""
Security infrastructure stack

This module defines the SecurityStack which provides foundational security
infrastructure including VPC, KMS encryption keys, security groups, and
logging configurations. It establishes the security perimeter for all
other components in the application.
"""

from aws_cdk import RemovalPolicy, Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from constructs import Construct
from loguru import logger


class SecurityStack(Stack):
    """
    CDK Stack that provides security infrastructure

    This stack creates and configures:
    - VPC with public and private subnets
    - KMS key for encryption at rest
    - Security groups with minimal required access
    - VPC Flow Logs for network monitoring
    - IAM policies for KMS key access

    The stack follows AWS security best practices including:
    - Network isolation using private subnets
    - Encryption at rest using customer-managed KMS keys
    - Least privilege access principles
    - Comprehensive logging and monitoring

    Attributes:
        vpc: The VPC created for network isolation
        kms_key: Customer-managed KMS key for encryption
        kms_key_policy: IAM policy for KMS key access
        flow_log_group: CloudWatch log group for VPC flow logs
        security_group: Security group with minimal required access
        subnet_ids: List of private subnet IDs for resource placement
    """

    def __init__(self, scope: Construct, construct_id: str, prefix: str = "dev", **kwargs) -> None:
        """
        Initialize the SecurityStack with VPC, KMS, and security configurations.

        Args:
            scope: The parent construct (typically the CDK App)
            construct_id: Unique identifier for this stack
            prefix: Resource name prefix for multi-environment support (default: "dev")
            **kwargs: Additional keyword arguments passed to the parent Stack
        """
        super().__init__(scope, construct_id, **kwargs)

        logger.info(f"Creating SecurityStack with prefix: {prefix}")

        # Create VPC with public and private subnets across 2 AZs
        self.vpc = ec2.Vpc(
            self,
            f"{prefix}-Vpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="Private", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, cidr_mask=24),
                ec2.SubnetConfiguration(name="Public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
            ],
        )

        # Create customer-managed KMS key for encryption at rest
        self.kms_key = kms.Key(
            self,
            f"{prefix}-KmsKey",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.DESTROY,  # deleted upon stack destroy
            alias=f"/{prefix}-comfyui",
            description=f"Shared KMS key for all encrypted resources for {prefix} comfyui app",
        )

        # Create IAM policy for KMS key access
        self.kms_key_policy = iam.ManagedPolicy(
            self,
            f"{prefix}-KmsKeyPolicy",
            managed_policy_name=f"{prefix}-kms-key-policy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["kms:Decrypt", "kms:DescribeKey", "kms:GenerateDataKey"],
                    resources=[self.kms_key.key_arn],
                )
            ],
        )

        # Add resource policy to KMS key for AWS service access
        self.kms_key.add_to_resource_policy(
            iam.PolicyStatement(
                actions=["kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:DescribeKey"],
                principals=[iam.ServicePrincipal("s3.amazonaws.com"), iam.ServicePrincipal("sqs.amazonaws.com")],
                resources=["*"],  # Shared key
            )
        )

        # Create CloudWatch log group for VPC flow logs
        self.flow_log_group = logs.LogGroup(
            self,
            f"{prefix}-VPCFlowLogs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,  # deleted upon stack destroy
            encryption_key=self.kms_key,
        )

        # Grant KMS permissions to CloudWatch Logs and S3 services
        self.kms_key.grant_encrypt_decrypt(iam.ServicePrincipal(f"logs.{Stack.of(self).region}.amazonaws.com"))
        self.kms_key.grant_encrypt_decrypt(iam.ServicePrincipal("s3.amazonaws.com"))

        # Enable VPC Flow Logs for network monitoring
        self.vpc.add_flow_log(
            f"{prefix}-FlowLog", destination=ec2.FlowLogDestination.to_cloud_watch_logs(self.flow_log_group)
        )

        # Create security group with full outbound internet access
        self.security_group = ec2.SecurityGroup(
            self,
            f"{prefix}-SecurityGroup",
            vpc=self.vpc,
            description=f"Security group for {prefix}",
            allow_all_outbound=True,
            security_group_name=f"{prefix}-security-group",
        )

        # Allow all inbound traffic from the same security group.
        # Required for multi-instance SageMaker Processing Jobs where
        # worker nodes need to communicate with the leader node.
        self.security_group.add_ingress_rule(
            peer=self.security_group,
            connection=ec2.Port.all_traffic(),
            description="Allow inter-node communication for multi-instance jobs",
        )

        # Get private subnet IDs for use by other stacks
        self.subnet_ids = self.vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS).subnet_ids

        # S3 Gateway VPC Endpoint — routes S3 traffic over the AWS backbone
        # instead of through the NAT Gateway, improving throughput and reducing cost
        self.s3_endpoint = self.vpc.add_gateway_endpoint(
            f"{prefix}-S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
        )
        logger.info("Created S3 Gateway VPC endpoint")
