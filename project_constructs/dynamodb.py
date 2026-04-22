from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
)
from aws_cdk import (
    aws_dynamodb as dynamodb,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_kms as kms,
)
from cdk_nag import NagSuppressions
from constructs import Construct


class GsiConfig:
    """Configuration for a single Global Secondary Index."""

    def __init__(self, index_name: str, partition_key: str, sort_key: str | None = None) -> None:
        self.index_name = index_name
        self.partition_key = partition_key
        self.sort_key = sort_key


class DynamoDbTemplate(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        table_name: str,
        kms_key: kms.Key,
        partition_key: str,
        sort_key: str | None = None,
        gsi_name: str | None = None,
        gsi_partition_key: str | None = None,
        global_secondary_indexes: list[GsiConfig] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id)
        self.sort_key_attribute = None
        if sort_key is not None:
            self.sort_key_attribute = dynamodb.Attribute(name=sort_key, type=dynamodb.AttributeType.STRING)

        self.partition_key = partition_key
        self.sort_key = sort_key
        self.gsi_name = gsi_name
        self.gsi_partition_key = gsi_partition_key
        self.table = dynamodb.Table(
            self,
            f"{construct_id}-Dynamodb",
            table_name=f"{construct_id}-{table_name}",
            partition_key=dynamodb.Attribute(name=self.partition_key, type=dynamodb.AttributeType.STRING),
            sort_key=self.sort_key_attribute,
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=kms_key,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Legacy single-GSI support (backward compatible)
        if gsi_name and gsi_partition_key:
            self.table.add_global_secondary_index(
                index_name=gsi_name,
                partition_key=dynamodb.Attribute(name=gsi_partition_key, type=dynamodb.AttributeType.STRING),
                projection_type=dynamodb.ProjectionType.ALL,
            )

        # Additional GSIs via GsiConfig list
        for gsi in global_secondary_indexes or []:
            gsi_kwargs: dict = {
                "index_name": gsi.index_name,
                "partition_key": dynamodb.Attribute(name=gsi.partition_key, type=dynamodb.AttributeType.STRING),
                "projection_type": dynamodb.ProjectionType.ALL,
            }
            if gsi.sort_key:
                gsi_kwargs["sort_key"] = dynamodb.Attribute(name=gsi.sort_key, type=dynamodb.AttributeType.STRING)
            self.table.add_global_secondary_index(**gsi_kwargs)

        # Table + GSI ARNs for IAM policies (queries on GSIs need index ARN)
        table_and_index_arns = [
            self.table.table_arn,
            f"{self.table.table_arn}/index/*",
        ]

        self.reader_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-{table_name}-DynamoDBReaderPolicy",
            managed_policy_name=f"{construct_id}-{table_name}-DynamoDBReaderPolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "dynamodb:GetItem",
                        "dynamodb:BatchGetItem",
                        "dynamodb:Query",
                        "dynamodb:Scan",
                    ],
                    resources=table_and_index_arns,
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["kms:Decrypt", "kms:DescribeKey"],
                    resources=[kms_key.key_arn],
                ),
            ],
        )

        self.writer_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-{table_name}-DynamoDBWriterPolicy",
            managed_policy_name=f"{construct_id}-{table_name}-DynamoDBWriterPolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "dynamodb:PutItem",
                        "dynamodb:UpdateItem",
                        "dynamodb:BatchWriteItem",
                        "dynamodb:DeleteItem",
                    ],
                    resources=[self.table.table_arn],
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["kms:Decrypt", "kms:GenerateDataKey"],
                    resources=[kms_key.key_arn],
                ),
            ],
        )

        self.read_write_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-{table_name}-DynamoDBReadWritePolicy",
            managed_policy_name=f"{construct_id}-{table_name}-DynamoDBReadWritePolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "dynamodb:GetItem",
                        "dynamodb:BatchGetItem",
                        "dynamodb:Query",
                        "dynamodb:Scan",
                    ],
                    resources=table_and_index_arns,
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "dynamodb:PutItem",
                        "dynamodb:UpdateItem",
                        "dynamodb:BatchWriteItem",
                        "dynamodb:DeleteItem",
                    ],
                    resources=[self.table.table_arn],
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "kms:Decrypt",
                        "kms:GenerateDataKey",
                        "kms:DescribeKey",
                    ],
                    resources=[kms_key.key_arn],
                ),
            ],
        )

        CfnOutput(
            self,
            f"{construct_id}-Dynamodb-Arn",
            value=self.table.table_arn,
            description="Dynamodb",
        )

        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "DynamoDB read policies need wildcard on index/* to support GSI queries",
                },
            ],
            apply_to_children=True,
        )
