> **Navigation:** [← Main README](../README.md) | [Extending Guide](../docs/EXTENDING.md)

# project_constructs/

Reusable CDK L3 constructs that all infrastructure stacks compose from. Stacks never create raw AWS resources directly — they describe *what* they need and the construct handles *how* it's built securely.

---

## Why L3 Constructs?

Each construct encapsulates one or more AWS resources together with:

- **Pre-built IAM managed policies** — granular read, write, and read-write policies that consuming stacks attach to their roles, enforcing least-privilege without boilerplate
- **Security defaults** — KMS encryption, VPC integration, access logging, and lifecycle rules baked in so every resource starts secure
- **cdk-nag suppressions** — known-acceptable warnings suppressed with documented reasons inside the construct, keeping stack code clean

Adding a new stack or pipeline step means composing existing constructs rather than re-implementing encryption, IAM, and compliance logic from scratch.

---

## Constructs

### Storage

| Construct | What it creates |
|---|---|
| `BucketTemplate` | S3 bucket with KMS encryption, SSL enforcement, versioning, access logging, 90-day expiry, and read/write/read-write managed policies |
| `DynamoDbTemplate` | DynamoDB table with PAY_PER_REQUEST billing, KMS encryption, point-in-time recovery, configurable keys/GSIs, and reader/writer managed policies |
| `SsmParameter` | SSM Parameter Store parameter (advanced tier) with read and write managed policies |

### Compute

| Construct | What it creates |
|---|---|
| `LambdaTemplate` | Lambda function (zip or container image) with VPC integration, CloudWatch logs, and automatic `schema/` bundling for zip deploys |
| `ReusableProcessingJob` | Complete SageMaker Processing Job definition with IAM role, bucket policies, optional trigger Lambda, and Pydantic-validated job config |
| `CodeBuildProject` | CodeBuild project with VPC integration, KMS-encrypted logs, and privileged mode for Docker builds |

### Pipeline & Orchestration

| Construct | What it creates |
|---|---|
| `PipelineConstruct` | SageMaker Pipeline from a list of processing jobs + DAG. Injects execution ID, rewrites S3 output URIs per-run, and wires step dependencies |
| `CodePipelineTemplate` | CodePipeline V2 with S3 source, configurable stages, and parallel execution mode |

### Container Registry

| Construct | What it creates |
|---|---|
| `EcrRepository` | ECR repo with image scanning, 10-image lifecycle, KMS encryption, and push/pull/auth managed policies |

### Messaging

| Construct | What it creates |
|---|---|
| `SqsTemplate` | SQS queue + DLQ with KMS encryption and read/write managed policies |
| `SnsTemplate` | KMS-encrypted SNS topic with publish and subscribe managed policies |

### Human Review (A2I)

| Construct | What it creates |
|---|---|
| `a2i/` | A2I flow definitions (via AwsCustomResource) and Liquid HTML task UI templates for video, image, and audio review |
| `WorkteamConstruct` | SageMaker private workteam backed by a Cognito user pool group |
| `CognitoConstruct` | Cognito user pool and client for A2I workforce authentication |

### Retrieval

| Construct | What it creates |
|---|---|
| `OpenSearchConstruct` | AOSS collection (VECTORSEARCH type) with encryption, network, and data access policies, plus kNN index creation via custom resource |
| `RetrievalConstruct` | Composes AOSS, SQS, ingest/query/load-test Lambdas, and optional setup jobs into the complete retrieval subsystem |
