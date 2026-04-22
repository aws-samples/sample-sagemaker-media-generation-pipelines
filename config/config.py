"""
Configuration management for processing jobs.

This module provides Pydantic models and utilities for loading and validating
configuration parameters for SageMaker processing jobs. It supports YAML-based
configuration files with strict validation for instance types, resource limits,
and container specifications.
"""

import os
from typing import Literal

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContainerConfig(BaseModel):
    """
    Configuration model for SageMaker processing job container specifications.

    This model defines the structure and validation rules for container
    configuration including entrypoint, arguments, instance specifications,
    and environment variables.

    Attributes:
        ContainerEntrypoint: List of strings defining the container entrypoint command
        ContainerArguments: List of strings for container command arguments
        InstanceCount: Number of instances to use (1-10)
        InstanceType: AWS instance type, restricted to supported ML instance types
        VolumeSizeInGB: EBS volume size in GB (50-1000)
        Environment: Dictionary of environment variables for the container
    """

    ContainerEntrypoint: list[str] = Field(
        default_factory=list, description="Container entrypoint command as a list of strings"
    )
    ContainerArguments: list[str] = Field(
        min_length=1, description="Container arguments as a list of strings (SageMaker requires at least 1)"
    )
    InstanceCount: int = Field(ge=1, le=40, description="Number of instances for the processing job")
    InstanceType: Literal[
        "ml.c5.xlarge",
        "ml.g4dn.2xlarge",
        "ml.g5.xlarge",
        "ml.g5.8xlarge",
        "ml.m5.xlarge",
        "ml.m5.2xlarge",
        "ml.m5.4xlarge",
    ]

    VolumeSizeInGB: int = Field(ge=50, le=125, description="EBS volume size in GB for the processing job")
    Environment: dict[str, str] = Field(default_factory=dict, description="Environment variables for the container")
    models_prefix: list[str] = Field(
        default_factory=list,
        description="S3 key prefixes inside the models bucket for this step's models",
    )
    ecr_image: str = Field(
        default="",
        description="ECR image to use if different from step name (e.g. vbench_t2v uses 'vbench' image)",
    )
    input_channel: str = Field(
        default="input",
        description="SageMaker input channel name for the primary input (mounted at /opt/ml/processing/input/<channel>)",
    )
    include_data_input: bool = Field(
        default=False,
        description="If true, also mount the DataStack input bucket as an 'input' channel (for accessing inputs.json)",
    )
    num_assets_per_prompt: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of assets (images/videos/audio) to generate per prompt. Injected as NUM_ASSETS_PER_PROMPT env var.",
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        strict=True,
    )


class S3Download(BaseModel):
    """
    Configuration for a single S3 download item.

    Attributes:
        url: HTTP/HTTPS URL to download from
        path: Destination path within the S3 models bucket (e.g. "wan_22_i2v/diffusion_models/model.safetensors")
        extract: If true, treat the downloaded file as a zip archive and extract
                 its contents into the directory specified by ``path``, then
                 delete the zip file.
    """

    url: str = Field(description="HTTP/HTTPS URL to download from")
    path: str = Field(description="Destination path within the S3 models bucket")
    extract: bool = Field(
        default=False,
        description="If true, extract the downloaded zip into the path directory and remove the archive",
    )

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )


class DynamoDBConfig(BaseModel):
    """
    Configuration for the DynamoDB table key schema.

    Attributes:
        partition_key: Name of the partition key attribute
        sort_key: Name of the sort key attribute
    """

    partition_key: str = Field(
        default="id",
        description="Name of the partition key attribute",
    )
    sort_key: str = Field(
        default="step",
        description="Name of the sort key attribute",
    )

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )


class ModelDownloadConfig(ContainerConfig):
    """
    Configuration for the model download SageMaker Processing Job.

    Extends ContainerConfig so it can be passed directly to
    ReusableProcessingJob.  Provides sensible defaults for the
    download-only workload (no GPU needed, large volume).

    Attributes:
        MaxRuntimeInSeconds: Maximum runtime before job is terminated
    """

    # Override ContainerConfig defaults for model download
    InstanceCount: int = Field(default=1, ge=1, le=40)
    InstanceType: Literal[
        "ml.c5.xlarge",
        "ml.g4dn.2xlarge",
        "ml.g5.xlarge",
        "ml.g5.8xlarge",
        "ml.m5.xlarge",
        "ml.m5.2xlarge",
    ] = "ml.m5.xlarge"
    VolumeSizeInGB: int = Field(default=125, ge=50, le=125)
    ContainerEntrypoint: list[str] = Field(
        default_factory=lambda: ["python3", "main.py"],
    )
    ContainerArguments: list[str] = Field(
        default_factory=lambda: ["--download"],
    )

    MaxRuntimeInSeconds: int = Field(
        default=86400,
        ge=600,
        le=604800,
        description="Maximum runtime in seconds (default 24h, max 7 days)",
    )


class A2IConfig(BaseModel):
    """
    Configuration for an Amazon A2I (Augmented AI) human review step.

    Attributes:
        workforce_arn: ARN of the SageMaker private workforce
        task_template_s3_uri: S3 URI of the Liquid HTML task template
        media_type: Type of media to review (image, video, audio)
        task_title: Title shown to human reviewers
        task_description: Description shown to human reviewers
        task_count: Number of human reviewers per object
        task_timeout_seconds: Time each reviewer has to complete a task
        max_concurrent_tasks: Maximum concurrent human review tasks
    """

    workteam_name: str = Field(
        default="",
        description="Name for the private workteam. If empty, derived from the A2I config key name.",
    )
    task_template_s3_uri: str = Field(
        default="",
        description="S3 URI of a custom Liquid HTML task template. If empty, a default template is generated.",
    )
    media_type: Literal["image", "video", "audio"] = Field(
        default="video",
        description="Type of media to present to human reviewers",
    )
    task_title: str = Field(
        default="Review generated media",
        description="Title shown to human reviewers in the worker portal",
    )
    task_description: str = Field(
        default="Review the generated media and rate its quality",
        description="Description shown to human reviewers",
    )
    task_count: int = Field(
        default=1,
        ge=1,
        le=9,
        description="Number of human reviewers per object",
    )
    task_timeout_seconds: int = Field(
        default=3600,
        ge=60,
        le=28800,
        description="Time in seconds each reviewer has to complete a task",
    )
    max_concurrent_tasks: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum number of concurrent human review tasks",
    )

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )


class LambdaStepConfig(BaseModel):
    """
    Configuration for a Lambda step in the SageMaker Pipeline.

    Lambda steps invoke an AWS Lambda function as part of the pipeline.
    Used for lightweight tasks like submitting A2I reviews.

    Attributes:
        lambda_path: Directory name under lambdas/ containing the Lambda code
        a2i_name: Name of the A2I config entry this step submits to (optional)
        media_type: Media type for A2I review (image, video, audio)
    """

    lambda_path: str = Field(
        description="Directory name under lambdas/ containing the Lambda code",
    )
    a2i_name: str = Field(
        default="",
        description="Name of the A2I config entry this step submits to. "
        "When set, the Lambda receives FLOW_DEFINITION_ARN and SOURCE_BUCKET env vars.",
    )
    media_type: Literal["image", "video", "audio"] = Field(
        default="video",
        description="Media type for A2I review filtering",
    )

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )


class SetupConfig(ContainerConfig):
    """
    Configuration for a standalone setup processing job.

    Extends ContainerConfig to add dataset-specific fields for setup jobs
    that download datasets, upload images, and generate prompts.

    Attributes:
        dataset_url: HuggingFace dataset URL (e.g. "1aurent/unsplash-lite")
        dataset_script: Filename of the dataset-specific loader script inside the container
        num_prompts: Number of video prompts to generate (1-50000)
        test_image_count: Maximum number of images to process (1-25000, default 25000)
    """

    dataset_url: str
    dataset_script: str = Field(
        description="Filename of the dataset-specific loader script inside the container (e.g. 'unsplash.py')"
    )
    num_prompts: int = Field(ge=1, le=50000)
    test_image_count: int = Field(ge=1, le=25000, default=25000)

    @model_validator(mode="after")
    def _check_prompts_le_images(self) -> "SetupConfig":
        if self.num_prompts > self.test_image_count:
            raise ValueError("num_prompts cannot exceed test_image_count")
        return self


class PipelineConfig(BaseModel):
    """
    Top-level configuration for the SageMaker Pipeline application.

    Contains per-step container configurations keyed by step name,
    S3 download items with URL and destination path, and a resource
    prefix for multi-environment support.

    Attributes:
        construct_id: Prefix for all resource names (kebab-case, default "dev")
        s3_downloads: List of S3 download items with url and path
        model_download: Configuration for the model download processing job
        steps: Per-step container configurations keyed by step name
    """

    construct_id: str = Field(
        default="dev",
        pattern=r"^[a-z][a-z0-9-]*$",
        description="Prefix for all resource names (kebab-case)",
    )
    s3_downloads: list[S3Download] = Field(
        default_factory=list,
        description="List of S3 download items with url and destination path",
    )
    dynamodb: DynamoDBConfig = Field(
        default_factory=DynamoDBConfig,
        description="DynamoDB table key schema configuration",
    )
    steps: dict[str, ContainerConfig] = Field(
        description="Per-step container configurations keyed by step name",
    )
    pipeline_graph: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Pipeline DAG: step name -> list of dependency step names. "
        "Empty list means no dependencies (root step).",
    )
    a2i: dict[str, A2IConfig] = Field(
        default_factory=dict,
        description="Optional A2I human review steps keyed by step name. "
        "Each entry creates a flow definition for human review of media assets.",
    )
    lambda_steps: dict[str, LambdaStepConfig] = Field(
        default_factory=dict,
        description="Optional Lambda steps keyed by step name. "
        "Each entry creates a Lambda function invoked as a pipeline step.",
    )
    setup: dict[str, SetupConfig] = Field(
        default_factory=dict,
        description="Optional standalone setup processing jobs keyed by job name. "
        "Each entry creates a ReusableProcessingJob with lambda_trigger=True inside RetrievalConstruct.",
    )
    retrieval: str | None = Field(
        default=None,
        description="Filename of a retrieval config in config/retrieval/ (e.g. 'retrieval.yaml'). "
        "When set, the RetrievalStack is created and the retrieval step can query OpenSearch.",
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        strict=True,
    )

    @model_validator(mode="after")
    def _check_a2i_name_references(self) -> "PipelineConfig":
        """Every lambda_step.a2i_name must reference a key in the a2i dict."""
        for ls_name, ls_cfg in self.lambda_steps.items():
            if ls_cfg.a2i_name and ls_cfg.a2i_name not in self.a2i:
                raise ValueError(
                    f"lambda_steps['{ls_name}'].a2i_name='{ls_cfg.a2i_name}' "
                    f"does not match any key in a2i (available: {list(self.a2i.keys())})"
                )
        return self


class RetrievalConfig(BaseModel):
    """
    Configuration model for the image retrieval pipeline.

    Validates retrieval-related parameters from YAML files in the
    ``config/retrieval/`` directory, including OpenSearch collection
    settings, embedding model, SQS queue tuning, and Lambda resource
    limits.

    Attributes:
        collection_name: AOSS collection name (3-32 lowercase alphanumeric/hyphen chars).
        embedding_model_id: Bedrock embedding model ID.
        index_name: OpenSearch index name for kNN queries.
        query_k: Number of nearest neighbours to retrieve (1-100).
        sqs_visibility_timeout_seconds: SQS visibility timeout (30-43200).
        sqs_max_receive_count: Max SQS retries before DLQ (1-10).
        ingest_lambda_timeout_seconds: Ingest Lambda timeout (1-900).
        ingest_lambda_memory_mb: Ingest Lambda memory in MB (128-10240).
    """

    collection_name: str = Field(
        pattern=r"^[a-z0-9-]{3,32}$",
        description="AOSS collection name (3-32 lowercase alphanumeric/hyphen characters)",
    )
    embedding_model_id: Literal[
        "amazon.titan-embed-image-v1",
        "amazon.nova-2-multimodal-embeddings-v1:0",
    ] = Field(
        default="amazon.titan-embed-image-v1",
        description="Bedrock embedding model ID",
    )
    embedding_dimension: Literal[256, 384, 1024, 3072] = Field(
        default=1024,
        description="Embedding vector dimension (Titan: 1024 only; Nova: 256, 384, 1024, or 3072)",
    )
    index_name: str = Field(
        description="OpenSearch index name for kNN queries",
    )
    query_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of nearest neighbours to retrieve",
    )
    sqs_visibility_timeout_seconds: int = Field(
        ge=30,
        le=43200,
        description="SQS visibility timeout in seconds",
    )
    sqs_max_receive_count: int = Field(
        ge=1,
        le=10,
        description="Max SQS receive count before routing to DLQ",
    )
    ingest_lambda_timeout_seconds: int = Field(
        ge=1,
        le=900,
        description="Ingest Lambda timeout in seconds",
    )
    ingest_lambda_memory_mb: int = Field(
        ge=128,
        le=10240,
        description="Ingest Lambda memory in MB",
    )
    ingest_lambda_reserved_concurrency: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Ingest Lambda reserved concurrency (None = unreserved)",
    )

    model_config = ConfigDict(extra="forbid", strict=True)

    @model_validator(mode="after")
    def _validate_embedding_dimension(self) -> "RetrievalConfig":
        """Titan only supports 1024-dim embeddings; Nova supports 256/384/1024/3072."""
        if self.embedding_model_id == "amazon.titan-embed-image-v1" and self.embedding_dimension != 1024:
            raise ValueError(
                f"amazon.titan-embed-image-v1 only supports embedding_dimension=1024, got {self.embedding_dimension}"
            )
        return self


class CicdConfig(BaseModel):
    """
    Configuration model for CI/CD pipeline settings.

    Loaded from config/cicd/cicd.yaml. Controls pipeline behavior including
    build compute, timeouts, test/deploy commands, source excludes, and
    optional stages (pipeline trigger, A2I smoke test).

    Attributes:
        enabled: Toggle pipeline instantiation in app.py
        notification_email: Optional email for pipeline failure notifications
        source_excludes: Glob patterns excluded from the S3 source asset
        compute_type: CodeBuild compute size
        timeout_minutes: Build timeout in minutes (1–480)
        test_commands: Maps each pipeline config filename to its pytest command string
        rollback: CloudFormation rollback behavior
        test_a2i: Enable the A2I Smoke Test stage
        pipeline_configs: Allowlist of pipeline config filenames to deploy
    """

    enabled: bool = True
    notification_email: str | None = None
    source_excludes: list[str] = Field(
        default_factory=lambda: [
            ".venv/",
            "cdk.out/",
            ".git/",
            "__pycache__/",
            ".hypothesis/",
            "node_modules/",
            "security/",
        ]
    )
    compute_type: Literal["SMALL", "MEDIUM", "LARGE", "X2_LARGE"] = "SMALL"
    timeout_minutes: int = Field(default=60, ge=1, le=480)
    test_commands: dict[str, str] = Field(
        default_factory=lambda: {
            "config_vrag.yaml": "uv run pytest tests/unit/ -x --no-header -q -n auto -m 'core or cicd or retrieval or processing or model_validation or steps_vrag or steps_t2v or steps_i2v or integration'",
        },
        description="Maps each pipeline config filename to its pytest command string.",
    )
    rollback: bool = True
    test_a2i: bool = False
    shared_prefix: str = Field(
        default="dev",
        min_length=3,
        max_length=3,
        pattern=r"^[a-z][a-z0-9]{2}$",
        description="3-letter prefix for shared stacks. Kept short to stay within AOSS 32-char policy name limits.",
    )
    pipeline_configs: list[str] = Field(
        default_factory=lambda: ["config_vrag.yaml"],
        min_length=1,
    )
    input_data: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Maps each pipeline config filename to a list of paths "
        "under sample_input_data/ to upload to the input bucket.",
    )

    model_config = ConfigDict(extra="forbid", strict=True)

    @model_validator(mode="after")
    def _check_test_commands_coverage(self) -> "CicdConfig":
        missing = [cfg for cfg in self.pipeline_configs if cfg not in self.test_commands]
        if missing:
            raise ValueError(f"pipeline_configs entries missing from test_commands: {missing}")
        return self


def get_cicd_config(config_fp: str = "cicd.yaml") -> CicdConfig:
    """
    Load and validate CI/CD configuration from a YAML file.

    This function reads a YAML configuration file from the config/cicd
    directory and validates it against the CicdConfig model.

    Args:
        config_fp: Filename of the configuration file in the config/cicd directory.
                  Defaults to "cicd.yaml".

    Returns:
        CicdConfig: Validated CI/CD configuration object.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
        pydantic.ValidationError: If the configuration doesn't match the schema.
    """
    logger.info("Loading CI/CD config from: {}", config_fp)

    with open("./config/cicd/" + config_fp, encoding="utf-8") as f:
        raw = yaml.full_load(f)

    config = CicdConfig(**raw)

    logger.debug("CI/CD pipeline enabled: {}", config.enabled)
    logger.debug("Pipeline configs: {}", config.pipeline_configs)

    return config


def get_retrieval_config(config_fp: str = "retrieval.yaml") -> RetrievalConfig:
    """
    Load and validate retrieval configuration from a YAML file.

    This function reads a YAML configuration file from the config/retrieval
    directory and validates it against the RetrievalConfig model.

    Args:
        config_fp: Filename of the configuration file in the config/retrieval directory.
                  Defaults to "retrieval.yaml".

    Returns:
        RetrievalConfig: Validated retrieval configuration object.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
        pydantic.ValidationError: If the configuration doesn't match the schema.
    """
    logger.info("Loading retrieval config from: {}", config_fp)

    with open("./config/retrieval/" + config_fp, encoding="utf-8") as f:
        raw = yaml.full_load(f.read())

    config = RetrievalConfig(**raw)

    logger.debug("Retrieval collection name: {}", config.collection_name)
    logger.debug("Retrieval index name: {}", config.index_name)

    return config


def get_configs(config_fp: str = "config_vrag.yaml") -> ContainerConfig:
    """
    Load and validate configuration from a YAML file.

    This function reads a YAML configuration file from the config directory
    and validates it against the ContainerConfig model. It provides type
    checking and validation for all configuration parameters.

    Args:
        config_fp: Filename of the configuration file in the config directory.
                  Defaults to "config_vrag.yaml".

    Returns:
        ContainerConfig: Validated configuration object with all parameters.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
        pydantic.ValidationError: If the configuration doesn't match the schema.

    Example:
        >>> config = get_configs("my_config.yaml")
        >>> print(config.InstanceType)
        'ml.g5.4xlarge'
    """
    with open("./config/pipeline/" + config_fp, encoding="utf-8") as f:
        cfg = ContainerConfig(**yaml.full_load(f))

    return cfg


def get_pipeline_config(config_fp: str = "config_vrag.yaml") -> PipelineConfig:
    """
    Load and validate pipeline configuration from a YAML file.

    This function reads a YAML configuration file from the config directory
    and validates it against the PipelineConfig model. It provides type
    checking and validation for all pipeline parameters including per-step
    container configurations, EFS download URLs, and construct_id.

    Args:
        config_fp: Filename of the configuration file in the config directory.
                  Defaults to "config_vrag.yaml".

    Returns:
        PipelineConfig: Validated pipeline configuration object.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
        pydantic.ValidationError: If the configuration doesn't match the schema.
    """
    logger.info("Loading pipeline config from: {}", config_fp)

    with open("./config/pipeline/" + config_fp, encoding="utf-8") as f:
        raw_text = f.read()

    # Resolve ${ENV_VAR} references so configs can pull from .env
    raw_text = os.path.expandvars(raw_text)
    raw = yaml.full_load(raw_text)

    config = PipelineConfig(**raw)

    logger.debug("Parsed construct_id: {}", config.construct_id)
    logger.debug("Parsed step names: {}", list(config.steps.keys()))

    return config
