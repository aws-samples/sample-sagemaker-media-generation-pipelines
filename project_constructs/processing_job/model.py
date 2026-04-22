"""
Pydantic models for SageMaker Processing Job definitions.

This module provides comprehensive data models for defining SageMaker
processing jobs using Pydantic for validation and type safety. The models
mirror the AWS SageMaker API structure while providing additional validation
and documentation.
"""

from pydantic import BaseModel, Field


class ClusterConfig(BaseModel):
    """
    Configuration for the compute cluster used by the processing job.

    Attributes:
        InstanceCount: Number of instances to use (1-16)
        InstanceType: AWS ML instance type (e.g., ml.g5.4xlarge)
        VolumeSizeInGB: EBS volume size in GB (50-125)
    """

    InstanceCount: int = Field(ge=1, le=40, description="Number of instances for the processing job")
    InstanceType: str = Field(description="AWS ML instance type for the processing job")
    VolumeSizeInGB: int = Field(ge=50, le=125, description="EBS volume size in GB (125 is the maximum)")


class ProcessingResources(BaseModel):
    """
    Resource configuration for the processing job.

    Attributes:
        ClusterConfig: Compute cluster configuration
    """

    ClusterConfig: ClusterConfig


class AppSpecification(BaseModel):
    """
    Application specification for the processing job container.

    Attributes:
        ImageUri: URI of the Docker image to use
        ContainerEntrypoint: Container entrypoint command
        ContainerArguments: Arguments to pass to the container (min 1 required by SageMaker)
    """

    ImageUri: str = Field(description="URI of the Docker image for the processing job")
    ContainerEntrypoint: list[str] = Field(description="Container entrypoint command as a list of strings")
    ContainerArguments: list[str] = Field(
        min_length=1,
        description="Arguments to pass to the container (SageMaker requires at least 1)",
    )


class S3Input(BaseModel):
    """
    S3 input configuration for processing job data sources.

    Attributes:
        S3Uri: S3 URI of the input data
        LocalPath: Local path where data will be mounted in container
        S3DataType: Type of S3 data (S3Prefix, ManifestFile, etc.)
        S3InputMode: How data is provided to container (File, Pipe)
    """

    S3Uri: str = Field(description="S3 URI of the input data")
    LocalPath: str = Field(description="Local path where data will be mounted in the container")
    S3DataType: str = Field(description="Type of S3 data (S3Prefix, ManifestFile, etc.)")
    S3InputMode: str = Field(description="How data is provided to container (File, Pipe)")
    S3DataDistributionType: str = Field(
        default="FullyReplicated",
        description="How data is distributed across instances (FullyReplicated, ShardedByS3Key)",
    )


class ProcessingInput(BaseModel):
    """
    Input configuration for processing job data sources.

    Attributes:
        InputName: Name identifier for this input
        S3Input: S3 input configuration
    """

    InputName: str = Field(description="Name identifier for this input")
    S3Input: S3Input


class S3Output(BaseModel):
    """
    S3 output configuration for processing job results.

    Attributes:
        S3Uri: S3 URI where output will be stored
        LocalPath: Local path in container where output is written
        S3UploadMode: When to upload output (Continuous, EndOfJob)
    """

    S3Uri: str = Field(description="S3 URI where output will be stored")
    LocalPath: str = Field(description="Local path in container where output is written")
    S3UploadMode: str = Field(description="When to upload output (Continuous, EndOfJob)")


class Output(BaseModel):
    """
    Output configuration for processing job results.

    Attributes:
        OutputName: Name identifier for this output
        S3Output: S3 output configuration
    """

    OutputName: str = Field(description="Name identifier for this output")
    S3Output: S3Output


class ProcessingOutputConfig(BaseModel):
    """
    Configuration for all processing job outputs.

    Attributes:
        Outputs: List of output configurations
    """

    Outputs: list[Output] = Field(description="List of output configurations")


class StoppingCondition(BaseModel):
    """
    Conditions for stopping the processing job.

    Attributes:
        MaxRuntimeInSeconds: Maximum runtime before job is stopped
    """

    MaxRuntimeInSeconds: int = Field(description="Maximum runtime in seconds before job is stopped")


class VpcConfig(BaseModel):
    """
    VPC configuration for network isolation.

    Attributes:
        SecurityGroupIds: List of security group IDs
        Subnets: List of subnet IDs
    """

    SecurityGroupIds: list[str] = Field(description="List of security group IDs for network access")
    Subnets: list[str] = Field(description="List of subnet IDs for network placement")


class NetworkConfig(BaseModel):
    """
    Network configuration for the processing job.

    Attributes:
        EnableInterContainerTrafficEncryption: Enable encryption between containers
        EnableNetworkIsolation: Enable network isolation
        VpcConfig: VPC configuration for network placement
    """

    EnableInterContainerTrafficEncryption: bool = Field(description="Enable encryption for inter-container traffic")
    EnableNetworkIsolation: bool = Field(description="Enable network isolation for the processing job")
    VpcConfig: VpcConfig


class Arguments(BaseModel):
    """
    Complete arguments configuration for the processing job.

    This model contains all the configuration needed to create a
    SageMaker processing job including resources, application spec,
    inputs, outputs, and network configuration.

    Attributes:
        ProcessingResources: Compute resource configuration
        AppSpecification: Container application specification
        RoleArn: IAM role ARN for the processing job
        ProcessingInputs: List of input configurations
        ProcessingOutputConfig: Output configuration
        Environment: Environment variables for the container
        StoppingCondition: Conditions for stopping the job
        NetworkConfig: Network and VPC configuration
    """

    ProcessingResources: ProcessingResources
    AppSpecification: AppSpecification
    RoleArn: str = Field(description="IAM role ARN for the processing job")
    ProcessingInputs: list[ProcessingInput] = Field(description="List of input configurations")
    ProcessingOutputConfig: ProcessingOutputConfig
    Environment: dict = Field(description="Environment variables for the container")
    StoppingCondition: StoppingCondition
    NetworkConfig: NetworkConfig


class ProcessingDefinition(BaseModel):
    """
    Complete processing job definition.

    This is the top-level model that defines a complete SageMaker
    processing job with all its configuration and dependencies.

    Attributes:
        Name: Name of the processing job
        Type: Type of the job (typically "Processing")
        Arguments: Complete job arguments and configuration
        DependsOn: Optional list of dependencies
    """

    Name: str = Field(description="Name of the processing job")
    Type: str = Field(description="Type of the job (typically 'Processing')")
    Arguments: Arguments
    DependsOn: list[str] | None = Field(default=None, description="Optional list of job dependencies")
