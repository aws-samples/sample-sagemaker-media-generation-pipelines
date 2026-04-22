"""
Unit tests for the Pydantic models used in SageMaker Processing Job definitions.

Tests verify model validation, serialization, field constraints,
and default behavior for all processing job data models.
"""

import pytest
from pydantic import ValidationError

from project_constructs.processing_job.model import (
    AppSpecification,
    Arguments,
    ClusterConfig,
    NetworkConfig,
    Output,
    ProcessingDefinition,
    ProcessingInput,
    ProcessingOutputConfig,
    ProcessingResources,
    S3Input,
    S3Output,
    StoppingCondition,
    VpcConfig,
)

pytestmark = pytest.mark.processing


class TestClusterConfig:
    """Tests for ClusterConfig model validation."""

    def test_valid_cluster_config(self):
        config = ClusterConfig(InstanceCount=1, InstanceType="ml.g5.xlarge", VolumeSizeInGB=125)
        assert config.InstanceCount == 1
        assert config.InstanceType == "ml.g5.xlarge"
        assert config.VolumeSizeInGB == 125

    def test_instance_count_minimum(self):
        with pytest.raises(ValidationError):
            ClusterConfig(InstanceCount=0, InstanceType="ml.g5.xlarge", VolumeSizeInGB=125)

    def test_instance_count_maximum(self):
        with pytest.raises(ValidationError):
            ClusterConfig(InstanceCount=41, InstanceType="ml.g5.xlarge", VolumeSizeInGB=125)

    def test_volume_size_minimum(self):
        with pytest.raises(ValidationError):
            ClusterConfig(InstanceCount=1, InstanceType="ml.g5.xlarge", VolumeSizeInGB=49)

    def test_volume_size_maximum(self):
        with pytest.raises(ValidationError):
            ClusterConfig(InstanceCount=1, InstanceType="ml.g5.xlarge", VolumeSizeInGB=126)

    def test_boundary_instance_count_1(self):
        config = ClusterConfig(InstanceCount=1, InstanceType="ml.g5.xlarge", VolumeSizeInGB=50)
        assert config.InstanceCount == 1

    def test_boundary_instance_count_16(self):
        config = ClusterConfig(InstanceCount=16, InstanceType="ml.g5.xlarge", VolumeSizeInGB=50)
        assert config.InstanceCount == 16

    def test_boundary_volume_50(self):
        config = ClusterConfig(InstanceCount=1, InstanceType="ml.g5.xlarge", VolumeSizeInGB=50)
        assert config.VolumeSizeInGB == 50

    def test_boundary_volume_125(self):
        config = ClusterConfig(InstanceCount=1, InstanceType="ml.g5.xlarge", VolumeSizeInGB=125)
        assert config.VolumeSizeInGB == 125


class TestAppSpecification:
    """Tests for AppSpecification model."""

    def test_valid_app_spec(self):
        spec = AppSpecification(
            ImageUri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:latest",
            ContainerEntrypoint=["/bin/bash", "./run.sh"],
            ContainerArguments=["300"],
        )
        assert spec.ImageUri == "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:latest"
        assert spec.ContainerEntrypoint == ["/bin/bash", "./run.sh"]
        assert spec.ContainerArguments == ["300"]

    def test_missing_image_uri_raises(self):
        with pytest.raises(ValidationError):
            AppSpecification(
                ContainerEntrypoint=["/bin/bash"],
                ContainerArguments=["300"],
            )


class TestS3InputOutput:
    """Tests for S3 input/output models."""

    def test_valid_s3_input(self):
        s3_input = S3Input(
            S3Uri="s3://my-bucket/input/",
            LocalPath="/opt/ml/processing/input/",
            S3DataType="S3Prefix",
            S3InputMode="File",
        )
        assert s3_input.S3Uri == "s3://my-bucket/input/"
        assert s3_input.S3InputMode == "File"

    def test_valid_processing_input(self):
        proc_input = ProcessingInput(
            InputName="my-input",
            S3Input=S3Input(
                S3Uri="s3://bucket/",
                LocalPath="/opt/ml/processing/input/",
                S3DataType="S3Prefix",
                S3InputMode="File",
            ),
        )
        assert proc_input.InputName == "my-input"

    def test_valid_s3_output(self):
        s3_output = S3Output(
            S3Uri="s3://my-bucket/output/",
            LocalPath="/opt/ml/processing/output/",
            S3UploadMode="Continuous",
        )
        assert s3_output.S3UploadMode == "Continuous"

    def test_valid_output(self):
        output = Output(
            OutputName="output",
            S3Output=S3Output(
                S3Uri="s3://bucket/output/",
                LocalPath="/opt/ml/processing/output/",
                S3UploadMode="EndOfJob",
            ),
        )
        assert output.OutputName == "output"
        assert output.S3Output.S3UploadMode == "EndOfJob"


class TestNetworkConfig:
    """Tests for VPC and network configuration models."""

    def test_valid_vpc_config(self):
        vpc = VpcConfig(
            SecurityGroupIds=["sg-12345"],
            Subnets=["subnet-abc", "subnet-def"],
        )
        assert len(vpc.SecurityGroupIds) == 1
        assert len(vpc.Subnets) == 2

    def test_valid_network_config(self):
        net = NetworkConfig(
            EnableInterContainerTrafficEncryption=True,
            EnableNetworkIsolation=False,
            VpcConfig=VpcConfig(
                SecurityGroupIds=["sg-12345"],
                Subnets=["subnet-abc"],
            ),
        )
        assert net.EnableInterContainerTrafficEncryption is True
        assert net.EnableNetworkIsolation is False


class TestStoppingCondition:
    """Tests for StoppingCondition model."""

    def test_valid_stopping_condition(self):
        sc = StoppingCondition(MaxRuntimeInSeconds=86400)
        assert sc.MaxRuntimeInSeconds == 86400

    def test_missing_max_runtime_raises(self):
        with pytest.raises(ValidationError):
            StoppingCondition()


class TestProcessingDefinition:
    """Tests for the complete ProcessingDefinition model."""

    def _build_definition(self, **overrides) -> ProcessingDefinition:
        """Helper to build a valid ProcessingDefinition with optional overrides."""
        defaults = dict(
            Name="TestJob",
            Type="Processing",
            Arguments=Arguments(
                ProcessingResources=ProcessingResources(
                    ClusterConfig=ClusterConfig(
                        InstanceCount=1,
                        InstanceType="ml.g5.xlarge",
                        VolumeSizeInGB=125,
                    )
                ),
                AppSpecification=AppSpecification(
                    ImageUri="123456789012.dkr.ecr.us-east-1.amazonaws.com/repo:latest",
                    ContainerEntrypoint=["/bin/bash", "./run.sh"],
                    ContainerArguments=["300"],
                ),
                RoleArn="arn:aws:iam::123456789012:role/test-role",
                ProcessingInputs=[],
                ProcessingOutputConfig=ProcessingOutputConfig(
                    Outputs=[
                        Output(
                            OutputName="output",
                            S3Output=S3Output(
                                S3Uri="s3://bucket/output/",
                                LocalPath="/opt/ml/processing/output/",
                                S3UploadMode="Continuous",
                            ),
                        )
                    ]
                ),
                Environment={"KEY": "VALUE"},
                StoppingCondition=StoppingCondition(MaxRuntimeInSeconds=86400),
                NetworkConfig=NetworkConfig(
                    EnableInterContainerTrafficEncryption=True,
                    EnableNetworkIsolation=False,
                    VpcConfig=VpcConfig(
                        SecurityGroupIds=["sg-12345"],
                        Subnets=["subnet-abc"],
                    ),
                ),
            ),
            DependsOn=[],
        )
        defaults.update(overrides)
        return ProcessingDefinition(**defaults)

    def test_valid_definition(self):
        definition = self._build_definition()
        assert definition.Name == "TestJob"
        assert definition.Type == "Processing"

    def test_depends_on_defaults_to_none(self):
        definition = ProcessingDefinition(
            Name="TestJob",
            Type="Processing",
            Arguments=self._build_definition().Arguments,
        )
        assert definition.DependsOn is None

    def test_model_dump_returns_dict(self):
        definition = self._build_definition()
        dumped = definition.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["Name"] == "TestJob"
        assert dumped["Type"] == "Processing"
        assert "Arguments" in dumped

    def test_model_dump_json_returns_string(self):
        definition = self._build_definition()
        json_str = definition.model_dump_json()
        assert isinstance(json_str, str)
        assert "TestJob" in json_str

    def test_definition_with_processing_inputs(self):
        definition = self._build_definition()
        definition.Arguments.ProcessingInputs = [
            ProcessingInput(
                InputName="input-data",
                S3Input=S3Input(
                    S3Uri="s3://input-bucket/",
                    LocalPath="/opt/ml/processing/input/input-data/",
                    S3DataType="S3Prefix",
                    S3InputMode="File",
                ),
            )
        ]
        dumped = definition.model_dump()
        assert len(dumped["Arguments"]["ProcessingInputs"]) == 1
        assert dumped["Arguments"]["ProcessingInputs"][0]["InputName"] == "input-data"

    def test_definition_environment_merge(self):
        """Test that environment dict can be merged like the construct does."""
        base_env = {"KEY1": "VAL1"}
        extra_env = {"LOCAL_OUTPUT_DIR": "/opt/ml/processing/output/"}
        merged = base_env | extra_env
        definition = self._build_definition()
        definition.Arguments.Environment = merged
        assert definition.Arguments.Environment["KEY1"] == "VAL1"
        assert definition.Arguments.Environment["LOCAL_OUTPUT_DIR"] == "/opt/ml/processing/output/"
