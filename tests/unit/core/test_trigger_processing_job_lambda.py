"""
Unit tests for the trigger_processing_job Lambda function.

Tests verify that the handler reads the processing job definition from
environment, generates a unique job name with timestamp, and calls
CreateProcessingJob with the correct arguments.
"""

import json
import os
from unittest.mock import MagicMock, patch

# Build a minimal valid processing job definition for the env var
_JOB_DEF = {
    "Name": "TestJob",
    "Type": "Processing",
    "Arguments": {
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m5.xlarge",
                "VolumeSizeInGB": 50,
            }
        },
        "AppSpecification": {
            "ImageUri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/test:latest",
            "ContainerEntrypoint": ["python3"],
            "ContainerArguments": ["run.py"],
        },
        "RoleArn": "arn:aws:iam::123456789012:role/test-role",
        "ProcessingInputs": [],
        "ProcessingOutputConfig": {"Outputs": []},
        "Environment": {},
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
        "NetworkConfig": {
            "EnableInterContainerTrafficEncryption": True,
            "EnableNetworkIsolation": False,
            "VpcConfig": {"SecurityGroupIds": ["sg-123"], "Subnets": ["subnet-1"]},
        },
    },
    "DependsOn": [],
}

os.environ["PROCESSING_JOB_DEFINITION"] = json.dumps(_JOB_DEF)
os.environ["JOB_NAME"] = "TestJob"

import pytest

import lambdas.trigger_processing_job.index as _module  # noqa: E402
from lambdas.trigger_processing_job.index import lambda_handler  # noqa: E402

pytestmark = pytest.mark.core


class TestLambdaHandler:
    """Tests for the trigger_processing_job lambda_handler."""

    @patch.object(_module, "boto3")
    def test_calls_create_processing_job(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.create_processing_job.return_value = {"ProcessingJobArn": "arn:job:1"}
        context = MagicMock()

        result = lambda_handler({}, context)

        mock_client.create_processing_job.assert_called_once()
        assert result == {"ProcessingJobArn": "arn:job:1"}

    @patch.object(_module, "boto3")
    def test_job_name_contains_timestamp_format(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.create_processing_job.return_value = {"ProcessingJobArn": "arn:job:1"}
        context = MagicMock()

        lambda_handler({}, context)

        call_kwargs = mock_client.create_processing_job.call_args
        job_name = call_kwargs.kwargs.get("ProcessingJobName") or call_kwargs[1].get("ProcessingJobName")
        assert job_name.startswith("TestJob-")
        # Verify timestamp portion is present (YYYY-MM-DD-HH-MM-SS)
        parts = job_name.split("-")
        assert len(parts) >= 7  # TestJob + 6 timestamp parts

    @patch.object(_module, "boto3")
    def test_passes_correct_arguments(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.create_processing_job.return_value = {"ProcessingJobArn": "arn:job:1"}
        context = MagicMock()

        lambda_handler({}, context)

        call_kwargs = mock_client.create_processing_job.call_args
        # Verify key arguments are passed from the definition
        assert "ProcessingResources" in (call_kwargs.kwargs or call_kwargs[1])
        assert "AppSpecification" in (call_kwargs.kwargs or call_kwargs[1])
        assert "RoleArn" in (call_kwargs.kwargs or call_kwargs[1])

    @patch.object(_module, "boto3")
    def test_raises_on_api_failure(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.create_processing_job.side_effect = Exception("ThrottlingException")
        context = MagicMock()

        with __import__("pytest").raises(Exception, match="ThrottlingException"):
            lambda_handler({}, context)
