"""
Unit tests for the trigger_pipeline Lambda function.

Tests verify that the handler reads PIPELINE_NAME from env,
calls StartPipelineExecution, and returns the execution ARN.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# Set env var before importing the module
os.environ["PIPELINE_NAME"] = "test-pipeline"

import lambdas.trigger_pipeline.index as _module  # noqa: E402
from lambdas.trigger_pipeline.index import lambda_handler  # noqa: E402

pytestmark = pytest.mark.core


class TestLambdaHandler:
    """Tests for the trigger_pipeline lambda_handler."""

    @patch.object(_module, "sagemaker_client")
    def test_returns_execution_arn(self, mock_sm):
        mock_sm.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test/exec/abc"
        }
        context = MagicMock()

        result = lambda_handler({}, context)

        assert result["PipelineExecutionArn"] == "arn:aws:sagemaker:us-east-1:123456789012:pipeline/test/exec/abc"

    @patch.object(_module, "sagemaker_client")
    def test_calls_start_pipeline_execution_with_correct_name(self, mock_sm):
        mock_sm.start_pipeline_execution.return_value = {"PipelineExecutionArn": "arn:exec:1"}
        context = MagicMock()

        lambda_handler({}, context)

        mock_sm.start_pipeline_execution.assert_called_once_with(PipelineName="test-pipeline")

    @patch.object(_module, "sagemaker_client")
    def test_raises_on_api_failure(self, mock_sm):
        mock_sm.start_pipeline_execution.side_effect = Exception("AccessDenied")
        context = MagicMock()

        with pytest.raises(Exception, match="AccessDenied"):
            lambda_handler({}, context)

    @patch.object(_module, "sagemaker_client")
    def test_event_payload_is_ignored(self, mock_sm):
        mock_sm.start_pipeline_execution.return_value = {"PipelineExecutionArn": "arn:exec:1"}
        context = MagicMock()

        result = lambda_handler({"some": "data"}, context)

        assert "PipelineExecutionArn" in result
