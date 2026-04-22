"""
Unit tests for the submit_a2i_review Lambda handler.

Tests verify asset listing, grouping by input_id, human loop creation,
and DynamoDB updates with mocked boto3 clients.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# Set env vars before importing the module
os.environ.setdefault("FLOW_DEFINITION_ARN", "arn:aws:sagemaker:us-east-1:123456789012:flow-definition/test")
os.environ.setdefault("SOURCE_BUCKET", "test-bucket")
os.environ.setdefault("DYNAMODB_TABLE_NAME", "test-table")
os.environ.setdefault("MEDIA_TYPE", "video")

import lambdas.submit_a2i_review.index as _module  # noqa: E402
from lambdas.submit_a2i_review.index import _parse_asset_name, lambda_handler  # noqa: E402

pytestmark = pytest.mark.core


class TestParseAssetName:
    """Tests for _parse_asset_name helper."""

    def test_known_model_ltx23(self) -> None:
        result = _parse_asset_name("honey-pancakes_ltx23_00001_.mp4")
        assert result["input_id"] == "honey-pancakes"
        assert result["model"] == "ltx23"
        assert result["generation_index"] == 0

    def test_known_model_with_generation_index(self) -> None:
        result = _parse_asset_name("honey-pancakes_ltx23_g1_00001_.mp4")
        assert result["input_id"] == "honey-pancakes"
        assert result["model"] == "ltx23"
        assert result["generation_index"] == 1

    def test_unknown_model_fallback(self) -> None:
        result = _parse_asset_name("some_random_file.mp4")
        assert result["input_id"] == "some_random_file"
        assert result["model"] == ""


class TestLambdaHandler:
    """Tests for the submit_a2i_review lambda_handler."""

    @patch.object(_module, "s3")
    @patch.object(_module, "a2i_runtime")
    @patch.object(_module, "dynamodb")
    def test_no_assets_returns_empty(self, mock_ddb, mock_a2i, mock_s3) -> None:
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        mock_s3.get_paginator.return_value = paginator

        result = lambda_handler({"execution_id": "exec-1"}, MagicMock())

        assert result["count"] == 0
        assert result["human_loops"] == []

    @patch.object(_module, "s3")
    @patch.object(_module, "a2i_runtime")
    @patch.object(_module, "dynamodb")
    def test_starts_human_loop_for_assets(self, mock_ddb, mock_a2i, mock_s3) -> None:
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": [{"Key": "exec-1/test-prompt_ltx23_00001_.mp4"}]}]
        mock_s3.get_paginator.return_value = paginator

        mock_table = MagicMock()
        mock_table.get_item.return_value = {"Item": {"prompt": "a cat"}}
        mock_ddb.Table.return_value = mock_table

        result = lambda_handler({"execution_id": "exec-1"}, MagicMock())

        assert result["count"] == 1
        mock_a2i.start_human_loop.assert_called_once()
        mock_table.update_item.assert_called_once()

    @patch.object(_module, "s3")
    @patch.object(_module, "a2i_runtime")
    @patch.object(_module, "dynamodb")
    def test_groups_assets_by_input_id(self, mock_ddb, mock_a2i, mock_s3) -> None:
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "exec-1/prompt1_ltx23_00001_.mp4"},
                    {"Key": "exec-1/prompt1_wan22_00001_.mp4"},
                    {"Key": "exec-1/prompt2_ltx23_00001_.mp4"},
                ]
            }
        ]
        mock_s3.get_paginator.return_value = paginator

        mock_table = MagicMock()
        mock_table.get_item.return_value = {"Item": {}}
        mock_ddb.Table.return_value = mock_table

        result = lambda_handler({"execution_id": "exec-1"}, MagicMock())

        # 2 input_ids -> 2 human loops
        assert result["count"] == 2
        assert mock_a2i.start_human_loop.call_count == 2
