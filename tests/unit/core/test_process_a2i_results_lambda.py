"""
Unit tests for the process_a2i_results Lambda handler.

Tests verify event parsing, A2I output extraction, and DynamoDB updates
with mocked boto3 clients.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

# Set env vars before importing the module
os.environ.setdefault("DYNAMODB_TABLE_NAME", "test-table")

import lambdas.process_a2i_results.index as _module  # noqa: E402
from lambdas.process_a2i_results.index import (  # noqa: E402
    _extract_selected,
    lambda_handler,
)

pytestmark = pytest.mark.core


class TestExtractSelected:
    """Tests for _extract_selected helper."""

    def test_radio_button_dict(self) -> None:
        output = {"humanAnswers": [{"answerContent": {"selected": {"file_a.mp4": True, "file_b.mp4": False}}}]}
        assert _extract_selected(output) == "file_a.mp4"

    def test_string_value(self) -> None:
        output = {"humanAnswers": [{"answerContent": {"selected": "file_a.mp4"}}]}
        assert _extract_selected(output) == "file_a.mp4"

    def test_majority_vote(self) -> None:
        output = {
            "humanAnswers": [
                {"answerContent": {"selected": {"a.mp4": True, "b.mp4": False}}},
                {"answerContent": {"selected": {"a.mp4": False, "b.mp4": True}}},
                {"answerContent": {"selected": {"a.mp4": False, "b.mp4": True}}},
            ]
        }
        assert _extract_selected(output) == "b.mp4"

    def test_no_answers_returns_none(self) -> None:
        assert _extract_selected({"humanAnswers": []}) is None

    def test_empty_selected_returns_none(self) -> None:
        output = {"humanAnswers": [{"answerContent": {"selected": ""}}]}
        assert _extract_selected(output) is None


class TestLambdaHandler:
    """Tests for the process_a2i_results lambda_handler."""

    @patch.object(_module, "s3")
    @patch.object(_module, "a2i_runtime")
    @patch.object(_module, "dynamodb")
    def test_eventbridge_event_processes_loop(self, mock_ddb, mock_a2i, mock_s3) -> None:
        mock_table = MagicMock()
        mock_table.query.return_value = {
            "Items": [
                {"id": "p1", "step": "t2v#ltx23#g0", "filename": "p1_ltx23_00001_.mp4"},
            ]
        }
        mock_ddb.Table.return_value = mock_table

        a2i_output = {"humanAnswers": [{"answerContent": {"selected": {"p1_ltx23_00001_.mp4": True}}}]}
        body = json.dumps(a2i_output).encode("utf-8")
        mock_s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=body))}

        event = {
            "detail-type": "SageMaker A2I HumanLoop Status Change",
            "detail": {
                "humanLoopName": "review-abc123",
                "humanLoopStatus": "Completed",
                "humanLoopOutput": {"outputS3Uri": "s3://bucket/output.json"},
            },
        }

        result = lambda_handler(event, MagicMock())

        assert result["count"] == 1
        mock_table.update_item.assert_called_once()

    @patch.object(_module, "s3")
    @patch.object(_module, "a2i_runtime")
    @patch.object(_module, "dynamodb")
    def test_sns_event_processes_loop(self, mock_ddb, mock_a2i, mock_s3) -> None:
        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": [{"id": "p1", "step": "t2v#ltx23#g0", "filename": "p1.mp4"}]}
        mock_ddb.Table.return_value = mock_table

        mock_a2i.describe_human_loop.return_value = {
            "HumanLoopStatus": "Completed",
            "HumanLoopOutput": {"OutputS3Uri": "s3://bucket/output.json"},
        }

        a2i_output = {"humanAnswers": [{"answerContent": {"selected": "p1.mp4"}}]}
        body = json.dumps(a2i_output).encode("utf-8")
        mock_s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=body))}

        event = {"Records": [{"Sns": {"Message": json.dumps({"humanLoopName": "review-xyz"})}}]}

        result = lambda_handler(event, MagicMock())

        assert result["count"] == 1

    @patch.object(_module, "dynamodb")
    def test_unrecognised_event_returns_empty(self, mock_ddb) -> None:
        result = lambda_handler({"unknown": "format"}, MagicMock())
        assert result["count"] == 0
        assert result["processed"] == []
