"""
Unit tests for the retrieval_query (ingestion) Lambda handler.

Tests verify SQS/S3 event processing, Bedrock embedding calls,
and OpenSearch indexing with mocked boto3 clients.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

# Set env vars before importing the module
os.environ.setdefault("RAG_BUCKET_NAME", "test-rag-bucket")
os.environ.setdefault("AOSS_ENDPOINT", "test-endpoint.aoss.amazonaws.com")
os.environ.setdefault("AOSS_INDEX_NAME", "test-vectors")

# Stub opensearchpy to avoid slow import of requests/urllib3 in xdist workers
import sys
import types

_oss_stub = types.ModuleType("opensearchpy")
_oss_stub.OpenSearch = MagicMock
_oss_stub.AWSV4SignerAuth = MagicMock
_oss_stub.RequestsHttpConnection = MagicMock
sys.modules["opensearchpy"] = _oss_stub

import lambdas.retrieval_query.index as _module  # noqa: E402
from lambdas.retrieval_query.index import lambda_handler  # noqa: E402

pytestmark = pytest.mark.retrieval


class TestLambdaHandler:
    """Tests for the retrieval_query lambda_handler."""

    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "s3_client")
    @patch.object(_module, "bedrock_client")
    def test_processes_single_image(self, mock_bedrock, mock_s3, mock_oss_factory) -> None:
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        image_bytes = b"fake-image-data"
        mock_s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=image_bytes))}

        embedding = [0.1] * 1024
        mock_bedrock.invoke_model.return_value = {
            "body": MagicMock(read=MagicMock(return_value=json.dumps({"embedding": embedding}).encode()))
        }

        event = {
            "Records": [
                {
                    "body": json.dumps(
                        {
                            "Records": [
                                {
                                    "s3": {
                                        "bucket": {"name": "test-rag-bucket"},
                                        "object": {"key": "images/cat.png"},
                                    }
                                }
                            ]
                        }
                    )
                }
            ]
        }

        result = lambda_handler(event, MagicMock())

        assert result["statusCode"] == 200
        assert result["processed"] == 1
        assert result["failed"] == 0
        mock_oss.index.assert_called_once()
        mock_s3.put_object.assert_called_once()

    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "s3_client")
    @patch.object(_module, "bedrock_client")
    def test_handles_processing_error(self, mock_bedrock, mock_s3, mock_oss_factory) -> None:
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        mock_s3.get_object.side_effect = Exception("S3 error")

        event = {
            "Records": [
                {
                    "messageId": "msg-1",
                    "body": json.dumps(
                        {
                            "Records": [
                                {
                                    "s3": {
                                        "bucket": {"name": "test-rag-bucket"},
                                        "object": {"key": "images/bad.png"},
                                    }
                                }
                            ]
                        }
                    ),
                }
            ]
        }

        result = lambda_handler(event, MagicMock())

        assert result["statusCode"] == 206
        assert result["failed"] == 1
        assert result["processed"] == 0

    @patch.object(_module, "get_oss_client")
    def test_empty_records(self, mock_oss_factory) -> None:
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        result = lambda_handler({"Records": []}, MagicMock())

        assert result["statusCode"] == 200
        assert result["processed"] == 0
