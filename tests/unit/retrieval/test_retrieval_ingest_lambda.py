"""
Unit tests for the retrieval_ingest Lambda function.

Tests verify SQS message parsing, S3 download/upload, Bedrock embedding
invocation, OpenSearch indexing, index creation, and batch item failure
reporting.
"""

import base64
import io
import json
import os
from unittest.mock import MagicMock, patch

import pytest

# Set env vars before importing the module (read at import time)
os.environ.setdefault("RETRIEVAL_BUCKET_NAME", "test-retrieval-bucket")
os.environ.setdefault("AOSS_ENDPOINT", "search-test.us-east-1.aoss.amazonaws.com")
os.environ.setdefault("AOSS_INDEX_NAME", "image-vectors")
os.environ.setdefault("EMBEDDING_MODEL_ID", "amazon.titan-embed-image-v1")

# Stub opensearchpy to avoid slow import of requests/urllib3 in xdist workers
import sys
import types

_oss_stub = types.ModuleType("opensearchpy")
_oss_stub.OpenSearch = MagicMock
_oss_stub.AWSV4SignerAuth = MagicMock
_oss_stub.RequestsHttpConnection = MagicMock
sys.modules["opensearchpy"] = _oss_stub

import lambdas.retrieval_ingest.index as _module  # noqa: E402
from lambdas.retrieval_ingest.index import lambda_handler  # noqa: E402

pytestmark = pytest.mark.retrieval


FAKE_EMBEDDING = [0.1] * 1024


def _sqs_event(records: list[dict]) -> dict:
    """Build an SQS event wrapping one or more S3 ObjectCreated records."""
    sqs_records = []
    for i, s3_rec in enumerate(records):
        sqs_records.append(
            {
                "messageId": f"msg-{i}",
                "body": json.dumps({"Records": [s3_rec]}),
            }
        )
    return {"Records": sqs_records}


def _s3_record(bucket: str = "source-bucket", key: str = "images/photo.jpg") -> dict:
    """Build a minimal S3 ObjectCreated record."""
    return {
        "s3": {
            "bucket": {"name": bucket},
            "object": {"key": key},
        }
    }


def _mock_bedrock_response(embedding: list[float] | None = None) -> MagicMock:
    """Return a mock Bedrock invoke_model response."""
    body_bytes = json.dumps({"embedding": embedding or FAKE_EMBEDDING}).encode()
    mock_body = MagicMock()
    mock_body.read.return_value = body_bytes
    return {"body": mock_body}


class TestSqsMessageParsing:
    """SQS message parsing extracts S3 bucket and key."""

    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "bedrock_client")
    @patch.object(_module, "s3_client")
    def test_extracts_bucket_and_key(self, mock_s3, mock_bedrock, mock_oss_factory):
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        mock_s3.get_object.return_value = {"Body": io.BytesIO(b"img-data")}
        mock_bedrock.invoke_model.return_value = _mock_bedrock_response()

        event = _sqs_event([_s3_record("my-bucket", "images/cat.png")])
        lambda_handler(event, MagicMock())

        mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="images/cat.png")

    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "bedrock_client")
    @patch.object(_module, "s3_client")
    def test_processes_multiple_sqs_records(self, mock_s3, mock_bedrock, mock_oss_factory):
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        mock_s3.get_object.return_value = {"Body": io.BytesIO(b"img")}
        mock_bedrock.invoke_model.return_value = _mock_bedrock_response()

        event = _sqs_event(
            [
                _s3_record("b", "images/a.jpg"),
                _s3_record("b", "images/b.jpg"),
            ]
        )
        result = lambda_handler(event, MagicMock())

        assert mock_s3.get_object.call_count == 2
        assert result["batchItemFailures"] == []


class TestS3DownloadAndBase64Upload:
    """S3 download and base64 encoding/upload."""

    @patch.object(_module, "RETRIEVAL_BUCKET", "test-retrieval-bucket")
    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "bedrock_client")
    @patch.object(_module, "s3_client")
    def test_downloads_image_and_uploads_base64(self, mock_s3, mock_bedrock, mock_oss_factory):
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        image_bytes = b"\x89PNG\r\n\x1a\nfake-image-data"
        mock_s3.get_object.return_value = {"Body": io.BytesIO(image_bytes)}
        mock_bedrock.invoke_model.return_value = _mock_bedrock_response()

        event = _sqs_event([_s3_record("src-bucket", "images/pic.png")])
        lambda_handler(event, MagicMock())

        expected_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mock_s3.put_object.assert_called_once_with(
            Bucket="test-retrieval-bucket",
            Key="base64/images/pic.png.txt",
            Body=expected_b64.encode("utf-8"),
        )


class TestBedrockEmbeddingInvocation:
    """Bedrock embedding invocation with correct model ID."""

    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "bedrock_client")
    @patch.object(_module, "s3_client")
    def test_invokes_bedrock_with_correct_model_and_payload(self, mock_s3, mock_bedrock, mock_oss_factory):
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        image_bytes = b"test-image"
        mock_s3.get_object.return_value = {"Body": io.BytesIO(image_bytes)}
        mock_bedrock.invoke_model.return_value = _mock_bedrock_response()

        event = _sqs_event([_s3_record()])
        lambda_handler(event, MagicMock())

        expected_b64 = base64.b64encode(image_bytes).decode("utf-8")
        expected_body = json.dumps(
            {
                "inputImage": expected_b64,
                "embeddingConfig": {"outputEmbeddingLength": 1024},
            }
        )
        mock_bedrock.invoke_model.assert_called_once_with(
            body=expected_body,
            modelId="amazon.titan-embed-image-v1",
            accept="application/json",
            contentType="application/json",
        )


class TestOpenSearchIndexing:
    """OpenSearch index call with all required fields."""

    @patch.object(_module, "RETRIEVAL_BUCKET", "test-retrieval-bucket")
    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "bedrock_client")
    @patch.object(_module, "s3_client")
    def test_indexes_document_with_all_four_fields(self, mock_s3, mock_bedrock, mock_oss_factory):
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        mock_s3.get_object.return_value = {"Body": io.BytesIO(b"data")}
        mock_bedrock.invoke_model.return_value = _mock_bedrock_response()

        event = _sqs_event([_s3_record("mybucket", "images/dog.jpg")])
        lambda_handler(event, MagicMock())

        mock_oss.index.assert_called_once()
        call_kwargs = mock_oss.index.call_args
        body = call_kwargs.kwargs.get("body") or call_kwargs[1].get("body")

        assert body["image_vector"] == FAKE_EMBEDDING
        assert body["description"] == "images/dog.jpg"
        assert body["image_s3_uri"] == "s3://mybucket/images/dog.jpg"
        assert body["image_base64_s3_uri"] == "s3://test-retrieval-bucket/base64/images/dog.jpg.txt"

    @patch.object(_module, "AOSS_INDEX_NAME", "image-vectors")
    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "bedrock_client")
    @patch.object(_module, "s3_client")
    def test_indexes_into_correct_index_name(self, mock_s3, mock_bedrock, mock_oss_factory):
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        mock_s3.get_object.return_value = {"Body": io.BytesIO(b"data")}
        mock_bedrock.invoke_model.return_value = _mock_bedrock_response()

        event = _sqs_event([_s3_record()])
        lambda_handler(event, MagicMock())

        call_kwargs = mock_oss.index.call_args
        index_name = call_kwargs.kwargs.get("index") or call_kwargs[1].get("index")
        assert index_name == "image-vectors"


class TestIndexCreation:
    """Index creation on first run when index doesn't exist."""

    @patch.object(_module, "AOSS_INDEX_NAME", "image-vectors")
    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "bedrock_client")
    @patch.object(_module, "s3_client")
    def test_creates_index_when_not_exists(self, mock_s3, mock_bedrock, mock_oss_factory):
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = False
        mock_oss_factory.return_value = mock_oss

        mock_s3.get_object.return_value = {"Body": io.BytesIO(b"data")}
        mock_bedrock.invoke_model.return_value = _mock_bedrock_response()

        event = _sqs_event([_s3_record()])
        lambda_handler(event, MagicMock())

        mock_oss.indices.create.assert_called_once()
        create_kwargs = mock_oss.indices.create.call_args
        body = create_kwargs.kwargs.get("body") or create_kwargs[1].get("body")
        assert body["settings"]["index.knn"] == "true"
        assert body["mappings"]["properties"]["image_vector"]["dimension"] == 1024

        # Verify exists() and create() use keyword arg for index name
        mock_oss.indices.exists.assert_called_with(index="image-vectors")
        assert create_kwargs.kwargs.get("index") == "image-vectors"

    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "bedrock_client")
    @patch.object(_module, "s3_client")
    def test_skips_index_creation_when_exists(self, mock_s3, mock_bedrock, mock_oss_factory):
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        mock_s3.get_object.return_value = {"Body": io.BytesIO(b"data")}
        mock_bedrock.invoke_model.return_value = _mock_bedrock_response()

        event = _sqs_event([_s3_record()])
        lambda_handler(event, MagicMock())

        mock_oss.indices.create.assert_not_called()


class TestBatchItemFailureReporting:
    """Batch item failure reporting on processing error."""

    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "bedrock_client")
    @patch.object(_module, "s3_client")
    def test_reports_failed_record_message_id(self, mock_s3, mock_bedrock, mock_oss_factory):
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        mock_s3.get_object.side_effect = Exception("NoSuchKey")

        event = _sqs_event([_s3_record()])
        result = lambda_handler(event, MagicMock())

        assert len(result["batchItemFailures"]) == 1
        assert result["batchItemFailures"][0]["itemIdentifier"] == "msg-0"

    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "bedrock_client")
    @patch.object(_module, "s3_client")
    def test_partial_failure_reports_only_failed_records(self, mock_s3, mock_bedrock, mock_oss_factory):
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        # First record succeeds, second fails
        mock_s3.get_object.side_effect = [
            {"Body": io.BytesIO(b"ok")},
            Exception("AccessDenied"),
        ]
        mock_bedrock.invoke_model.return_value = _mock_bedrock_response()

        event = _sqs_event(
            [
                _s3_record("b", "images/good.jpg"),
                _s3_record("b", "images/bad.jpg"),
            ]
        )
        result = lambda_handler(event, MagicMock())

        assert len(result["batchItemFailures"]) == 1
        assert result["batchItemFailures"][0]["itemIdentifier"] == "msg-1"

    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "bedrock_client")
    @patch.object(_module, "s3_client")
    def test_empty_event_returns_no_failures(self, mock_s3, mock_bedrock, mock_oss_factory):
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        result = lambda_handler({"Records": []}, MagicMock())

        assert result == {"batchItemFailures": []}

    @patch.object(_module, "get_oss_client")
    @patch.object(_module, "bedrock_client")
    @patch.object(_module, "s3_client")
    def test_bedrock_failure_reports_batch_item_failure(self, mock_s3, mock_bedrock, mock_oss_factory):
        mock_oss = MagicMock()
        mock_oss.indices.exists.return_value = True
        mock_oss_factory.return_value = mock_oss

        mock_s3.get_object.return_value = {"Body": io.BytesIO(b"data")}
        mock_bedrock.invoke_model.side_effect = Exception("ThrottlingException")

        event = _sqs_event([_s3_record()])
        result = lambda_handler(event, MagicMock())

        assert len(result["batchItemFailures"]) == 1
        assert result["batchItemFailures"][0]["itemIdentifier"] == "msg-0"
