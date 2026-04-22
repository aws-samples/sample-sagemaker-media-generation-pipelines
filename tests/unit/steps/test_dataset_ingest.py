"""
Unit tests for the dataset_ingest container logic.

Covers:
- resolve_description: real-shaped row dicts, "nan" handling, missing ai key, empty keywords
- download_image: mock HTTP responses (200, 404, timeout)
- upload_to_s3: mock boto3 PutObject, key format, retry on ClientError
- generate_prompt: mock Strands Agent, VisualEntry output format
- verify_aoss: mock OpenSearch client, count comparison logic
- cleanup: mock S3 list/delete + AOSS delete-by-query
- import_dataset_script: dynamic module import

**Validates: Requirements 11.1, 11.2, 11.3**
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest
import requests
from botocore.exceptions import ClientError

from processing_job.dataset_ingest.main import (
    _delete_aoss_documents,
    _delete_s3_prefix,
    cleanup,
    generate_prompt,
    import_dataset_script,
    verify_aoss,
    write_visual_entries,
)
from processing_job.dataset_ingest.unsplash import (
    _flatten_row,
    download_image,
    resolve_description,
    upload_to_s3,
)

pytestmark = pytest.mark.steps_setup


# ---------------------------------------------------------------------------
# Unit tests: resolve_description
# ---------------------------------------------------------------------------


class TestResolveDescription:
    """resolve_description fallback chain with real-shaped row dicts."""

    def test_ai_description_primary(self) -> None:
        row = {
            "ai": {"description": "silhouette of structure under red sky"},
            "description": "fallback",
            "keywords": [{"keyword": "kw1"}],
        }
        assert resolve_description(row) == "silhouette of structure under red sky"

    def test_description_secondary_when_ai_nan(self) -> None:
        row = {
            "ai": {"description": "nan"},
            "description": "a beautiful sunset",
            "keywords": [{"keyword": "kw1"}],
        }
        assert resolve_description(row) == "a beautiful sunset"

    def test_description_secondary_when_ai_missing(self) -> None:
        row = {
            "ai": {},
            "description": "mountain landscape",
            "keywords": [{"keyword": "kw1"}],
        }
        assert resolve_description(row) == "mountain landscape"

    def test_keywords_tertiary(self) -> None:
        row = {
            "ai": {"description": "nan"},
            "description": "nan",
            "keywords": [{"keyword": "sunset"}, {"keyword": "beach"}],
        }
        assert resolve_description(row) == "sunset, beach"

    def test_none_when_all_absent(self) -> None:
        row = {
            "ai": {"description": "nan"},
            "description": "nan",
            "keywords": [],
        }
        assert resolve_description(row) is None

    def test_none_when_no_ai_key(self) -> None:
        row = {"description": "nan", "keywords": []}
        assert resolve_description(row) is None

    def test_nan_case_insensitive(self) -> None:
        row = {
            "ai": {"description": "NaN"},
            "description": "NaN",
            "keywords": [{"keyword": "nature"}],
        }
        assert resolve_description(row) == "nature"

    def test_whitespace_only_treated_as_empty(self) -> None:
        row = {
            "ai": {"description": "   "},
            "description": "  ",
            "keywords": [{"keyword": "tree"}],
        }
        assert resolve_description(row) == "tree"

    def test_empty_keywords_list(self) -> None:
        row = {
            "ai": {"description": "nan"},
            "description": "",
            "keywords": [],
        }
        assert resolve_description(row) is None

    def test_keywords_with_empty_strings_skipped(self) -> None:
        row = {
            "ai": {},
            "description": "nan",
            "keywords": [{"keyword": ""}, {"keyword": "  "}, {"keyword": "valid"}],
        }
        assert resolve_description(row) == "valid"

    def test_ai_description_stripped(self) -> None:
        row = {"ai": {"description": "  padded text  "}}
        assert resolve_description(row) == "padded text"

    def test_missing_all_keys(self) -> None:
        row: dict = {}
        assert resolve_description(row) is None

    def test_ai_none_value(self) -> None:
        row = {"ai": {"description": None}, "description": "fallback"}
        assert resolve_description(row) == "fallback"


# ---------------------------------------------------------------------------
# Unit tests: download_image
# ---------------------------------------------------------------------------


class TestDownloadImage:
    """download_image with mock HTTP responses."""

    @patch("processing_job.dataset_ingest.unsplash.requests.get")
    def test_successful_download(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.content = b"\xff\xd8\xff\xe0fake-jpeg-data"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = download_image("https://images.unsplash.com/photo-123")
        assert result == b"\xff\xd8\xff\xe0fake-jpeg-data"
        mock_get.assert_called_once_with("https://images.unsplash.com/photo-123", timeout=30)

    @patch("processing_job.dataset_ingest.unsplash.requests.get")
    def test_404_raises(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_resp

        with pytest.raises(requests.HTTPError, match="404"):
            download_image("https://images.unsplash.com/photo-missing")

    @patch("processing_job.dataset_ingest.unsplash.requests.get")
    def test_timeout_raises(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.Timeout("Connection timed out")

        with pytest.raises(requests.Timeout):
            download_image("https://images.unsplash.com/photo-slow")

    @patch("processing_job.dataset_ingest.unsplash.requests.get")
    def test_custom_timeout(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.content = b"data"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        download_image("https://example.com/img.jpg", timeout=10)
        mock_get.assert_called_once_with("https://example.com/img.jpg", timeout=10)


# ---------------------------------------------------------------------------
# Unit tests: upload_to_s3
# ---------------------------------------------------------------------------


class TestUploadToS3:
    """upload_to_s3 with mock boto3 PutObject."""

    def test_successful_upload(self) -> None:
        mock_s3 = MagicMock()
        result = upload_to_s3(mock_s3, "my-bucket", "images/abc123.jpg", b"data")
        assert result is True
        mock_s3.put_object.assert_called_once_with(Bucket="my-bucket", Key="images/abc123.jpg", Body=b"data")

    def test_key_format(self) -> None:
        mock_s3 = MagicMock()
        photo_id = "xYz789"
        key = f"images/{photo_id}.jpg"
        upload_to_s3(mock_s3, "bucket", key, b"img")
        call_args = mock_s3.put_object.call_args
        assert call_args.kwargs["Key"] == "images/xYz789.jpg"
        assert call_args.kwargs["Key"].startswith("images/")
        assert call_args.kwargs["Key"].endswith(".jpg")

    @patch("processing_job.dataset_ingest.unsplash.time.sleep")
    def test_retry_on_client_error_then_success(self, mock_sleep: MagicMock) -> None:
        error_response = {"Error": {"Code": "InternalError", "Message": "Service error"}}
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = [
            ClientError(error_response, "PutObject"),
            ClientError(error_response, "PutObject"),
            None,  # success on 3rd attempt
        ]

        result = upload_to_s3(mock_s3, "bucket", "images/test.jpg", b"data")
        assert result is True
        assert mock_s3.put_object.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("processing_job.dataset_ingest.unsplash.time.sleep")
    def test_all_retries_exhausted(self, mock_sleep: MagicMock) -> None:
        error_response = {"Error": {"Code": "InternalError", "Message": "Service error"}}
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = ClientError(error_response, "PutObject")

        result = upload_to_s3(mock_s3, "bucket", "images/fail.jpg", b"data")
        assert result is False
        assert mock_s3.put_object.call_count == 3

    @patch("processing_job.dataset_ingest.unsplash.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep: MagicMock) -> None:
        error_response = {"Error": {"Code": "InternalError", "Message": "err"}}
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = [
            ClientError(error_response, "PutObject"),
            ClientError(error_response, "PutObject"),
            None,
        ]

        upload_to_s3(mock_s3, "bucket", "images/t.jpg", b"d")
        # Backoff: 1*2^0=1, 1*2^1=2
        assert mock_sleep.call_args_list == [call(1), call(2)]


# ---------------------------------------------------------------------------
# Unit tests: generate_prompt
# ---------------------------------------------------------------------------


class TestGeneratePrompt:
    """generate_prompt with mock Strands Agent and S3 client."""

    def _mock_s3_client(self, image_bytes: bytes = b"fake-jpeg-data") -> MagicMock:
        mock_s3 = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = image_bytes
        mock_s3.get_object.return_value = {"Body": mock_body}
        return mock_s3

    def test_valid_prompt_generation(self) -> None:
        mock_agent = MagicMock()
        mock_agent.return_value = "A cinematic slow pan across a misty mountain"
        mock_s3 = self._mock_s3_client()

        result = generate_prompt(mock_agent, "s3://my-bucket/images/abc.jpg", mock_s3)
        assert result == "A cinematic slow pan across a misty mountain"
        mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="images/abc.jpg")
        # Agent should be called with a list of content blocks
        call_args = mock_agent.call_args[0][0]
        assert len(call_args) == 2
        assert "text" in call_args[0]
        assert "image" in call_args[1]

    def test_empty_response_raises(self) -> None:
        mock_agent = MagicMock()
        mock_agent.return_value = "   "
        mock_s3 = self._mock_s3_client()

        with pytest.raises(ValueError, match="empty response"):
            generate_prompt(mock_agent, "s3://bucket/images/test.jpg", mock_s3)

    def test_agent_exception_propagates(self) -> None:
        mock_agent = MagicMock()
        mock_agent.side_effect = RuntimeError("Bedrock timeout")
        mock_s3 = self._mock_s3_client()

        with pytest.raises(RuntimeError, match="Bedrock timeout"):
            generate_prompt(mock_agent, "s3://bucket/images/test.jpg", mock_s3)

    def test_response_stripped(self) -> None:
        mock_agent = MagicMock()
        mock_agent.return_value = "  padded prompt text  "
        mock_s3 = self._mock_s3_client()

        result = generate_prompt(mock_agent, "s3://bucket/images/desc.jpg", mock_s3)
        assert result == "padded prompt text"


# ---------------------------------------------------------------------------
# Unit tests: write_visual_entries
# ---------------------------------------------------------------------------


class TestWriteVisualEntries:
    """write_visual_entries writes JSON array to inputs_t2v.json."""

    def test_writes_correct_file(self) -> None:
        d = tempfile.mkdtemp()
        entries = [{"id": "a", "prompt": "p1", "image": "s3://b/images/a.jpg"}]
        path = write_visual_entries(entries, d)
        assert os.path.exists(path)
        assert path.endswith("inputs_t2v.json")

    def test_content_matches_entries(self) -> None:
        d = tempfile.mkdtemp()
        entries = [
            {"id": "x", "prompt": "prompt x", "image": "s3://bucket/images/x.jpg"},
            {"id": "y", "prompt": "prompt y", "image": "s3://bucket/images/y.jpg"},
        ]
        path = write_visual_entries(entries, d)
        with open(path) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[0]["id"] == "x"
        assert loaded[1]["prompt"] == "prompt y"

    def test_empty_entries(self) -> None:
        d = tempfile.mkdtemp()
        path = write_visual_entries([], d)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == []

    def test_creates_output_dir(self) -> None:
        d = os.path.join(tempfile.mkdtemp(), "nested", "dir")
        write_visual_entries([{"id": "a", "prompt": "p", "image": "s3://b/i/a.jpg"}], d)
        assert os.path.exists(os.path.join(d, "inputs_t2v.json"))


# ---------------------------------------------------------------------------
# Unit tests: verify_aoss
# ---------------------------------------------------------------------------


class TestVerifyAoss:
    """verify_aoss with mock OpenSearch client."""

    @patch("processing_job.dataset_ingest.main.time.sleep")
    def test_count_meets_expected(self, mock_sleep: MagicMock) -> None:
        mock_oss = MagicMock()
        mock_oss.count.return_value = {"count": 100}

        result = verify_aoss(mock_oss, "test-index", 100, poll_interval=1, timeout=10)
        assert result == 100
        mock_oss.count.assert_called_with(index="test-index")

    @patch("processing_job.dataset_ingest.main.time.sleep")
    def test_count_exceeds_expected(self, mock_sleep: MagicMock) -> None:
        mock_oss = MagicMock()
        mock_oss.count.return_value = {"count": 150}

        result = verify_aoss(mock_oss, "idx", 100, poll_interval=1, timeout=10)
        assert result == 150

    @patch("processing_job.dataset_ingest.main.time.time")
    @patch("processing_job.dataset_ingest.main.time.sleep")
    def test_count_below_expected_after_timeout(self, mock_sleep: MagicMock, mock_time: MagicMock) -> None:
        # Simulate time progression: start=0, then 5, then 15 (past timeout=10)
        mock_time.side_effect = [0, 5, 5, 10, 10, 15]
        mock_oss = MagicMock()
        mock_oss.count.return_value = {"count": 50}

        result = verify_aoss(mock_oss, "idx", 100, poll_interval=1, timeout=10)
        assert result == 50

    @patch("processing_job.dataset_ingest.main.time.sleep")
    def test_count_query_exception_handled(self, mock_sleep: MagicMock) -> None:
        mock_oss = MagicMock()
        # First call fails, second succeeds with enough count
        mock_oss.count.side_effect = [
            Exception("Connection error"),
            {"count": 100},
        ]

        result = verify_aoss(mock_oss, "idx", 100, poll_interval=1, timeout=60)
        assert result == 100


# ---------------------------------------------------------------------------
# Unit tests: cleanup (_delete_s3_prefix, _delete_aoss_documents, cleanup)
# ---------------------------------------------------------------------------


class TestDeleteS3Prefix:
    """_delete_s3_prefix with mock S3 paginator."""

    def test_deletes_objects(self) -> None:
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "images/a.jpg"}, {"Key": "images/b.jpg"}]},
        ]

        count = _delete_s3_prefix(mock_s3, "bucket", "images/")
        assert count == 2
        mock_s3.delete_objects.assert_called_once_with(
            Bucket="bucket",
            Delete={"Objects": [{"Key": "images/a.jpg"}, {"Key": "images/b.jpg"}]},
        )

    def test_empty_prefix(self) -> None:
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"Contents": []}]

        count = _delete_s3_prefix(mock_s3, "bucket", "images/")
        assert count == 0
        mock_s3.delete_objects.assert_not_called()

    def test_multiple_pages(self) -> None:
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "images/a.jpg"}]},
            {"Contents": [{"Key": "images/b.jpg"}, {"Key": "images/c.jpg"}]},
        ]

        count = _delete_s3_prefix(mock_s3, "bucket", "images/")
        assert count == 3
        assert mock_s3.delete_objects.call_count == 2


class TestDeleteAossDocuments:
    """_delete_aoss_documents with mock OpenSearch client."""

    def test_deletes_documents(self) -> None:
        mock_oss = MagicMock()
        mock_oss.count.return_value = {"count": 50}
        mock_oss.delete_by_query.return_value = {"deleted": 50}

        count = _delete_aoss_documents(mock_oss, "test-index")
        assert count == 50
        mock_oss.delete_by_query.assert_called_once_with(
            index="test-index",
            body={"query": {"match_all": {}}},
            refresh=True,
        )

    def test_empty_index(self) -> None:
        mock_oss = MagicMock()
        mock_oss.count.return_value = {"count": 0}

        count = _delete_aoss_documents(mock_oss, "test-index")
        assert count == 0
        mock_oss.delete_by_query.assert_not_called()

    def test_delete_failure_raises(self) -> None:
        mock_oss = MagicMock()
        mock_oss.count.return_value = {"count": 10}
        mock_oss.delete_by_query.side_effect = Exception("AOSS error")

        with pytest.raises(Exception, match="AOSS error"):
            _delete_aoss_documents(mock_oss, "test-index")


class TestCleanup:
    """cleanup orchestrates S3 prefix deletion + AOSS document deletion."""

    @patch("processing_job.dataset_ingest.main._delete_aoss_documents")
    @patch("processing_job.dataset_ingest.main._delete_s3_prefix")
    def test_cleanup_calls_both(self, mock_s3_delete: MagicMock, mock_aoss_delete: MagicMock) -> None:
        mock_s3_delete.side_effect = [10, 5]  # images/ then base64/
        mock_aoss_delete.return_value = 10

        cleanup(MagicMock(), "bucket", MagicMock(), "index")

        assert mock_s3_delete.call_count == 2
        calls = mock_s3_delete.call_args_list
        assert calls[0].args[1:] == ("bucket", "images/")
        assert calls[1].args[1:] == ("bucket", "base64/")
        mock_aoss_delete.assert_called_once()


# ---------------------------------------------------------------------------
# Unit tests: import_dataset_script
# ---------------------------------------------------------------------------


class TestImportDatasetScript:
    """import_dataset_script dynamic module import."""

    def test_missing_script_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset module"):
            import_dataset_script("nonexistent_module_xyz.py")

    def test_missing_load_and_upload_raises(self) -> None:
        with patch("processing_job.dataset_ingest.main.importlib.import_module") as mock_import:
            mock_module = MagicMock(spec=[])  # no attributes
            mock_import.return_value = mock_module

            with pytest.raises(AttributeError, match="missing required"):
                import_dataset_script("unsplash.py")

    def test_valid_script_returns_module(self) -> None:
        with patch("processing_job.dataset_ingest.main.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.load_and_upload = MagicMock()
            mock_import.return_value = mock_module

            result = import_dataset_script("unsplash.py")
            assert result is mock_module
            mock_import.assert_called_once_with("unsplash")


# ---------------------------------------------------------------------------
# Unit tests: _flatten_row
# ---------------------------------------------------------------------------


class TestFlattenRow:
    """_flatten_row merges nested photo dict to top level."""

    def test_nested_photo_dict_flattened(self) -> None:
        row = {
            "photo": {"id": "abc123", "image_url": "https://example.com/img.jpg", "description": "nan"},
            "ai": {"description": "a sunset"},
            "keywords": [{"keyword": "nature"}],
        }
        flat = _flatten_row(row)
        assert flat["id"] == "abc123"
        assert flat["image_url"] == "https://example.com/img.jpg"
        assert flat["description"] == "nan"
        assert flat["ai"] == {"description": "a sunset"}
        assert "photo" not in flat

    def test_already_flat_row_unchanged(self) -> None:
        row = {"id": "abc", "image_url": "https://x.com/i.jpg", "ai": {"description": "test"}}
        flat = _flatten_row(row)
        assert flat["id"] == "abc"
        assert flat["image_url"] == "https://x.com/i.jpg"

    def test_empty_photo_dict(self) -> None:
        row = {"photo": {}, "ai": {"description": "test"}}
        flat = _flatten_row(row)
        assert "photo" not in flat
        assert flat["ai"] == {"description": "test"}

    def test_no_photo_key(self) -> None:
        row = {"ai": {"description": "test"}, "keywords": []}
        flat = _flatten_row(row)
        assert flat == {"ai": {"description": "test"}, "keywords": []}
