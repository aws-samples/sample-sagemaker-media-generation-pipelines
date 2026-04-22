"""
Property-based tests for the model download script.

Uses Hypothesis to verify universal properties across randomly generated inputs.
"""

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from processing_job.model_download.main import download_and_upload

# --- Strategies ---

# Generate valid URL-like strings
urls = st.from_regex(r"https://example\.com/[a-z0-9]{1,20}\.[a-z]{2,4}", fullmatch=True)

# Generate valid S3 key paths (non-empty, no leading slash)
s3_keys = st.from_regex(r"[a-z][a-z0-9/]{0,30}[a-z0-9]\.[a-z]{2,4}", fullmatch=True)

# Generate valid S3 prefix paths for extract items (directory-like, no trailing slash)
s3_prefixes = st.from_regex(r"[a-z][a-z0-9]{0,10}(/[a-z0-9]{1,10}){0,3}", fullmatch=True)


class TestS3ExistenceCheckDeterminesDownloadBehavior:
    """Property 1: S3 existence check determines download behavior (non-extract items).

    **Validates: Requirements 2.1, 2.2, 2.3**

    For any non-extract download item and any S3 existence state:
    - If s3_key_exists returns True → status is "skipped"
    - If s3_key_exists returns False → status is "success"
    """

    @given(
        url=urls,
        s3_key=s3_keys,
        key_exists=st.booleans(),
    )
    @settings(max_examples=100)
    def test_skip_iff_s3_key_exists(self, url: str, s3_key: str, key_exists: bool) -> None:
        """Item is downloaded iff S3 key does not exist."""
        item = {"url": url, "path": s3_key, "extract": False}

        with (
            patch("processing_job.model_download.main.s3_key_exists", return_value=key_exists),
            patch("processing_job.model_download.main.get_file_size", return_value=1024),
            patch("processing_job.model_download.main.simple_upload") as mock_simple,
            patch("processing_job.model_download.main.parallel_download_to_s3"),
            patch("processing_job.model_download.main.stream_upload_to_s3"),
        ):
            result = download_and_upload(item, "test-bucket")

        if key_exists:
            assert result["status"] == "skipped", f"Expected 'skipped' when key exists, got '{result['status']}'"
            assert result["reason"] == "already exists"
            mock_simple.assert_not_called()
        else:
            assert result["status"] == "success", f"Expected 'success' when key missing, got '{result['status']}'"


class TestS3PrefixExistenceCheckDeterminesDownloadBehavior:
    """Property 2: S3 prefix existence check determines download behavior (extract items).

    **Validates: Requirements 2.4**

    For any download item with extract=True and any S3 prefix existence state:
    - If s3_prefix_has_objects returns True → status is "skipped"
    - If s3_prefix_has_objects returns False → status is "success" with extracted=True
    """

    @given(
        url=urls,
        s3_prefix=s3_prefixes,
        prefix_has_objects=st.booleans(),
    )
    @settings(max_examples=100)
    def test_skip_iff_s3_prefix_has_objects(self, url: str, s3_prefix: str, prefix_has_objects: bool) -> None:
        """Item is downloaded iff no objects exist under the S3 prefix."""
        item = {"url": url, "path": s3_prefix, "extract": True}

        with (
            patch(
                "processing_job.model_download.main.s3_prefix_has_objects",
                return_value=prefix_has_objects,
            ),
            patch(
                "processing_job.model_download.main.download_to_local",
            ) as mock_download,
            patch(
                "processing_job.model_download.main.extract_and_upload_zip",
            ) as mock_extract,
        ):
            result = download_and_upload(item, "test-bucket")

        if prefix_has_objects:
            assert result["status"] == "skipped", (
                f"Expected 'skipped' when prefix has objects, got '{result['status']}'"
            )
            assert result["reason"] == "already exists"
            mock_download.assert_not_called()
            mock_extract.assert_not_called()
        else:
            assert result["status"] == "success", f"Expected 'success' when prefix empty, got '{result['status']}'"
            assert result["extracted"] is True
            mock_download.assert_called_once()
            mock_extract.assert_called_once()


import json
import os
import tempfile
import uuid

import pytest

from processing_job.model_download.main import main

pytestmark = pytest.mark.processing


def _is_valid_json(s: str) -> bool:
    """Return True if s is valid JSON."""
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


# Strategy: random strings that are NOT valid JSON
non_json_strings = st.text(min_size=1).filter(lambda s: not _is_valid_json(s))


class TestMalformedOrMissingManifestCausesNonZeroExit:
    """Property 3: Malformed or missing manifest causes non-zero exit.

    **Validates: Requirements 4.4**

    For any string that is not valid JSON (or a missing file path),
    calling main() should exit with a non-zero exit code.
    """

    @given(malformed_content=non_json_strings)
    @settings(max_examples=100)
    def test_malformed_manifest_causes_nonzero_exit(self, malformed_content: str) -> None:
        """main() exits non-zero when manifest contains non-JSON content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write(malformed_content)
            tmp_path = tmp.name

        try:
            with (
                patch("processing_job.model_download.main._get_manifest_path", return_value=tmp_path),
                patch.dict(os.environ, {"MODELS_BUCKET": "test-bucket"}),
            ):
                try:
                    main()
                    raise AssertionError("main() should have called sys.exit but did not")
                except SystemExit as exc:
                    assert exc.code != 0, f"Expected non-zero exit code, got {exc.code}"
        finally:
            os.unlink(tmp_path)

    @given(data=st.data())
    @settings(max_examples=20)
    def test_missing_manifest_causes_nonzero_exit(self, data: st.DataObject) -> None:
        """main() exits non-zero when manifest file does not exist."""
        missing_path = os.path.join(tempfile.gettempdir(), f"nonexistent_{uuid.uuid4().hex}.json")

        with (
            patch("processing_job.model_download.main._get_manifest_path", return_value=missing_path),
            patch.dict(os.environ, {"MODELS_BUCKET": "test-bucket"}),
        ):
            try:
                main()
                raise AssertionError("main() should have called sys.exit but did not")
            except SystemExit as exc:
                assert exc.code != 0, f"Expected non-zero exit code, got {exc.code}"
