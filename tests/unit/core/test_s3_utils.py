"""
Unit and property-based tests for processing_job/common/s3_utils.py.

Covers:
- is_s3_uri: S3 URI detection
- download_s3_to_temp: S3 download with mocked client
- cleanup_temp_file: temp file removal and missing-file handling

**Validates: Requirements 7b.1, 7b.2, 7b.4**
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from processing_job.common.s3_utils import cleanup_temp_file, download_s3_to_temp, is_s3_uri

pytestmark = pytest.mark.core


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# S3 bucket names: 3-63 lowercase alphanumeric chars (simplified)
_bucket_st = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789"),
    min_size=3,
    max_size=20,
)

# S3 key segments: non-empty alphanumeric with common extensions
_key_segment_st = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_-"),
    min_size=1,
    max_size=20,
)

_extension_st = st.sampled_from([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".json", ""])

# Full S3 URI strategy
_s3_uri_st = st.builds(
    lambda bucket, segments, ext: f"s3://{bucket}/{'/'.join(segments)}{ext}",
    bucket=_bucket_st,
    segments=st.lists(_key_segment_st, min_size=1, max_size=3),
    ext=_extension_st,
)

# Non-S3 path strategies: local paths, http URLs, empty strings, random text
_local_path_st = st.builds(
    lambda segments, ext: "/".join(segments) + ext,
    segments=st.lists(_key_segment_st, min_size=1, max_size=3),
    ext=_extension_st,
)

_non_s3_path_st = st.one_of(
    _local_path_st,
    st.builds(lambda p: f"http://example.com/{p}", _key_segment_st),
    st.builds(lambda p: f"https://example.com/{p}", _key_segment_st),
    st.builds(lambda p: f"gs://{p}", _key_segment_st),
    st.just(""),
    st.just("/tmp/image.jpg"),
    st.just("./relative/path.png"),
)


# ---------------------------------------------------------------------------
# Unit tests: is_s3_uri
# ---------------------------------------------------------------------------


class TestIsS3Uri:
    """is_s3_uri returns True for s3:// prefixed strings, False otherwise."""

    def test_s3_uri_returns_true(self) -> None:
        assert is_s3_uri("s3://bucket/key/image.jpg") is True

    def test_s3_uri_root_returns_true(self) -> None:
        assert is_s3_uri("s3://bucket/") is True

    def test_s3_uri_bare_bucket_returns_true(self) -> None:
        assert is_s3_uri("s3://mybucket") is True

    def test_local_path_returns_false(self) -> None:
        assert is_s3_uri("/tmp/image.jpg") is False

    def test_relative_path_returns_false(self) -> None:
        assert is_s3_uri("images/photo.png") is False

    def test_http_url_returns_false(self) -> None:
        assert is_s3_uri("http://example.com/image.jpg") is False

    def test_https_url_returns_false(self) -> None:
        assert is_s3_uri("https://example.com/image.jpg") is False

    def test_empty_string_returns_false(self) -> None:
        assert is_s3_uri("") is False

    def test_s3_without_slashes_returns_false(self) -> None:
        assert is_s3_uri("s3:bucket/key") is False

    def test_uppercase_s3_returns_false(self) -> None:
        assert is_s3_uri("S3://bucket/key") is False


# ---------------------------------------------------------------------------
# Unit tests: download_s3_to_temp
# ---------------------------------------------------------------------------


class TestDownloadS3ToTemp:
    """download_s3_to_temp downloads from S3 to a temp file using a mocked client."""

    def test_downloads_to_temp_file(self, tmp_path: object) -> None:
        mock_client = MagicMock()
        result = download_s3_to_temp("s3://mybucket/images/photo.jpg", s3_client=mock_client)

        assert os.path.exists(result)
        assert result.endswith(".jpg")
        mock_client.download_file.assert_called_once_with("mybucket", "images/photo.jpg", result)

        # Cleanup
        os.remove(result)

    def test_preserves_png_extension(self) -> None:
        mock_client = MagicMock()
        result = download_s3_to_temp("s3://bucket/path/to/img.png", s3_client=mock_client)

        assert result.endswith(".png")
        mock_client.download_file.assert_called_once()

        os.remove(result)

    def test_no_extension_key(self) -> None:
        mock_client = MagicMock()
        result = download_s3_to_temp("s3://bucket/noext", s3_client=mock_client)

        mock_client.download_file.assert_called_once_with("bucket", "noext", result)

        os.remove(result)

    def test_nested_key_path(self) -> None:
        mock_client = MagicMock()
        result = download_s3_to_temp("s3://mybucket/a/b/c/deep.webp", s3_client=mock_client)

        assert result.endswith(".webp")
        mock_client.download_file.assert_called_once_with("mybucket", "a/b/c/deep.webp", result)

        os.remove(result)

    def test_creates_boto3_client_when_none(self) -> None:
        with patch.dict("sys.modules", {"boto3": MagicMock()}) as _:
            import boto3 as mock_boto3  # noqa: F811

            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            result = download_s3_to_temp("s3://bucket/key.jpg")

            mock_boto3.client.assert_called_once_with("s3")
            mock_client.download_file.assert_called_once()

            os.remove(result)


# ---------------------------------------------------------------------------
# Unit tests: cleanup_temp_file
# ---------------------------------------------------------------------------


class TestCleanupTempFile:
    """cleanup_temp_file removes existing files and handles missing files gracefully."""

    def test_removes_existing_file(self, tmp_path: object) -> None:
        f = tmp_path / "temp_image.jpg"  # type: ignore[operator]
        f.write_text("fake image data")
        assert f.exists()

        cleanup_temp_file(str(f))
        assert not f.exists()

    def test_missing_file_no_error(self) -> None:
        cleanup_temp_file("/tmp/nonexistent_file_abc123xyz.jpg")

    def test_missing_file_no_error_empty_path(self, tmp_path: object) -> None:
        cleanup_temp_file(str(tmp_path / "does_not_exist.png"))  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Property 8: S3 URI detection — s3:// prefix triggers download, other paths are local
# ---------------------------------------------------------------------------


class TestS3UriDetectionProperty:
    """Property-based tests for S3 URI detection.

    **Validates: Requirements 7b.1, 7b.2**
    """

    # Feature: vrag-llm-container, Property 8: S3 URI detection — s3:// prefix triggers download, other paths are local

    @given(uri=_s3_uri_st)
    @settings(max_examples=100)
    def test_s3_uris_detected(self, uri: str) -> None:
        """Any string starting with s3:// is detected as an S3 URI."""
        assert is_s3_uri(uri) is True

    @given(path=_non_s3_path_st)
    @settings(max_examples=100)
    def test_non_s3_paths_not_detected(self, path: str) -> None:
        """Any string NOT starting with s3:// is not detected as an S3 URI."""
        assert is_s3_uri(path) is False

    @given(uri=_s3_uri_st)
    @settings(max_examples=100)
    def test_s3_uri_download_returns_temp_path(self, uri: str) -> None:
        """For any s3:// URI, download_s3_to_temp returns a local temp path."""
        mock_client = MagicMock()
        result = download_s3_to_temp(uri, s3_client=mock_client)

        assert os.path.exists(result)
        assert not result.startswith("s3://")
        mock_client.download_file.assert_called_once()

        os.remove(result)


# ---------------------------------------------------------------------------
# Property 9: Temp file cleanup after i2v processing
# ---------------------------------------------------------------------------


class TestTempFileCleanupProperty:
    """Property-based tests for temp file cleanup.

    **Validates: Requirements 7b.4**
    """

    # Feature: vrag-llm-container, Property 9: Temp file cleanup after i2v processing

    @given(
        content=st.binary(min_size=1, max_size=100),
        ext=_extension_st,
    )
    @settings(max_examples=100)
    def test_cleanup_removes_temp_file(self, content: bytes, ext: str) -> None:
        """For any temp file, cleanup_temp_file removes it from disk."""
        import tempfile as _tf

        fd, path = _tf.mkstemp(suffix=ext)
        os.close(fd)
        with open(path, "wb") as fh:
            fh.write(content)
        assert os.path.exists(path)

        cleanup_temp_file(path)
        assert not os.path.exists(path)

    @given(
        name=st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789"),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_cleanup_missing_file_no_error(self, name: str) -> None:
        """For any non-existent path, cleanup_temp_file does not raise."""
        import tempfile as _tf

        path = os.path.join(_tf.gettempdir(), f"{name}_missing_pbt.tmp")
        # Ensure it doesn't exist
        if os.path.exists(path):
            os.remove(path)
        cleanup_temp_file(path)
