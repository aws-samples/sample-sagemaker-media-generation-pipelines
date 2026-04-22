"""
Unit tests for the model download processing job script.

Tests verify that files are deleted after upload to S3 to conserve disk space.
"""

import io
import json
import os
import shutil
import tempfile
import zipfile
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from processing_job.model_download import main as download_module
from processing_job.model_download.main import (
    _download_with_retry,
    download_and_upload,
    download_to_local,
    extract_and_upload_zip,
    s3_key_exists,
    s3_prefix_has_objects,
)

pytestmark = pytest.mark.processing


def _make_urlopen_mock(data: bytes):
    """Create a mock urlopen context manager returning data as a file-like stream."""
    stream = io.BytesIO(data)
    mock_resp = MagicMock()
    mock_resp.read = stream.read
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestDownloadToLocal:
    """Tests for download_to_local function."""

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "subdir", "nested", "file.bin")
            mock_resp = _make_urlopen_mock(b"test data")

            with patch("processing_job.model_download.main.urlopen", return_value=mock_resp):
                download_to_local("http://example.com/file.bin", local_path)

            assert os.path.exists(local_path)
            with open(local_path, "rb") as f:
                assert f.read() == b"test data"


class TestExtractAndUploadZip:
    """Tests for extract_and_upload_zip function."""

    def test_uploads_all_files_from_zip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "test.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("file1.txt", "content1")
                zf.writestr("subdir/file2.txt", "content2")

            mock_s3 = MagicMock()
            with patch("processing_job.model_download.main.s3_client", mock_s3):
                extract_and_upload_zip(zip_path, "test-bucket", "models/prefix")

            assert mock_s3.put_object.call_count == 2
            keys = [c.kwargs["Key"] for c in mock_s3.put_object.call_args_list]
            assert "models/prefix/file1.txt" in keys
            assert "models/prefix/subdir/file2.txt" in keys


class TestDownloadAndUploadCleanup:
    """Tests for file cleanup after S3 upload."""

    def test_extract_deletes_local_zip_after_upload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("processing_job.model_download.main.LOCAL_DOWNLOAD_DIR", tmpdir):
                src_zip = os.path.join(tmpdir, "_src.zip")
                with zipfile.ZipFile(src_zip, "w") as zf:
                    zf.writestr("model.bin", b"model data")

                def mock_download(url, path):
                    shutil.copy(src_zip, path)

                mock_s3 = MagicMock()
                with (
                    patch("processing_job.model_download.main.download_to_local", side_effect=mock_download),
                    patch("processing_job.model_download.main.s3_client", mock_s3),
                ):
                    result = download_and_upload(
                        {"url": "http://example.com/models.zip", "path": "models/out", "extract": True},
                        "test-bucket",
                    )

                assert result["status"] == "success"
                assert result["extracted"] is True
                assert not os.path.exists(os.path.join(tmpdir, "models.zip"))

    def test_extract_cleans_up_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("processing_job.model_download.main.LOCAL_DOWNLOAD_DIR", tmpdir):
                bad_zip = os.path.join(tmpdir, "_bad.zip")
                with open(bad_zip, "wb") as f:
                    f.write(b"not a zip")

                def mock_download(url, path):
                    shutil.copy(bad_zip, path)

                with patch("processing_job.model_download.main.download_to_local", side_effect=mock_download):
                    with pytest.raises(zipfile.BadZipFile):
                        download_and_upload(
                            {"url": "http://example.com/bad.zip", "path": "out", "extract": True},
                            "test-bucket",
                        )

                assert not os.path.exists(os.path.join(tmpdir, "bad.zip"))

    def test_simple_upload_no_local_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("processing_job.model_download.main.LOCAL_DOWNLOAD_DIR", tmpdir):
                with (
                    patch("processing_job.model_download.main.get_file_size", return_value=1024),
                    patch("processing_job.model_download.main.simple_upload"),
                ):
                    result = download_and_upload(
                        {"url": "http://example.com/small.bin", "path": "models/small.bin"},
                        "test-bucket",
                    )

                assert result["status"] == "success"
                real_files = [f for f in os.listdir(tmpdir) if not f.startswith("_")]
                assert len(real_files) == 0


class TestMainCleanup:
    """Tests for cleanup at the end of main().

    Each test writes downloads.json into its own temp directory and patches
    ``__file__`` on the module so ``main()`` reads from there.  This avoids
    a race condition when pytest-xdist runs tests in parallel — previously
    both tests shared the real module path and could delete each other's
    manifest.
    """

    def test_cleanup_download_dir_at_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            leftover = os.path.join(tmpdir, "leftover.tmp")
            with open(leftover, "w") as f:
                f.write("leftover")

            # Write manifest into an isolated temp dir
            manifest_dir = tempfile.mkdtemp()
            manifest = os.path.join(manifest_dir, "downloads.json")
            with open(manifest, "w") as f:
                json.dump([], f)

            try:
                with (
                    patch.object(download_module, "_get_manifest_path", return_value=manifest),
                    patch.object(download_module, "LOCAL_DOWNLOAD_DIR", tmpdir),
                    patch.dict(os.environ, {"MODELS_BUCKET": "test-bucket"}),
                    patch.object(download_module, "s3_client", MagicMock()),
                ):
                    download_module.main()
            finally:
                shutil.rmtree(manifest_dir, ignore_errors=True)

            assert not os.path.exists(leftover), "Leftover files should be cleaned up"

    def test_cleanup_happens_even_with_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            leftover = os.path.join(tmpdir, "leftover.tmp")
            with open(leftover, "w") as f:
                f.write("leftover")

            downloads = [{"url": "http://example.com/fail.bin", "path": "fail.bin"}]

            # Write manifest into an isolated temp dir
            manifest_dir = tempfile.mkdtemp()
            manifest = os.path.join(manifest_dir, "downloads.json")
            with open(manifest, "w") as f:
                json.dump(downloads, f)

            try:
                with (
                    patch.object(download_module, "_get_manifest_path", return_value=manifest),
                    patch.object(download_module, "LOCAL_DOWNLOAD_DIR", tmpdir),
                    patch.dict(os.environ, {"MODELS_BUCKET": "test-bucket"}),
                    patch.object(download_module, "s3_client", MagicMock()),
                    patch.object(download_module, "_download_with_retry", side_effect=Exception("network error")),
                    pytest.raises(SystemExit),
                ):
                    download_module.main()
            finally:
                shutil.rmtree(manifest_dir, ignore_errors=True)

            assert not os.path.exists(leftover), "Cleanup should happen even on failure"


class TestS3KeyExists:
    """Tests for s3_key_exists() — validates Requirement 2.1."""

    def test_returns_true_when_key_exists(self):
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {"ContentLength": 1024}
        with patch("processing_job.model_download.main.s3_client", mock_s3):
            assert s3_key_exists("my-bucket", "models/file.bin") is True
        mock_s3.head_object.assert_called_once_with(Bucket="my-bucket", Key="models/file.bin")

    def test_returns_false_on_404(self):
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        with patch("processing_job.model_download.main.s3_client", mock_s3):
            assert s3_key_exists("my-bucket", "models/missing.bin") is False

    def test_returns_false_on_other_client_error(self):
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject")
        with patch("processing_job.model_download.main.s3_client", mock_s3):
            assert s3_key_exists("my-bucket", "models/forbidden.bin") is False

    def test_returns_false_on_unexpected_exception(self):
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = RuntimeError("connection reset")
        with patch("processing_job.model_download.main.s3_client", mock_s3):
            assert s3_key_exists("my-bucket", "models/error.bin") is False


class TestS3PrefixHasObjects:
    """Tests for s3_prefix_has_objects() — validates Requirement 2.4."""

    def test_returns_true_when_objects_exist(self):
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {"KeyCount": 1, "Contents": [{"Key": "prefix/file.bin"}]}
        with patch("processing_job.model_download.main.s3_client", mock_s3):
            assert s3_prefix_has_objects("my-bucket", "prefix/") is True
        mock_s3.list_objects_v2.assert_called_once_with(Bucket="my-bucket", Prefix="prefix/", MaxKeys=1)

    def test_returns_false_when_no_objects(self):
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {"KeyCount": 0}
        with patch("processing_job.model_download.main.s3_client", mock_s3):
            assert s3_prefix_has_objects("my-bucket", "empty-prefix/") is False

    def test_returns_false_on_client_error(self):
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "ListObjectsV2"
        )
        with patch("processing_job.model_download.main.s3_client", mock_s3):
            assert s3_prefix_has_objects("my-bucket", "denied-prefix/") is False

    def test_returns_false_on_unexpected_exception(self):
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.side_effect = RuntimeError("timeout")
        with patch("processing_job.model_download.main.s3_client", mock_s3):
            assert s3_prefix_has_objects("my-bucket", "timeout-prefix/") is False

    def test_returns_false_when_key_count_missing(self):
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {}
        with patch("processing_job.model_download.main.s3_client", mock_s3):
            assert s3_prefix_has_objects("my-bucket", "weird-prefix/") is False


class TestDownloadAndUploadSkipBehavior:
    """Tests for skip behavior in download_and_upload() — validates Requirements 2.2, 2.3, 2.4."""

    def test_non_extract_skips_when_key_exists(self):
        """When s3_key_exists returns True for a non-extract item, no download occurs."""
        mock_s3 = MagicMock()
        with (
            patch("processing_job.model_download.main.s3_client", mock_s3),
            patch("processing_job.model_download.main.s3_key_exists", return_value=True),
            patch("processing_job.model_download.main.get_file_size") as mock_get_size,
            patch("processing_job.model_download.main.simple_upload") as mock_simple,
            patch("processing_job.model_download.main.parallel_download_to_s3") as mock_parallel,
            patch("processing_job.model_download.main.stream_upload_to_s3") as mock_stream,
        ):
            result = download_and_upload(
                {"url": "http://example.com/model.bin", "path": "models/model.bin"},
                "test-bucket",
            )

        assert result == {"status": "skipped", "path": "models/model.bin", "reason": "already exists"}
        mock_get_size.assert_not_called()
        mock_simple.assert_not_called()
        mock_parallel.assert_not_called()
        mock_stream.assert_not_called()

    def test_non_extract_downloads_when_key_missing(self):
        """When s3_key_exists returns False for a non-extract item, download proceeds."""
        with (
            patch("processing_job.model_download.main.s3_key_exists", return_value=False),
            patch("processing_job.model_download.main.get_file_size", return_value=1024),
            patch("processing_job.model_download.main.simple_upload") as mock_simple,
        ):
            result = download_and_upload(
                {"url": "http://example.com/model.bin", "path": "models/model.bin"},
                "test-bucket",
            )

        assert result["status"] == "success"
        mock_simple.assert_called_once()

    def test_extract_skips_when_prefix_has_objects(self):
        """When s3_prefix_has_objects returns True for an extract item, no download occurs."""
        with (
            patch("processing_job.model_download.main.s3_prefix_has_objects", return_value=True),
            patch("processing_job.model_download.main.download_to_local") as mock_dl,
            patch("processing_job.model_download.main.extract_and_upload_zip") as mock_extract,
        ):
            result = download_and_upload(
                {"url": "http://example.com/models.zip", "path": "models/prefix", "extract": True},
                "test-bucket",
            )

        assert result == {"status": "skipped", "path": "models/prefix", "reason": "already exists"}
        mock_dl.assert_not_called()
        mock_extract.assert_not_called()

    def test_extract_downloads_when_prefix_empty(self):
        """When s3_prefix_has_objects returns False for an extract item, download proceeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid zip for extraction
            src_zip = os.path.join(tmpdir, "_src.zip")
            with zipfile.ZipFile(src_zip, "w") as zf:
                zf.writestr("file.bin", b"data")

            def mock_download(url, path):
                shutil.copy(src_zip, path)

            mock_s3 = MagicMock()
            with (
                patch("processing_job.model_download.main.LOCAL_DOWNLOAD_DIR", tmpdir),
                patch("processing_job.model_download.main.s3_prefix_has_objects", return_value=False),
                patch("processing_job.model_download.main.download_to_local", side_effect=mock_download),
                patch("processing_job.model_download.main.s3_client", mock_s3),
            ):
                result = download_and_upload(
                    {"url": "http://example.com/models.zip", "path": "models/prefix", "extract": True},
                    "test-bucket",
                )

        assert result["status"] == "success"
        assert result["extracted"] is True


class TestMainSummaryAndConfig:
    """Tests for main() summary logging and LOCAL_DOWNLOAD_DIR — validates Requirements 2.5, 4.1."""

    def test_local_download_dir_is_codebuild_compatible(self):
        """LOCAL_DOWNLOAD_DIR should be /tmp/model_download, not the SageMaker path."""
        assert download_module.LOCAL_DOWNLOAD_DIR == "/tmp/model_download"

    def test_summary_logs_skipped_and_downloaded_counts(self):
        """main() logs Skipped and Downloaded counts in the final summary."""
        downloads = [
            {"url": "http://example.com/a.bin", "path": "models/a.bin"},
            {"url": "http://example.com/b.bin", "path": "models/b.bin"},
            {"url": "http://example.com/c.bin", "path": "models/c.bin"},
        ]

        # First item skipped, second and third downloaded
        call_count = 0

        def mock_download_and_upload(item, bucket, index=0, total=0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"status": "skipped", "path": item["path"], "reason": "already exists"}
            return {"status": "success", "path": item["path"]}

        manifest_dir = tempfile.mkdtemp()
        manifest = os.path.join(manifest_dir, "downloads.json")
        with open(manifest, "w") as f:
            json.dump(downloads, f)

        log_messages = []

        def capture_log(msg, *args, **kwargs):
            try:
                log_messages.append(msg.format(*args))
            except (IndexError, KeyError):
                log_messages.append(msg)

        try:
            with (
                patch.object(download_module, "_get_manifest_path", return_value=manifest),
                patch.object(download_module, "LOCAL_DOWNLOAD_DIR", tempfile.mkdtemp()),
                patch.dict(os.environ, {"MODELS_BUCKET": "test-bucket"}),
                patch.object(download_module, "s3_client", MagicMock()),
                patch.object(download_module, "_download_with_retry", side_effect=mock_download_and_upload),
                patch.object(download_module, "get_file_size", return_value=None),
                patch.object(download_module.logger, "info", side_effect=capture_log),
            ):
                download_module.main()
        finally:
            shutil.rmtree(manifest_dir, ignore_errors=True)

        # Check that skipped and downloaded counts appear in log output
        skipped_logs = [m for m in log_messages if "Skipped" in m and "1" in m]
        downloaded_logs = [m for m in log_messages if "Downloaded" in m and "2" in m]
        assert skipped_logs, f"Expected 'Skipped: 1' in logs, got: {log_messages}"
        assert downloaded_logs, f"Expected 'Downloaded: 2' in logs, got: {log_messages}"

    def test_summary_logs_all_skipped(self):
        """When all items are skipped, summary reflects 0 downloaded."""
        downloads = [
            {"url": "http://example.com/a.bin", "path": "models/a.bin"},
            {"url": "http://example.com/b.bin", "path": "models/b.bin"},
        ]

        def mock_download_and_upload(item, bucket, index=0, total=0):
            return {"status": "skipped", "path": item["path"], "reason": "already exists"}

        manifest_dir = tempfile.mkdtemp()
        manifest = os.path.join(manifest_dir, "downloads.json")
        with open(manifest, "w") as f:
            json.dump(downloads, f)

        log_messages = []

        def capture_log(msg, *args, **kwargs):
            try:
                log_messages.append(msg.format(*args))
            except (IndexError, KeyError):
                log_messages.append(msg)

        try:
            with (
                patch.object(download_module, "_get_manifest_path", return_value=manifest),
                patch.object(download_module, "LOCAL_DOWNLOAD_DIR", tempfile.mkdtemp()),
                patch.dict(os.environ, {"MODELS_BUCKET": "test-bucket"}),
                patch.object(download_module, "s3_client", MagicMock()),
                patch.object(download_module, "_download_with_retry", side_effect=mock_download_and_upload),
                patch.object(download_module, "get_file_size", return_value=None),
                patch.object(download_module.logger, "info", side_effect=capture_log),
            ):
                download_module.main()
        finally:
            shutil.rmtree(manifest_dir, ignore_errors=True)

        skipped_logs = [m for m in log_messages if "Skipped" in m and "2" in m]
        downloaded_logs = [m for m in log_messages if "Downloaded" in m and "0" in m]
        assert skipped_logs, f"Expected 'Skipped: 2' in logs, got: {log_messages}"
        assert downloaded_logs, f"Expected 'Downloaded: 0' in logs, got: {log_messages}"


class TestDownloadWithRetry:
    """_download_with_retry exponential backoff behaviour."""

    _item = {"url": "https://example.com/model.bin", "path": "models/model.bin"}

    def test_succeeds_on_first_attempt(self):
        with patch.object(download_module, "download_and_upload", return_value={"status": "success"}) as mock:
            result = _download_with_retry(self._item, "bucket", index=1, total=1)
        assert result == {"status": "success"}
        assert mock.call_count == 1

    def test_succeeds_on_second_attempt(self):
        with (
            patch.object(
                download_module,
                "download_and_upload",
                side_effect=[ConnectionError("timeout"), {"status": "success"}],
            ) as mock,
            patch.object(download_module.time, "sleep") as mock_sleep,
        ):
            result = _download_with_retry(self._item, "bucket", index=1, total=1)
        assert result == {"status": "success"}
        assert mock.call_count == 2
        mock_sleep.assert_called_once_with(30)

    def test_backoff_doubles_each_retry(self):
        with (
            patch.object(
                download_module,
                "download_and_upload",
                side_effect=[ConnectionError("1"), ConnectionError("2"), {"status": "success"}],
            ),
            patch.object(download_module.time, "sleep") as mock_sleep,
        ):
            result = _download_with_retry(self._item, "bucket", index=1, total=1)
        assert result == {"status": "success"}
        assert mock_sleep.call_args_list == [
            ((30,),),
            ((60,),),
        ]

    def test_raises_after_all_retries_exhausted(self):
        with (
            patch.object(
                download_module,
                "download_and_upload",
                side_effect=ConnectionError("persistent failure"),
            ),
            patch.object(download_module.time, "sleep"),
        ):
            with pytest.raises(ConnectionError, match="persistent failure"):
                _download_with_retry(self._item, "bucket", index=1, total=1)

    def test_retry_count_matches_max_retries(self):
        with (
            patch.object(
                download_module,
                "download_and_upload",
                side_effect=ConnectionError("fail"),
            ) as mock,
            patch.object(download_module.time, "sleep"),
        ):
            with pytest.raises(ConnectionError):
                _download_with_retry(self._item, "bucket", index=1, total=1)
        assert mock.call_count == download_module.MAX_RETRIES

    def test_skipped_result_not_retried(self):
        with patch.object(
            download_module,
            "download_and_upload",
            return_value={"status": "skipped", "path": "models/model.bin", "reason": "already exists"},
        ) as mock:
            result = _download_with_retry(self._item, "bucket", index=1, total=1)
        assert result["status"] == "skipped"
        assert mock.call_count == 1
