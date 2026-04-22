"""
Unit tests for the open_images_v7 dataset loader and updated multimodal prompt generation.

Covers:
- load_and_upload interface: return schema (keys: id, description, s3_uri)
- _download_csv: successful download, nrows calculation, error handling
- _passes_resolution_filter: within bounds, outside bounds, boundary values
- _check_image_dimensions: successful header read, network error, corrupt data
- _process_image: download failure, timeout, description placeholder
- generate_prompt (main.py): S3 download, content block structure, error handling

**Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 13.10**
"""

from __future__ import annotations

import sys
from io import BytesIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

# Make the bare `unsplash` import in open_images_v7.py resolve to the
# full package path used in the test environment.
import processing_job.dataset_ingest.unsplash as _unsplash_mod

sys.modules.setdefault("unsplash", _unsplash_mod)

import processing_job.dataset_ingest.open_images_v7 as oiv7  # noqa: E402
from processing_job.dataset_ingest.main import generate_prompt  # noqa: E402

pytestmark = pytest.mark.steps_setup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_jpeg_bytes() -> bytes:
    """Create a minimal valid JPEG via PIL so Image.open succeeds."""
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (640, 480))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_csv_df(n: int = 5) -> pd.DataFrame:
    """Build a small DataFrame mimicking the Open Images CSV schema."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "ImageID": f"img_{i:04d}",
                "Subset": "train",
                "OriginalURL": f"https://flickr.com/photos/{i}.jpg",
                "OriginalLandingURL": f"https://flickr.com/landing/{i}",
                "License": "https://creativecommons.org/licenses/by/2.0/",
                "Author": f"author_{i}",
                "Title": f"Title {i}",
                "Rotation": 0.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. TestLoadAndUploadInterface
# ---------------------------------------------------------------------------


class TestLoadAndUploadInterface:
    """Verify load_and_upload returns list of dicts with keys id, description, s3_uri."""

    @patch.object(oiv7, "_download_csv")
    @patch.object(oiv7, "_check_image_dimensions")
    @patch.object(oiv7, "download_image")
    @patch.object(oiv7, "resize_image")
    @patch.object(oiv7, "upload_to_s3", return_value=True)
    @patch.object(oiv7, "_key_exists", return_value=False)
    @patch("PIL.Image.open")
    def test_returns_list_of_dicts_with_correct_keys(
        self,
        mock_pil_open: MagicMock,
        mock_key_exists: MagicMock,
        mock_upload: MagicMock,
        mock_resize: MagicMock,
        mock_download: MagicMock,
        mock_check_dims: MagicMock,
        mock_csv: MagicMock,
    ) -> None:
        fake_jpeg = _fake_jpeg_bytes()
        mock_csv.return_value = _make_csv_df(3)
        mock_check_dims.return_value = (640, 480)
        mock_download.return_value = fake_jpeg
        mock_resize.return_value = fake_jpeg

        # PIL.Image.open for the _process_image dimension check
        mock_img = MagicMock()
        mock_img.size = (640, 480)
        mock_pil_open.return_value = mock_img

        with (
            patch.object(oiv7, "DATASET_URL", "https://example.com/test.csv"),
            patch.object(oiv7, "MIN_WIDTH", 100),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 100),
            patch.object(oiv7, "MAX_HEIGHT", 720),
            patch.object(oiv7, "CSV_READ_MULTIPLIER", 3),
        ):
            results = oiv7.load_and_upload(MagicMock(), "test-bucket", 10)

        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, dict)
            assert set(item.keys()) == {"id", "description", "s3_uri"}
            assert item["id"]
            assert item["description"]
            assert item["s3_uri"].startswith("s3://")
            assert "/images/" in item["s3_uri"]

    @patch.object(oiv7, "_download_csv")
    @patch.object(oiv7, "_check_image_dimensions")
    @patch.object(oiv7, "download_image")
    @patch.object(oiv7, "resize_image")
    @patch.object(oiv7, "upload_to_s3", return_value=True)
    @patch.object(oiv7, "_key_exists", return_value=False)
    @patch("PIL.Image.open")
    def test_description_equals_image_id(
        self,
        mock_pil_open: MagicMock,
        mock_key_exists: MagicMock,
        mock_upload: MagicMock,
        mock_resize: MagicMock,
        mock_download: MagicMock,
        mock_check_dims: MagicMock,
        mock_csv: MagicMock,
    ) -> None:
        fake_jpeg = _fake_jpeg_bytes()
        mock_csv.return_value = _make_csv_df(2)
        mock_check_dims.return_value = (640, 480)
        mock_download.return_value = fake_jpeg
        mock_resize.return_value = fake_jpeg

        mock_img = MagicMock()
        mock_img.size = (640, 480)
        mock_pil_open.return_value = mock_img

        with (
            patch.object(oiv7, "DATASET_URL", "https://example.com/test.csv"),
            patch.object(oiv7, "MIN_WIDTH", 100),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 100),
            patch.object(oiv7, "MAX_HEIGHT", 720),
            patch.object(oiv7, "CSV_READ_MULTIPLIER", 3),
        ):
            results = oiv7.load_and_upload(MagicMock(), "test-bucket", 10)

        for item in results:
            assert item["description"] == item["id"]


# ---------------------------------------------------------------------------
# 1b. TestLoadAndUploadBatching
# ---------------------------------------------------------------------------


class TestLoadAndUploadBatching:
    """Verify load_and_upload paginates through CSV batches until target is met."""

    @patch.object(oiv7, "_download_csv")
    @patch.object(oiv7, "_check_image_dimensions")
    @patch.object(oiv7, "download_image")
    @patch.object(oiv7, "resize_image")
    @patch.object(oiv7, "upload_to_s3", return_value=True)
    @patch.object(oiv7, "_key_exists", return_value=False)
    @patch("PIL.Image.open")
    def test_fetches_second_batch_when_first_insufficient(
        self,
        mock_pil_open: MagicMock,
        mock_key_exists: MagicMock,
        mock_upload: MagicMock,
        mock_resize: MagicMock,
        mock_download: MagicMock,
        mock_check_dims: MagicMock,
        mock_csv: MagicMock,
    ) -> None:
        """When first batch yields fewer than limit, a second batch is fetched."""
        fake_jpeg = _fake_jpeg_bytes()
        # First batch: 2 rows, second batch: 2 rows — target is 3
        mock_csv.side_effect = [_make_csv_df(2), _make_csv_df(2)]
        mock_check_dims.return_value = (640, 480)
        mock_download.return_value = fake_jpeg
        mock_resize.return_value = fake_jpeg

        mock_img = MagicMock()
        mock_img.size = (640, 480)
        mock_pil_open.return_value = mock_img

        with (
            patch.object(oiv7, "DATASET_URL", "https://example.com/test.csv"),
            patch.object(oiv7, "MIN_WIDTH", 100),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 100),
            patch.object(oiv7, "MAX_HEIGHT", 720),
            patch.object(oiv7, "CSV_READ_MULTIPLIER", 1),
        ):
            results = oiv7.load_and_upload(MagicMock(), "test-bucket", 3)

        assert len(results) == 3
        assert mock_csv.call_count == 2
        # First call: offset=0, second call: offset=2
        mock_csv.assert_any_call("https://example.com/test.csv", 3, skiprows=0)
        mock_csv.assert_any_call("https://example.com/test.csv", 3, skiprows=2)

    @patch.object(oiv7, "_download_csv")
    @patch.object(oiv7, "_check_image_dimensions")
    @patch.object(oiv7, "download_image")
    @patch.object(oiv7, "resize_image")
    @patch.object(oiv7, "upload_to_s3", return_value=True)
    @patch.object(oiv7, "_key_exists", return_value=False)
    @patch("PIL.Image.open")
    def test_stops_when_csv_exhausted(
        self,
        mock_pil_open: MagicMock,
        mock_key_exists: MagicMock,
        mock_upload: MagicMock,
        mock_resize: MagicMock,
        mock_download: MagicMock,
        mock_check_dims: MagicMock,
        mock_csv: MagicMock,
    ) -> None:
        """When CSV returns empty DataFrame, stops and returns what it has."""
        fake_jpeg = _fake_jpeg_bytes()
        mock_csv.side_effect = [_make_csv_df(2), pd.DataFrame()]
        mock_check_dims.return_value = (640, 480)
        mock_download.return_value = fake_jpeg
        mock_resize.return_value = fake_jpeg

        mock_img = MagicMock()
        mock_img.size = (640, 480)
        mock_pil_open.return_value = mock_img

        with (
            patch.object(oiv7, "DATASET_URL", "https://example.com/test.csv"),
            patch.object(oiv7, "MIN_WIDTH", 100),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 100),
            patch.object(oiv7, "MAX_HEIGHT", 720),
            patch.object(oiv7, "CSV_READ_MULTIPLIER", 1),
        ):
            results = oiv7.load_and_upload(MagicMock(), "test-bucket", 10)

        assert len(results) == 2
        assert mock_csv.call_count == 2

    @patch.object(oiv7, "_download_csv")
    @patch.object(oiv7, "_check_image_dimensions")
    @patch.object(oiv7, "download_image")
    @patch.object(oiv7, "resize_image")
    @patch.object(oiv7, "upload_to_s3", return_value=True)
    @patch.object(oiv7, "_key_exists", return_value=False)
    @patch("PIL.Image.open")
    def test_skips_filtered_images_and_continues_to_next_batch(
        self,
        mock_pil_open: MagicMock,
        mock_key_exists: MagicMock,
        mock_upload: MagicMock,
        mock_resize: MagicMock,
        mock_download: MagicMock,
        mock_check_dims: MagicMock,
        mock_csv: MagicMock,
    ) -> None:
        """Images that fail resolution filter are skipped; batching continues."""
        fake_jpeg = _fake_jpeg_bytes()
        # First batch: 3 rows, all filtered out (too small)
        # Second batch: 3 rows, all pass
        mock_csv.side_effect = [_make_csv_df(3), _make_csv_df(3)]
        mock_check_dims.return_value = None  # not used in _handle_row anymore
        mock_download.return_value = fake_jpeg
        mock_resize.return_value = fake_jpeg

        # First 3 images return small dims (filtered), next 3 return good dims
        small_img = MagicMock()
        small_img.size = (50, 50)
        good_img = MagicMock()
        good_img.size = (640, 480)
        mock_pil_open.side_effect = [small_img, small_img, small_img, good_img, good_img, good_img]

        with (
            patch.object(oiv7, "DATASET_URL", "https://example.com/test.csv"),
            patch.object(oiv7, "MIN_WIDTH", 200),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 200),
            patch.object(oiv7, "MAX_HEIGHT", 720),
            patch.object(oiv7, "CSV_READ_MULTIPLIER", 1),
        ):
            results = oiv7.load_and_upload(MagicMock(), "test-bucket", 3)

        assert len(results) == 3
        assert mock_csv.call_count == 2

    @patch.object(oiv7, "_download_csv")
    @patch.object(oiv7, "_check_image_dimensions")
    @patch.object(oiv7, "download_image")
    @patch.object(oiv7, "resize_image")
    @patch.object(oiv7, "upload_to_s3", return_value=True)
    @patch.object(oiv7, "_key_exists", return_value=False)
    @patch("PIL.Image.open")
    def test_single_batch_sufficient(
        self,
        mock_pil_open: MagicMock,
        mock_key_exists: MagicMock,
        mock_upload: MagicMock,
        mock_resize: MagicMock,
        mock_download: MagicMock,
        mock_check_dims: MagicMock,
        mock_csv: MagicMock,
    ) -> None:
        """When first batch has enough images, no second batch is fetched."""
        fake_jpeg = _fake_jpeg_bytes()
        mock_csv.return_value = _make_csv_df(5)
        mock_check_dims.return_value = (640, 480)
        mock_download.return_value = fake_jpeg
        mock_resize.return_value = fake_jpeg

        mock_img = MagicMock()
        mock_img.size = (640, 480)
        mock_pil_open.return_value = mock_img

        with (
            patch.object(oiv7, "DATASET_URL", "https://example.com/test.csv"),
            patch.object(oiv7, "MIN_WIDTH", 100),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 100),
            patch.object(oiv7, "MAX_HEIGHT", 720),
            patch.object(oiv7, "CSV_READ_MULTIPLIER", 1),
        ):
            results = oiv7.load_and_upload(MagicMock(), "test-bucket", 3)

        assert len(results) >= 3
        assert mock_csv.call_count == 1

    def test_raises_when_dataset_url_not_set(self) -> None:
        """Raises ValueError when DATASET_URL is empty."""
        with patch.object(oiv7, "DATASET_URL", ""):
            with pytest.raises(ValueError, match="DATASET_URL"):
                oiv7.load_and_upload(MagicMock(), "test-bucket", 10)


# ---------------------------------------------------------------------------
# 2. TestCsvDownloadAndParsing
# ---------------------------------------------------------------------------


class TestCsvDownloadAndParsing:
    """Test _download_csv: successful download, nrows, skiprows, and error handling."""

    @patch("processing_job.dataset_ingest.open_images_v7.pd.read_csv")
    def test_successful_download_returns_dataframe(self, mock_read_csv: MagicMock) -> None:
        expected_df = _make_csv_df(10)
        mock_read_csv.return_value = expected_df

        result = oiv7._download_csv("https://example.com/data.csv", nrows=10)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        mock_read_csv.assert_called_once_with("https://example.com/data.csv", nrows=10)

    @patch("processing_job.dataset_ingest.open_images_v7.pd.read_csv")
    def test_nrows_passed_correctly(self, mock_read_csv: MagicMock) -> None:
        mock_read_csv.return_value = _make_csv_df(15)

        oiv7._download_csv("https://example.com/data.csv", nrows=15)

        mock_read_csv.assert_called_once_with("https://example.com/data.csv", nrows=15)

    @patch("processing_job.dataset_ingest.open_images_v7.pd.read_csv")
    def test_skiprows_reads_header_then_skips(self, mock_read_csv: MagicMock) -> None:
        header_df = _make_csv_df(0)
        data_df = _make_csv_df(5)
        mock_read_csv.side_effect = [header_df, data_df]

        result = oiv7._download_csv("https://example.com/data.csv", nrows=5, skiprows=100)

        assert len(result) == 5
        assert mock_read_csv.call_count == 2
        # First call reads header only
        mock_read_csv.assert_any_call("https://example.com/data.csv", nrows=0)

    @patch("processing_job.dataset_ingest.open_images_v7.pd.read_csv")
    def test_failed_download_raises_runtime_error(self, mock_read_csv: MagicMock) -> None:
        mock_read_csv.side_effect = Exception("Connection refused")

        with pytest.raises(RuntimeError, match="https://example.com/bad.csv"):
            oiv7._download_csv("https://example.com/bad.csv", nrows=10)


# ---------------------------------------------------------------------------
# 3. TestResolutionFiltering
# ---------------------------------------------------------------------------


class TestResolutionFiltering:
    """Test _passes_resolution_filter with various dimension/bound combinations."""

    def test_image_within_bounds_passes(self) -> None:
        with (
            patch.object(oiv7, "MIN_WIDTH", 200),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 200),
            patch.object(oiv7, "MAX_HEIGHT", 720),
        ):
            assert oiv7._passes_resolution_filter(640, 480) is True

    def test_image_below_min_width_fails(self) -> None:
        with (
            patch.object(oiv7, "MIN_WIDTH", 200),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 200),
            patch.object(oiv7, "MAX_HEIGHT", 720),
        ):
            assert oiv7._passes_resolution_filter(100, 480) is False

    def test_image_above_max_width_fails(self) -> None:
        with (
            patch.object(oiv7, "MIN_WIDTH", 200),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 200),
            patch.object(oiv7, "MAX_HEIGHT", 720),
        ):
            assert oiv7._passes_resolution_filter(1920, 480) is False

    def test_image_below_min_height_fails(self) -> None:
        with (
            patch.object(oiv7, "MIN_WIDTH", 200),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 200),
            patch.object(oiv7, "MAX_HEIGHT", 720),
        ):
            assert oiv7._passes_resolution_filter(640, 100) is False

    def test_image_above_max_height_fails(self) -> None:
        with (
            patch.object(oiv7, "MIN_WIDTH", 200),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 200),
            patch.object(oiv7, "MAX_HEIGHT", 720),
        ):
            assert oiv7._passes_resolution_filter(640, 1080) is False

    def test_boundary_exactly_at_min(self) -> None:
        with (
            patch.object(oiv7, "MIN_WIDTH", 200),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 200),
            patch.object(oiv7, "MAX_HEIGHT", 720),
        ):
            assert oiv7._passes_resolution_filter(200, 200) is True

    def test_boundary_exactly_at_max(self) -> None:
        with (
            patch.object(oiv7, "MIN_WIDTH", 200),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 200),
            patch.object(oiv7, "MAX_HEIGHT", 720),
        ):
            assert oiv7._passes_resolution_filter(1280, 720) is True

    def test_one_pixel_below_min_width(self) -> None:
        with (
            patch.object(oiv7, "MIN_WIDTH", 200),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 200),
            patch.object(oiv7, "MAX_HEIGHT", 720),
        ):
            assert oiv7._passes_resolution_filter(199, 400) is False

    def test_one_pixel_above_max_height(self) -> None:
        with (
            patch.object(oiv7, "MIN_WIDTH", 200),
            patch.object(oiv7, "MAX_WIDTH", 1280),
            patch.object(oiv7, "MIN_HEIGHT", 200),
            patch.object(oiv7, "MAX_HEIGHT", 720),
        ):
            assert oiv7._passes_resolution_filter(640, 721) is False


# ---------------------------------------------------------------------------
# 4. TestHeaderCheck
# ---------------------------------------------------------------------------


class TestHeaderCheck:
    """Test _check_image_dimensions: successful read, network error, corrupt data."""

    @patch("processing_job.dataset_ingest.open_images_v7.requests.get")
    @patch("processing_job.dataset_ingest.open_images_v7.Image.open")
    def test_successful_header_read_returns_dimensions(self, mock_pil_open: MagicMock, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_content.return_value = [b"\xff" * 16384]
        mock_resp.close = MagicMock()
        mock_get.return_value = mock_resp

        mock_img = MagicMock()
        mock_img.size = (800, 600)
        mock_pil_open.return_value = mock_img

        result = oiv7._check_image_dimensions("https://flickr.com/photo.jpg")

        assert result == (800, 600)
        mock_get.assert_called_once_with("https://flickr.com/photo.jpg", stream=True, timeout=10)

    @patch("processing_job.dataset_ingest.open_images_v7.requests.get")
    def test_network_error_returns_none(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.ConnectionError("Network unreachable")

        result = oiv7._check_image_dimensions("https://flickr.com/photo.jpg")

        assert result is None

    @patch("processing_job.dataset_ingest.open_images_v7.requests.get")
    @patch("processing_job.dataset_ingest.open_images_v7.Image.open")
    def test_corrupt_data_returns_none(self, mock_pil_open: MagicMock, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_content.return_value = [b"\x00" * 16384]
        mock_resp.close = MagicMock()
        mock_get.return_value = mock_resp

        mock_pil_open.side_effect = Exception("Cannot identify image file")

        result = oiv7._check_image_dimensions("https://flickr.com/corrupt.jpg")

        assert result is None


# ---------------------------------------------------------------------------
# 5. TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error scenarios: download failure, timeout, CSV failure."""

    @patch.object(oiv7, "download_image")
    def test_download_404_skips_image(self, mock_download: MagicMock) -> None:
        mock_download.side_effect = requests.HTTPError("404 Not Found")

        result = oiv7._process_image(MagicMock(), "bucket", "img_001", "https://flickr.com/missing.jpg")

        assert result is None

    @patch.object(oiv7, "download_image")
    def test_timeout_skips_image(self, mock_download: MagicMock) -> None:
        mock_download.side_effect = requests.Timeout("Connection timed out")

        result = oiv7._process_image(MagicMock(), "bucket", "img_002", "https://flickr.com/slow.jpg")

        assert result is None

    @patch("processing_job.dataset_ingest.open_images_v7.pd.read_csv")
    def test_csv_download_failure_raises_runtime_error(self, mock_read_csv: MagicMock) -> None:
        mock_read_csv.side_effect = Exception("HTTP 500")

        with pytest.raises(RuntimeError, match="Failed to download CSV"):
            oiv7._download_csv("https://example.com/broken.csv", nrows=10)


# ---------------------------------------------------------------------------
# 6. TestDescriptionPlaceholder
# ---------------------------------------------------------------------------


class TestDescriptionPlaceholder:
    """Verify _process_image returns ImageID as description."""

    @patch.object(oiv7, "download_image")
    @patch.object(oiv7, "resize_image")
    @patch.object(oiv7, "upload_to_s3", return_value=True)
    @patch.object(oiv7, "_key_exists", return_value=False)
    def test_description_is_image_id(
        self,
        mock_key_exists: MagicMock,
        mock_upload: MagicMock,
        mock_resize: MagicMock,
        mock_download: MagicMock,
    ) -> None:
        fake_jpeg = _fake_jpeg_bytes()
        mock_download.return_value = fake_jpeg
        mock_resize.return_value = fake_jpeg

        with (
            patch.object(oiv7, "MIN_WIDTH", 1),
            patch.object(oiv7, "MAX_WIDTH", 5000),
            patch.object(oiv7, "MIN_HEIGHT", 1),
            patch.object(oiv7, "MAX_HEIGHT", 5000),
        ):
            result = oiv7._process_image(MagicMock(), "bucket", "abc123XYZ", "https://flickr.com/img.jpg")

        assert result is not None
        assert result["description"] == "abc123XYZ"
        assert result["id"] == "abc123XYZ"


# ---------------------------------------------------------------------------
# 7. TestGeneratePromptMultimodal
# ---------------------------------------------------------------------------


class TestGeneratePromptMultimodal:
    """Test updated generate_prompt from main.py with multimodal content."""

    def _mock_s3_client(self, image_bytes: bytes = b"fake-jpeg-data") -> MagicMock:
        mock_s3 = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = image_bytes
        mock_s3.get_object.return_value = {"Body": mock_body}
        return mock_s3

    def test_downloads_image_bytes_from_s3(self) -> None:
        mock_agent = MagicMock()
        mock_agent.return_value = "A cinematic prompt"
        mock_s3 = self._mock_s3_client(b"test-image-bytes")

        generate_prompt(mock_agent, "s3://my-bucket/images/abc.jpg", mock_s3)

        mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="images/abc.jpg")

    def test_passes_content_blocks_with_correct_structure(self) -> None:
        mock_agent = MagicMock()
        mock_agent.return_value = "A cinematic prompt"
        image_bytes = b"raw-image-data"
        mock_s3 = self._mock_s3_client(image_bytes)

        generate_prompt(mock_agent, "s3://bucket/images/test.jpg", mock_s3)

        call_args = mock_agent.call_args[0][0]
        assert len(call_args) == 2

        # First block: text instruction
        assert "text" in call_args[0]
        assert isinstance(call_args[0]["text"], str)
        assert len(call_args[0]["text"]) > 0

        # Second block: image with correct structure
        assert "image" in call_args[1]
        assert call_args[1]["image"]["format"] == "jpeg"
        assert call_args[1]["image"]["source"]["bytes"] == image_bytes

    def test_empty_response_raises_value_error(self) -> None:
        mock_agent = MagicMock()
        mock_agent.return_value = "   "
        mock_s3 = self._mock_s3_client()

        with pytest.raises(ValueError, match="empty response"):
            generate_prompt(mock_agent, "s3://bucket/images/test.jpg", mock_s3)

    def test_s3_download_failure_propagates(self) -> None:
        mock_agent = MagicMock()
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = Exception("Access Denied")

        with pytest.raises(Exception, match="Access Denied"):
            generate_prompt(mock_agent, "s3://bucket/images/test.jpg", mock_s3)
