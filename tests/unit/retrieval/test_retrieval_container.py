"""
Unit tests for the retrieval container (processing_job/retrieval/main.py).

Tests verify prompt reading from input channel, OpenSearch kNN query
construction and result parsing, image download from S3, S3 URI parsing,
file extension extraction, and output file naming with document ID.

**Validates: Requirements 9.4, 9.5, 9.6, 15.6**
"""

import json
from unittest.mock import MagicMock

import pytest

from processing_job.retrieval.main import (
    download_image,
    get_file_extension,
    parse_s3_uri,
    read_prompts,
    search_images,
)

pytestmark = pytest.mark.retrieval


# ---------------------------------------------------------------------------
# read_prompts
# ---------------------------------------------------------------------------
class TestReadPrompts:
    """Prompt reading from input channel directory."""

    def test_reads_txt_files(self, tmp_path) -> None:
        (tmp_path / "prompt1.txt").write_text("a sunset over the ocean")
        result = read_prompts(str(tmp_path))
        assert result == ["a sunset over the ocean"]

    def test_reads_json_files_with_retrieval_query(self, tmp_path) -> None:
        data = {"retrieval_query": "mountain landscape"}
        (tmp_path / "query.json").write_text(json.dumps(data))
        result = read_prompts(str(tmp_path))
        assert result == ["mountain landscape"]

    def test_reads_json_files_with_prompt_field(self, tmp_path) -> None:
        data = {"prompt": "city skyline at night"}
        (tmp_path / "query.json").write_text(json.dumps(data))
        result = read_prompts(str(tmp_path))
        assert result == ["city skyline at night"]

    def test_reads_json_list_of_prompts(self, tmp_path) -> None:
        data = [{"retrieval_query": "cat"}, {"retrieval_query": "dog"}]
        (tmp_path / "queries.json").write_text(json.dumps(data))
        result = read_prompts(str(tmp_path))
        assert result == ["cat", "dog"]

    def test_reads_multiple_txt_files_sorted(self, tmp_path) -> None:
        (tmp_path / "b.txt").write_text("second")
        (tmp_path / "a.txt").write_text("first")
        result = read_prompts(str(tmp_path))
        assert result == ["first", "second"]

    def test_ignores_non_txt_non_json_files(self, tmp_path) -> None:
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "prompt.txt").write_text("valid prompt")
        result = read_prompts(str(tmp_path))
        assert result == ["valid prompt"]

    def test_returns_empty_for_nonexistent_directory(self) -> None:
        result = read_prompts("/nonexistent/path")
        assert result == []

    def test_returns_empty_for_empty_directory(self, tmp_path) -> None:
        result = read_prompts(str(tmp_path))
        assert result == []

    def test_skips_empty_txt_files(self, tmp_path) -> None:
        (tmp_path / "empty.txt").write_text("")
        (tmp_path / "valid.txt").write_text("hello")
        result = read_prompts(str(tmp_path))
        assert result == ["hello"]


# ---------------------------------------------------------------------------
# search_images
# ---------------------------------------------------------------------------
class TestSearchImages:
    """OpenSearch kNN query construction and result parsing."""

    def test_returns_correct_results_from_knn_response(self) -> None:
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "doc-1",
                        "_score": 0.95,
                        "_source": {"image_s3_uri": "s3://bucket/images/cat.jpg"},
                    },
                    {
                        "_id": "doc-2",
                        "_score": 0.88,
                        "_source": {"image_s3_uri": "s3://bucket/images/dog.png"},
                    },
                ]
            }
        }

        query_vector = [0.1] * 1024
        results = search_images(mock_client, query_vector, "image-vectors", k=5)

        assert len(results) == 2
        assert results[0]["doc_id"] == "doc-1"
        assert results[0]["image_s3_uri"] == "s3://bucket/images/cat.jpg"
        assert results[0]["score"] == 0.95
        assert results[1]["doc_id"] == "doc-2"

    def test_constructs_knn_query_with_correct_structure(self) -> None:
        mock_client = MagicMock()
        mock_client.search.return_value = {"hits": {"hits": []}}

        query_vector = [0.5] * 1024
        search_images(mock_client, query_vector, "my-index", k=10)

        call_kwargs = mock_client.search.call_args
        body = call_kwargs.kwargs.get("body") or call_kwargs[1].get("body")
        index = call_kwargs.kwargs.get("index") or call_kwargs[1].get("index")

        assert index == "my-index"
        assert body["size"] == 10
        assert body["query"]["knn"]["image_vector"]["vector"] == query_vector
        assert body["query"]["knn"]["image_vector"]["k"] == 10

    def test_returns_empty_list_for_no_hits(self) -> None:
        mock_client = MagicMock()
        mock_client.search.return_value = {"hits": {"hits": []}}

        results = search_images(mock_client, [0.1] * 1024, "idx", k=5)
        assert results == []

    def test_handles_missing_source_fields_gracefully(self) -> None:
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "doc-x", "_score": 0.5, "_source": {}},
                ]
            }
        }

        results = search_images(mock_client, [0.0] * 1024, "idx", k=3)
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc-x"
        assert results[0]["image_s3_uri"] == ""


# ---------------------------------------------------------------------------
# download_image
# ---------------------------------------------------------------------------
class TestDownloadImage:
    """S3 download returns bytes."""

    def test_downloads_and_returns_bytes(self) -> None:
        mock_s3 = MagicMock()
        body_mock = MagicMock()
        body_mock.read.return_value = b"\x89PNG-image-data"
        mock_s3.get_object.return_value = {"Body": body_mock}

        result = download_image(mock_s3, "my-bucket", "images/photo.jpg")

        assert result == b"\x89PNG-image-data"
        mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="images/photo.jpg")


# ---------------------------------------------------------------------------
# parse_s3_uri
# ---------------------------------------------------------------------------
class TestParseS3Uri:
    """Parses valid URIs, raises ValueError for invalid."""

    def test_parses_valid_uri(self) -> None:
        bucket, key = parse_s3_uri("s3://my-bucket/images/photo.jpg")
        assert bucket == "my-bucket"
        assert key == "images/photo.jpg"

    def test_parses_uri_with_nested_path(self) -> None:
        bucket, key = parse_s3_uri("s3://bucket/a/b/c/d.png")
        assert bucket == "bucket"
        assert key == "a/b/c/d.png"

    def test_raises_for_missing_s3_prefix(self) -> None:
        with pytest.raises(ValueError, match="must start with s3://"):
            parse_s3_uri("https://bucket/key")

    def test_raises_for_missing_key(self) -> None:
        with pytest.raises(ValueError, match="missing bucket or key"):
            parse_s3_uri("s3://bucket-only")

    def test_raises_for_empty_string(self) -> None:
        with pytest.raises(ValueError):
            parse_s3_uri("")

    def test_raises_for_bucket_with_trailing_slash_only(self) -> None:
        with pytest.raises(ValueError, match="missing bucket or key"):
            parse_s3_uri("s3://bucket/")


# ---------------------------------------------------------------------------
# get_file_extension
# ---------------------------------------------------------------------------
class TestGetFileExtension:
    """Extracts extension, defaults to .jpg."""

    def test_extracts_jpg_extension(self) -> None:
        assert get_file_extension("images/photo.jpg") == ".jpg"

    def test_extracts_png_extension(self) -> None:
        assert get_file_extension("images/photo.png") == ".png"

    def test_extracts_webp_extension(self) -> None:
        assert get_file_extension("path/to/image.webp") == ".webp"

    def test_defaults_to_jpg_when_no_extension(self) -> None:
        assert get_file_extension("images/photo") == ".jpg"

    def test_defaults_to_jpg_for_empty_string(self) -> None:
        assert get_file_extension("") == ".jpg"


# ---------------------------------------------------------------------------
# Output file naming
# ---------------------------------------------------------------------------
class TestOutputFileNaming:
    """Document ID used as filename with correct extension."""

    def test_output_filename_uses_doc_id_and_extension(self) -> None:
        doc_id = "abc123"
        s3_key = "images/sunset.png"
        ext = get_file_extension(s3_key)
        output_filename = f"{doc_id}{ext}"
        assert output_filename == "abc123.png"

    def test_output_filename_defaults_to_jpg_without_extension(self) -> None:
        doc_id = "xyz789"
        s3_key = "images/noext"
        ext = get_file_extension(s3_key)
        output_filename = f"{doc_id}{ext}"
        assert output_filename == "xyz789.jpg"

    def test_output_filename_preserves_jpeg_extension(self) -> None:
        doc_id = "doc-42"
        s3_key = "photos/beach.jpeg"
        ext = get_file_extension(s3_key)
        output_filename = f"{doc_id}{ext}"
        assert output_filename == "doc-42.jpeg"
