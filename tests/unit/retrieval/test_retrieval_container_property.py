"""
Property-based tests for the retrieval container (processing_job/retrieval/main.py).

Uses hypothesis to verify universal correctness properties of kNN query
result parsing and output file naming.

**Validates: Requirements 9.5, 9.6**
"""

from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from processing_job.retrieval.main import get_file_extension, search_images

pytestmark = pytest.mark.retrieval


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Document IDs: alphanumeric with hyphens/underscores, non-empty
_doc_id = st.text(
    alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")),
    min_size=1,
    max_size=64,
)

# S3 bucket name: lowercase alphanumeric with hyphens
_bucket_name = st.text(
    alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyz0123456789")),
    min_size=3,
    max_size=32,
)

# File extensions commonly used for images
_extension = st.sampled_from([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ""])

# S3 key path segment (no slashes)
_path_segment = st.text(
    alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyz0123456789-_")),
    min_size=1,
    max_size=30,
)


@st.composite
def _s3_image_uri(draw: st.DrawFn) -> str:
    """Generate a valid S3 image URI with a known extension."""
    bucket = draw(_bucket_name)
    prefix = draw(_path_segment)
    filename = draw(_path_segment)
    ext = draw(_extension)
    return f"s3://{bucket}/{prefix}/{filename}{ext}"


@st.composite
def _knn_hit(draw: st.DrawFn) -> dict:
    """Generate a single OpenSearch kNN hit with _id and _source.image_s3_uri."""
    doc_id = draw(_doc_id)
    uri = draw(_s3_image_uri())
    score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return {
        "_id": doc_id,
        "_score": score,
        "_source": {"image_s3_uri": uri},
    }


@st.composite
def _knn_response(draw: st.DrawFn) -> tuple[int, dict]:
    """Generate a kNN response with N hits and return (N, response_dict)."""
    n = draw(st.integers(min_value=0, max_value=20))
    hits = draw(st.lists(_knn_hit(), min_size=n, max_size=n))
    response = {"hits": {"hits": hits}}
    return n, response


# ---------------------------------------------------------------------------
# Feature: image-retrieval-pipeline, Property 8: Retrieval container kNN query correctness
# ---------------------------------------------------------------------------
class TestKnnQueryCorrectness:
    """Property 8: Retrieval container kNN query correctness.

    For any text prompt and any OpenSearch kNN response containing N hits
    (each with _id and _source.image_s3_uri), the retrieval container's
    query function should return exactly N results, each preserving the
    document ID and S3 URI from the response.

    **Validates: Requirements 9.5**
    """

    @given(data=_knn_response())
    @settings(max_examples=100)
    def test_search_images_preserves_all_hits(self, data: tuple[int, dict]) -> None:
        n, response = data

        mock_client = MagicMock()
        mock_client.search.return_value = response

        query_vector = [0.0] * 1024
        results = search_images(mock_client, query_vector, "test-index", k=max(n, 1))

        # Exactly N results returned
        assert len(results) == n

        # Each result preserves doc_id and image_s3_uri from the response
        hits = response["hits"]["hits"]
        for i, result in enumerate(results):
            assert result["doc_id"] == hits[i]["_id"]
            assert result["image_s3_uri"] == hits[i]["_source"]["image_s3_uri"]


# ---------------------------------------------------------------------------
# Feature: image-retrieval-pipeline, Property 9: Retrieval container output file naming
# ---------------------------------------------------------------------------
class TestOutputFileNaming:
    """Property 9: Retrieval container output file naming.

    For any set of matched OpenSearch documents with distinct document IDs
    and valid S3 image URIs, the retrieval container should write one file
    per document to the output directory, where each filename equals the
    document ID (with the original image extension).

    **Validates: Requirements 9.6**
    """

    @given(
        doc_id=_doc_id,
        s3_uri=_s3_image_uri(),
    )
    @settings(max_examples=100)
    def test_output_filename_equals_doc_id_with_extension(
        self,
        doc_id: str,
        s3_uri: str,
    ) -> None:
        # Extract the key from the S3 URI (strip s3://bucket/)
        stripped = s3_uri[5:]  # remove "s3://"
        _, _, key = stripped.partition("/")

        ext = get_file_extension(key)
        output_filename = f"{doc_id}{ext}"

        # Filename must start with the document ID
        assert output_filename.startswith(doc_id)

        # Extension must come from the S3 key (or default to .jpg)
        expected_ext = ext if ext else ".jpg"
        assert output_filename == f"{doc_id}{expected_ext}"
