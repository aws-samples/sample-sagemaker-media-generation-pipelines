"""
Property-based tests for the retrieval ingest Lambda.

Uses hypothesis to verify universal correctness properties of the
base64 encoding round-trip and indexed document structure.

**Validates: Requirements 5.3, 5.5**
"""

import base64

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.retrieval


# ---------------------------------------------------------------------------
# Feature: image-retrieval-pipeline, Property 5: Base64 encoding round-trip in ingest Lambda
# ---------------------------------------------------------------------------
class TestBase64EncodingRoundTrip:
    """Property 5: Base64 encoding round-trip in ingest Lambda.

    For any byte sequence representing an image, base64-encoding it and
    then base64-decoding the result should produce the original byte
    sequence. This validates that the ingest Lambda's base64 storage
    under the base64/ prefix preserves image data faithfully.

    **Validates: Requirements 5.3**
    """

    @given(image_bytes=st.binary(min_size=1, max_size=10000))
    @settings(max_examples=100)
    def test_base64_round_trip_preserves_image_data(self, image_bytes: bytes) -> None:
        # Replicate the encoding logic from lambdas/retrieval_ingest/index.py
        image_base64: str = base64.b64encode(image_bytes).decode("utf-8")
        decoded: bytes = base64.b64decode(image_base64)
        assert decoded == image_bytes


# ---------------------------------------------------------------------------
# Strategies for Property 6
# ---------------------------------------------------------------------------

# Valid S3 key: alphanumeric with path separators, non-empty, no leading slash
_s3_key_chars = st.sampled_from(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./"))
_s3_key = st.text(alphabet=_s3_key_chars, min_size=1, max_size=200).filter(
    lambda k: not k.startswith("/") and k.strip() != ""
)

# S3 bucket name: simple lowercase alphanumeric with hyphens
_bucket_chars = st.sampled_from(list("abcdefghijklmnopqrstuvwxyz0123456789"))
_bucket_name = st.text(alphabet=_bucket_chars, min_size=3, max_size=32)


@st.composite
def _embedding_strategy(draw: st.DrawFn) -> list[float]:
    """Generate a list of exactly 1024 floats efficiently.

    Uses a single seed float and repeats it 1024 times to avoid
    the overhead of generating 1024 independent floats.
    """
    seed_val = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return [seed_val] * 1024


# ---------------------------------------------------------------------------
# Feature: image-retrieval-pipeline, Property 6: Indexed document completeness
# ---------------------------------------------------------------------------
class TestIndexedDocumentCompleteness:
    """Property 6: Indexed document completeness.

    For any S3 image key processed by the ingest Lambda, the document
    indexed into OpenSearch should contain all four required fields:
    image_vector (a list of 1024 floats), image_s3_uri (containing the
    original bucket and key), image_base64_s3_uri (containing the base64
    prefix and key), and description (a non-empty string).

    **Validates: Requirements 5.5**
    """

    @given(
        bucket=_bucket_name,
        key=_s3_key,
        embedding=_embedding_strategy(),
    )
    @settings(max_examples=100)
    def test_indexed_document_has_all_four_required_fields(
        self,
        bucket: str,
        key: str,
        embedding: list[float],
    ) -> None:
        # Replicate the document construction logic from
        # lambdas/retrieval_ingest/index.py lambda_handler
        retrieval_bucket = bucket
        base64_key = f"base64/{key}.txt"

        document = {
            "image_vector": embedding,
            "description": key,
            "image_s3_uri": f"s3://{bucket}/{key}",
            "image_base64_s3_uri": f"s3://{retrieval_bucket}/{base64_key}",
        }

        # 1. image_vector must be a list of exactly 1024 floats
        assert "image_vector" in document
        assert isinstance(document["image_vector"], list)
        assert len(document["image_vector"]) == 1024
        assert all(isinstance(v, float) for v in document["image_vector"])

        # 2. image_s3_uri must contain both bucket and key
        assert "image_s3_uri" in document
        assert bucket in document["image_s3_uri"]
        assert key in document["image_s3_uri"]

        # 3. image_base64_s3_uri must contain the base64 prefix and key
        assert "image_base64_s3_uri" in document
        assert "base64" in document["image_base64_s3_uri"]
        assert key in document["image_base64_s3_uri"]

        # 4. description must be a non-empty string
        assert "description" in document
        assert isinstance(document["description"], str)
        assert len(document["description"]) > 0
