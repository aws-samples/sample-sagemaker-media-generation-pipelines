"""
Property-based tests for RetrievalConfig and PipelineConfig retrieval field.

Uses hypothesis to generate random valid/invalid inputs and verify that
Pydantic validation behaves correctly across the input space.

**Validates: Requirements 1.1, 1.2, 1.4, 2.1, 2.3, 2.4**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from config.config import ContainerConfig, PipelineConfig, RetrievalConfig
from tests.unit.processing.conftest import _valid_step

pytestmark = pytest.mark.retrieval


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

valid_collection_name = st.from_regex(r"^[a-z0-9-]{3,32}$", fullmatch=True)
valid_index_name = st.text(min_size=1, max_size=64).filter(lambda s: s.strip() != "")
valid_query_k = st.integers(min_value=1, max_value=100)
valid_sqs_visibility = st.integers(min_value=30, max_value=43200)
valid_sqs_max_receive = st.integers(min_value=1, max_value=10)
valid_lambda_timeout = st.integers(min_value=1, max_value=900)
valid_lambda_memory = st.integers(min_value=128, max_value=10240)

RETRIEVAL_FIELD_NAMES = frozenset(
    {
        "collection_name",
        "embedding_model_id",
        "index_name",
        "query_k",
        "sqs_visibility_timeout_seconds",
        "sqs_max_receive_count",
        "ingest_lambda_timeout_seconds",
        "ingest_lambda_memory_mb",
    }
)


# ---------------------------------------------------------------------------
# Feature: image-retrieval-pipeline, Property 1: RetrievalConfig field validation
# ---------------------------------------------------------------------------
class TestRetrievalConfigFieldValidation:
    """Property 1: RetrievalConfig field validation.

    For any dictionary of field values where every value is within its
    specified constraints, RetrievalConfig(**d) should succeed and produce
    a model whose field values equal the input values. For any dictionary
    where at least one value violates its constraint, RetrievalConfig(**d)
    should raise pydantic.ValidationError.

    **Validates: Requirements 1.1, 1.4**
    """

    @given(
        collection_name=valid_collection_name,
        query_k=valid_query_k,
        sqs_visibility=valid_sqs_visibility,
        sqs_max_receive=valid_sqs_max_receive,
        lambda_timeout=valid_lambda_timeout,
        lambda_memory=valid_lambda_memory,
    )
    @settings(max_examples=100)
    def test_valid_inputs_accepted(
        self,
        collection_name: str,
        query_k: int,
        sqs_visibility: int,
        sqs_max_receive: int,
        lambda_timeout: int,
        lambda_memory: int,
    ) -> None:
        cfg = RetrievalConfig(
            collection_name=collection_name,
            index_name="test-index",
            query_k=query_k,
            sqs_visibility_timeout_seconds=sqs_visibility,
            sqs_max_receive_count=sqs_max_receive,
            ingest_lambda_timeout_seconds=lambda_timeout,
            ingest_lambda_memory_mb=lambda_memory,
        )
        assert cfg.collection_name == collection_name
        assert cfg.query_k == query_k
        assert cfg.sqs_visibility_timeout_seconds == sqs_visibility
        assert cfg.sqs_max_receive_count == sqs_max_receive
        assert cfg.ingest_lambda_timeout_seconds == lambda_timeout
        assert cfg.ingest_lambda_memory_mb == lambda_memory

    @given(query_k=st.integers().filter(lambda x: x < 1 or x > 100))
    @settings(max_examples=100)
    def test_invalid_query_k_rejected(self, query_k: int) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(
                collection_name="valid-name",
                index_name="idx",
                query_k=query_k,
                sqs_visibility_timeout_seconds=60,
                sqs_max_receive_count=3,
                ingest_lambda_timeout_seconds=300,
                ingest_lambda_memory_mb=2048,
            )

    @given(sqs_vis=st.integers().filter(lambda x: x < 30 or x > 43200))
    @settings(max_examples=100)
    def test_invalid_sqs_visibility_rejected(self, sqs_vis: int) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(
                collection_name="valid-name",
                index_name="idx",
                query_k=5,
                sqs_visibility_timeout_seconds=sqs_vis,
                sqs_max_receive_count=3,
                ingest_lambda_timeout_seconds=300,
                ingest_lambda_memory_mb=2048,
            )

    @given(sqs_max=st.integers().filter(lambda x: x < 1 or x > 10))
    @settings(max_examples=100)
    def test_invalid_sqs_max_receive_rejected(self, sqs_max: int) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(
                collection_name="valid-name",
                index_name="idx",
                query_k=5,
                sqs_visibility_timeout_seconds=60,
                sqs_max_receive_count=sqs_max,
                ingest_lambda_timeout_seconds=300,
                ingest_lambda_memory_mb=2048,
            )

    @given(timeout=st.integers().filter(lambda x: x < 1 or x > 900))
    @settings(max_examples=100)
    def test_invalid_lambda_timeout_rejected(self, timeout: int) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(
                collection_name="valid-name",
                index_name="idx",
                query_k=5,
                sqs_visibility_timeout_seconds=60,
                sqs_max_receive_count=3,
                ingest_lambda_timeout_seconds=timeout,
                ingest_lambda_memory_mb=2048,
            )

    @given(memory=st.integers().filter(lambda x: x < 128 or x > 10240))
    @settings(max_examples=100)
    def test_invalid_lambda_memory_rejected(self, memory: int) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(
                collection_name="valid-name",
                index_name="idx",
                query_k=5,
                sqs_visibility_timeout_seconds=60,
                sqs_max_receive_count=3,
                ingest_lambda_timeout_seconds=300,
                ingest_lambda_memory_mb=memory,
            )

    @given(
        bad_name=st.text(min_size=0, max_size=50).filter(
            lambda s: not __import__("re").fullmatch(r"[a-z0-9-]{3,32}", s)
        )
    )
    @settings(max_examples=100)
    def test_invalid_collection_name_rejected(self, bad_name: str) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(
                collection_name=bad_name,
                index_name="idx",
                query_k=5,
                sqs_visibility_timeout_seconds=60,
                sqs_max_receive_count=3,
                ingest_lambda_timeout_seconds=300,
                ingest_lambda_memory_mb=2048,
            )


# ---------------------------------------------------------------------------
# Feature: image-retrieval-pipeline, Property 2: RetrievalConfig rejects unknown fields
# ---------------------------------------------------------------------------
class TestRetrievalConfigRejectsUnknownFields:
    """Property 2: RetrievalConfig rejects unknown fields.

    For any valid RetrievalConfig dictionary and any additional key-value
    pair where the key is not a recognized field name, constructing
    RetrievalConfig with the extra field should raise ValidationError.

    **Validates: Requirements 1.2**
    """

    @given(
        extra_key=st.text(min_size=1, max_size=30).filter(
            lambda k: k not in RETRIEVAL_FIELD_NAMES and k.isidentifier()
        ),
        extra_value=st.one_of(st.text(max_size=20), st.integers(), st.booleans()),
    )
    @settings(max_examples=100)
    def test_extra_fields_rejected(self, extra_key: str, extra_value: object) -> None:
        d = {
            "collection_name": "my-images",
            "index_name": "image-vectors",
            "query_k": 5,
            "sqs_visibility_timeout_seconds": 960,
            "sqs_max_receive_count": 3,
            "ingest_lambda_timeout_seconds": 300,
            "ingest_lambda_memory_mb": 2048,
            extra_key: extra_value,
        }
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RetrievalConfig(**d)


# ---------------------------------------------------------------------------
# Feature: image-retrieval-pipeline, Property 3: PipelineConfig retrieval field passthrough
# ---------------------------------------------------------------------------
class TestPipelineConfigRetrievalFieldPassthrough:
    """Property 3: PipelineConfig retrieval field passthrough.

    For any valid PipelineConfig dictionary and any string value for the
    retrieval field, parsing the config should store that exact string in
    config.retrieval without loading or validating the referenced file.
    When the retrieval field is omitted, config.retrieval should be None.

    **Validates: Requirements 2.1, 2.3, 2.4**
    """

    @given(filename=st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_retrieval_filename_stored_as_is(self, filename: str) -> None:
        cfg = PipelineConfig(
            steps={"s": ContainerConfig(**_valid_step())},
            retrieval=filename,
        )
        assert cfg.retrieval == filename

    def test_retrieval_defaults_to_none(self) -> None:
        cfg = PipelineConfig(steps={"s": ContainerConfig(**_valid_step())})
        assert cfg.retrieval is None
