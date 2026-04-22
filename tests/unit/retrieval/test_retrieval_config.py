"""
Unit tests for RetrievalConfig, get_retrieval_config(), PipelineConfig retrieval field,
and ContainerConfig InstanceType update.

**Validates: Requirements 15.2, 15.3, 15.4, 15.8**
"""

import pytest
import yaml
from pydantic import ValidationError

from config.config import (
    ContainerConfig,
    PipelineConfig,
    RetrievalConfig,
    get_retrieval_config,
)
from tests.unit.processing.conftest import _valid_step

pytestmark = pytest.mark.retrieval


def _valid_retrieval_dict() -> dict:
    """Return a minimal valid RetrievalConfig dict."""
    return {
        "collection_name": "my-images",
        "index_name": "image-vectors",
        "query_k": 5,
        "sqs_visibility_timeout_seconds": 960,
        "sqs_max_receive_count": 3,
        "ingest_lambda_timeout_seconds": 300,
        "ingest_lambda_memory_mb": 2048,
    }


# ---------------------------------------------------------------------------
# RetrievalConfig: valid config accepted
# ---------------------------------------------------------------------------
class TestRetrievalConfigValid:
    """Valid RetrievalConfig round-trips correctly."""

    def test_minimal_valid_config(self) -> None:
        cfg = RetrievalConfig(**_valid_retrieval_dict())
        assert cfg.collection_name == "my-images"
        assert cfg.index_name == "image-vectors"
        assert cfg.query_k == 5

    def test_all_fields_stored(self) -> None:
        d = _valid_retrieval_dict()
        cfg = RetrievalConfig(**d)
        assert cfg.sqs_visibility_timeout_seconds == 960
        assert cfg.sqs_max_receive_count == 3
        assert cfg.ingest_lambda_timeout_seconds == 300
        assert cfg.ingest_lambda_memory_mb == 2048


# ---------------------------------------------------------------------------
# RetrievalConfig: defaults applied
# ---------------------------------------------------------------------------
class TestRetrievalConfigDefaults:
    """Default values are applied for embedding_model_id and query_k."""

    def test_embedding_model_id_default(self) -> None:
        cfg = RetrievalConfig(**_valid_retrieval_dict())
        assert cfg.embedding_model_id == "amazon.titan-embed-image-v1"

    def test_query_k_default(self) -> None:
        d = _valid_retrieval_dict()
        del d["query_k"]
        cfg = RetrievalConfig(**d)
        assert cfg.query_k == 5

    def test_custom_embedding_model_id(self) -> None:
        d = _valid_retrieval_dict()
        d["embedding_model_id"] = "amazon.nova-2-multimodal-embeddings-v1:0"
        cfg = RetrievalConfig(**d)
        assert cfg.embedding_model_id == "amazon.nova-2-multimodal-embeddings-v1:0"

    def test_invalid_embedding_model_id_rejected(self) -> None:
        d = _valid_retrieval_dict()
        d["embedding_model_id"] = "custom-model-v2"
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_titan_rejects_non_1024_dimension(self) -> None:
        d = _valid_retrieval_dict()
        d["embedding_model_id"] = "amazon.titan-embed-image-v1"
        d["embedding_dimension"] = 256
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_nova_accepts_all_dimensions(self) -> None:
        d = _valid_retrieval_dict()
        d["embedding_model_id"] = "amazon.nova-2-multimodal-embeddings-v1:0"
        for dim in (256, 384, 1024, 3072):
            d["embedding_dimension"] = dim
            cfg = RetrievalConfig(**d)
            assert cfg.embedding_dimension == dim


# ---------------------------------------------------------------------------
# RetrievalConfig: extra='forbid' rejects unknown fields
# ---------------------------------------------------------------------------
class TestRetrievalConfigExtraForbid:
    """extra='forbid' rejects unknown keys."""

    def test_extra_field_rejected(self) -> None:
        d = _valid_retrieval_dict()
        d["unknown_field"] = "bad"
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RetrievalConfig(**d)

    def test_extra_numeric_field_rejected(self) -> None:
        d = _valid_retrieval_dict()
        d["extra_count"] = 42
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RetrievalConfig(**d)


# ---------------------------------------------------------------------------
# RetrievalConfig: invalid collection_name pattern
# ---------------------------------------------------------------------------
class TestRetrievalConfigCollectionName:
    """collection_name must match ^[a-z0-9-]{3,32}$."""

    def test_uppercase_rejected(self) -> None:
        d = _valid_retrieval_dict()
        d["collection_name"] = "MyImages"
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_special_chars_rejected(self) -> None:
        d = _valid_retrieval_dict()
        d["collection_name"] = "my_images!"
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_too_short_rejected(self) -> None:
        d = _valid_retrieval_dict()
        d["collection_name"] = "ab"
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_too_long_rejected(self) -> None:
        d = _valid_retrieval_dict()
        d["collection_name"] = "a" * 33
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_valid_boundary_3_chars(self) -> None:
        d = _valid_retrieval_dict()
        d["collection_name"] = "abc"
        cfg = RetrievalConfig(**d)
        assert cfg.collection_name == "abc"

    def test_valid_boundary_32_chars(self) -> None:
        d = _valid_retrieval_dict()
        d["collection_name"] = "a" * 32
        cfg = RetrievalConfig(**d)
        assert cfg.collection_name == "a" * 32


# ---------------------------------------------------------------------------
# RetrievalConfig: out-of-range values for bounded fields
# ---------------------------------------------------------------------------
class TestRetrievalConfigBoundedFields:
    """Out-of-range values rejected for each bounded field."""

    def test_query_k_below_min(self) -> None:
        d = _valid_retrieval_dict()
        d["query_k"] = 0
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_query_k_above_max(self) -> None:
        d = _valid_retrieval_dict()
        d["query_k"] = 101
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_sqs_visibility_timeout_below_min(self) -> None:
        d = _valid_retrieval_dict()
        d["sqs_visibility_timeout_seconds"] = 29
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_sqs_visibility_timeout_above_max(self) -> None:
        d = _valid_retrieval_dict()
        d["sqs_visibility_timeout_seconds"] = 43201
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_sqs_max_receive_count_below_min(self) -> None:
        d = _valid_retrieval_dict()
        d["sqs_max_receive_count"] = 0
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_sqs_max_receive_count_above_max(self) -> None:
        d = _valid_retrieval_dict()
        d["sqs_max_receive_count"] = 11
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_ingest_lambda_timeout_below_min(self) -> None:
        d = _valid_retrieval_dict()
        d["ingest_lambda_timeout_seconds"] = 0
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_ingest_lambda_timeout_above_max(self) -> None:
        d = _valid_retrieval_dict()
        d["ingest_lambda_timeout_seconds"] = 901
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_ingest_lambda_memory_below_min(self) -> None:
        d = _valid_retrieval_dict()
        d["ingest_lambda_memory_mb"] = 127
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)

    def test_ingest_lambda_memory_above_max(self) -> None:
        d = _valid_retrieval_dict()
        d["ingest_lambda_memory_mb"] = 10241
        with pytest.raises(ValidationError):
            RetrievalConfig(**d)


# ---------------------------------------------------------------------------
# get_retrieval_config(): loads valid YAML
# ---------------------------------------------------------------------------
class TestGetRetrievalConfig:
    """get_retrieval_config() loader function tests."""

    def test_loads_valid_yaml(self, tmp_path, monkeypatch) -> None:
        retrieval_dir = tmp_path / "config" / "retrieval"
        retrieval_dir.mkdir(parents=True)
        yaml_file = retrieval_dir / "test.yaml"
        yaml_file.write_text(yaml.dump(_valid_retrieval_dict()))

        monkeypatch.chdir(tmp_path)
        cfg = get_retrieval_config("test.yaml")
        assert cfg.collection_name == "my-images"
        assert cfg.index_name == "image-vectors"

    def test_raises_file_not_found(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            get_retrieval_config("nonexistent.yaml")

    def test_raises_validation_error_for_invalid_yaml(self, tmp_path, monkeypatch) -> None:
        retrieval_dir = tmp_path / "config" / "retrieval"
        retrieval_dir.mkdir(parents=True)
        yaml_file = retrieval_dir / "bad.yaml"
        yaml_file.write_text(yaml.dump({"collection_name": "AB!", "index_name": "x"}))

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValidationError):
            get_retrieval_config("bad.yaml")


# ---------------------------------------------------------------------------
# PipelineConfig retrieval field
# ---------------------------------------------------------------------------
class TestPipelineConfigRetrievalField:
    """PipelineConfig retrieval field tests."""

    def test_defaults_to_none(self) -> None:
        cfg = PipelineConfig(steps={"s": ContainerConfig(**_valid_step())})
        assert cfg.retrieval is None

    def test_accepts_string_filename(self) -> None:
        cfg = PipelineConfig(
            steps={"s": ContainerConfig(**_valid_step())},
            retrieval="retrieval.yaml",
        )
        assert cfg.retrieval == "retrieval.yaml"

    def test_existing_config_parsing_unaffected(self) -> None:
        cfg = PipelineConfig(
            construct_id="test",
            steps={"s": ContainerConfig(**_valid_step())},
        )
        assert cfg.construct_id == "test"
        assert cfg.retrieval is None
        assert "s" in cfg.steps


# ---------------------------------------------------------------------------
# ContainerConfig: ml.c5.xlarge accepted
# ---------------------------------------------------------------------------
class TestContainerConfigInstanceType:
    """ml.c5.xlarge is accepted as a valid InstanceType."""

    def test_c5_xlarge_accepted(self) -> None:
        step = _valid_step()
        step["InstanceType"] = "ml.c5.xlarge"
        cfg = ContainerConfig(**step)
        assert cfg.InstanceType == "ml.c5.xlarge"
