"""Shared fixtures and helpers for retrieval/ tests.

Provides _valid_retrieval_config() helper for creating valid
RetrievalConfig instances in tests.
"""

from config.config import RetrievalConfig


def _valid_retrieval_config() -> RetrievalConfig:
    """Return a valid RetrievalConfig for testing."""
    return RetrievalConfig(
        collection_name="test-images",
        index_name="test-vectors",
        sqs_visibility_timeout_seconds=960,
        sqs_max_receive_count=3,
        ingest_lambda_timeout_seconds=300,
        ingest_lambda_memory_mb=2048,
    )
