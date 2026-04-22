"""Shared fixtures, helpers, and Hypothesis profile for unit tests.

Constants live in step_names.py (importable by test modules).
This conftest re-exports them so pytest fixtures can also use them.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# Stub opensearchpy before any Lambda module import to avoid pulling in
# requests/urllib3 (slow import, high memory usage across xdist workers).
if "opensearchpy" not in sys.modules:
    _oss_stub = types.ModuleType("opensearchpy")
    _oss_stub.OpenSearch = MagicMock
    _oss_stub.AWSV4SignerAuth = MagicMock
    _oss_stub.RequestsHttpConnection = MagicMock
    sys.modules["opensearchpy"] = _oss_stub

from hypothesis import HealthCheck, settings

# Ensure at least one downloads manifest exists before any CDK test tries
# to bundle processing_job/ as an S3 asset.  In production, app.py writes
# per-config files ({construct_id}_downloads.json) at synth time.  Tests
# instantiate stacks with construct_id="dev", so we create that one.
# Any *_downloads.json already on disk (from a prior synth) is left as-is.
_dl_dir = Path(__file__).resolve().parents[2] / "processing_job" / "model_download"
_dl_dir.mkdir(parents=True, exist_ok=True)
_dl_default = _dl_dir / "dev_downloads.json"
if not _dl_default.exists():
    _dl_default.write_text("[]\n")

from tests.unit.step_names import *  # noqa: F401,F403

# Hypothesis CI profile
settings.register_profile("ci", max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
settings.load_profile("ci")

from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3


def _mock_from_asset(*args, **kwargs):
    """Mock for lambda_.Code.from_asset — returns inline code."""
    return lambda_.Code.from_inline("def handler(event, context): pass")


def _mock_s3_asset(scope, construct_id, **kwargs):
    """Mock for s3_assets.Asset — avoids copying .venv/ into staging."""
    mock = MagicMock()
    bucket = s3.Bucket(scope, f"{construct_id}-mock-bucket")
    mock.bucket = bucket
    mock.s3_object_key = "mock-key.zip"
    mock.grant_read = MagicMock()
    return mock
