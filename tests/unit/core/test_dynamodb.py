"""
Unit tests for DynamoDB operations, config, utilities, and log_outputs scripts.

Tests cover:
- DynamoDBConfig model validation
- DynamoDBOperations (put_item, get_item, update_attribute) with mocked boto3
- scan_output_files utility
- t2v/log_outputs.py
- i2v/log_outputs.py
- vbench/run.py DynamoDB integration
- DataStack with dynamodb_config parameter
- Pipeline env vars only contain DYNAMODB_TABLE_NAME
"""

import json
import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from config.config import DynamoDBConfig
from tests.unit.conftest import STEP_0, STEP_1

pytestmark = pytest.mark.core


# ---------------------------------------------------------------------------
# DynamoDBConfig model tests
# ---------------------------------------------------------------------------


class TestDynamoDBConfig:
    def test_defaults(self):
        cfg = DynamoDBConfig()
        assert cfg.partition_key == "id"
        assert cfg.sort_key == "step"

    def test_custom_keys(self):
        cfg = DynamoDBConfig(partition_key="my_pk", sort_key="my_sk")
        assert cfg.partition_key == "my_pk"
        assert cfg.sort_key == "my_sk"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DynamoDBConfig(partition_key="pk", sort_key="sk", extra="bad")

    def test_round_trip(self):
        cfg = DynamoDBConfig(partition_key="my_pk", sort_key="my_sk")
        restored = DynamoDBConfig(**cfg.model_dump())
        assert restored == cfg


# ---------------------------------------------------------------------------
# DynamoDBOperations tests (mocked boto3)
# ---------------------------------------------------------------------------


class TestDynamoDBOperations:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        # Import after env var is set
        from processing_job.common.dynamodb import DynamoDBOperations

        self.DynamoDBOperations = DynamoDBOperations

    def _make_ops(self):
        with patch("processing_job.common.dynamodb.boto3") as mock_boto3:
            mock_table = MagicMock()
            mock_resource = MagicMock()
            mock_resource.Table.return_value = mock_table
            mock_boto3.resource.return_value = mock_resource
            ops = self.DynamoDBOperations(table_name="test-table")
            return ops, mock_table

    def test_put_item_success(self):
        ops, mock_table = self._make_ops()
        result = ops.put_item(id="abc-123", step=STEP_0, data={"model": "ltx"})
        assert result is True
        mock_table.put_item.assert_called_once()
        call_item = mock_table.put_item.call_args[1]["Item"]
        assert call_item["id"] == "abc-123"
        assert call_item["step"] == STEP_0
        assert call_item["model"] == "ltx"

    def test_put_item_serializes_dict_as_json(self):
        ops, mock_table = self._make_ops()
        ops.put_item(id="x", step="s", data={"metrics": {"score": 0.9}})
        call_item = mock_table.put_item.call_args[1]["Item"]
        assert call_item["metrics"] == json.dumps({"score": 0.9})

    def test_put_item_client_error_returns_false(self):
        ops, mock_table = self._make_ops()
        from botocore.exceptions import ClientError

        mock_table.put_item.side_effect = ClientError({"Error": {"Message": "boom"}}, "PutItem")
        result = ops.put_item(id="x", step="s", data={})
        assert result is False

    def test_get_item_found(self):
        ops, mock_table = self._make_ops()
        mock_table.get_item.return_value = {"Item": {"id": "abc", "step": "vbench", "score": "0.95"}}
        result = ops.get_item(id="abc", step="vbench")
        assert result["id"] == "abc"
        assert result["score"] == 0.95  # deserialized from string

    def test_get_item_not_found(self):
        ops, mock_table = self._make_ops()
        mock_table.get_item.return_value = {}
        result = ops.get_item(id="missing", step="x")
        assert result == {}

    def test_get_item_deserializes_decimal(self):
        ops, mock_table = self._make_ops()
        mock_table.get_item.return_value = {"Item": {"id": "a", "step": "s", "count": Decimal("42")}}
        result = ops.get_item(id="a", step="s")
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_get_item_deserializes_decimal_float(self):
        ops, mock_table = self._make_ops()
        mock_table.get_item.return_value = {"Item": {"id": "a", "step": "s", "score": Decimal("3.14")}}
        result = ops.get_item(id="a", step="s")
        assert result["score"] == 3.14
        assert isinstance(result["score"], float)

    def test_get_item_deserializes_json_string(self):
        ops, mock_table = self._make_ops()
        mock_table.get_item.return_value = {"Item": {"id": "a", "step": "s", "data": '{"key": "val"}'}}
        result = ops.get_item(id="a", step="s")
        assert result["data"] == {"key": "val"}

    def test_update_attribute_success(self):
        ops, mock_table = self._make_ops()
        result = ops.update_attribute(id="abc", step="s", attr="status", value="done")
        assert result is True
        mock_table.update_item.assert_called_once()

    def test_update_attribute_serializes_dict(self):
        ops, mock_table = self._make_ops()
        ops.update_attribute(id="x", step="s", attr="meta", value={"k": "v"})
        call_kwargs = mock_table.update_item.call_args[1]
        assert call_kwargs["ExpressionAttributeValues"][":v"] == json.dumps({"k": "v"})

    def test_update_attribute_client_error_returns_false(self):
        ops, mock_table = self._make_ops()
        from botocore.exceptions import ClientError

        mock_table.update_item.side_effect = ClientError({"Error": {"Message": "boom"}}, "UpdateItem")
        result = ops.update_attribute(id="x", step="s", attr="a", value="v")
        assert result is False

    def test_serialize_converts_list_to_json(self):
        result = self.DynamoDBOperations._serialize({"tags": ["a", "b"]})
        assert result["tags"] == json.dumps(["a", "b"])

    def test_serialize_converts_int_to_str(self):
        result = self.DynamoDBOperations._serialize({"size": 1024})
        assert result["size"] == "1024"


# ---------------------------------------------------------------------------
# scan_output_files tests
# ---------------------------------------------------------------------------


class TestScanOutputFiles:
    def test_scans_all_files(self, tmp_path):
        (tmp_path / "video.mp4").write_text("data")
        (tmp_path / "image.png").write_text("data")

        from processing_job.common.utils import scan_output_files

        results = scan_output_files(str(tmp_path))
        assert len(results) == 2
        filenames = {r["filename"] for r in results}
        assert filenames == {"video.mp4", "image.png"}

    def test_filters_by_extension(self, tmp_path):
        (tmp_path / "video.mp4").write_text("data")
        (tmp_path / "readme.txt").write_text("data")

        from processing_job.common.utils import scan_output_files

        results = scan_output_files(str(tmp_path), extensions=(".mp4",))
        assert len(results) == 1
        assert results[0]["filename"] == "video.mp4"

    def test_returns_correct_metadata(self, tmp_path):
        (tmp_path / "test.mp4").write_bytes(b"x" * 100)

        from processing_job.common.utils import scan_output_files

        results = scan_output_files(str(tmp_path))
        assert results[0]["size_bytes"] == 100
        assert results[0]["extension"] == ".mp4"
        assert results[0]["filename"] == "test.mp4"

    def test_empty_directory(self, tmp_path):
        from processing_job.common.utils import scan_output_files

        results = scan_output_files(str(tmp_path))
        assert results == []

    def test_recursive_scan(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.mp4").write_text("data")

        from processing_job.common.utils import scan_output_files

        results = scan_output_files(str(tmp_path))
        assert len(results) == 1
        assert results[0]["filename"] == "nested.mp4"


# ---------------------------------------------------------------------------
# common/log_outputs.py tests
# ---------------------------------------------------------------------------


class TestLogOutputs:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("DYNAMODB_TABLE_NAME", "test-table")
        monkeypatch.setenv("STEP_NAME", STEP_0)
        monkeypatch.setenv("OUTPUT_S3_URI", "s3://test-output-bucket")
        monkeypatch.setenv("EXECUTION_ID", "exec-abc123")

    def _reload_and_run(self, mock_db, **kwargs):
        """Reload log_outputs (picks up env vars) and run main with a mocked DynamoDBOperations."""
        import importlib

        import processing_job.common.log_outputs as mod

        with patch("processing_job.common.dynamodb.boto3"):
            with patch.object(mod, "DynamoDBOperations", return_value=mock_db):
                importlib.reload(mod)
                with patch.object(mod, "DynamoDBOperations", return_value=mock_db):
                    mod.main(**kwargs)

    def test_logs_each_file(self, monkeypatch, tmp_path):
        """Each output file gets a put_item call with the correct step name."""
        (tmp_path / "vid1.mp4").write_bytes(b"x" * 1000)
        (tmp_path / "vid2.mp4").write_bytes(b"x" * 2000)
        monkeypatch.setenv("LOCAL_OUTPUT_DIR", str(tmp_path))

        mock_db = MagicMock()
        mock_db.put_item.return_value = True
        self._reload_and_run(mock_db, extensions=(".mp4",), extra={"model": "ltx", "mode": STEP_0})

        assert mock_db.put_item.call_count == 2
        for call in mock_db.put_item.call_args_list:
            assert call[1]["step"] == STEP_0
            assert call[1]["data"]["model"] == "ltx"
            assert call[1]["data"]["mode"] == STEP_0

    def test_step_name_from_env(self, monkeypatch, tmp_path):
        """STEP_NAME env var controls the step field."""
        monkeypatch.setenv("STEP_NAME", STEP_1)
        (tmp_path / "out.mp4").write_bytes(b"x" * 500)
        monkeypatch.setenv("LOCAL_OUTPUT_DIR", str(tmp_path))

        mock_db = MagicMock()
        mock_db.put_item.return_value = True
        self._reload_and_run(mock_db)

        assert mock_db.put_item.call_args[1]["step"] == STEP_1

    def test_no_files_does_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LOCAL_OUTPUT_DIR", str(tmp_path))

        mock_db = MagicMock()
        self._reload_and_run(mock_db)

        mock_db.put_item.assert_not_called()

    def test_uses_uuid_as_id(self, monkeypatch, tmp_path):
        (tmp_path / "v.mp4").write_bytes(b"x" * 500)
        monkeypatch.setenv("LOCAL_OUTPUT_DIR", str(tmp_path))

        mock_db = MagicMock()
        mock_db.put_item.return_value = True
        self._reload_and_run(mock_db)

        call_id = mock_db.put_item.call_args[1]["id"]
        uuid.UUID(call_id)  # validates it's a proper UUID

    def test_extra_fields_included_in_data(self, monkeypatch, tmp_path):
        (tmp_path / "v.mp4").write_bytes(b"x" * 100)
        monkeypatch.setenv("LOCAL_OUTPUT_DIR", str(tmp_path))

        mock_db = MagicMock()
        mock_db.put_item.return_value = True
        self._reload_and_run(mock_db, extra={"model": "wan", "mode": STEP_1})

        data = mock_db.put_item.call_args[1]["data"]
        assert data["model"] == "wan"
        assert data["mode"] == STEP_1
        assert "filename" in data
        assert "timestamp" in data

    def test_filters_by_extension(self, monkeypatch, tmp_path):
        (tmp_path / "vid.mp4").write_bytes(b"x" * 100)
        (tmp_path / "readme.txt").write_bytes(b"x" * 100)
        monkeypatch.setenv("LOCAL_OUTPUT_DIR", str(tmp_path))

        mock_db = MagicMock()
        mock_db.put_item.return_value = True
        self._reload_and_run(mock_db, extensions=(".mp4",))

        assert mock_db.put_item.call_count == 1
        assert mock_db.put_item.call_args[1]["data"]["filename"] == "vid.mp4"

    def test_s3_uri_included_in_data(self, monkeypatch, tmp_path):
        """Each DynamoDB item includes the full S3 URI for the output file."""
        (tmp_path / "clip.mp4").write_bytes(b"x" * 100)
        monkeypatch.setenv("LOCAL_OUTPUT_DIR", str(tmp_path))
        monkeypatch.setenv("OUTPUT_S3_URI", "s3://my-bucket")

        mock_db = MagicMock()
        mock_db.put_item.return_value = True
        self._reload_and_run(mock_db, extensions=(".mp4",))

        data = mock_db.put_item.call_args[1]["data"]
        assert data["s3_uri"] == "s3://my-bucket/clip.mp4"

    def test_s3_uri_empty_when_env_not_set(self, monkeypatch, tmp_path):
        """s3_uri is empty string when OUTPUT_S3_URI env var is not set."""
        (tmp_path / "clip.mp4").write_bytes(b"x" * 100)
        monkeypatch.setenv("LOCAL_OUTPUT_DIR", str(tmp_path))
        monkeypatch.delenv("OUTPUT_S3_URI", raising=False)

        mock_db = MagicMock()
        mock_db.put_item.return_value = True
        self._reload_and_run(mock_db, extensions=(".mp4",))

        data = mock_db.put_item.call_args[1]["data"]
        assert data["s3_uri"] == ""

    def _reload_and_run_with_meta_mock(self, mock_db, fake_meta, **kwargs):
        """Like _reload_and_run but also patches get_video_metadata after reload."""
        import importlib

        import processing_job.common.log_outputs as mod

        with patch("processing_job.common.dynamodb.boto3"):
            with patch.object(mod, "DynamoDBOperations", return_value=mock_db):
                importlib.reload(mod)
                with patch.object(mod, "DynamoDBOperations", return_value=mock_db):
                    with patch.object(mod, "get_video_metadata", return_value=fake_meta) as gvm:
                        mod.main(**kwargs)
                        return gvm

    def test_metadata_flag_adds_video_metadata(self, monkeypatch, tmp_path):
        """When metadata=True, video files get width/height/fps/duration/frame_count in data."""
        (tmp_path / "clip.mp4").write_bytes(b"x" * 500)
        monkeypatch.setenv("LOCAL_OUTPUT_DIR", str(tmp_path))

        mock_db = MagicMock()
        mock_db.put_item.return_value = True

        fake_meta = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 5.0, "frame_count": 150}
        self._reload_and_run_with_meta_mock(mock_db, fake_meta, extensions=(".mp4",), metadata=True)

        data = mock_db.put_item.call_args[1]["data"]
        assert data["width"] == 1920
        assert data["height"] == 1080
        assert data["fps"] == 30.0
        assert data["duration"] == 5.0
        assert data["frame_count"] == 150

    def test_metadata_flag_skips_non_video_files(self, monkeypatch, tmp_path):
        """metadata=True should not call get_video_metadata for non-video extensions."""
        (tmp_path / "image.png").write_bytes(b"x" * 100)
        monkeypatch.setenv("LOCAL_OUTPUT_DIR", str(tmp_path))

        mock_db = MagicMock()
        mock_db.put_item.return_value = True

        fake_meta = {"width": -1, "height": -1, "fps": -1, "duration": -1, "frame_count": -1}
        gvm = self._reload_and_run_with_meta_mock(mock_db, fake_meta, extensions=(".png",), metadata=True)
        gvm.assert_not_called()

        data = mock_db.put_item.call_args[1]["data"]
        assert "width" not in data


# ---------------------------------------------------------------------------
# get_video_metadata tests
# ---------------------------------------------------------------------------


class TestGetVideoMetadata:
    """Tests for get_video_metadata in common/utils.py."""

    def test_returns_metadata_on_success(self):
        mock_clip = MagicMock()
        mock_clip.w = 1920
        mock_clip.h = 1080
        mock_clip.fps = 30.0
        mock_clip.duration = 10.0

        mock_moviepy = MagicMock()
        mock_moviepy.VideoFileClip.return_value = mock_clip

        with patch.dict("sys.modules", {"moviepy": mock_moviepy}):
            from processing_job.common.utils import get_video_metadata

            result = get_video_metadata("/tmp/test.mp4")

        assert result == {
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "duration": 10.0,
            "frame_count": 300,
        }
        mock_clip.close.assert_called_once()

    def test_returns_negative_ones_on_error(self):
        mock_moviepy = MagicMock()
        mock_moviepy.VideoFileClip.side_effect = RuntimeError("bad file")

        with patch.dict("sys.modules", {"moviepy": mock_moviepy}):
            from processing_job.common.utils import get_video_metadata

            result = get_video_metadata("/tmp/corrupt.mp4")

        assert result == {
            "width": -1,
            "height": -1,
            "fps": -1,
            "duration": -1,
            "frame_count": -1,
        }

    def test_frame_count_calculation(self):
        mock_clip = MagicMock()
        mock_clip.w = 640
        mock_clip.h = 480
        mock_clip.fps = 24.0
        mock_clip.duration = 5.5

        mock_moviepy = MagicMock()
        mock_moviepy.VideoFileClip.return_value = mock_clip

        with patch.dict("sys.modules", {"moviepy": mock_moviepy}):
            from processing_job.common.utils import get_video_metadata

            result = get_video_metadata("/tmp/clip.mp4")

        assert result["frame_count"] == int(24.0 * 5.5)


# ---------------------------------------------------------------------------
# vbench/run.py DynamoDB integration tests
# ---------------------------------------------------------------------------


class TestVbenchDynamoDB:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("DYNAMODB_TABLE_NAME", "test-table")

    def test_format_json_for_dynamodb(self, tmp_path):
        """Test that vbench JSON is reformatted per-video with metrics."""
        vbench_data = {
            "background_consistency": [
                0.95,
                [{"video_path": "/videos/test_vid.mp4", "video_results": 0.95}],
            ],
            "dynamic_degree": [
                1.0,
                [{"video_path": "/videos/test_vid.mp4", "video_results": True}],
            ],
        }
        json_file = tmp_path / "eval_results.json"
        json_file.write_text(json.dumps(vbench_data))

        from processing_job.vbench.main import format_json_for_dynamodb

        result = format_json_for_dynamodb(str(json_file))

        assert "test_vid" in result
        assert result["test_vid"]["background_consistency"] == 0.95
        assert result["test_vid"]["dynamic_degree"] == 1  # True → 1

    def test_format_json_dynamic_degree_false(self, tmp_path):
        """dynamic_degree False → 0."""
        vbench_data = {
            "dynamic_degree": [
                0.0,
                [{"video_path": "/videos/v.mp4", "video_results": False}],
            ],
        }
        json_file = tmp_path / "eval_results.json"
        json_file.write_text(json.dumps(vbench_data))

        from processing_job.vbench.main import format_json_for_dynamodb

        result = format_json_for_dynamodb(str(json_file))

        assert result["v"]["dynamic_degree"] == 0


# ---------------------------------------------------------------------------
# DataStack with dynamodb_config tests
# ---------------------------------------------------------------------------


class TestDataStackDynamoDBConfig:
    def test_data_stack_creates_dynamodb_table(self):
        from unittest.mock import patch as _patch

        import aws_cdk as cdk
        from aws_cdk import assertions
        from aws_cdk import aws_lambda as lambda_

        from config.config import DynamoDBConfig, PipelineConfig
        from infrastructure.data import DataStack
        from infrastructure.security import SecurityStack

        def _mock(*a, **kw):
            return lambda_.Code.from_inline("def handler(event, context): pass")

        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")
        with _patch.object(lambda_.Code, "from_asset", side_effect=_mock):
            sec = SecurityStack(app, "Sec", prefix="dev", env=env)
            stack = DataStack(
                app,
                "Data",
                security_stack=sec,
                dynamodb_config=DynamoDBConfig(),
                pipeline_config=PipelineConfig(construct_id="dev", s3_downloads=[], steps={}),
                prefix="dev",
                env=env,
            )
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::DynamoDB::Table", 1)

    def test_data_stack_dynamodb_has_correct_key_schema(self):
        from unittest.mock import patch as _patch

        import aws_cdk as cdk
        from aws_cdk import assertions
        from aws_cdk import aws_lambda as lambda_

        from config.config import DynamoDBConfig, PipelineConfig
        from infrastructure.data import DataStack
        from infrastructure.security import SecurityStack

        def _mock(*a, **kw):
            return lambda_.Code.from_inline("def handler(event, context): pass")

        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")
        with _patch.object(lambda_.Code, "from_asset", side_effect=_mock):
            sec = SecurityStack(app, "Sec", prefix="dev", env=env)
            stack = DataStack(
                app,
                "Data",
                security_stack=sec,
                dynamodb_config=DynamoDBConfig(),
                pipeline_config=PipelineConfig(construct_id="dev", s3_downloads=[], steps={}),
                prefix="dev",
                env=env,
            )
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::DynamoDB::Table",
            {
                "KeySchema": assertions.Match.array_with(
                    [
                        assertions.Match.object_like({"AttributeName": "id", "KeyType": "HASH"}),
                        assertions.Match.object_like({"AttributeName": "step", "KeyType": "RANGE"}),
                    ]
                ),
            },
        )


# ---------------------------------------------------------------------------
# Pipeline env vars test
# ---------------------------------------------------------------------------


class TestPipelineEnvVars:
    def test_pipeline_only_passes_dynamodb_table_name(self):
        """Verify processing jobs get DYNAMODB_TABLE_NAME, not old pk/sk vars."""
        from unittest.mock import patch as _patch

        import aws_cdk as cdk
        from aws_cdk import aws_lambda as lambda_

        from config.config import (
            ContainerConfig,
            DynamoDBConfig,
            PipelineConfig,
        )
        from infrastructure.data import DataStack
        from infrastructure.pipeline import PipelineStack
        from infrastructure.security import SecurityStack

        def _mock(*a, **kw):
            return lambda_.Code.from_inline("def handler(event, context): pass")

        cfg = ContainerConfig(
            InstanceCount=1,
            InstanceType="ml.g5.xlarge",
            VolumeSizeInGB=125,
            ContainerEntrypoint=["/bin/bash", "./run_job.sh"],
            ContainerArguments=["300"],
        )
        pipeline_config = PipelineConfig(
            construct_id="dev",
            s3_downloads=[],
            steps={STEP_0: cfg},
        )

        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")

        with _patch.object(lambda_.Code, "from_asset", side_effect=_mock):
            sec = SecurityStack(app, "Sec", prefix="dev", env=env)
            data = DataStack(
                app,
                "Data",
                security_stack=sec,
                dynamodb_config=DynamoDBConfig(),
                pipeline_config=pipeline_config,
                prefix="dev",
                env=env,
            )
            PipelineStack(
                app,
                "Pipeline",
                security_stack=sec,
                data_stack=data,
                pipeline_config=pipeline_config,
                prefix="dev",
                env=env,
            )

        # Check the pipeline.py source — environment dict should have
        # DYNAMODB_TABLE_NAME but NOT DYNAMODB_PARTITION_KEY or DYNAMODB_SORT_KEY
        import inspect

        from infrastructure import pipeline as pipeline_mod

        source = inspect.getsource(pipeline_mod.PipelineStack.__init__)
        assert "DYNAMODB_TABLE_NAME" in source
        assert "STEP_NAME" in source
        assert "DYNAMODB_PARTITION_KEY" not in source
        assert "DYNAMODB_SORT_KEY" not in source
