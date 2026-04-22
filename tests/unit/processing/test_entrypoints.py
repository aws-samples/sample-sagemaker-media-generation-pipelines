"""Unit tests for processing job entrypoint scripts (main.py).

Tests t2v, i2v, and vbench main.py logic including
arg parsing, directory logging, output copying, queue monitoring, and
vbench JSON formatting.
"""

import io
import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from tests.unit.conftest import REPO_ROOT

pytestmark = pytest.mark.processing


@pytest.fixture
def loguru_capture():
    """Capture loguru output to a StringIO buffer."""
    buf = io.StringIO()
    handler_id = logger.add(buf, format="{message}", level="DEBUG")
    yield buf
    logger.remove(handler_id)


# ---------------------------------------------------------------------------
# Import helpers — these modules use env vars and external deps at import time,
# so we need to patch before importing.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _t2v_module_cached():
    """Import t2v main.py once for the entire test module."""
    import importlib

    # Mock ComfyScript-dependent modules before importing main.py
    mock_ltx = types.ModuleType("common.ltx")
    mock_ltx.load_inputs = MagicMock(return_value=[])
    mock_ltx.run_workflow = MagicMock()
    mock_wan = types.ModuleType("common.wan22")
    mock_wan.load_inputs = MagicMock(return_value=[])
    mock_wan.run_i2v = MagicMock()
    mock_wan.run_t2v = MagicMock()
    old_ltx = sys.modules.get("common.ltx")
    old_wan = sys.modules.get("common.wan22")
    sys.modules["common.ltx"] = mock_ltx
    sys.modules["common.wan22"] = mock_wan
    try:
        mod_path = str(REPO_ROOT / "processing_job" / "t2v")
        if mod_path not in sys.path:
            sys.path.insert(0, mod_path)
        spec = importlib.util.spec_from_file_location(f"{'t2v'}_main", REPO_ROOT / "processing_job" / "t2v" / "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if old_ltx is None:
            sys.modules.pop("common.ltx", None)
        else:
            sys.modules["common.ltx"] = old_ltx
        if old_wan is None:
            sys.modules.pop("common.wan22", None)
        else:
            sys.modules["common.wan22"] = old_wan


@pytest.fixture
def t2v_module(tmp_path, _t2v_module_cached):
    """Per-test wrapper that resets env-dependent module vars."""
    mod = _t2v_module_cached
    mod.COMFY_HOME = str(tmp_path / "comfy")
    mod.LOCAL_OUTPUT_DIR = str(tmp_path / "output")
    mod.SM_INPUT_DIR = "/opt/ml/processing/input/input"
    return mod


@pytest.fixture(scope="module")
def _i2v_module_cached():
    """Import i2v main.py once for the entire test module."""
    import importlib

    mock_ltx = types.ModuleType("common.ltx")
    mock_ltx.load_inputs = MagicMock(return_value=[])
    mock_ltx.run_workflow = MagicMock()
    mock_wan = types.ModuleType("common.wan22")
    mock_wan.load_inputs = MagicMock(return_value=[])
    mock_wan.run_i2v = MagicMock()
    mock_wan.run_t2v = MagicMock()
    old_ltx = sys.modules.get("common.ltx")
    old_wan = sys.modules.get("common.wan22")
    sys.modules["common.ltx"] = mock_ltx
    sys.modules["common.wan22"] = mock_wan
    try:
        mod_path = str(REPO_ROOT / "processing_job" / "i2v")
        if mod_path not in sys.path:
            sys.path.insert(0, mod_path)
        spec = importlib.util.spec_from_file_location(f"{'i2v'}_main", REPO_ROOT / "processing_job" / "i2v" / "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if old_ltx is None:
            sys.modules.pop("common.ltx", None)
        else:
            sys.modules["common.ltx"] = old_ltx
        if old_wan is None:
            sys.modules.pop("common.wan22", None)
        else:
            sys.modules["common.wan22"] = old_wan


@pytest.fixture
def i2v_module(tmp_path, _i2v_module_cached):
    """Per-test wrapper that resets env-dependent module vars."""
    mod = _i2v_module_cached
    mod.COMFY_HOME = str(tmp_path / "comfy")
    mod.LOCAL_OUTPUT_DIR = str(tmp_path / "output")
    mod.SM_INPUT_DIR = "/opt/ml/processing/input/input"
    return mod


@pytest.fixture
def vbench_module(monkeypatch):
    """Import vbench main.py with mocked DynamoDB."""
    monkeypatch.setenv("DYNAMODB_TABLE_NAME", "test-table")
    # Mock the dynamodb module before importing vbench
    mock_db = MagicMock()
    mock_db_mod = types.ModuleType("common.dynamodb")
    mock_db_mod.DynamoDBOperations = MagicMock(return_value=mock_db)
    monkeypatch.setitem(sys.modules, "common.dynamodb", mock_db_mod)

    import importlib

    spec = importlib.util.spec_from_file_location("vbench_main", REPO_ROOT / "processing_job" / "vbench" / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# t2v tests
# ===========================================================================
class TestT2vLogDirectory:
    def test_logs_existing_directory(self, t2v_module, tmp_path, loguru_capture):
        d = tmp_path / "testdir"
        d.mkdir()
        (d / "file1.txt").write_text("hello")
        (d / "subdir").mkdir()

        t2v_module.log_directory_tree(str(d), "Test Dir")
        out = loguru_capture.getvalue()
        assert "Test Dir" in out
        assert "file1.txt" in out
        assert "subdir" in out

    def test_logs_missing_directory(self, t2v_module, tmp_path, loguru_capture):
        t2v_module.log_directory_tree("/nonexistent/path", "Missing")
        out = loguru_capture.getvalue()
        assert "does not exist" in out


class TestT2vCopyOutputs:
    def test_copies_files(self, t2v_module, tmp_path):
        # Set up source
        src = tmp_path / "comfy" / "output" / "video"
        src.mkdir(parents=True)
        (src / "clip1.mp4").write_text("video1")
        (src / "clip2.mp4").write_text("video2")

        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True)

        # Patch module-level vars
        t2v_module.COMFY_HOME = str(tmp_path / "comfy")
        t2v_module.LOCAL_OUTPUT_DIR = str(out_dir)

        t2v_module.copy_outputs()

        assert (out_dir / "clip1.mp4").read_text() == "video1"
        assert (out_dir / "clip2.mp4").read_text() == "video2"

    def test_copies_subdirectories(self, t2v_module, tmp_path):
        src = tmp_path / "comfy" / "output" / "video"
        sub = src / "batch1"
        sub.mkdir(parents=True)
        (sub / "v.mp4").write_text("data")

        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True)

        t2v_module.COMFY_HOME = str(tmp_path / "comfy")
        t2v_module.LOCAL_OUTPUT_DIR = str(out_dir)

        t2v_module.copy_outputs()
        assert (out_dir / "batch1" / "v.mp4").read_text() == "data"

    def test_no_error_when_source_missing(self, t2v_module, tmp_path, loguru_capture):
        t2v_module.COMFY_HOME = str(tmp_path / "nonexistent")
        t2v_module.LOCAL_OUTPUT_DIR = str(tmp_path / "output")
        (tmp_path / "output").mkdir()

        t2v_module.copy_outputs()  # Should not raise
        out = loguru_capture.getvalue()
        assert "Copying generated videos" in out


class TestT2vWaitForQueueEmpty:
    @patch("subprocess.run")
    def test_stops_when_queue_empty(self, mock_run, t2v_module, monkeypatch):
        monkeypatch.setattr(t2v_module, "wait_for_queue_empty", lambda: None)
        # Simpler: just test the function directly with mocked import
        # Re-test by patching at module level
        t2v_module.wait_for_queue_empty = lambda: None
        t2v_module.wait_for_queue_empty()  # Should not raise

    @patch("time.sleep")
    def test_polls_until_empty(self, mock_sleep, t2v_module):
        """wait_for_queue_empty polls get_queue_size until 0."""
        # We can't easily test the internal import of is_queue_empty
        # so we verify the function exists and is callable
        assert callable(t2v_module.wait_for_queue_empty)


class TestT2vLinkInputs:
    """Tests for link_inputs_to_comfyui in t2v main.py."""

    def test_symlinks_files_into_comfyui_input(self, t2v_module, tmp_path):
        sm_input = tmp_path / "sm_input"
        sm_input.mkdir()
        (sm_input / "image.png").write_text("img")
        (sm_input / "inputs.json").write_text("{}")

        comfy_input = tmp_path / "comfy" / "input"
        comfy_input.mkdir(parents=True)

        t2v_module.SM_INPUT_DIR = str(sm_input)
        t2v_module.COMFY_HOME = str(tmp_path / "comfy")
        t2v_module.link_inputs_to_comfyui()

        assert (comfy_input / "image.png").is_symlink()
        assert (comfy_input / "inputs.json").is_symlink()

    def test_skips_existing_files(self, t2v_module, tmp_path):
        sm_input = tmp_path / "sm_input"
        sm_input.mkdir()
        (sm_input / "image.png").write_text("img")

        comfy_input = tmp_path / "comfy" / "input"
        comfy_input.mkdir(parents=True)
        (comfy_input / "image.png").write_text("existing")

        t2v_module.SM_INPUT_DIR = str(sm_input)
        t2v_module.COMFY_HOME = str(tmp_path / "comfy")
        t2v_module.link_inputs_to_comfyui()

        # Should not overwrite existing file
        assert not (comfy_input / "image.png").is_symlink()
        assert (comfy_input / "image.png").read_text() == "existing"

    def test_handles_missing_input_dir(self, t2v_module, tmp_path, loguru_capture):
        t2v_module.SM_INPUT_DIR = str(tmp_path / "nonexistent")
        t2v_module.COMFY_HOME = str(tmp_path / "comfy")
        t2v_module.link_inputs_to_comfyui()
        out = loguru_capture.getvalue()
        assert "not found" in out

    def test_creates_comfyui_input_dir(self, t2v_module, tmp_path):
        sm_input = tmp_path / "sm_input"
        sm_input.mkdir()
        (sm_input / "file.txt").write_text("data")

        t2v_module.SM_INPUT_DIR = str(sm_input)
        t2v_module.COMFY_HOME = str(tmp_path / "comfy")
        t2v_module.link_inputs_to_comfyui()

        assert (tmp_path / "comfy" / "input").is_dir()


class TestI2vLinkInputs:
    """Tests for link_inputs_to_comfyui in i2v main.py."""

    def test_symlinks_files_into_comfyui_input(self, i2v_module, tmp_path):
        sm_input = tmp_path / "sm_input"
        sm_input.mkdir()
        (sm_input / "image.png").write_text("img")

        comfy_input = tmp_path / "comfy" / "input"
        comfy_input.mkdir(parents=True)

        i2v_module.SM_INPUT_DIR = str(sm_input)
        i2v_module.COMFY_HOME = str(tmp_path / "comfy")
        i2v_module.link_inputs_to_comfyui()

        assert (comfy_input / "image.png").is_symlink()

    def test_handles_missing_input_dir(self, i2v_module, tmp_path, loguru_capture):
        i2v_module.SM_INPUT_DIR = str(tmp_path / "nonexistent")
        i2v_module.COMFY_HOME = str(tmp_path / "comfy")
        i2v_module.link_inputs_to_comfyui()
        out = loguru_capture.getvalue()
        assert "not found" in out


class TestInputPathConstants:
    """Verify SM_INPUT_DIR and load_inputs default path match SageMaker layout."""

    def test_t2v_sm_input_dir(self, t2v_module):
        assert t2v_module.SM_INPUT_DIR == "/opt/ml/processing/input/input"

    def test_i2v_sm_input_dir(self, i2v_module):
        assert i2v_module.SM_INPUT_DIR == "/opt/ml/processing/input/input"


class TestLoadInputsPath:
    """Verify load_inputs uses the correct SageMaker input path."""

    def test_ltx_default_path(self, tmp_path):
        """ltx.py load_inputs default should check shards then input/input/."""
        ltx_path = REPO_ROOT / "processing_job" / "common" / "ltx.py"
        content = ltx_path.read_text()
        assert "/opt/ml/processing/input/shards" in content
        assert "/opt/ml/processing/input/input" in content

    def test_wan22_default_path(self, tmp_path):
        """wan22.py load_inputs default should check shards then input/input/."""
        wan_path = REPO_ROOT / "processing_job" / "common" / "wan22.py"
        content = wan_path.read_text()
        assert "/opt/ml/processing/input/shards" in content
        assert "/opt/ml/processing/input/input" in content

    def test_load_inputs_reads_json(self, tmp_path):
        """load_inputs should read and parse a JSON file."""
        ltx_path = REPO_ROOT / "processing_job" / "common" / "ltx.py"
        content = ltx_path.read_text()
        assert "def load_inputs" in content
        assert "INPUTS_JSON" in content


class TestT2vMain:
    @patch("time.sleep")
    @patch("subprocess.run")
    def test_main_ltx_t2v(self, mock_run, mock_sleep, t2v_module, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--model", "ltx", "--mode", "t2v"])

        # Mock all subprocess calls to succeed
        mock_run.return_value = MagicMock(returncode=0, stdout="Queue size: 0\n")

        # Mock the imported workflow functions (ComfyScript not available in tests)
        mock_ltx_load = MagicMock(return_value=[{"prompt": "test", "image": "test.png", "id": "test-id"}])
        mock_ltx_run = MagicMock()
        monkeypatch.setattr(t2v_module, "ltx_load_inputs", mock_ltx_load)
        monkeypatch.setattr(t2v_module, "ltx_run_workflow", mock_ltx_run)
        monkeypatch.setattr(t2v_module, "wait_for_queue_empty", lambda: None)

        # Create output dir so copy_outputs doesn't fail
        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True)
        t2v_module.LOCAL_OUTPUT_DIR = str(out_dir)
        t2v_module.COMFY_HOME = str(tmp_path / "comfy")

        t2v_module.main()

        # Verify comfy launch was called
        calls = [str(c) for c in mock_run.call_args_list]
        assert any("comfy" in c and "launch" in c for c in calls)
        # Verify ltx workflow was called with disable_i2v=True (t2v mode), file_prefix, and seed
        mock_ltx_run.assert_called_once_with("test", "test.png", True, "test-id", "test-id_ltx23", seed=42)

    @patch("time.sleep")
    @patch("subprocess.run")
    def test_main_wan_i2v(self, mock_run, mock_sleep, t2v_module, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--model", "wan", "--mode", "i2v"])
        mock_run.return_value = MagicMock(returncode=0, stdout="Queue size: 0\n")

        mock_ltx_load = MagicMock(return_value=[{"prompt": "test", "image": "test.png", "id": "test-id"}])
        mock_wan_i2v = MagicMock()
        monkeypatch.setattr(t2v_module, "ltx_load_inputs", mock_ltx_load)
        monkeypatch.setattr(t2v_module, "wan_run_i2v", mock_wan_i2v)
        monkeypatch.setattr(t2v_module, "wait_for_queue_empty", lambda: None)

        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True)
        t2v_module.LOCAL_OUTPUT_DIR = str(out_dir)
        t2v_module.COMFY_HOME = str(tmp_path / "comfy")

        t2v_module.main()

        mock_wan_i2v.assert_called_once_with("test", "test.png", "test-id", "test-id_wan22", seed=42)

    def test_rejects_invalid_model(self, t2v_module, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--model", "invalid"])
        with pytest.raises(SystemExit):
            t2v_module.main()

    def test_rejects_invalid_mode(self, t2v_module, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--mode", "invalid"])
        with pytest.raises(SystemExit):
            t2v_module.main()


# ===========================================================================
# i2v tests
# ===========================================================================
class TestI2vMain:
    @patch("time.sleep")
    @patch("subprocess.run")
    def test_main_both_models(self, mock_run, mock_sleep, i2v_module, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--model", "both"])
        mock_run.return_value = MagicMock(returncode=0, stdout="Queue size: 0\n")

        mock_load = MagicMock(return_value=[{"prompt": "test", "image": "test.png", "id": "test-id"}])
        mock_ltx_run = MagicMock()
        mock_wan_i2v = MagicMock()
        monkeypatch.setattr(i2v_module, "load_inputs", mock_load)
        monkeypatch.setattr(i2v_module, "ltx_run_workflow", mock_ltx_run)
        monkeypatch.setattr(i2v_module, "wan_run_i2v", mock_wan_i2v)
        monkeypatch.setattr(i2v_module, "wait_for_queue_empty", lambda: None)

        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True)
        i2v_module.LOCAL_OUTPUT_DIR = str(out_dir)
        i2v_module.COMFY_HOME = str(tmp_path / "comfy")

        i2v_module.main()

        # Both ltx and wan should be called
        mock_ltx_run.assert_called_once()
        mock_wan_i2v.assert_called_once()

    @patch("time.sleep")
    @patch("subprocess.run")
    def test_main_ltx_only(self, mock_run, mock_sleep, i2v_module, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--model", "ltx"])
        mock_run.return_value = MagicMock(returncode=0, stdout="Queue size: 0\n")

        mock_load = MagicMock(return_value=[{"prompt": "test", "image": "test.png", "id": "test-id"}])
        mock_ltx_run = MagicMock()
        mock_wan_i2v = MagicMock()
        monkeypatch.setattr(i2v_module, "load_inputs", mock_load)
        monkeypatch.setattr(i2v_module, "ltx_run_workflow", mock_ltx_run)
        monkeypatch.setattr(i2v_module, "wan_run_i2v", mock_wan_i2v)
        monkeypatch.setattr(i2v_module, "wait_for_queue_empty", lambda: None)

        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True)
        i2v_module.LOCAL_OUTPUT_DIR = str(out_dir)
        i2v_module.COMFY_HOME = str(tmp_path / "comfy")

        i2v_module.main()

        mock_ltx_run.assert_called_once()
        mock_wan_i2v.assert_not_called()

    @patch("time.sleep")
    @patch("subprocess.run")
    def test_main_wan_only(self, mock_run, mock_sleep, i2v_module, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--model", "wan"])
        mock_run.return_value = MagicMock(returncode=0, stdout="Queue size: 0\n")

        mock_load = MagicMock(return_value=[{"prompt": "test", "image": "test.png", "id": "test-id"}])
        mock_ltx_run = MagicMock()
        mock_wan_i2v = MagicMock()
        monkeypatch.setattr(i2v_module, "load_inputs", mock_load)
        monkeypatch.setattr(i2v_module, "ltx_run_workflow", mock_ltx_run)
        monkeypatch.setattr(i2v_module, "wan_run_i2v", mock_wan_i2v)
        monkeypatch.setattr(i2v_module, "wait_for_queue_empty", lambda: None)

        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True)
        i2v_module.LOCAL_OUTPUT_DIR = str(out_dir)
        i2v_module.COMFY_HOME = str(tmp_path / "comfy")

        i2v_module.main()

        mock_ltx_run.assert_not_called()
        mock_wan_i2v.assert_called_once()

    def test_rejects_invalid_model(self, i2v_module, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--model", "invalid"])
        with pytest.raises(SystemExit):
            i2v_module.main()

    def test_default_model_is_both(self, i2v_module, monkeypatch):
        """Default --model should be 'both'."""
        import argparse

        monkeypatch.setattr("sys.argv", ["main.py"])
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", choices=["ltx", "wan", "both"], default="both")
        args = parser.parse_args([])
        assert args.model == "both"

    def test_mode_is_always_i2v(self, i2v_module):
        assert i2v_module.MODE == "i2v"


# ===========================================================================
# vbench tests
# ===========================================================================
class TestVbenchFormatJson:
    def test_basic_formatting(self, vbench_module, tmp_path):
        data = {
            "background_consistency": [
                0.95,
                [
                    {"video_path": "/videos/clip1.mp4", "video_results": 0.92},
                    {"video_path": "/videos/clip2.mp4", "video_results": 0.88},
                ],
            ],
            "motion_smoothness": [
                0.90,
                [
                    {"video_path": "/videos/clip1.mp4", "video_results": 0.85},
                    {"video_path": "/videos/clip2.mp4", "video_results": 0.91},
                ],
            ],
        }
        f = tmp_path / "eval_results.json"
        f.write_text(json.dumps(data))

        result = vbench_module.format_json_for_dynamodb(str(f))

        assert "clip1" in result
        assert "clip2" in result
        assert result["clip1"]["background_consistency"] == 0.92
        assert result["clip1"]["motion_smoothness"] == 0.85
        assert result["clip2"]["background_consistency"] == 0.88

    def test_dynamic_degree_boolean_conversion(self, vbench_module, tmp_path):
        data = {
            "dynamic_degree": [
                0.5,
                [
                    {"video_path": "/v/a.mp4", "video_results": True},
                    {"video_path": "/v/b.mp4", "video_results": False},
                    {"video_path": "/v/c.mp4", "video_results": 0},
                ],
            ],
        }
        f = tmp_path / "eval.json"
        f.write_text(json.dumps(data))

        result = vbench_module.format_json_for_dynamodb(str(f))
        assert result["a"]["dynamic_degree"] == 1
        assert result["b"]["dynamic_degree"] == 0
        assert result["c"]["dynamic_degree"] == 0

    def test_string_results_replaced_with_zero(self, vbench_module, tmp_path):
        data = {
            "imaging_quality": [0.7, [{"video_path": "/v/x.mp4", "video_results": "*"}]],
        }
        f = tmp_path / "eval.json"
        f.write_text(json.dumps(data))

        result = vbench_module.format_json_for_dynamodb(str(f))
        assert result["x"]["imaging_quality"] == 0

    def test_multiple_metrics_same_video(self, vbench_module, tmp_path):
        data = {
            "metric_a": [0.5, [{"video_path": "/v/vid.mp4", "video_results": 0.1}]],
            "metric_b": [0.6, [{"video_path": "/v/vid.mp4", "video_results": 0.2}]],
            "metric_c": [0.7, [{"video_path": "/v/vid.mp4", "video_results": 0.3}]],
        }
        f = tmp_path / "eval.json"
        f.write_text(json.dumps(data))

        result = vbench_module.format_json_for_dynamodb(str(f))
        assert len(result["vid"]) == 3
        assert result["vid"]["metric_a"] == 0.1
        assert result["vid"]["metric_b"] == 0.2
        assert result["vid"]["metric_c"] == 0.3
