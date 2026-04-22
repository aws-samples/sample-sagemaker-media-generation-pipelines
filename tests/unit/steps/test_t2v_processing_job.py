"""
Unit tests for t2v processing job scripts.

Tests cover:
- is_queue_empty.py: Queue polling with mocked HTTP responses
- run_job.sh: CLI argument parsing and validation
"""

import importlib.util
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests.exceptions

from tests.unit.conftest import REPO_ROOT

pytestmark = pytest.mark.steps_t2v


def _import_module_from_path(name: str, filepath: str):
    """Import a Python module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_is_queue_empty = _import_module_from_path(
    "is_queue_empty",
    str(REPO_ROOT / "processing_job" / "common" / "is_queue_empty.py"),
)
main = _is_queue_empty.get_queue_size
ENDPOINT = _is_queue_empty.ENDPOINT


# ---------------------------------------------------------------------------
# is_queue_empty.py tests
# ---------------------------------------------------------------------------


class TestIsQueueEmpty:
    """Tests for the is_queue_empty main() function."""

    @patch("is_queue_empty.requests.get")
    def test_returns_queue_size_when_zero(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"exec_info": {"queue_remaining": 0}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        assert main() == 0
        mock_get.assert_called_once_with(ENDPOINT, timeout=10)

    @patch("is_queue_empty.requests.get")
    def test_returns_queue_size_when_nonzero(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"exec_info": {"queue_remaining": 5}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        assert main() == 5

    @patch("is_queue_empty.requests.get")
    def test_returns_none_on_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=MagicMock(text="Server Error")
        )
        mock_get.return_value = mock_response

        assert main() is None

    @patch("is_queue_empty.requests.get")
    def test_returns_none_on_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()
        assert main() is None

    @patch("is_queue_empty.requests.get")
    def test_returns_none_on_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout()
        assert main() is None

    @patch("is_queue_empty.requests.get")
    def test_returns_none_on_generic_request_exception(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("fail")
        assert main() is None

    @patch("is_queue_empty.requests.get")
    def test_returns_queue_size(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"exec_info": {"queue_remaining": 3}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        assert main() == 3

    def test_endpoint_is_localhost(self):
        assert ENDPOINT == "http://127.0.0.1:8188/prompt"


# ---------------------------------------------------------------------------
# run_job.sh CLI argument tests
# ---------------------------------------------------------------------------

RUN_JOB_SH = str(REPO_ROOT / "processing_job" / "t2v" / "run_job.sh")


class TestRunJobShArgs:
    """Tests for run_job.sh argument parsing and validation."""

    def _run_sh(self, args: list[str], expect_exit: bool = True) -> subprocess.CompletedProcess:
        """Run run_job.sh with given args.

        Args:
            args: CLI arguments to pass.
            expect_exit: If True, the script should exit quickly (validation error).
                If False, the script passes validation and will hang on comfy/sleep,
                so we catch the TimeoutExpired and return stdout/stderr from it.
        """
        try:
            return subprocess.run(
                ["bash", RUN_JOB_SH] + args,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except subprocess.TimeoutExpired as e:
            if expect_exit:
                raise
            # Script passed validation but hung on comfy/sleep — that's expected
            return subprocess.CompletedProcess(
                args=e.cmd,
                returncode=0,
                stdout=e.stdout.decode() if e.stdout else "",
                stderr=e.stderr.decode() if e.stderr else "",
            )

    def test_rejects_invalid_model(self):
        result = self._run_sh(["--model", "invalid", "--mode", "i2v"])
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "must be 'ltx' or 'wan'" in combined

    def test_rejects_invalid_mode(self):
        result = self._run_sh(["--model", "ltx", "--mode", "invalid"])
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "must be 'i2v' or 't2v'" in combined

    def test_rejects_unknown_argument(self):
        result = self._run_sh(["--unknown", "value"])
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "Unknown arg" in combined

    def test_accepts_ltx_i2v(self):
        result = self._run_sh(["--model", "ltx", "--mode", "i2v"], expect_exit=False)
        combined = result.stdout + result.stderr
        assert "Model: ltx, Mode: i2v" in combined

    def test_accepts_ltx_t2v(self):
        result = self._run_sh(["--model", "ltx", "--mode", "t2v"], expect_exit=False)
        combined = result.stdout + result.stderr
        assert "Model: ltx, Mode: t2v" in combined

    def test_accepts_wan_i2v(self):
        result = self._run_sh(["--model", "wan", "--mode", "i2v"], expect_exit=False)
        combined = result.stdout + result.stderr
        assert "Model: wan, Mode: i2v" in combined

    def test_accepts_wan_t2v(self):
        result = self._run_sh(["--model", "wan", "--mode", "t2v"], expect_exit=False)
        combined = result.stdout + result.stderr
        assert "Model: wan, Mode: t2v" in combined

    def test_defaults_to_ltx_i2v(self):
        result = self._run_sh([], expect_exit=False)
        combined = result.stdout + result.stderr
        assert "Model: ltx, Mode: i2v" in combined
