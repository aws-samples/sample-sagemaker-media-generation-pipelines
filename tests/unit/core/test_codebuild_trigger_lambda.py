"""
Unit tests for the codebuild_trigger Lambda function.

Tests verify that the cr.Provider on_event handler starts all CodeBuild
projects (fire-and-forget), handles Delete requests as no-ops, and returns
proper dicts for the cr.Provider framework.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# Set env var before importing the module
os.environ["PROJECT_NAMES"] = "project-a,project-b"

import lambdas.codebuild_trigger.index as _module  # noqa: E402
from lambdas.codebuild_trigger.index import handler  # noqa: E402

pytestmark = pytest.mark.core


def _make_event(request_type: str = "Create", physical_id: str | None = None) -> dict:
    event = {
        "RequestType": request_type,
        "LogicalResourceId": "BuildTriggerCR",
        "StackId": "arn:aws:cloudformation:us-east-1:123456789012:stack/test/guid",
        "RequestId": "unique-id-1234",
        "ResourceProperties": {
            "ProjectNames": "project-a,project-b",
        },
    }
    if physical_id:
        event["PhysicalResourceId"] = physical_id
    return event


class TestHandlerDelete:
    """Delete requests should return immediately with PhysicalResourceId."""

    @patch.object(_module, "codebuild")
    def test_delete_returns_dict(self, mock_cb):
        event = _make_event("Delete", physical_id="old-physical-id")
        result = handler(event, MagicMock())

        assert isinstance(result, dict)
        assert result["PhysicalResourceId"] == "old-physical-id"
        mock_cb.start_build.assert_not_called()

    @patch.object(_module, "codebuild")
    def test_delete_without_physical_id_uses_logical_id(self, mock_cb):
        event = _make_event("Delete")
        result = handler(event, MagicMock())

        assert result["PhysicalResourceId"] == "BuildTriggerCR"


class TestHandlerCreate:
    """Create requests should start builds and return success dict."""

    @patch.object(_module, "codebuild")
    def test_all_builds_started(self, mock_cb):
        mock_cb.start_build.side_effect = [
            {"build": {"id": "project-a:build-1"}},
            {"build": {"id": "project-b:build-2"}},
        ]

        result = handler(_make_event(), MagicMock())

        assert mock_cb.start_build.call_count == 2
        assert isinstance(result, dict)
        assert "PhysicalResourceId" in result
        assert result["Data"]["BuildCount"] == "2"

    @patch.object(_module, "codebuild")
    def test_returns_build_ids_in_data(self, mock_cb):
        mock_cb.start_build.side_effect = [
            {"build": {"id": "project-a:build-1"}},
            {"build": {"id": "project-b:build-2"}},
        ]

        result = handler(_make_event(), MagicMock())

        assert "project-a:build-1" in result["Data"]["BuildIds"]
        assert "project-b:build-2" in result["Data"]["BuildIds"]

    @patch.object(_module, "codebuild")
    def test_no_polling_after_start(self, mock_cb):
        """Fire-and-forget: no batch_get_builds calls."""
        mock_cb.start_build.side_effect = [
            {"build": {"id": "project-a:build-1"}},
            {"build": {"id": "project-b:build-2"}},
        ]

        handler(_make_event(), MagicMock())

        mock_cb.batch_get_builds.assert_not_called()

    @patch.object(_module, "codebuild")
    def test_exception_propagates(self, mock_cb):
        """cr.Provider catches exceptions and sends FAILED to CloudFormation."""
        mock_cb.start_build.side_effect = Exception("AccessDenied")

        with pytest.raises(Exception, match="AccessDenied"):
            handler(_make_event(), MagicMock())

    @patch.object(_module, "codebuild")
    def test_single_project(self, mock_cb):
        """Works with a single project name."""
        with patch.dict(os.environ, {"PROJECT_NAMES": "solo-project"}):
            mock_cb.start_build.return_value = {"build": {"id": "solo:1"}}

            result = handler(_make_event(), MagicMock())

            mock_cb.start_build.assert_called_once_with(projectName="solo-project")
            assert result["Data"]["BuildCount"] == "1"


class TestHandlerUpdate:
    """Update requests should re-trigger builds (same as Create)."""

    @patch.object(_module, "codebuild")
    def test_update_starts_builds(self, mock_cb):
        mock_cb.start_build.side_effect = [
            {"build": {"id": "project-a:build-3"}},
            {"build": {"id": "project-b:build-4"}},
        ]

        result = handler(_make_event("Update"), MagicMock())

        assert mock_cb.start_build.call_count == 2
        assert result["Data"]["BuildCount"] == "2"


class TestHandlerDoesNotUseCfnresponse:
    """Verify the handler does NOT use cfnresponse or send_response.

    cr.Provider framework handles CloudFormation callbacks.
    """

    def test_no_cfnresponse_or_urllib_in_source(self):
        import importlib

        source = importlib.util.find_spec("lambdas.codebuild_trigger.index")
        with open(source.origin) as f:
            lines = f.readlines()
        # Check that no line actually imports cfnresponse/urllib (ignore comments/docstrings)
        import_lines = [l.strip() for l in lines if l.strip().startswith(("import ", "from "))]
        for line in import_lines:
            assert "cfnresponse" not in line, f"Should not import cfnresponse: {line}"
            assert "urllib" not in line, f"Should not import urllib: {line}"
        # Check no send_response function definition
        func_lines = [l.strip() for l in lines if l.strip().startswith("def ")]
        func_names = [l.split("(")[0].replace("def ", "") for l in func_lines]
        assert "send_response" not in func_names

    @patch.object(_module, "codebuild")
    def test_handler_returns_dict_not_none(self, mock_cb):
        mock_cb.start_build.return_value = {"build": {"id": "x:1"}}
        result = handler(_make_event(), MagicMock())
        assert result is not None
        assert isinstance(result, dict)
