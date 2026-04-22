"""Unit tests for infrastructure/cicd_pipeline/resolve_stacks.py.

Tests the resolve_stacks() function which determines which CDK stacks
to deploy for a given pipeline config and phase.

The shared CodeBuildStack is deployed separately by ``make deploy``
alongside CiCdPipelineStack and is NOT included in per-config deploys.
"""

from unittest.mock import mock_open, patch

import pytest
import yaml

from infrastructure.cicd_pipeline.resolve_stacks import resolve_stacks

pytestmark = pytest.mark.cicd


def _mock_config(cfg: dict) -> str:
    """Return YAML string for a config dict."""
    return yaml.dump(cfg)


class TestResolveStacksPhase1:
    """Phase 1 returns consumer stacks only (no Data/CodeBuild)."""

    def test_phase1_always_includes_pipeline_stack(self) -> None:
        cfg = {"steps": {"t2v": {}}}
        with patch("builtins.open", mock_open(read_data=_mock_config(cfg))):
            stacks = resolve_stacks("config_vrag.yaml", "dev", phase=1)
        assert "dev-PipelineStack" in stacks

    def test_phase1_never_includes_data_or_codebuild(self) -> None:
        cfg = {"steps": {"t2v": {}}}
        with patch("builtins.open", mock_open(read_data=_mock_config(cfg))):
            stacks = resolve_stacks("config_vrag.yaml", "dev", phase=1)
        assert "dev-DataStack" not in stacks
        assert "dev-CodeBuildStack" not in stacks

    def test_phase1_with_a2i_includes_a2i_stack(self) -> None:
        cfg = {
            "a2i": {"review": {"task_template": "t.html"}},
            "lambda_steps": {"submit": {"a2i_name": "review"}},
        }
        with patch("builtins.open", mock_open(read_data=_mock_config(cfg))):
            stacks = resolve_stacks("config_vrag.yaml", "dev", phase=1)
        assert "dev-A2IStack" in stacks
        assert "dev-PipelineStack" in stacks

    def test_phase1_without_a2i_no_a2i_stack(self) -> None:
        cfg = {"steps": {"t2v": {}}}
        with patch("builtins.open", mock_open(read_data=_mock_config(cfg))):
            stacks = resolve_stacks("config_vrag.yaml", "dev", phase=1)
        assert "dev-A2IStack" not in stacks


class TestResolveStacksPhase2:
    """Phase 2 returns DataStack + consumer stacks. No CodeBuildStack."""

    def test_phase2_includes_data(self) -> None:
        cfg = {"steps": {"t2v": {}}}
        with patch("builtins.open", mock_open(read_data=_mock_config(cfg))):
            stacks = resolve_stacks("config_vrag.yaml", "dev", phase=2)
        assert "dev-DataStack" in stacks
        assert "dev-CodeBuildStack" not in stacks

    def test_phase2_includes_pipeline_stack(self) -> None:
        cfg = {"steps": {"t2v": {}}}
        with patch("builtins.open", mock_open(read_data=_mock_config(cfg))):
            stacks = resolve_stacks("config_vrag.yaml", "dev", phase=2)
        assert "dev-PipelineStack" in stacks
        assert "dev-CodeBuildStack" not in stacks

    def test_phase2_with_a2i(self) -> None:
        cfg = {
            "a2i": {"review": {"task_template": "t.html"}},
            "lambda_steps": {"submit": {"a2i_name": "review"}},
        }
        with patch("builtins.open", mock_open(read_data=_mock_config(cfg))):
            stacks = resolve_stacks("config_vrag.yaml", "cfg", phase=2)
        assert stacks == [
            "cfg-DataStack",
            "cfg-A2IStack",
            "cfg-PipelineStack",
        ]


class TestResolveStacksA2IDetection:
    """A2I stack is only included when lambda_steps reference a2i names."""

    def test_a2i_without_lambda_steps_not_included(self) -> None:
        cfg = {"a2i": {"review": {}}}
        with patch("builtins.open", mock_open(read_data=_mock_config(cfg))):
            stacks = resolve_stacks("config_vrag.yaml", "dev", phase=2)
        assert "dev-A2IStack" not in stacks

    def test_lambda_steps_without_a2i_not_included(self) -> None:
        cfg = {"lambda_steps": {"submit": {"a2i_name": "review"}}}
        with patch("builtins.open", mock_open(read_data=_mock_config(cfg))):
            stacks = resolve_stacks("config_vrag.yaml", "dev", phase=2)
        assert "dev-A2IStack" not in stacks

    def test_mismatched_a2i_name_not_included(self) -> None:
        cfg = {
            "a2i": {"review": {}},
            "lambda_steps": {"submit": {"a2i_name": "other"}},
        }
        with patch("builtins.open", mock_open(read_data=_mock_config(cfg))):
            stacks = resolve_stacks("config_vrag.yaml", "dev", phase=2)
        assert "dev-A2IStack" not in stacks

    def test_prefix_propagated(self) -> None:
        cfg = {"steps": {"t2v": {}}}
        with patch("builtins.open", mock_open(read_data=_mock_config(cfg))):
            stacks = resolve_stacks("config_vrag.yaml", "prod", phase=2)
        assert all(s.startswith("prod-") for s in stacks)
        assert "prod-DataStack" in stacks
        assert "prod-PipelineStack" in stacks
