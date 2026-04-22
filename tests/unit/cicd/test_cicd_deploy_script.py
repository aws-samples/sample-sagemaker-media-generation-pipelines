"""
Unit tests for the CI/CD deploy script's dynamic stack discovery logic.

The deploy stage runs an inline Python snippet that reads the pipeline config
YAML and determines which stacks to deploy. This must mirror the conditional
logic in app.py — specifically the A2I stack detection.

The shared CodeBuildStack is deployed separately by ``make deploy``
alongside CiCdPipelineStack and is NOT included in per-config deploys.

These tests validate the stack discovery logic in isolation (no CDK needed)
by running the same algorithm against each real pipeline config YAML.
"""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cicd


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config" / "pipeline"

# All pipeline config files
ALL_CONFIGS = sorted(f.name for f in CONFIG_DIR.glob("config*.yaml"))


def _discover_stacks(cfg_file: str, prefix: str) -> list[str]:
    """Replicate the deploy script's Phase 2 stack discovery logic.

    Phase 2 returns DataStack + (A2IStack if needed) + PipelineStack.
    CodeBuildStack is NOT included — it's deployed by ``make deploy``.
    """
    with open(CONFIG_DIR / cfg_file) as f:
        cfg = yaml.safe_load(f)

    stacks = [f"{prefix}-DataStack"]

    a2i = cfg.get("a2i") or {}
    ls = cfg.get("lambda_steps") or {}
    refs = {v.get("a2i_name") for v in ls.values() if v.get("a2i_name")}
    if refs & set(a2i.keys()):
        stacks.append(f"{prefix}-A2IStack")

    stacks.append(f"{prefix}-PipelineStack")
    return stacks


def _discover_phase1_stacks(cfg_file: str, prefix: str) -> list[str]:
    """Replicate the deploy script's Phase 1 stack discovery logic.

    Phase 1 deploys consumer stacks first (PipelineStack + A2IStack)
    to break cross-stack CloudFormation imports before CodeBuildStack
    tries to remove old exports.
    """
    with open(CONFIG_DIR / cfg_file) as f:
        cfg = yaml.safe_load(f)

    stacks = []

    a2i = cfg.get("a2i") or {}
    ls = cfg.get("lambda_steps") or {}
    refs = {v.get("a2i_name") for v in ls.values() if v.get("a2i_name")}
    if refs & set(a2i.keys()):
        stacks.append(f"{prefix}-A2IStack")

    stacks.append(f"{prefix}-PipelineStack")
    return stacks


def _discover_stacks_app_py(cfg_file: str, prefix: str) -> list[str]:
    """Replicate app.py's stack creation logic for comparison.

    app.py creates A2IStack when referenced_a2i_names & set(a2i.keys())
    is non-empty. This must produce the same result as _discover_stacks.
    """
    from config.config import get_pipeline_config

    pipeline_config = get_pipeline_config(cfg_file)

    stacks = [f"{prefix}-DataStack"]

    referenced_a2i_names = {ls_cfg.a2i_name for ls_cfg in pipeline_config.lambda_steps.values() if ls_cfg.a2i_name}
    active_a2i_names = referenced_a2i_names & set(pipeline_config.a2i.keys())
    if active_a2i_names:
        stacks.append(f"{prefix}-A2IStack")

    stacks.append(f"{prefix}-PipelineStack")
    return stacks


class TestDeployScriptStackDiscovery:
    """Validate the deploy script discovers the correct stacks for each config."""

    @pytest.mark.parametrize("cfg_file", ALL_CONFIGS)
    def test_always_includes_data_and_pipeline(self, cfg_file: str) -> None:
        """Every config must deploy DataStack and PipelineStack."""
        stacks = _discover_stacks(cfg_file, "test")
        assert "test-DataStack" in stacks
        assert "test-PipelineStack" in stacks
        assert "test-CodeBuildStack" not in stacks

    @pytest.mark.parametrize("cfg_file", ALL_CONFIGS)
    def test_pipeline_stack_is_last(self, cfg_file: str) -> None:
        """PipelineStack must be deployed last (depends on all others)."""
        stacks = _discover_stacks(cfg_file, "test")
        assert stacks[-1] == "test-PipelineStack"

    @pytest.mark.parametrize("cfg_file", ALL_CONFIGS)
    def test_deploy_script_matches_app_py(self, cfg_file: str) -> None:
        """Deploy script stack list must match app.py's conditional logic."""
        deploy_stacks = _discover_stacks(cfg_file, "test")
        app_stacks = _discover_stacks_app_py(cfg_file, "test")
        assert deploy_stacks == app_stacks, (
            f"Mismatch for {cfg_file}: deploy_script={deploy_stacks}, app_py={app_stacks}"
        )

    def test_config_with_no_a2i(self) -> None:
        """Config without A2I → A2IStack must NOT be included."""
        cfg = {"steps": {"agent": {}}, "pipeline_graph": {"agent": []}}
        stacks = TestDeployScriptA2IDetection._discover_from_dict(cfg, "test")
        assert "test-A2IStack" not in stacks

    def test_config_with_a2i(self) -> None:
        """Config with matching A2I + lambda_steps → A2IStack must be included."""
        cfg = {
            "steps": {"agent": {}},
            "a2i": {"vid_t2v": {"media_type": "video"}},
            "lambda_steps": {"submit_a2i_t2v": {"a2i_name": "vid_t2v", "lambda_path": "submit_a2i_review"}},
        }
        stacks = TestDeployScriptA2IDetection._discover_from_dict(cfg, "test")
        assert "test-A2IStack" in stacks

    @pytest.mark.parametrize("cfg_file", ALL_CONFIGS)
    def test_no_duplicate_stacks(self, cfg_file: str) -> None:
        """Stack list must not contain duplicates."""
        stacks = _discover_stacks(cfg_file, "test")
        assert len(stacks) == len(set(stacks))


class TestDeployScriptPhase1:
    """Tests for Phase 1 (consumer-first) deploy order.

    Phase 1 deploys PipelineStack (and A2IStack if needed) before
    CodeBuildStack to break cross-stack CloudFormation imports.
    """

    @pytest.mark.parametrize("cfg_file", ALL_CONFIGS)
    def test_phase1_always_includes_pipeline_stack(self, cfg_file: str) -> None:
        """Phase 1 must always include PipelineStack."""
        stacks = _discover_phase1_stacks(cfg_file, "test")
        assert "test-PipelineStack" in stacks

    @pytest.mark.parametrize("cfg_file", ALL_CONFIGS)
    def test_phase1_never_includes_data_or_codebuild(self, cfg_file: str) -> None:
        """Phase 1 must NOT include DataStack or CodeBuildStack."""
        stacks = _discover_phase1_stacks(cfg_file, "test")
        assert "test-DataStack" not in stacks
        assert "test-CodeBuildStack" not in stacks

    @pytest.mark.parametrize("cfg_file", ALL_CONFIGS)
    def test_phase1_pipeline_stack_is_last(self, cfg_file: str) -> None:
        """PipelineStack must be last in Phase 1."""
        stacks = _discover_phase1_stacks(cfg_file, "test")
        assert stacks[-1] == "test-PipelineStack"

    def test_phase1_no_a2i(self) -> None:
        """Config without A2I → Phase 1 is just PipelineStack."""
        cfg = {"steps": {"agent": {}}, "pipeline_graph": {"agent": []}}
        stacks = self._phase1_from_dict(cfg, "test")
        assert stacks == ["test-PipelineStack"]

    def test_phase1_with_a2i(self) -> None:
        """Config with A2I → Phase 1 includes A2IStack before PipelineStack."""
        cfg = {
            "a2i": {"vid_t2v": {"media_type": "video"}},
            "lambda_steps": {"submit": {"a2i_name": "vid_t2v", "lambda_path": "submit_a2i_review"}},
        }
        stacks = self._phase1_from_dict(cfg, "test")
        assert stacks == ["test-A2IStack", "test-PipelineStack"]

    @staticmethod
    def _phase1_from_dict(cfg: dict, prefix: str) -> list[str]:
        """Run Phase 1 discovery on an in-memory config dict."""
        stacks = []
        a2i = cfg.get("a2i") or {}
        ls = cfg.get("lambda_steps") or {}
        refs = {v.get("a2i_name") for v in ls.values() if v.get("a2i_name")}
        if refs & set(a2i.keys()):
            stacks.append(f"{prefix}-A2IStack")
        stacks.append(f"{prefix}-PipelineStack")
        return stacks

    @pytest.mark.parametrize("cfg_file", ALL_CONFIGS)
    def test_phase1_is_subset_of_phase2(self, cfg_file: str) -> None:
        """Every stack in Phase 1 must also appear in Phase 2."""
        phase1 = _discover_phase1_stacks(cfg_file, "test")
        phase2 = _discover_stacks(cfg_file, "test")
        assert set(phase1).issubset(set(phase2))


class TestDeployScriptA2IDetection:
    """Test edge cases in A2I detection logic."""

    def test_a2i_without_lambda_steps_not_included(self) -> None:
        """Config with a2i section but no lambda_steps referencing it → no A2IStack."""
        cfg = {
            "a2i": {"review": {"media_type": "video"}},
            "lambda_steps": {},
        }
        stacks = self._discover_from_dict(cfg, "test")
        assert "test-A2IStack" not in stacks

    def test_lambda_steps_without_a2i_not_included(self) -> None:
        """Config with lambda_steps referencing a2i_name but no a2i section → no A2IStack."""
        cfg = {
            "lambda_steps": {"submit": {"a2i_name": "review", "lambda_path": "submit_a2i_review"}},
        }
        stacks = self._discover_from_dict(cfg, "test")
        assert "test-A2IStack" not in stacks

    def test_mismatched_a2i_name_not_included(self) -> None:
        """lambda_steps references a2i_name that doesn't exist in a2i section → no A2IStack."""
        cfg = {
            "a2i": {"video_review": {"media_type": "video"}},
            "lambda_steps": {"submit": {"a2i_name": "wrong_name", "lambda_path": "submit_a2i_review"}},
        }
        stacks = self._discover_from_dict(cfg, "test")
        assert "test-A2IStack" not in stacks

    def test_matching_a2i_name_included(self) -> None:
        """lambda_steps references a2i_name that exists in a2i section → A2IStack included."""
        cfg = {
            "a2i": {"video_review": {"media_type": "video"}},
            "lambda_steps": {"submit": {"a2i_name": "video_review", "lambda_path": "submit_a2i_review"}},
        }
        stacks = self._discover_from_dict(cfg, "test")
        assert "test-A2IStack" in stacks

    def test_empty_config_no_a2i(self) -> None:
        """Config with no a2i and no lambda_steps → no A2IStack."""
        cfg = {}
        stacks = self._discover_from_dict(cfg, "test")
        assert "test-A2IStack" not in stacks

    @staticmethod
    def _discover_from_dict(cfg: dict, prefix: str) -> list[str]:
        """Run the deploy script's stack discovery on an in-memory config dict."""
        stacks = [f"{prefix}-DataStack"]
        a2i = cfg.get("a2i") or {}
        ls = cfg.get("lambda_steps") or {}
        refs = {v.get("a2i_name") for v in ls.values() if v.get("a2i_name")}
        if refs & set(a2i.keys()):
            stacks.append(f"{prefix}-A2IStack")
        stacks.append(f"{prefix}-PipelineStack")
        return stacks
