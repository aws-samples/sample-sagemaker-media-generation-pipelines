"""Unit tests for infrastructure/cicd_pipeline/buildspecs.py.

Verifies each buildspec factory returns a dict with the correct
structure (version, phases, commands).
"""

import pytest

from infrastructure.cicd_pipeline.buildspecs import (
    cdk_synth,
    deploy,
    lint_and_synth,
    model_download,
    trigger_pipeline,
    unit_test,
    upload_input,
)

pytestmark = pytest.mark.cicd


def _assert_valid_buildspec(bs: dict) -> None:
    """Assert common buildspec structure."""
    assert bs["version"] == "0.2"
    assert "phases" in bs
    # Every phase must have a "commands" list
    for phase_name, phase_body in bs["phases"].items():
        assert "commands" in phase_body
        assert isinstance(phase_body["commands"], list)
        assert len(phase_body["commands"]) > 0


class TestLintAndSynthBuildspec:
    def test_structure(self) -> None:
        bs = lint_and_synth("config_vrag.yaml")
        _assert_valid_buildspec(bs)

    def test_build_contains_pre_commit(self) -> None:
        cmds = " ".join(lint_and_synth("config_vrag.yaml")["phases"]["build"]["commands"])
        assert "pre-commit run --all-files" in cmds

    def test_build_contains_cdk_synth(self) -> None:
        bs = lint_and_synth("config_t2i.yaml")
        synth_cmds = [c for c in bs["phases"]["build"]["commands"] if "cdk synth" in c]
        assert len(synth_cmds) == 1
        assert "config_t2i.yaml" in synth_cmds[0]

    def test_build_initializes_git_repo(self) -> None:
        cmds = " ".join(lint_and_synth("config_vrag.yaml")["phases"]["build"]["commands"])
        assert "git init" in cmds

    def test_build_uses_set_e(self) -> None:
        assert "set -e" in lint_and_synth("config_vrag.yaml")["phases"]["build"]["commands"]


class TestUnitTestBuildspec:
    def test_structure(self) -> None:
        bs = unit_test("uv run pytest tests/unit/ -x")
        _assert_valid_buildspec(bs)

    def test_build_contains_test_command(self) -> None:
        cmd = "uv run pytest tests/unit/ -x -m 'core or cicd'"
        assert cmd in unit_test(cmd)["phases"]["build"]["commands"]

    def test_build_uses_set_e(self) -> None:
        assert "set -e" in unit_test("uv run pytest")["phases"]["build"]["commands"]


class TestCdkSynthBuildspec:
    def test_structure(self) -> None:
        bs = cdk_synth("config_vrag.yaml")
        _assert_valid_buildspec(bs)

    def test_build_contains_synth_with_config(self) -> None:
        bs = cdk_synth("config_t2i.yaml")
        synth_cmds = [c for c in bs["phases"]["build"]["commands"] if "cdk synth" in c]
        assert len(synth_cmds) == 1
        assert "config_t2i.yaml" in synth_cmds[0]


class TestDeployBuildspec:
    def test_structure(self) -> None:
        bs = deploy("python3 deploy.py")
        _assert_valid_buildspec(bs)

    def test_build_contains_deploy_script(self) -> None:
        script = "python3 deploy.py --phase 2"
        bs = deploy(script)
        assert script in bs["phases"]["build"]["commands"]


class TestModelDownloadBuildspec:
    def test_structure(self) -> None:
        bs = model_download()
        _assert_valid_buildspec(bs)

    def test_install_has_python_runtime(self) -> None:
        bs = model_download()
        assert "runtime-versions" in bs["phases"]["install"]

    def test_build_runs_main(self) -> None:
        bs = model_download()
        cmds = " ".join(bs["phases"]["build"]["commands"])
        assert "model_download/main.py" in cmds


class TestUploadInputBuildspec:
    def test_structure(self) -> None:
        bs = upload_input(["aws s3 sync data/ s3://bucket/"])
        _assert_valid_buildspec(bs)

    def test_build_contains_sync_commands(self) -> None:
        sync = ["aws s3 sync a/ s3://b/a", "aws s3 sync c/ s3://b/c"]
        bs = upload_input(sync)
        assert bs["phases"]["build"]["commands"] == sync


class TestTriggerPipelineBuildspec:
    def test_structure(self) -> None:
        bs = trigger_pipeline()
        _assert_valid_buildspec(bs)

    def test_build_invokes_lambda(self) -> None:
        bs = trigger_pipeline()
        cmds = " ".join(bs["phases"]["build"]["commands"])
        assert "lambda invoke" in cmds


from hypothesis import given, settings
from hypothesis import strategies as st


class TestBuildspecContainsPerConfigCommand:
    """
    Property 5: For any pipeline config filename and its corresponding test
    command string, buildspecs.unit_test(test_command, cfg_file) output
    contains that exact test command string in its build phase commands.

    **Validates: Requirements 4.1, 4.2, 4.3**
    """

    @given(
        cfg_file=st.from_regex(r"[a-z][a-z0-9_]{0,20}\.yaml", fullmatch=True),
        test_command=st.text(min_size=1, max_size=200).filter(lambda s: s.isprintable()),
    )
    @settings(max_examples=100, deadline=None)
    def test_test_command_appears_in_buildspec(self, cfg_file: str, test_command: str) -> None:
        """
        **Validates: Requirements 4.1, 4.2, 4.3**
        """
        bs = unit_test(test_command)
        build_cmds = bs["phases"]["build"]["commands"]
        assert test_command in build_cmds

    @given(
        cfg_file=st.from_regex(r"[a-z][a-z0-9_]{0,20}\.yaml", fullmatch=True),
        test_command=st.text(min_size=1, max_size=200).filter(lambda s: s.isprintable()),
    )
    @settings(max_examples=100, deadline=None)
    def test_cdk_synth_references_cfg_file(self, cfg_file: str, test_command: str) -> None:
        """
        **Validates: Requirements 4.1, 4.2, 4.3**
        """
        bs = lint_and_synth(cfg_file)
        build_cmds = bs["phases"]["build"]["commands"]
        synth_cmds = [c for c in build_cmds if "cdk synth" in c]
        assert len(synth_cmds) == 1
        assert cfg_file in synth_cmds[0]
