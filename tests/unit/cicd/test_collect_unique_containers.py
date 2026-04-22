"""
Unit tests for collect_unique_containers and collect_unique_step_names.

Tests verify deduplication across multiple configs, setup job inclusion,
shared_prefix usage, and ecr_image override resolution.
"""

from unittest.mock import patch

import pytest

from config.config import ContainerConfig, PipelineConfig, SetupConfig
from infrastructure.cicd_pipeline.helpers import collect_unique_containers, collect_unique_step_names

pytestmark = pytest.mark.cicd


def _cfg() -> ContainerConfig:
    return ContainerConfig(
        InstanceCount=1,
        InstanceType="ml.g5.xlarge",
        VolumeSizeInGB=125,
        ContainerEntrypoint=["/bin/bash", "./run_job.sh"],
        ContainerArguments=["300"],
    )


def _setup_cfg() -> SetupConfig:
    return SetupConfig(
        InstanceCount=1,
        InstanceType="ml.c5.xlarge",
        VolumeSizeInGB=50,
        ContainerEntrypoint=["python3", "main.py"],
        ContainerArguments=["--setup"],
        dataset_url="https://example.com/data",
        dataset_script="loader.py",
        num_prompts=10,
        test_image_count=100,
    )


def _mock_get_pipeline_config(configs: dict[str, PipelineConfig]):
    """Return a side_effect function that maps cfg_file -> PipelineConfig."""

    def _get(cfg_file: str) -> PipelineConfig:
        return configs[cfg_file]

    return _get


_PATCH_TARGET = "config.config.get_pipeline_config"


class TestCollectUniqueContainers:
    """Tests for collect_unique_containers deduplication."""

    def test_steps_from_multiple_configs_deduplicated(self) -> None:
        """Same step name in two configs appears only once."""
        configs = {
            "a.yaml": PipelineConfig(construct_id="a", s3_downloads=[], steps={"t2v": _cfg(), "i2v": _cfg()}),
            "b.yaml": PipelineConfig(construct_id="b", s3_downloads=[], steps={"t2v": _cfg(), "t2a": _cfg()}),
        }
        with patch(_PATCH_TARGET, side_effect=_mock_get_pipeline_config(configs)):
            result = collect_unique_containers(["a.yaml", "b.yaml"], "dev")
        names = [e["container"] for e in result]
        assert names.count("t2v") == 1
        assert "i2v" in names
        assert "t2a" in names

    def test_setup_jobs_included(self) -> None:
        """Setup jobs from config.setup are included in the result."""
        configs = {
            "a.yaml": PipelineConfig(
                construct_id="a",
                s3_downloads=[],
                steps={"t2v": _cfg()},
                setup={"dataset_ingest": _setup_cfg()},
            ),
        }
        with patch(_PATCH_TARGET, side_effect=_mock_get_pipeline_config(configs)):
            result = collect_unique_containers(["a.yaml"], "dev")
        names = [e["container"] for e in result]
        assert "dataset_ingest" in names

    def test_all_entries_use_shared_prefix(self) -> None:
        """Every entry's prefix equals the shared_prefix argument."""
        configs = {
            "a.yaml": PipelineConfig(construct_id="a", s3_downloads=[], steps={"t2v": _cfg()}),
        }
        with patch(_PATCH_TARGET, side_effect=_mock_get_pipeline_config(configs)):
            result = collect_unique_containers(["a.yaml"], "shared")
        for entry in result:
            assert entry["prefix"] == "shared"

    def test_ecr_image_override_resolved(self) -> None:
        """Step with ecr_image override uses the override name."""
        step = _cfg()
        step.ecr_image = "vbench"
        configs = {
            "a.yaml": PipelineConfig(construct_id="a", s3_downloads=[], steps={"vbench_t2v": step}),
        }
        with patch(_PATCH_TARGET, side_effect=_mock_get_pipeline_config(configs)):
            result = collect_unique_containers(["a.yaml"], "dev")
        names = [e["container"] for e in result]
        assert "vbench" in names
        assert "vbench_t2v" not in names

    def test_setup_job_collision_with_step_deduplicated(self) -> None:
        """Setup job with same name as a step is deduplicated."""
        configs = {
            "a.yaml": PipelineConfig(
                construct_id="a",
                s3_downloads=[],
                steps={"shared_step": _cfg()},
                setup={"shared_step": _setup_cfg()},
            ),
        }
        with patch(_PATCH_TARGET, side_effect=_mock_get_pipeline_config(configs)):
            result = collect_unique_containers(["a.yaml"], "dev")
        names = [e["container"] for e in result]
        assert names.count("shared_step") == 1


class TestCollectUniqueStepNames:
    """Tests for collect_unique_step_names returning just name strings."""

    def test_returns_name_strings(self) -> None:
        configs = {
            "a.yaml": PipelineConfig(construct_id="a", s3_downloads=[], steps={"t2v": _cfg(), "i2v": _cfg()}),
        }
        with patch(_PATCH_TARGET, side_effect=_mock_get_pipeline_config(configs)):
            result = collect_unique_step_names(["a.yaml"], "dev")
        assert isinstance(result, list)
        assert all(isinstance(n, str) for n in result)
        assert "t2v" in result
        assert "i2v" in result


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings
from hypothesis import strategies as st


def _step_name_strategy():
    """Generate valid step names (lowercase alphanumeric + underscore)."""
    return st.from_regex(r"[a-z][a-z0-9_]{1,15}", fullmatch=True)


def _pipeline_config_strategy():
    """Generate a mock PipelineConfig with random steps, ecr_image overrides, and setup jobs."""
    step_entry = st.fixed_dictionaries(
        {
            "name": _step_name_strategy(),
            "ecr_image": st.one_of(st.just(""), _step_name_strategy()),
        }
    )
    return st.fixed_dictionaries(
        {
            "steps": st.lists(step_entry, min_size=1, max_size=5),
            "setup": st.lists(_step_name_strategy(), min_size=0, max_size=3),
        }
    )


def _build_mock_configs(config_dicts: list[dict]) -> dict[str, PipelineConfig]:
    """Build a dict of cfg_file -> PipelineConfig from generated dicts."""
    configs = {}
    for i, d in enumerate(config_dicts):
        steps = {}
        for entry in d["steps"]:
            cfg = _cfg()
            cfg.ecr_image = entry["ecr_image"]
            steps[entry["name"]] = cfg
        setup = {}
        for name in d["setup"]:
            setup[name] = _setup_cfg()
        cfg_file = f"config_{i}.yaml"
        configs[cfg_file] = PipelineConfig(
            construct_id=f"c{i}",
            s3_downloads=[],
            steps=steps,
            setup=setup,
        )
    return configs


class TestCollectUniqueContainersProperty:
    """Property 1: Container deduplication completeness and prefix invariant.

    **Validates: Requirements 1.3, 1.5, 1.6, 5.1, 5.2**
    """

    @given(
        config_dicts=st.lists(_pipeline_config_strategy(), min_size=1, max_size=3),
        shared_prefix=st.from_regex(r"[a-z][a-z0-9]{2}", fullmatch=True),
    )
    @settings(max_examples=100)
    def test_deduplication_and_prefix_invariant(self, config_dicts, shared_prefix) -> None:
        """Every unique container appears exactly once, every prefix equals shared_prefix."""
        configs = _build_mock_configs(config_dicts)
        cfg_files = list(configs.keys())

        with patch(_PATCH_TARGET, side_effect=lambda f: configs[f]):
            result = collect_unique_containers(cfg_files, shared_prefix)

        # Collect expected unique containers
        expected_containers: set[str] = set()
        for cfg in configs.values():
            for step_name, step_cfg in cfg.steps.items():
                expected_containers.add(step_cfg.ecr_image or step_name)
            for setup_name in cfg.setup:
                expected_containers.add(setup_name)

        result_names = [e["container"] for e in result]

        # No duplicates
        assert len(result_names) == len(set(result_names)), "Duplicate containers found"

        # Every expected container is present
        assert set(result_names) == expected_containers

        # Every entry's prefix equals shared_prefix
        for entry in result:
            assert entry["prefix"] == shared_prefix
