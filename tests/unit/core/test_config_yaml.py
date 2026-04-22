"""Tests that validate the actual config/pipeline/config_vrag.yaml file."""

from pathlib import Path

import pytest
import yaml

from config.config import PipelineConfig

pytestmark = pytest.mark.core


CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "pipeline" / "config_vrag.yaml"


@pytest.fixture
def raw_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def pipeline_config(raw_config):
    return PipelineConfig(**raw_config)


class TestConfigYamlLoads:
    """config_vrag.yaml loads and validates through PipelineConfig."""

    def test_loads_without_error(self, pipeline_config):
        assert pipeline_config is not None

    def test_has_steps(self, pipeline_config):
        assert len(pipeline_config.steps) > 0

    def test_pipeline_graph_present(self, pipeline_config):
        assert len(pipeline_config.pipeline_graph) > 0

    def test_pipeline_graph_references_valid_steps(self, pipeline_config):
        valid_names = set(pipeline_config.steps.keys()) | set(pipeline_config.lambda_steps.keys())
        for step, deps in pipeline_config.pipeline_graph.items():
            assert step in valid_names, f"Graph step not in steps or lambda_steps: {step}"
            for dep in deps:
                assert dep in valid_names, f"Graph dep not in steps or lambda_steps: {dep}"

    def test_has_s3_downloads(self, pipeline_config):
        assert len(pipeline_config.s3_downloads) > 0

    def test_dynamodb_keys(self, pipeline_config):
        assert pipeline_config.dynamodb.partition_key == "id"
        assert pipeline_config.dynamodb.sort_key == "step"


class TestS3DownloadPaths:
    """s3_download paths must not have a models/ prefix."""

    def test_no_models_prefix(self, pipeline_config):
        for dl in pipeline_config.s3_downloads:
            assert not dl.path.startswith("models/"), f"Path should not start with 'models/': {dl.path}"

    def test_all_urls_are_https(self, pipeline_config):
        for dl in pipeline_config.s3_downloads:
            assert dl.url.startswith("https://"), f"Non-HTTPS URL: {dl.url}"

    def test_paths_are_nonempty(self, pipeline_config):
        for dl in pipeline_config.s3_downloads:
            assert len(dl.path.strip()) > 0

    def test_no_duplicate_paths(self, pipeline_config):
        paths = [dl.path for dl in pipeline_config.s3_downloads]
        assert len(paths) == len(set(paths)), "Duplicate download paths found"

    def test_extract_only_on_zip_urls(self, pipeline_config):
        for dl in pipeline_config.s3_downloads:
            if dl.extract:
                assert dl.url.endswith(".zip"), f"extract=true but URL is not a zip: {dl.url}"


class TestStepConfigs:
    """Step configurations are valid."""

    def test_all_steps_have_entrypoint(self, pipeline_config):
        for name, cfg in pipeline_config.steps.items():
            assert len(cfg.ContainerEntrypoint) > 0, f"{name} missing entrypoint"

    def test_all_steps_have_arguments(self, pipeline_config):
        for name, cfg in pipeline_config.steps.items():
            assert len(cfg.ContainerArguments) > 0, f"{name} missing arguments"

    def test_instance_types_are_valid(self, pipeline_config):
        valid = {"ml.c5.xlarge", "ml.g4dn.2xlarge", "ml.g5.xlarge", "ml.g5.8xlarge", "ml.m5.xlarge", "ml.m5.2xlarge"}
        for name, cfg in pipeline_config.steps.items():
            assert cfg.InstanceType in valid, f"{name} has invalid type: {cfg.InstanceType}"
