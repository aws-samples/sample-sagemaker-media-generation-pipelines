"""
Unit tests for multi-config CI/CD pipeline stack synthesis.

Validates that the CiCdPipelineStack correctly creates independent pipelines
for each config file, with proper resource isolation and naming. Tests use
the real pipeline config files from config/pipeline/.
"""

import pytest

from config.config import CicdConfig
from infrastructure.cicd_pipeline.stack import CiCdPipelineStack
from tests.unit.cicd.conftest import _create_cicd_pipeline_stack

pytestmark = pytest.mark.cicd


# ── Real multi-config from cicd.yaml ─────────────────────────────────

_ALL_FIVE = CicdConfig(
    pipeline_configs=[
        "config_vrag.yaml",
        "config_i2v.yaml",
        "config_motionart.yaml",
        "config_t2a.yaml",
        "config_t2i.yaml",
    ],
    test_commands={
        "config_vrag.yaml": "uv run pytest tests/unit/ -x --no-header -q",
        "config_i2v.yaml": "uv run pytest tests/unit/ -x --no-header -q",
        "config_motionart.yaml": "uv run pytest tests/unit/ -x --no-header -q",
        "config_t2a.yaml": "uv run pytest tests/unit/ -x --no-header -q",
        "config_t2i.yaml": "uv run pytest tests/unit/ -x --no-header -q",
    },
    shared_prefix="dev",
)


@pytest.fixture(scope="module")
def all_five_template():
    """Synthesize the 5-config CiCdPipelineStack once for the entire module."""
    _, template = _create_cicd_pipeline_stack(_ALL_FIVE)
    return template


class TestMultiConfigPipelineCreation:
    """Tests for creating all 5 pipelines in a single stack."""

    def test_creates_pipelines(self, all_five_template) -> None:
        """5 config pipelines (container pipeline is in its own stack)."""
        all_five_template.resource_count_is("AWS::CodePipeline::Pipeline", 5)

    def test_creates_five_failure_rules(self, all_five_template) -> None:
        """Each pipeline gets its own EventBridge failure rule."""
        all_five_template.resource_count_is("AWS::Events::Rule", 5)

    def test_creates_codebuild_projects(self, all_five_template) -> None:
        """Each config gets deploy + lint + test + model-download + trigger + per-step build projects."""
        # Count varies with number of steps per config; just verify > 30
        projects = all_five_template.find_resources("AWS::CodeBuild::Project")
        assert len(projects) >= 30

    def test_shared_sns_topic(self, all_five_template) -> None:
        """All pipelines share a single SNS notification topic."""
        all_five_template.resource_count_is("AWS::SNS::Topic", 1)

    def test_shared_artifact_bucket(self, all_five_template) -> None:
        """All pipelines share artifact and logging buckets (+ per-config mock source asset buckets)."""
        resources = all_five_template.find_resources("AWS::S3::Bucket")
        # 2 real buckets (artifact + logging) + 5 per-config mock source asset buckets
        assert len(resources) == 7


class TestMultiConfigPipelineNaming:
    """Tests for pipeline naming with real config construct_ids."""

    def test_config_pipelines_have_five_stages(self, all_five_template) -> None:
        """Config pipelines have Source, QualityGate, Deploy, ModelDownloadAndUpload, TriggerPipeline."""
        pipelines = all_five_template.find_resources("AWS::CodePipeline::Pipeline")
        for logical_id, resource in pipelines.items():
            stages = resource["Properties"]["Stages"]
            stage_names = [s["Name"] for s in stages]
            assert stage_names == [
                "Source",
                "QualityGate",
                "Deploy",
                "ModelDownloadAndUpload",
                "TriggerPipeline",
            ], f"Pipeline {logical_id} has wrong stages: {stage_names}"


class TestMultiConfigDeployActions:
    """Tests for deploy action configuration across configs."""

    def test_deploy_action_is_first_in_deploy_stage(self, all_five_template) -> None:
        """Deploy action has run_order=1 in Deploy stage for config pipelines."""
        pipelines = all_five_template.find_resources("AWS::CodePipeline::Pipeline")
        for logical_id, resource in pipelines.items():
            stages = resource["Properties"]["Stages"]
            deploy_stages = [s for s in stages if s["Name"] == "Deploy"]
            deploy_action = next(a for a in deploy_stages[0]["Actions"] if a["Name"] == "Deploy")
            assert deploy_action["RunOrder"] == 1, f"Pipeline {logical_id}"

    def test_config_pipelines_deploy_only_in_deploy_stage(self, all_five_template) -> None:
        """Config pipelines have only Deploy action in Deploy stage (containers built separately)."""
        pipelines = all_five_template.find_resources("AWS::CodePipeline::Pipeline")
        for logical_id, resource in pipelines.items():
            stages = resource["Properties"]["Stages"]
            deploy_stages = [s for s in stages if s["Name"] == "Deploy"]
            actions = deploy_stages[0]["Actions"]
            assert len(actions) == 1, f"Pipeline {logical_id} should have only Deploy action"
            assert actions[0]["Name"] == "Deploy"
            assert actions[0]["RunOrder"] == 1


class TestMultiConfigModelDownload:
    """Tests for model download stage across configs."""

    def test_model_download_parallel_with_upload(self, all_five_template) -> None:
        """TriggerModelDownload and UploadInputData both have run_order=1."""
        pipelines = all_five_template.find_resources("AWS::CodePipeline::Pipeline")
        for logical_id, resource in pipelines.items():
            stages = resource["Properties"]["Stages"]
            md_stages = [s for s in stages if s["Name"] == "ModelDownloadAndUpload"]
            for action in md_stages[0]["Actions"]:
                assert action["RunOrder"] == 1, f"Pipeline {logical_id}, action {action['Name']} should be run_order=1"


class TestMultiConfigTriggerStage:
    """Tests for trigger stage across configs."""

    def test_approval_before_trigger(self, all_five_template) -> None:
        """ApproveExecution (run_order=1) before TriggerSageMakerPipeline (run_order=2)."""
        pipelines = all_five_template.find_resources("AWS::CodePipeline::Pipeline")
        for logical_id, resource in pipelines.items():
            stages = resource["Properties"]["Stages"]
            trigger_stages = [s for s in stages if s["Name"] == "TriggerPipeline"]
            actions = {a["Name"]: a["RunOrder"] for a in trigger_stages[0]["Actions"]}
            assert actions["ApproveExecution"] == 1
            assert actions["TriggerSageMakerPipeline"] == 2


class TestCfgLabelAllConfigs:
    """Tests for _cfg_label with all real config filenames."""

    def test_config_yaml(self) -> None:
        assert CiCdPipelineStack._cfg_label("config_vrag.yaml") == "vrag"

    def test_config_i2v(self) -> None:
        assert CiCdPipelineStack._cfg_label("config_i2v.yaml") == "i2v"

    def test_config_motionart(self) -> None:
        assert CiCdPipelineStack._cfg_label("config_motionart.yaml") == "motionart"

    def test_config_t2a(self) -> None:
        assert CiCdPipelineStack._cfg_label("config_t2a.yaml") == "t2a"

    def test_config_t2i(self) -> None:
        assert CiCdPipelineStack._cfg_label("config_t2i.yaml") == "t2i"


class TestReadConfigPrefix:
    """Tests for _read_config_prefix with all real config files."""

    def test_config_yaml_prefix(self) -> None:
        assert CiCdPipelineStack._read_config_prefix("config_vrag.yaml") == "vr"

    def test_config_i2v_prefix(self) -> None:
        assert CiCdPipelineStack._read_config_prefix("config_i2v.yaml") == "i2v"

    def test_config_motionart_prefix(self) -> None:
        assert CiCdPipelineStack._read_config_prefix("config_motionart.yaml") == "ma"

    def test_config_t2a_prefix(self) -> None:
        assert CiCdPipelineStack._read_config_prefix("config_t2a.yaml") == "t2a"

    def test_config_t2i_prefix(self) -> None:
        assert CiCdPipelineStack._read_config_prefix("config_t2i.yaml") == "t2i"


class TestMultiConfigInputData:
    """Tests for UploadInputData with input_data mapping."""

    def test_with_input_data_mapping(self) -> None:
        """Config with input_data creates upload commands."""
        config = CicdConfig(
            pipeline_configs=["config_vrag.yaml"],
            input_data={"config_vrag.yaml": ["inputs_video.json", "images/"]},
        )
        _, template = _create_cicd_pipeline_stack(config)
        # 1 config pipeline (container pipeline is in its own stack)
        template.resource_count_is("AWS::CodePipeline::Pipeline", 1)

    def test_without_input_data_mapping(self) -> None:
        """Config without input_data still creates pipeline (no-op upload)."""
        config = CicdConfig(
            pipeline_configs=["config_vrag.yaml"],
            input_data={},
        )
        _, template = _create_cicd_pipeline_stack(config)
        template.resource_count_is("AWS::CodePipeline::Pipeline", 1)
