"""Tests that validate Dockerfile and buildspec correctness.

Catches issues like:
- uv PATH not matching HOME (the bug that broke all 3 CodeBuild jobs)
- Missing COPY for required files (common/requirements.txt, etc.)
- Buildspec not copying files that Dockerfile expects
- Entrypoint scripts referenced in config_vrag.yaml actually exist
- ENV consistency across Dockerfiles
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSING_DIR = REPO_ROOT / "processing_job"
CONFIG_PATH = REPO_ROOT / "config" / "pipeline" / "config_vrag.yaml"

from tests.unit.conftest import (
    COMFY_STEPS as STEPS_WITH_COMFY,
)
from tests.unit.conftest import (
    DOCKERFILE_STEPS as ALL_STEPS,
)
from tests.unit.conftest import (
    VIDEOGEN_COMFY_STEPS as STEPS_WITH_VIDEOGEN,
)

pytestmark = pytest.mark.processing


def read_dockerfile(step: str) -> str:
    return (PROCESSING_DIR / step / "Dockerfile").read_text()


def read_buildspec(step: str) -> str:
    """Read the root buildspec.yml (used by all CodeBuild projects)."""
    return (PROCESSING_DIR / "buildspec.yml").read_text()


def parse_env_from_dockerfile(content: str) -> dict[str, str]:
    """Extract ENV key=value pairs from a Dockerfile."""
    envs = {}
    for m in re.finditer(r"ENV\s+(.+?)(?=\n(?:RUN|COPY|WORKDIR|FROM|ARG|ENTRYPOINT|CMD|#|\Z))", content, re.DOTALL):
        block = m.group(1)
        # Handle multi-line ENV with backslash continuations
        block = block.replace("\\\n", " ")
        for pair in re.finditer(r'(\w+)=("[^"]*"|[^\s\\]+)', block):
            key, val = pair.group(1), pair.group(2).strip('"')
            envs[key] = val
    return envs


def parse_env_path(content: str) -> list[str]:
    """Extract all ENV PATH=... declarations."""
    return re.findall(r'ENV\s+PATH="([^"]+)"', content)


@pytest.fixture
def config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# uv PATH consistency — the exact bug we hit
# ---------------------------------------------------------------------------
class TestUvPathConsistency:
    """uv installs to $HOME/.local/bin. PATH must include that, not /root/.local/bin
    when HOME is overridden."""

    # Steps that install uv via curl (CUDA-based containers)
    UV_CURL_STEPS = [s for s in ALL_STEPS if "curl -LsSf https://astral.sh/uv/install.sh" in read_dockerfile(s)]

    @pytest.mark.parametrize("step", UV_CURL_STEPS)
    def test_uv_path_matches_home(self, step):
        content = read_dockerfile(step)
        envs = parse_env_from_dockerfile(content)
        paths = parse_env_path(content)

        home = envs.get("HOME", "/root")
        home = home.rstrip("/")
        expected_bin = f"{home}/.local/bin"

        # At least one PATH declaration must contain the correct bin dir
        all_paths = ":".join(paths)
        assert expected_bin in all_paths or "/root/.local/bin" in all_paths and home == "/root", (
            f"{step}: uv installs to $HOME/.local/bin ({expected_bin}) but PATH is set to {all_paths}"
        )

    # Steps that explicitly set PATH in their Dockerfile
    PATH_STEPS = [s for s in ALL_STEPS if any(re.match(r"ENV\s+PATH=", l) for l in read_dockerfile(s).splitlines())]

    @pytest.mark.parametrize("step", PATH_STEPS)
    def test_uv_used_before_path_is_set(self, step):
        """uv must not be invoked before PATH is updated."""
        content = read_dockerfile(step)
        lines = content.splitlines()

        path_set_line = None
        for i, line in enumerate(lines):
            if re.match(r"ENV\s+PATH=", line):
                path_set_line = i
                break

        # Check no RUN line before path_set_line uses uv
        for i, line in enumerate(lines[:path_set_line]):
            if line.strip().startswith("RUN") and "uv " in line:
                pytest.fail(f"{step}: line {i + 1} uses 'uv' before PATH is set on line {path_set_line + 1}")


# ---------------------------------------------------------------------------
# Dockerfile structure
# ---------------------------------------------------------------------------
class TestDockerfileStructure:
    """Basic structural checks for all Dockerfiles."""

    @pytest.mark.parametrize("step", ALL_STEPS)
    def test_dockerfile_exists(self, step):
        assert (PROCESSING_DIR / step / "Dockerfile").exists()

    def test_root_buildspec_exists(self):
        assert (PROCESSING_DIR / "buildspec.yml").exists()

    @pytest.mark.parametrize("step", ALL_STEPS)
    def test_has_from_instruction(self, step):
        content = read_dockerfile(step)
        assert re.search(r"^FROM\s", content, re.MULTILINE)

    @pytest.mark.parametrize("step", ALL_STEPS)
    def test_apt_lists_cleaned(self, step):
        """apt-get update should be followed by rm -rf /var/lib/apt/lists/*"""
        content = read_dockerfile(step)
        if "apt-get update" in content:
            assert "rm -rf /var/lib/apt/lists/*" in content, f"{step}: apt-get update without cleanup"

    @pytest.mark.parametrize("step", ALL_STEPS)
    def test_copies_common_requirements(self, step):
        """Every Dockerfile must COPY common/requirements.txt."""
        content = read_dockerfile(step)
        assert "common/requirements.txt" in content, f"{step}: missing COPY for common/requirements.txt"

    @pytest.mark.parametrize("step", ALL_STEPS)
    def test_installs_common_requirements(self, step):
        """Every Dockerfile must install common requirements."""
        content = read_dockerfile(step)
        assert "common-req" in content or "common-requirements" in content, (
            f"{step}: not installing common requirements"
        )


# ---------------------------------------------------------------------------
# Buildspec copies what Dockerfile expects
# ---------------------------------------------------------------------------
class TestBuildspecDockerfileConsistency:
    """Buildspec must copy all files that the Dockerfile COPYs."""

    @pytest.mark.parametrize("step", [s for s in ALL_STEPS if s != "agent"])
    def test_buildspec_copies_common(self, step):
        """Every buildspec must cp -r common/ into the step dir."""
        bs = read_buildspec(step)
        assert "cp -r common/" in bs or "cp -r common " in bs, f"{step}: buildspec doesn't copy common/ directory"

    @pytest.mark.parametrize("step", STEPS_WITH_VIDEOGEN)
    def test_dockerfile_references_videogen_requirements(self, step):
        df = read_dockerfile(step)
        assert "videogen-requirements" in df or "videogen-req" in df, (
            f"{step} Dockerfile must install videogen requirements from common/"
        )


# ---------------------------------------------------------------------------
# Config entrypoints match actual files
# ---------------------------------------------------------------------------
class TestConfigEntrypointsExist:
    """Container entrypoint scripts referenced in config_vrag.yaml must exist."""

    def test_entrypoint_scripts_exist(self, config):
        for step_name, step_cfg in config["steps"].items():
            # Steps with ecr_image override share another step's directory
            dir_name = step_cfg.get("ecr_image") or step_name
            entrypoint = step_cfg.get("ContainerEntrypoint", [])
            # Find the script in the entrypoint (last arg that looks like a file)
            for arg in entrypoint:
                if arg.startswith("./") or arg.endswith(".py") or arg.endswith(".sh"):
                    script = arg.lstrip("./")
                    path = PROCESSING_DIR / dir_name / script
                    assert path.exists(), f"Step '{step_name}' entrypoint references '{arg}' but {path} doesn't exist"

    def test_config_step_dirs_exist(self, config):
        for step_name, step_cfg in config["steps"].items():
            dir_name = step_cfg.get("ecr_image") or step_name
            assert (PROCESSING_DIR / dir_name).is_dir(), (
                f"Step '{step_name}' (dir '{dir_name}') directory doesn't exist"
            )


# ---------------------------------------------------------------------------
# Input path consistency — scripts must use /opt/ml/processing/input/input/
# ---------------------------------------------------------------------------
class TestInputPathConsistency:
    """Scripts must use the correct SageMaker input path (input/input/)."""

    @pytest.mark.parametrize("script", ["ltx.py", "wan22.py"])
    def test_load_inputs_uses_correct_path(self, script):
        """load_inputs default must point to /opt/ml/processing/input/input."""
        path = PROCESSING_DIR / "common" / script
        assert path.exists(), f"common/{script} must exist"
        content = path.read_text()
        if "load_inputs" in content and "INPUTS_JSON" in content:
            assert "/opt/ml/processing/input/input" in content, (
                f"common/{script}: load_inputs default path must use "
                f"/opt/ml/processing/input/input (SageMaker channel 'input')"
            )

    @pytest.mark.parametrize("step", STEPS_WITH_VIDEOGEN)
    def test_main_has_link_inputs_function(self, step):
        """main.py must link SageMaker inputs into ComfyUI's input dir."""
        content = (PROCESSING_DIR / step / "main.py").read_text()
        assert "link_inputs_to_comfyui" in content, f"{step}/main.py must call link_inputs_to_comfyui"

    @pytest.mark.parametrize("step", STEPS_WITH_COMFY)
    def test_sm_input_dir_constant(self, step):
        """main.py must define SM_INPUT_DIR = /opt/ml/processing/input/input."""
        content = (PROCESSING_DIR / step / "main.py").read_text()
        assert "/opt/ml/processing/input/input" in content, f"{step}/main.py must define SM_INPUT_DIR with correct path"


# ---------------------------------------------------------------------------
# ComfyUI-specific checks (t2v, i2v)
# ---------------------------------------------------------------------------
class TestComfyDockerfiles:
    """Checks specific to containers that use ComfyUI."""

    @pytest.mark.parametrize("step", STEPS_WITH_COMFY)
    def test_has_comfy_home_env(self, step):
        envs = parse_env_from_dockerfile(read_dockerfile(step))
        assert "COMFY_HOME" in envs

    @pytest.mark.parametrize("step", STEPS_WITH_VIDEOGEN)
    def test_models_from_s3(self, step):
        """Models are downloaded from S3 as a SageMaker ProcessingInput."""
        content = read_dockerfile(step)
        assert "S3" in content or "s3" in content.lower() or "Models" in content, (
            f"{step}: missing comment about S3 models input"
        )

    @pytest.mark.parametrize("step", STEPS_WITH_COMFY)
    def test_installs_comfy_cli(self, step):
        content = read_dockerfile(step)
        assert "comfy-cli" in content

    @pytest.mark.parametrize("step", STEPS_WITH_COMFY)
    def test_clones_comfyscript(self, step):
        content = read_dockerfile(step)
        assert "ComfyScript" in content


# ---------------------------------------------------------------------------
# Run scripts reference common.log_outputs
# ---------------------------------------------------------------------------
class TestRunScripts:
    """Run scripts must log to DynamoDB via common.log_outputs."""

    @pytest.mark.parametrize("step", STEPS_WITH_COMFY)
    def test_run_job_py_exists(self, step):
        assert (PROCESSING_DIR / step / "main.py").exists(), f"{step}/main.py doesn't exist"

    @pytest.mark.parametrize("step", STEPS_WITH_COMFY)
    def test_logs_to_dynamodb(self, step):
        script = (PROCESSING_DIR / step / "main.py").read_text()
        assert "common.log_outputs" in script, f"{step}/main.py doesn't call common.log_outputs"

    @pytest.mark.parametrize("step", STEPS_WITH_COMFY)
    def test_copies_output_to_sagemaker_dir(self, step):
        script = (PROCESSING_DIR / step / "main.py").read_text()
        assert "LOCAL_OUTPUT_DIR" in script or "/opt/ml/processing/output" in script

    @pytest.mark.parametrize("step", STEPS_WITH_COMFY)
    def test_lists_input_directory(self, step):
        script = (PROCESSING_DIR / step / "main.py").read_text()
        assert "/opt/ml/processing/input" in script, f"{step}/main.py doesn't list input directory contents"

    @pytest.mark.parametrize("step", STEPS_WITH_COMFY)
    def test_has_main_guard(self, step):
        script = (PROCESSING_DIR / step / "main.py").read_text()
        assert 'if __name__ == "__main__"' in script, f"{step}/main.py missing __main__ guard"


# ---------------------------------------------------------------------------
# Runtime module availability — every local import must resolve at build time
# ---------------------------------------------------------------------------
class TestRuntimeModulesExist:
    """Every module imported at runtime in main.py must exist in the Docker
    build context (the step directory after buildspec copies common/)."""

    @pytest.mark.parametrize("step", STEPS_WITH_COMFY)
    def test_is_queue_empty_exists_in_common(self, step):
        """is_queue_empty.py must be present in common/ (shared by all ComfyUI steps)."""
        path = PROCESSING_DIR / "common" / "is_queue_empty.py"
        assert path.exists(), "common/is_queue_empty.py is missing — will cause ModuleNotFoundError at runtime"

    @pytest.mark.parametrize("step", STEPS_WITH_COMFY)
    def test_is_queue_empty_has_get_queue_size(self, step):
        """common/is_queue_empty.py must export get_queue_size."""
        content = (PROCESSING_DIR / "common" / "is_queue_empty.py").read_text()
        assert "def get_queue_size" in content, (
            "common/is_queue_empty.py missing get_queue_size() — main.py imports this function"
        )

    @pytest.mark.parametrize("step", STEPS_WITH_COMFY)
    def test_local_imports_resolvable(self, step):
        """All non-stdlib, non-third-party imports in main.py must resolve to
        a file in the step directory or common/.

        At build time, buildspec copies processing_job/common/ into the step
        directory, so we check both the step dir and the source common/ dir.
        """
        main_py = (PROCESSING_DIR / step / "main.py").read_text()
        # Match "from X import" or "import X" at statement level
        import_re = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE)
        # Modules that are third-party or stdlib — skip these
        skip = {
            "argparse",
            "json",
            "os",
            "shutil",
            "subprocess",
            "sys",
            "time",
            "pathlib",
            "typing",
            "re",
            "glob",
            "io",
            "math",
            "copy",
            "urllib",
            "loguru",
            "requests",
            "boto3",
            "botocore",
            "yaml",
            "PIL",
            "pydantic",
            "comfy_script",
            "comfy_script.runtime",
            "comfy_script.runtime.nodes",
            "processing_job",  # fallback import path used in try/except blocks
            "is_queue_empty",  # fallback import; resolves via common/ copied into step dir at build time
        }
        missing = []
        for m in import_re.finditer(main_py):
            mod = m.group(1) or m.group(2)
            if mod in skip or mod.split(".")[0] in skip:
                continue
            top = mod.split(".")[0]
            # Check step dir, and also processing_job/ root (buildspec copies
            # common/ into step dir at build time)
            candidates = [
                PROCESSING_DIR / step / f"{top}.py",
                PROCESSING_DIR / step / top / "__init__.py",
                PROCESSING_DIR / f"{top}.py",
                PROCESSING_DIR / top,  # directory (e.g. common/)
            ]
            parts = mod.split(".")
            if len(parts) > 1:
                # e.g. common.ltx -> processing_job/common/ltx.py
                candidates.append(PROCESSING_DIR / "/".join(parts[:-1]) / f"{parts[-1]}.py")
                # Also check inside step dir
                candidates.append(PROCESSING_DIR / step / "/".join(parts[:-1]) / f"{parts[-1]}.py")
            if not any(c.exists() for c in candidates):
                missing.append(mod)
        assert not missing, (
            f"{step}/main.py imports modules not found in build context: "
            f"{missing}. These will cause ModuleNotFoundError at runtime."
        )
