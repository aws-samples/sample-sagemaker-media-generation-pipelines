"""Test constants — config-derived values are hardcoded, filesystem-derived values scan the repo."""

from pathlib import Path

# === Hardcoded test-only constants (never derived from real config) ===
STEP_NAMES: list[str] = ["step_a", "step_b", "vbench_step_a"]
PRIMARY_STEPS: list[str] = ["step_a", "step_b"]
STEP_0 = "step_a"
STEP_1 = "step_b"
STEP_0_DASHED = "step-a"
STEP_1_DASHED = "step-b"
VBENCH_STEPS: list[str] = ["vbench_step_a"]

# === Filesystem-derived constants (scan processing_job/ for Dockerfiles) ===
REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESSING_DIR = REPO_ROOT / "processing_job"
_ALL_STEP_DIRS: list[str] = sorted(
    d.name for d in _PROCESSING_DIR.iterdir() if d.is_dir() and (d / "Dockerfile").exists()
)

# Steps that use ComfyUI — detected by checking for comfy-cli in Dockerfile
COMFY_STEPS: list[str] = [s for s in _ALL_STEP_DIRS if "comfy-cli" in (_PROCESSING_DIR / s / "Dockerfile").read_text()]

# Video-generation ComfyUI steps (use videogen-requirements, link_inputs_to_comfyui)
VIDEOGEN_COMFY_STEPS: list[str] = [
    s
    for s in COMFY_STEPS
    if "videogen-req" in (_PROCESSING_DIR / s / "Dockerfile").read_text()
    or "videogen-requirements" in (_PROCESSING_DIR / s / "Dockerfile").read_text()
]

# Steps used for Dockerfile/buildspec checks (all dirs with a Dockerfile,
# excluding model_download which is a utility, not a processing step)
DOCKERFILE_STEPS: list[str] = [s for s in _ALL_STEP_DIRS if s != "model_download"]
