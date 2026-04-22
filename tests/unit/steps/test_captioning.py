"""Tests for the captioning processing job."""

from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.steps_captioning


REPO_ROOT = Path(__file__).resolve().parents[3]
CAPTIONING_DIR = REPO_ROOT / "processing_job" / "captioning"


class TestPromptFile:
    """Verify prompt.txt exists and is loaded correctly."""

    def test_prompt_file_exists(self) -> None:
        assert (CAPTIONING_DIR / "prompt.txt").is_file()

    def test_prompt_file_not_empty(self) -> None:
        text = (CAPTIONING_DIR / "prompt.txt").read_text().strip()
        assert len(text) > 50, "prompt.txt should contain a meaningful prompt"

    def test_prompt_file_no_leading_trailing_whitespace(self) -> None:
        raw = (CAPTIONING_DIR / "prompt.txt").read_text()
        # Allow a single trailing newline (POSIX standard, enforced by end-of-file-fixer)
        assert raw == raw.strip() + "\n" or raw == raw.strip(), (
            "prompt.txt should not have leading/trailing whitespace (except optional trailing newline)"
        )


class TestScanImages:
    """Test the scan_images function."""

    def test_finds_supported_extensions(self, tmp_path: Path) -> None:
        from processing_job.captioning.main import scan_images

        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
            (tmp_path / f"img{ext}").write_bytes(b"\x00")
        result = scan_images(str(tmp_path))
        assert len(result) == 6

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        from processing_job.captioning.main import scan_images

        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "photo.png").write_bytes(b"\x89PNG")
        result = scan_images(str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "photo.png"

    def test_recursive_scan(self, tmp_path: Path) -> None:
        from processing_job.captioning.main import scan_images

        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "a.jpg").write_bytes(b"\x00")
        (sub / "b.png").write_bytes(b"\x00")
        result = scan_images(str(tmp_path))
        assert len(result) == 2

    def test_empty_directory(self, tmp_path: Path) -> None:
        from processing_job.captioning.main import scan_images

        result = scan_images(str(tmp_path))
        assert result == []

    def test_sorted_output(self, tmp_path: Path) -> None:
        from processing_job.captioning.main import scan_images

        for name in ("c.png", "a.png", "b.png"):
            (tmp_path / name).write_bytes(b"\x00")
        result = scan_images(str(tmp_path))
        names = [p.name for p in result]
        assert names == ["a.png", "b.png", "c.png"]


class TestFindModelPath:
    """Test the find_model_path function."""

    def test_finds_model_subdir(self, tmp_path: Path) -> None:
        from processing_job.captioning.main import find_model_path

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model = models_dir / "qwen2_5_vl_7b"
        model.mkdir()
        with (
            patch("processing_job.captioning.main.MODELS_DIR", str(tmp_path)),
            patch("processing_job.captioning.main.CAPTION_MODEL_NAME", "qwen2_5_vl_7b"),
        ):
            assert find_model_path() == str(model)

    def test_finds_config_json_fallback(self, tmp_path: Path) -> None:
        from processing_job.captioning.main import find_model_path

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "config.json").write_text("{}")
        with patch("processing_job.captioning.main.MODELS_DIR", str(tmp_path)):
            assert find_model_path() == str(models_dir)

    def test_raises_when_no_model(self, tmp_path: Path) -> None:
        from processing_job.captioning.main import find_model_path

        with (
            patch("processing_job.captioning.main.MODELS_DIR", str(tmp_path)),
            patch("processing_job.captioning.main.CAPTION_MODEL_NAME", "nonexistent"),
        ):
            with pytest.raises(FileNotFoundError, match="Could not find model"):
                find_model_path()


class TestCaptionPromptLoading:
    """Test that CAPTION_PROMPT is loaded from prompt.txt or env var."""

    def test_loads_from_prompt_file(self) -> None:
        # Re-import to get the module-level value
        from processing_job.captioning.main import CAPTION_PROMPT

        assert "motion" in CAPTION_PROMPT.lower() or "describe" in CAPTION_PROMPT.lower()

    def test_env_var_overrides_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CAPTION_PROMPT", "custom prompt from env")
        # Need to reload the module to pick up the env var
        import importlib

        import processing_job.captioning.main as mod

        importlib.reload(mod)
        assert mod.CAPTION_PROMPT == "custom prompt from env"
        # Restore
        monkeypatch.delenv("CAPTION_PROMPT", raising=False)
        importlib.reload(mod)


class TestImageExtensions:
    """Verify IMAGE_EXTENSIONS constant."""

    def test_contains_common_formats(self) -> None:
        from processing_job.captioning.main import IMAGE_EXTENSIONS

        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            assert ext in IMAGE_EXTENSIONS


class TestDockerfileAndBuildspec:
    """Verify captioning container build files."""

    def test_dockerfile_exists(self) -> None:
        assert (CAPTIONING_DIR / "Dockerfile").is_file()

    def test_buildspec_exists(self) -> None:
        assert (CAPTIONING_DIR / "buildspec.yml").is_file()

    def test_requirements_exists(self) -> None:
        assert (CAPTIONING_DIR / "requirements.txt").is_file()

    def test_dockerfile_copies_prompt(self) -> None:
        content = (CAPTIONING_DIR / "Dockerfile").read_text()
        # The COPY . . instruction copies everything including prompt.txt
        assert "COPY . ." in content

    def test_requirements_has_transformers(self) -> None:
        reqs = (CAPTIONING_DIR / "requirements.txt").read_text()
        assert "transformers" in reqs

    def test_requirements_has_torch(self) -> None:
        reqs = (CAPTIONING_DIR / "requirements.txt").read_text()
        assert "torch" in reqs

    def test_requirements_has_qwen_vl_utils(self) -> None:
        """qwen-vl-utils is required for Qwen2.5-VL image preprocessing."""
        reqs = (CAPTIONING_DIR / "requirements.txt").read_text()
        assert "qwen-vl-utils" in reqs


class TestGenerateCaptionSignature:
    """Verify generate_caption uses qwen_vl_utils and apply_chat_template."""

    def test_uses_qwen_vl_utils(self) -> None:
        source = (CAPTIONING_DIR / "main.py").read_text()
        assert "qwen_vl_utils" in source

    def test_uses_apply_chat_template(self) -> None:
        source = (CAPTIONING_DIR / "main.py").read_text()
        assert "apply_chat_template" in source


class TestCaptioningEnvironmentConfig:
    """Verify CAPTION_PROMPT is loaded from prompt.txt (not env var)."""

    def test_caption_prompt_loaded_from_file(self) -> None:
        from processing_job.captioning.main import CAPTION_PROMPT

        assert len(CAPTION_PROMPT) > 50
        assert "motion" in CAPTION_PROMPT.lower() or "describe" in CAPTION_PROMPT.lower()


class TestCaptioningOutputFormat:
    """Verify captioning output JSON contains fields needed by downstream flf2v."""

    def test_output_json_has_prompt_and_image_keys(self, tmp_path: Path) -> None:
        """Each JSON sidecar must include 'prompt' and 'image' for flf2v consumption."""
        source = (CAPTIONING_DIR / "main.py").read_text()
        # The VisualEntry model must set both fields so i2v's load_inputs() can read them
        assert "prompt=" in source or '"prompt"' in source or "'prompt'" in source
        assert "image=" in source or '"image"' in source or "'image'" in source

    def test_prompt_field_equals_caption(self) -> None:
        """The 'prompt' field should be set to the generated caption text."""
        source = (CAPTIONING_DIR / "main.py").read_text()
        # After captioning, prompt is assigned from caption (via VisualEntry or dict)
        assert "prompt=caption" in source or 'entry["prompt"] = caption' in source

    def test_image_field_is_filename(self) -> None:
        """The 'image' field should be the original image filename (not full path)."""
        source = (CAPTIONING_DIR / "main.py").read_text()
        assert "img_path.name" in source

    def test_images_copied_to_output_dir(self) -> None:
        """Original images must be copied to output so i2v can access them."""
        source = (CAPTIONING_DIR / "main.py").read_text()
        assert "shutil.copy2" in source
        assert "import shutil" in source

    def test_sidecar_metadata_written(self) -> None:
        """A _caption_metadata.json sidecar must be written for log_outputs."""
        source = (CAPTIONING_DIR / "main.py").read_text()
        assert "_caption_metadata.json" in source
