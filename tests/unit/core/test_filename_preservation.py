# Feature: unit-test-reorganization, Property 1: Filename preservation across reorganization
"""
Unit test verifying no test files remain in the flat tests/unit/ directory.

Property 1: For any test file that existed in the original flat tests/unit/
directory and was moved to a subdirectory, the filename (basename) shall be
identical to the original. Additionally, no test_*.py files should remain
in the flat tests/unit/ directory.

**Validates: Requirements 1.9**
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.core

REPO_ROOT = Path(__file__).resolve().parents[3]
UNIT_DIR = REPO_ROOT / "tests" / "unit"

# Files that are allowed to remain in the flat tests/unit/ directory
ALLOWED_ROOT_FILES = {"__init__.py", "conftest.py", "step_names.py", "inputs.json"}

# Expected subdirectories
EXPECTED_SUBDIRS = {"core", "cicd", "retrieval", "processing", "model_validation", "steps", "integration"}


class TestFilenamePreservation:
    """No test files remain in the flat tests/unit/ directory."""

    def test_no_test_files_in_root(self) -> None:
        """
        **Validates: Requirements 1.9**

        No test_*.py files should exist directly in tests/unit/.
        """
        root_test_files = list(UNIT_DIR.glob("test_*.py"))
        assert root_test_files == [], (
            f"Test files found in flat tests/unit/ directory (should be in subdirectories): "
            f"{[f.name for f in root_test_files]}"
        )

    def test_only_allowed_files_in_root(self) -> None:
        """Only infrastructure files remain in tests/unit/ root."""
        root_files = {f.name for f in UNIT_DIR.iterdir() if f.is_file() and not f.name.startswith(".")}
        unexpected = root_files - ALLOWED_ROOT_FILES
        assert not unexpected, f"Unexpected files in tests/unit/ root: {unexpected}"

    def test_all_subdirectories_exist(self) -> None:
        """All expected subdirectories exist."""
        existing_dirs = {d.name for d in UNIT_DIR.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))}
        missing = EXPECTED_SUBDIRS - existing_dirs
        assert not missing, f"Missing subdirectories: {missing}"

    def test_all_subdirectories_have_init(self) -> None:
        """Every subdirectory has an __init__.py."""
        for dirname in EXPECTED_SUBDIRS:
            subdir = UNIT_DIR / dirname
            if subdir.exists():
                assert (subdir / "__init__.py").exists(), f"{dirname}/ missing __init__.py"
