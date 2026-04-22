# Feature: unit-test-reorganization, Property 2: Marker-directory consistency
"""
Property test verifying every test file has the correct pytestmark for its directory.

Property 2: For any test file in any Test_Category subdirectory, the file
shall contain a module-level pytestmark assignment that matches the expected
marker for that subdirectory.

**Validates: Requirements 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9**
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.core

REPO_ROOT = Path(__file__).resolve().parents[3]
UNIT_DIR = REPO_ROOT / "tests" / "unit"

# Mapping from directory name to expected marker(s)
DIRECTORY_MARKER_MAP: dict[str, str | list[str]] = {
    "core": "core",
    "cicd": "cicd",
    "retrieval": "retrieval",
    "processing": "processing",
    "model_validation": "model_validation",
    "integration": "integration",
    # steps/ uses per-step markers — checked separately
}

# steps/ files map to specific step markers
STEPS_MARKER_PREFIX = "steps_"


def _extract_pytestmark(filepath: Path) -> list[str]:
    """Extract pytestmark marker names from a test file using AST parsing."""
    source = filepath.read_text()
    tree = ast.parse(source)
    markers: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "pytestmark":
                    # pytestmark = pytest.mark.core
                    if isinstance(node.value, ast.Attribute):
                        markers.append(node.value.attr)
                    # pytestmark = [pytest.mark.core, pytest.mark.cicd]
                    elif isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Attribute):
                                markers.append(elt.attr)
    return markers


def _get_test_files(subdir: Path) -> list[Path]:
    """Get all test_*.py files in a subdirectory (non-recursive)."""
    return sorted(subdir.glob("test_*.py"))


class TestMarkerDirectoryConsistency:
    """Every test file has the correct pytestmark for its directory."""

    @pytest.mark.parametrize(
        "dirname,expected_marker",
        list(DIRECTORY_MARKER_MAP.items()),
    )
    def test_directory_marker_matches(self, dirname: str, expected_marker: str) -> None:
        """
        **Validates: Requirements 2.2-2.9**
        """
        subdir = UNIT_DIR / dirname
        if not subdir.exists():
            pytest.skip(f"{dirname}/ does not exist")

        test_files = _get_test_files(subdir)
        assert test_files, f"No test files found in {dirname}/"

        for filepath in test_files:
            markers = _extract_pytestmark(filepath)
            assert expected_marker in markers, (
                f"{filepath.name} in {dirname}/ missing pytestmark = pytest.mark.{expected_marker}; "
                f"found markers: {markers}"
            )

    def test_steps_directory_uses_step_markers(self) -> None:
        """Files in steps/ must use a steps_* marker."""
        steps_dir = UNIT_DIR / "steps"
        if not steps_dir.exists():
            pytest.skip("steps/ does not exist")

        test_files = _get_test_files(steps_dir)
        assert test_files, "No test files found in steps/"

        for filepath in test_files:
            markers = _extract_pytestmark(filepath)
            has_step_marker = any(m.startswith(STEPS_MARKER_PREFIX) for m in markers)
            assert has_step_marker, f"{filepath.name} in steps/ must have a steps_* marker; found markers: {markers}"
