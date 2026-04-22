# Feature: unit-test-reorganization, Property 7: No config file reads during test execution
"""
Unit test verifying step_names.py does not read any config files.

Property 7: For any test module import or test function execution, no file
under config/pipeline/, config/cicd/, or config/retrieval/ shall be opened.

**Validates: Requirements 8.1, 8.2, 8.6**
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.core

REPO_ROOT = Path(__file__).resolve().parents[3]
STEP_NAMES_PATH = REPO_ROOT / "tests" / "unit" / "step_names.py"


class TestStepNamesNoConfigReads:
    """step_names.py must not import yaml or open config files."""

    def test_no_yaml_import(self) -> None:
        """step_names.py should not import yaml."""
        source = STEP_NAMES_PATH.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "yaml", "step_names.py must not import yaml"
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "yaml", "step_names.py must not import from yaml"

    def test_no_config_path_strings(self) -> None:
        """step_names.py should not reference config/ directory paths."""
        source = STEP_NAMES_PATH.read_text()
        forbidden = ["config/pipeline/", "config/cicd/", "config/retrieval/"]
        for pattern in forbidden:
            assert pattern not in source, f"step_names.py must not reference '{pattern}'"

    def test_no_open_calls_on_config(self) -> None:
        """step_names.py should not call open() on any config file."""
        source = STEP_NAMES_PATH.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Check for open(...) calls
                if isinstance(func, ast.Name) and func.id == "open":
                    # Check if any string arg references config/
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            assert "config/" not in arg.value, f"step_names.py must not open config files: {arg.value}"

    def test_no_yaml_load_calls(self) -> None:
        """step_names.py should not call yaml.safe_load or yaml.full_load."""
        source = STEP_NAMES_PATH.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in (
                    "safe_load",
                    "full_load",
                    "load",
                ):
                    if isinstance(func.value, ast.Name) and func.value.id == "yaml":
                        pytest.fail("step_names.py must not call yaml load functions")
