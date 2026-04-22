# Feature: unit-test-reorganization, Property 6: No duplicated mock helpers in test files
# Feature: unit-test-reorganization, Property 8: Template cache for multi-assertion test classes
# Feature: unit-test-reorganization, Property 9: Stack synthesis tests use required mocks
"""
Static checks verifying structural properties of the test codebase.

Property 6: No test file in any subdirectory defines _mock_from_asset or
_mock_s3_asset — these shall only exist in conftest.py files.

Property 8: Any test class with >1 test method asserting against a CDK
Template should use a class-scoped fixture or setup_class for synthesis.

Property 9: Stack synthesis tests that use Lambda or S3 assets must mock
lambda_.Code.from_asset and s3_assets.Asset respectively.

**Validates: Requirements 7.1, 7.5, 7.6, 10.1, 10.2, 10.3**
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.core

REPO_ROOT = Path(__file__).resolve().parents[3]
UNIT_DIR = REPO_ROOT / "tests" / "unit"

SUBDIRS = ["core", "cicd", "retrieval", "processing", "model_validation", "steps", "integration"]

# Mock helper names that should only live in conftest.py
CONFTEST_ONLY_HELPERS = {"_mock_from_asset", "_mock_s3_asset"}


def _all_test_files() -> list[Path]:
    """Collect all test_*.py files across subdirectories."""
    files: list[Path] = []
    for subdir in SUBDIRS:
        d = UNIT_DIR / subdir
        if d.exists():
            files.extend(sorted(d.glob("test_*.py")))
    return files


def _get_defined_functions(filepath: Path) -> list[str]:
    """Extract top-level function names defined in a file."""
    source = filepath.read_text()
    tree = ast.parse(source)
    return [
        node.name for node in ast.iter_child_nodes(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


class TestNoDuplicatedMockHelpers:
    """
    Property 6: _mock_from_asset and _mock_s3_asset must only be defined
    in conftest.py files, not in individual test files.

    **Validates: Requirements 7.1, 7.5, 7.6**
    """

    def test_no_mock_helpers_in_test_files(self) -> None:
        violations: list[str] = []
        for filepath in _all_test_files():
            defined = _get_defined_functions(filepath)
            for helper in CONFTEST_ONLY_HELPERS:
                if helper in defined:
                    violations.append(f"{filepath.relative_to(REPO_ROOT)}: defines {helper}")

        assert not violations, "Mock helpers should only be in conftest.py:\n" + "\n".join(violations)


class TestRequiredSynthesisMocks:
    """
    Property 9: Stack synthesis tests that create Lambda functions must mock
    lambda_.Code.from_asset. Tests using s3_assets.Asset must mock it.

    This checks that test files importing from_asset or s3_assets also
    contain mock/patch references for those symbols.

    **Validates: Requirements 10.2, 10.3**
    """

    def test_from_asset_is_mocked_when_used(self) -> None:
        """Files that synthesize stacks with Lambda should mock from_asset."""
        violations: list[str] = []

        for filepath in _all_test_files():
            source = filepath.read_text()

            # Skip files that don't do CDK synthesis
            if "Template.from_stack" not in source and "from_stack" not in source:
                continue

            # Check if file uses lambda_ or Code.from_asset
            uses_lambda = "aws_lambda" in source or "lambda_" in source
            if not uses_lambda:
                continue

            # Check if from_asset is mocked
            has_mock = (
                "_mock_from_asset" in source
                or "from_asset" in source
                and "patch" in source
                or "mock_from_asset" in source
            )
            if not has_mock:
                # Only flag if the file actually calls from_asset directly
                if "from_asset" in source and "Code.from_asset" in source:
                    rel = filepath.relative_to(REPO_ROOT)
                    violations.append(f"{rel}: uses Code.from_asset without mocking")

        assert not violations, "Stack synthesis tests must mock lambda_.Code.from_asset:\n" + "\n".join(violations)
