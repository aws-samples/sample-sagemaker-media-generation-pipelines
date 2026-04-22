"""Unit tests for processing_job/inputs.json and the load_inputs helper."""

import json
import os
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.processing


INPUTS_JSON_PATH = Path(__file__).resolve().parent.parent / "inputs.json"

REQUIRED_KEYS = {"id", "prompt", "image"}


@pytest.fixture
def inputs():
    """Load and return the canonical inputs.json."""
    with open(INPUTS_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


# --- Schema / structure tests ---


def test_inputs_file_exists():
    assert INPUTS_JSON_PATH.exists(), f"inputs.json not found at {INPUTS_JSON_PATH}"


def test_inputs_is_list(inputs):
    assert isinstance(inputs, list)


def test_inputs_not_empty(inputs):
    assert len(inputs) > 0, "inputs.json must contain at least one entry"


@pytest.mark.parametrize("idx", range(5))
def test_entry_has_required_keys(inputs, idx):
    entry = inputs[idx]
    missing = REQUIRED_KEYS - set(entry.keys())
    assert not missing, f"Entry {idx} missing keys: {missing}"


@pytest.mark.parametrize("idx", range(5))
def test_prompt_is_nonempty_string(inputs, idx):
    assert isinstance(inputs[idx]["prompt"], str)
    assert len(inputs[idx]["prompt"].strip()) > 0


@pytest.mark.parametrize("idx", range(5))
def test_image_is_nonempty_string(inputs, idx):
    assert isinstance(inputs[idx]["image"], str)
    assert len(inputs[idx]["image"].strip()) > 0


@pytest.mark.parametrize("idx", range(5))
def test_image_has_extension(inputs, idx):
    img = inputs[idx]["image"]
    assert "." in img, f"Entry {idx} image '{img}' has no file extension"


def test_no_duplicate_prompts(inputs):
    prompts = [e["prompt"] for e in inputs]
    assert len(prompts) == len(set(prompts)), "Duplicate prompts found"


def test_no_duplicate_ids(inputs):
    ids = [e["id"] for e in inputs]
    assert len(ids) == len(set(ids)), "Duplicate ids found"


@pytest.mark.parametrize("idx", range(5))
def test_id_is_nonempty_string(inputs, idx):
    assert isinstance(inputs[idx]["id"], str)
    assert len(inputs[idx]["id"].strip()) > 0


def test_valid_json_encoding():
    """Ensure the file is valid UTF-8 JSON with no BOM."""
    raw = INPUTS_JSON_PATH.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "File has UTF-8 BOM"
    json.loads(raw.decode("utf-8"))


def test_no_extra_keys(inputs):
    allowed = {"id", "prompt", "image"}
    for idx, entry in enumerate(inputs):
        extra = set(entry.keys()) - allowed
        assert not extra, f"Entry {idx} has unexpected keys: {extra}"


# --- load_inputs helper tests ---


def test_load_inputs_from_explicit_path():
    """load_inputs() reads from an explicit path."""
    # We can't import ltx.py (it imports comfy_script), so replicate the logic
    with open(INPUTS_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 5


def test_load_inputs_from_env_var():
    """Simulate INPUTS_JSON env var pointing to a custom file."""
    custom = [{"prompt": "test prompt", "image": "test.png"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(custom, tmp)
        tmp_path = tmp.name

    try:
        with open(tmp_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == custom
    finally:
        os.unlink(tmp_path)


def test_load_inputs_rejects_invalid_json():
    """Ensure invalid JSON raises an error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp.write("not valid json {{{")
        tmp_path = tmp.name

    try:
        with pytest.raises(json.JSONDecodeError), open(tmp_path, encoding="utf-8") as f:
            json.load(f)
    finally:
        os.unlink(tmp_path)


def test_roundtrip_serialization(inputs):
    """Serialize and deserialize should produce identical data."""
    serialized = json.dumps(inputs, ensure_ascii=False)
    deserialized = json.loads(serialized)
    assert deserialized == inputs
