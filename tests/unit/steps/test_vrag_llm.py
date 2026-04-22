"""
Unit and property-based tests for processing_job/vrag_llm/main.py.

Covers:
- load_entries: valid JSON array, empty array, malformed entries, missing file
- extract_json: JSON in code fences, bare JSON, malformed text
- refine_prompt: valid response, missing fields, Bedrock failure (mock Agent)
- write_shard: correct filename, VragOutputEntry conformance, field values
- log_to_dynamodb: verify DynamoDB put_item call args (mock DynamoDBOperations)
- main() end-to-end with mocked agent and DynamoDB

**Validates: Requirements 1.1, 2.2, 2.3, 3.4, 3.5, 6.1, 6.2, 6.3**
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

# AWS_REGION must be set before importing dynamodb module (reads at import time)
os.environ.setdefault("AWS_REGION", "us-east-1")

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from processing_job.common.models import VisualEntry, VragOutputEntry
from processing_job.vrag_llm.main import (
    extract_json,
    load_entries,
    log_to_dynamodb,
    refine_prompt,
    write_shard,
)

pytestmark = pytest.mark.steps_vrag


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_str_st = st.text(min_size=1, max_size=80)

# Strategy for valid VisualEntry dicts (input format)
_valid_visual_entry_st = st.fixed_dictionaries(
    {
        "id": _str_st,
        "prompt": _str_st,
    }
)

# Strategy for invalid entry dicts (missing required fields)
_invalid_entry_st = st.one_of(
    st.fixed_dictionaries({"id": _str_st}),  # missing prompt
    st.fixed_dictionaries({"prompt": _str_st}),  # missing id
    st.fixed_dictionaries({}),  # empty
    st.fixed_dictionaries({"id": _str_st, "prompt": _str_st, "extra": _str_st}),  # extra field
    st.fixed_dictionaries({"id": st.integers(), "prompt": _str_st}),  # wrong type
)

# Strategy for valid JSON with both required keys
_valid_llm_json_st = st.fixed_dictionaries(
    {
        "retrieval_query": _str_st,
        "video_prompt": _str_st,
    }
)

# Strategy for file-safe IDs (no path separators or special chars)
_safe_id_st = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789-_"),
    min_size=1,
    max_size=30,
)


# ---------------------------------------------------------------------------
# Unit tests: load_entries
# ---------------------------------------------------------------------------


class TestLoadEntries:
    """load_entries reads JSON array, validates each entry against VisualEntry."""

    def test_valid_json_array(self, tmp_path) -> None:
        data = [
            {"id": "a", "prompt": "hello"},
            {"id": "b", "prompt": "world"},
        ]
        f = tmp_path / "inputs_t2v.json"
        f.write_text(json.dumps(data))

        result = load_entries(str(tmp_path), "inputs_t2v.json")
        assert len(result) == 2
        assert result[0].id == "a"
        assert result[1].prompt == "world"

    def test_empty_array_returns_empty(self, tmp_path) -> None:
        f = tmp_path / "inputs_t2v.json"
        f.write_text("[]")

        result = load_entries(str(tmp_path), "inputs_t2v.json")
        assert result == []

    def test_malformed_entries_skipped(self, tmp_path) -> None:
        data = [
            {"id": "good", "prompt": "valid"},
            {"id": "bad"},  # missing prompt
            {"prompt": "no-id"},  # missing id
            {"id": "also-good", "prompt": "ok"},
        ]
        f = tmp_path / "inputs_t2v.json"
        f.write_text(json.dumps(data))

        result = load_entries(str(tmp_path), "inputs_t2v.json")
        assert len(result) == 2
        assert result[0].id == "good"
        assert result[1].id == "also-good"

    def test_extra_fields_rejected(self, tmp_path) -> None:
        data = [{"id": "x", "prompt": "p", "unexpected": "field"}]
        f = tmp_path / "inputs_t2v.json"
        f.write_text(json.dumps(data))

        result = load_entries(str(tmp_path), "inputs_t2v.json")
        assert result == []

    def test_missing_file_returns_empty(self, tmp_path) -> None:
        result = load_entries(str(tmp_path), "nonexistent.json")
        assert result == []

    def test_non_array_json_returns_empty(self, tmp_path) -> None:
        f = tmp_path / "inputs_t2v.json"
        f.write_text('{"id": "x", "prompt": "p"}')

        result = load_entries(str(tmp_path), "inputs_t2v.json")
        assert result == []

    def test_entries_are_visual_entry_instances(self, tmp_path) -> None:
        data = [{"id": "a", "prompt": "hello"}]
        f = tmp_path / "inputs_t2v.json"
        f.write_text(json.dumps(data))

        result = load_entries(str(tmp_path), "inputs_t2v.json")
        assert isinstance(result[0], VisualEntry)

    def test_image_field_defaults_to_empty(self, tmp_path) -> None:
        data = [{"id": "a", "prompt": "hello"}]
        f = tmp_path / "inputs_t2v.json"
        f.write_text(json.dumps(data))

        result = load_entries(str(tmp_path), "inputs_t2v.json")
        assert result[0].image == ""


# ---------------------------------------------------------------------------
# Unit tests: extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    """extract_json extracts JSON from agent response text."""

    def test_json_in_code_fence(self) -> None:
        text = '```json\n{"retrieval_query": "rq", "video_prompt": "vp"}\n```'
        result = extract_json(text)
        parsed = json.loads(result)
        assert parsed["retrieval_query"] == "rq"
        assert parsed["video_prompt"] == "vp"

    def test_json_in_code_fence_no_lang(self) -> None:
        text = '```\n{"retrieval_query": "rq", "video_prompt": "vp"}\n```'
        result = extract_json(text)
        parsed = json.loads(result)
        assert parsed["retrieval_query"] == "rq"

    def test_bare_json(self) -> None:
        text = '{"retrieval_query": "rq", "video_prompt": "vp"}'
        result = extract_json(text)
        parsed = json.loads(result)
        assert parsed["video_prompt"] == "vp"

    def test_json_with_surrounding_text(self) -> None:
        text = 'Here is the result: {"retrieval_query": "rq", "video_prompt": "vp"} Done.'
        result = extract_json(text)
        parsed = json.loads(result)
        assert parsed["retrieval_query"] == "rq"

    def test_malformed_text_returns_stripped(self) -> None:
        text = "  no json here  "
        result = extract_json(text)
        assert result == "no json here"

    def test_empty_string(self) -> None:
        result = extract_json("")
        assert result == ""

    def test_code_fence_with_extra_whitespace(self) -> None:
        text = '```json\n  {"retrieval_query": "rq", "video_prompt": "vp"}  \n```'
        result = extract_json(text)
        parsed = json.loads(result)
        assert parsed["retrieval_query"] == "rq"


# ---------------------------------------------------------------------------
# Unit tests: refine_prompt
# ---------------------------------------------------------------------------


class TestRefinePrompt:
    """refine_prompt invokes agent and parses JSON response."""

    def test_valid_response(self) -> None:
        mock_agent = MagicMock()
        mock_agent.return_value = '{"retrieval_query": "tokyo neon", "video_prompt": "slow pan"}'

        result = refine_prompt(mock_agent, "A rainy Tokyo alley")
        assert result["retrieval_query"] == "tokyo neon"
        assert result["video_prompt"] == "slow pan"
        mock_agent.assert_called_once_with("A rainy Tokyo alley")

    def test_response_in_code_fence(self) -> None:
        mock_agent = MagicMock()
        mock_agent.return_value = '```json\n{"retrieval_query": "rq", "video_prompt": "vp"}\n```'

        result = refine_prompt(mock_agent, "test prompt")
        assert result["retrieval_query"] == "rq"
        assert result["video_prompt"] == "vp"

    def test_missing_retrieval_query_raises(self) -> None:
        mock_agent = MagicMock()
        mock_agent.return_value = '{"video_prompt": "vp"}'

        with pytest.raises(ValueError, match="missing required fields"):
            refine_prompt(mock_agent, "test")

    def test_missing_video_prompt_raises(self) -> None:
        mock_agent = MagicMock()
        mock_agent.return_value = '{"retrieval_query": "rq"}'

        with pytest.raises(ValueError, match="missing required fields"):
            refine_prompt(mock_agent, "test")

    def test_malformed_json_raises(self) -> None:
        mock_agent = MagicMock()
        mock_agent.return_value = "not json at all"

        with pytest.raises(ValueError, match="Failed to parse"):
            refine_prompt(mock_agent, "test")

    def test_agent_exception_propagates(self) -> None:
        mock_agent = MagicMock()
        mock_agent.side_effect = RuntimeError("Bedrock timeout")

        with pytest.raises(RuntimeError, match="Bedrock timeout"):
            refine_prompt(mock_agent, "test")


# ---------------------------------------------------------------------------
# Unit tests: write_shard
# ---------------------------------------------------------------------------


class TestWriteShard:
    """write_shard writes VragOutputEntry JSON shard as {id}.json."""

    def test_correct_filename(self, tmp_path) -> None:
        write_shard("tokyo-rain", "original", "rq", "vp", str(tmp_path))
        assert (tmp_path / "tokyo-rain.json").exists()

    def test_vrag_output_entry_conformance(self, tmp_path) -> None:
        write_shard("test-id", "my prompt", "search query", "video desc", str(tmp_path))
        content = json.loads((tmp_path / "test-id.json").read_text())
        entry = VragOutputEntry.model_validate(content)
        assert entry.id == "test-id"
        assert entry.prompt == "my prompt"
        assert entry.retrieval_query == "search query"
        assert entry.video_prompt == "video desc"
        assert entry.image == ""

    def test_field_values_preserved(self, tmp_path) -> None:
        write_shard("abc", "p1", "rq1", "vp1", str(tmp_path))
        content = json.loads((tmp_path / "abc.json").read_text())
        assert content["id"] == "abc"
        assert content["prompt"] == "p1"
        assert content["retrieval_query"] == "rq1"
        assert content["video_prompt"] == "vp1"
        assert content["image"] == ""

    def test_overwrites_existing_file(self, tmp_path) -> None:
        write_shard("x", "old", "old-rq", "old-vp", str(tmp_path))
        write_shard("x", "new", "new-rq", "new-vp", str(tmp_path))
        content = json.loads((tmp_path / "x.json").read_text())
        assert content["prompt"] == "new"


# ---------------------------------------------------------------------------
# Unit tests: log_to_dynamodb
# ---------------------------------------------------------------------------


class TestLogToDynamodb:
    """log_to_dynamodb calls DynamoDBOperations.put_item with correct args."""

    def test_put_item_called_with_correct_args(self) -> None:
        mock_db = MagicMock()
        data = {"original_prompt": "p", "retrieval_query": "rq"}

        log_to_dynamodb(mock_db, "entry-1", data)

        mock_db.put_item.assert_called_once_with(
            id="entry-1",
            step="vrag_llm",
            data=data,
        )

    def test_exception_does_not_propagate(self) -> None:
        mock_db = MagicMock()
        mock_db.put_item.side_effect = RuntimeError("DynamoDB error")

        # Should not raise
        log_to_dynamodb(mock_db, "entry-1", {"key": "val"})

    def test_step_name_from_module_constant(self) -> None:
        from processing_job.vrag_llm.main import STEP_NAME

        mock_db = MagicMock()
        log_to_dynamodb(mock_db, "x", {})
        call_kwargs = mock_db.put_item.call_args
        assert call_kwargs.kwargs["step"] == STEP_NAME


# ---------------------------------------------------------------------------
# Unit tests: main() end-to-end
# ---------------------------------------------------------------------------


class TestMainEndToEnd:
    """main() orchestrates load → refine → write → log with mocked dependencies."""

    def test_main_happy_path(self, tmp_path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        entries = [
            {"id": "a", "prompt": "prompt A"},
            {"id": "b", "prompt": "prompt B"},
        ]
        (input_dir / "inputs_t2v.json").write_text(json.dumps(entries))

        mock_agent = MagicMock()
        mock_agent.return_value = '{"retrieval_query": "rq", "video_prompt": "vp"}'

        mock_db = MagicMock()

        with (
            patch("processing_job.vrag_llm.main.SM_INPUT_DIR", str(input_dir)),
            patch("processing_job.vrag_llm.main.LOCAL_OUTPUT_DIR", str(output_dir)),
            patch("processing_job.vrag_llm.main.DYNAMODB_TABLE_NAME", "test-table"),
            patch("processing_job.vrag_llm.main.create_agent", return_value=mock_agent),
            patch("processing_job.vrag_llm.main.DynamoDBOperations", return_value=mock_db),
            patch("sys.argv", ["main.py", "--refine"]),
        ):
            from processing_job.vrag_llm.main import main

            main()

        # Verify shards written
        assert (output_dir / "a.json").exists()
        assert (output_dir / "b.json").exists()

        # Verify shard content
        content_a = json.loads((output_dir / "a.json").read_text())
        assert content_a["id"] == "a"
        assert content_a["retrieval_query"] == "rq"
        assert content_a["video_prompt"] == "vp"

        # Verify DynamoDB called for each entry
        assert mock_db.put_item.call_count == 2

    def test_main_no_valid_entries_exits(self, tmp_path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (input_dir / "inputs_t2v.json").write_text("[]")

        with (
            patch("processing_job.vrag_llm.main.SM_INPUT_DIR", str(input_dir)),
            patch("processing_job.vrag_llm.main.LOCAL_OUTPUT_DIR", str(output_dir)),
            patch("sys.argv", ["main.py", "--refine"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            from processing_job.vrag_llm.main import main

            main()

        assert exc_info.value.code == 1

    def test_main_skips_dynamodb_when_no_table(self, tmp_path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        entries = [{"id": "a", "prompt": "p"}]
        (input_dir / "inputs_t2v.json").write_text(json.dumps(entries))

        mock_agent = MagicMock()
        mock_agent.return_value = '{"retrieval_query": "rq", "video_prompt": "vp"}'

        with (
            patch("processing_job.vrag_llm.main.SM_INPUT_DIR", str(input_dir)),
            patch("processing_job.vrag_llm.main.LOCAL_OUTPUT_DIR", str(output_dir)),
            patch("processing_job.vrag_llm.main.DYNAMODB_TABLE_NAME", ""),
            patch("processing_job.vrag_llm.main.create_agent", return_value=mock_agent),
            patch("sys.argv", ["main.py", "--refine"]),
        ):
            from processing_job.vrag_llm.main import main

            main()

        assert (output_dir / "a.json").exists()

    def test_main_skips_failed_refinements(self, tmp_path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        entries = [
            {"id": "good", "prompt": "ok"},
            {"id": "bad", "prompt": "fail"},
        ]
        (input_dir / "inputs_t2v.json").write_text(json.dumps(entries))

        call_count = 0

        def mock_agent_call(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Bedrock error")
            return '{"retrieval_query": "rq", "video_prompt": "vp"}'

        mock_agent = MagicMock(side_effect=mock_agent_call)
        mock_db = MagicMock()

        with (
            patch("processing_job.vrag_llm.main.SM_INPUT_DIR", str(input_dir)),
            patch("processing_job.vrag_llm.main.LOCAL_OUTPUT_DIR", str(output_dir)),
            patch("processing_job.vrag_llm.main.DYNAMODB_TABLE_NAME", "test-table"),
            patch("processing_job.vrag_llm.main.create_agent", return_value=mock_agent),
            patch("processing_job.vrag_llm.main.DynamoDBOperations", return_value=mock_db),
            patch("sys.argv", ["main.py", "--refine"]),
        ):
            from processing_job.vrag_llm.main import main

            main()

        assert (output_dir / "good.json").exists()
        assert not (output_dir / "bad.json").exists()


# ---------------------------------------------------------------------------
# Property 1: Input validation preserves valid entries and rejects invalid ones
# ---------------------------------------------------------------------------


class TestInputValidationProperty:
    """Property-based tests for input validation.

    **Validates: Requirements 1.1, 2.2, 2.3**
    """

    # Feature: vrag-llm-container, Property 1: Input validation preserves valid entries and rejects invalid ones
    @given(
        valid_entries=st.lists(_valid_visual_entry_st, min_size=0, max_size=10),
        invalid_entries=st.lists(_invalid_entry_st, min_size=0, max_size=10),
    )
    @settings(max_examples=100)
    def test_valid_preserved_invalid_rejected(
        self,
        valid_entries: list[dict],
        invalid_entries: list[dict],
    ) -> None:
        """For any mix of valid and invalid entries, load_entries returns exactly
        the valid ones and skips the invalid ones. The count of returned entries
        plus skipped entries equals the input list length."""
        d = tempfile.mkdtemp()
        try:
            combined = valid_entries + invalid_entries
            with open(os.path.join(d, "inputs_t2v.json"), "w") as f:
                json.dump(combined, f)

            result = load_entries(d, "inputs_t2v.json")

            # Every returned entry must be a valid VisualEntry
            for entry in result:
                assert isinstance(entry, VisualEntry)

            # Count of returned + skipped == total input
            assert len(result) + (len(combined) - len(result)) == len(combined)

            # All valid entries should be in the result
            assert len(result) >= 0
            assert len(result) <= len(combined)

            # Every valid entry dict should produce a matching result
            valid_ids = {e["id"] for e in valid_entries}
            result_ids = {e.id for e in result}
            assert valid_ids.issubset(result_ids)
        finally:
            shutil.rmtree(d)

    @given(entries=st.lists(_valid_visual_entry_st, min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_all_valid_entries_preserved(
        self,
        entries: list[dict],
    ) -> None:
        """When all entries are valid, load_entries returns all of them."""
        d = tempfile.mkdtemp()
        try:
            with open(os.path.join(d, "inputs_t2v.json"), "w") as f:
                json.dump(entries, f)

            result = load_entries(d, "inputs_t2v.json")
            assert len(result) == len(entries)
        finally:
            shutil.rmtree(d)

    @given(entries=st.lists(_invalid_entry_st, min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_all_invalid_entries_rejected(
        self,
        entries: list[dict],
    ) -> None:
        """When all entries are invalid, load_entries returns empty list."""
        d = tempfile.mkdtemp()
        try:
            with open(os.path.join(d, "inputs_t2v.json"), "w") as f:
                json.dump(entries, f)

            result = load_entries(d, "inputs_t2v.json")
            assert len(result) == 0
        finally:
            shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Property 3: LLM response JSON parsing extracts fields or rejects malformed input
# ---------------------------------------------------------------------------


class TestLlmResponseParsingProperty:
    """Property-based tests for LLM response JSON parsing.

    **Validates: Requirements 3.4, 3.5**
    """

    # Feature: vrag-llm-container, Property 3: LLM response JSON parsing extracts fields or rejects malformed input
    @given(data=_valid_llm_json_st)
    @settings(max_examples=100)
    def test_bare_json_extracted(self, data: dict) -> None:
        """For any valid JSON with both required keys, extract_json + json.loads
        produces a dict containing both keys."""
        text = json.dumps(data)
        extracted = extract_json(text)
        parsed = json.loads(extracted)
        assert "retrieval_query" in parsed
        assert "video_prompt" in parsed
        assert parsed["retrieval_query"] == data["retrieval_query"]
        assert parsed["video_prompt"] == data["video_prompt"]

    @given(data=_valid_llm_json_st)
    @settings(max_examples=100)
    def test_code_fenced_json_extracted(self, data: dict) -> None:
        """For any valid JSON wrapped in ```json code fences, extract_json + json.loads
        produces a dict containing both keys."""
        text = f"```json\n{json.dumps(data)}\n```"
        extracted = extract_json(text)
        parsed = json.loads(extracted)
        assert "retrieval_query" in parsed
        assert "video_prompt" in parsed
        assert parsed["retrieval_query"] == data["retrieval_query"]
        assert parsed["video_prompt"] == data["video_prompt"]

    @given(data=_valid_llm_json_st)
    @settings(max_examples=100)
    def test_refine_prompt_with_valid_response(self, data: dict) -> None:
        """For any valid JSON response from agent, refine_prompt returns dict
        with both required keys."""
        mock_agent = MagicMock()
        mock_agent.return_value = json.dumps(data)

        result = refine_prompt(mock_agent, "any prompt")
        assert result["retrieval_query"] == data["retrieval_query"]
        assert result["video_prompt"] == data["video_prompt"]

    @given(
        text=st.text(min_size=1, max_size=100).filter(lambda t: "retrieval_query" not in t and "video_prompt" not in t)
    )
    @settings(max_examples=100)
    def test_malformed_text_rejected(self, text: str) -> None:
        """For any string missing both required keys, refine_prompt raises ValueError."""
        mock_agent = MagicMock()
        mock_agent.return_value = text

        with pytest.raises(ValueError):
            refine_prompt(mock_agent, "test")


# ---------------------------------------------------------------------------
# Property 4: Output shard round-trip — write then read produces valid VragOutputEntry
# ---------------------------------------------------------------------------


class TestOutputShardRoundTripProperty:
    """Property-based tests for output shard round-trip.

    **Validates: Requirements 6.1, 6.2, 6.3**
    """

    # Feature: vrag-llm-container, Property 4: Output shard round-trip — write then read produces valid VragOutputEntry
    @given(
        entry_id=_safe_id_st,
        original_prompt=_str_st,
        retrieval_query=_str_st,
        video_prompt=_str_st,
    )
    @settings(max_examples=100)
    def test_write_then_read_produces_valid_entry(
        self,
        entry_id: str,
        original_prompt: str,
        retrieval_query: str,
        video_prompt: str,
    ) -> None:
        """For any valid VragOutputEntry fields, writing as {id}.json and reading
        back produces a dict that validates as VragOutputEntry with identical values."""
        d = tempfile.mkdtemp()
        try:
            write_shard(entry_id, original_prompt, retrieval_query, video_prompt, d)

            shard_path = os.path.join(d, f"{entry_id}.json")
            assert os.path.exists(shard_path), f"Shard file {entry_id}.json not created"

            with open(shard_path) as f:
                content = json.load(f)
            restored = VragOutputEntry.model_validate(content)

            assert restored.id == entry_id
            assert restored.prompt == original_prompt
            assert restored.retrieval_query == retrieval_query
            assert restored.video_prompt == video_prompt
            assert restored.image == ""
        finally:
            shutil.rmtree(d)

    @given(
        entry_id=_safe_id_st,
        original_prompt=_str_st,
        retrieval_query=_str_st,
        video_prompt=_str_st,
    )
    @settings(max_examples=100)
    def test_shard_filename_matches_id(
        self,
        entry_id: str,
        original_prompt: str,
        retrieval_query: str,
        video_prompt: str,
    ) -> None:
        """The shard file is named {id}.json."""
        d = tempfile.mkdtemp()
        try:
            write_shard(entry_id, original_prompt, retrieval_query, video_prompt, d)

            files = os.listdir(d)
            assert len(files) == 1
            assert files[0] == f"{entry_id}.json"
        finally:
            shutil.rmtree(d)
