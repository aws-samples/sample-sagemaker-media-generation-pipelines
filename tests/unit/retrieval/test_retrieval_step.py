"""
Unit and property-based tests for retrieval step modifications.

Covers:
- read_input_shards: VragOutputEntry shards, legacy format, empty dir
- write_visual_entry_shard: correct filename, VisualEntry conformance, field mapping
- id-to-image-URI mapping preservation
- backward compatibility with old prompt-only input format
- no image binary files in output directory

**Validates: Requirements 6b.1, 6b.3, 6b.4**
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

os.environ.setdefault("AWS_REGION", "us-east-1")

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from processing_job.common.models import VisualEntry
from processing_job.retrieval.main import read_input_shards, read_prompts, write_visual_entry_shard

pytestmark = pytest.mark.retrieval


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_str_st = st.text(min_size=1, max_size=80)

# File-safe IDs (no path separators or special chars)
_safe_id_st = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789-_"),
    min_size=1,
    max_size=30,
)

# S3 URI strategy
_s3_uri_st = st.builds(
    lambda bucket, key: f"s3://{bucket}/{key}",
    bucket=st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789"),
        min_size=3,
        max_size=20,
    ),
    key=st.builds(
        lambda segments, ext: "/".join(segments) + ext,
        segments=st.lists(
            st.text(
                alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_-"),
                min_size=1,
                max_size=15,
            ),
            min_size=1,
            max_size=3,
        ),
        ext=st.sampled_from([".jpg", ".jpeg", ".png", ".webp"]),
    ),
)

# Valid VragOutputEntry dict strategy
_vrag_output_entry_st = st.fixed_dictionaries(
    {
        "id": _safe_id_st,
        "prompt": _str_st,
        "image": st.just(""),
        "retrieval_query": _str_st,
        "video_prompt": _str_st,
    }
)

# Image binary extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Unit tests: read_input_shards
# ---------------------------------------------------------------------------


class TestReadInputShards:
    """read_input_shards reads JSON shard files from input directory."""

    def test_reads_vrag_output_entry_shards(self, tmp_path) -> None:
        shard = {
            "id": "tokyo-rain",
            "prompt": "original prompt",
            "image": "",
            "retrieval_query": "tokyo neon rain alley",
            "video_prompt": "Slow tracking shot through rainy Tokyo alley",
        }
        (tmp_path / "tokyo-rain.json").write_text(json.dumps(shard))

        result = read_input_shards(str(tmp_path))
        assert len(result) == 1
        assert result[0]["id"] == "tokyo-rain"
        assert result[0]["retrieval_query"] == "tokyo neon rain alley"
        assert result[0]["video_prompt"] == "Slow tracking shot through rainy Tokyo alley"

    def test_reads_multiple_shards_sorted(self, tmp_path) -> None:
        for name in ["b-entry", "a-entry", "c-entry"]:
            shard = {
                "id": name,
                "prompt": f"p-{name}",
                "image": "",
                "retrieval_query": f"rq-{name}",
                "video_prompt": f"vp-{name}",
            }
            (tmp_path / f"{name}.json").write_text(json.dumps(shard))

        result = read_input_shards(str(tmp_path))
        assert len(result) == 3
        # Sorted by filename
        assert result[0]["id"] == "a-entry"
        assert result[1]["id"] == "b-entry"
        assert result[2]["id"] == "c-entry"

    def test_legacy_format_dict(self, tmp_path) -> None:
        """Legacy format: a single dict with prompt field (no retrieval_query/video_prompt)."""
        shard = {"id": "legacy-1", "prompt": "a simple prompt", "image": ""}
        (tmp_path / "legacy-1.json").write_text(json.dumps(shard))

        result = read_input_shards(str(tmp_path))
        assert len(result) == 1
        assert result[0]["id"] == "legacy-1"
        assert result[0]["prompt"] == "a simple prompt"

    def test_empty_directory(self, tmp_path) -> None:
        result = read_input_shards(str(tmp_path))
        assert result == []

    def test_nonexistent_directory(self, tmp_path) -> None:
        result = read_input_shards(str(tmp_path / "nonexistent"))
        assert result == []

    def test_skips_non_json_files(self, tmp_path) -> None:
        (tmp_path / "readme.txt").write_text("not a shard")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        shard = {"id": "valid", "prompt": "p", "image": "", "retrieval_query": "rq", "video_prompt": "vp"}
        (tmp_path / "valid.json").write_text(json.dumps(shard))

        result = read_input_shards(str(tmp_path))
        assert len(result) == 1
        assert result[0]["id"] == "valid"

    def test_skips_malformed_json(self, tmp_path) -> None:
        (tmp_path / "bad.json").write_text("{invalid json")
        shard = {"id": "good", "prompt": "p", "image": "", "retrieval_query": "rq", "video_prompt": "vp"}
        (tmp_path / "good.json").write_text(json.dumps(shard))

        result = read_input_shards(str(tmp_path))
        assert len(result) == 1
        assert result[0]["id"] == "good"

    def test_skips_non_dict_json(self, tmp_path) -> None:
        (tmp_path / "array.json").write_text("[1, 2, 3]")
        shard = {"id": "ok", "prompt": "p", "image": "", "retrieval_query": "rq", "video_prompt": "vp"}
        (tmp_path / "ok.json").write_text(json.dumps(shard))

        result = read_input_shards(str(tmp_path))
        assert len(result) == 1
        assert result[0]["id"] == "ok"


# ---------------------------------------------------------------------------
# Unit tests: write_visual_entry_shard
# ---------------------------------------------------------------------------


class TestWriteVisualEntryShard:
    """write_visual_entry_shard writes VisualEntry JSON shard as {id}.json."""

    def test_correct_filename(self, tmp_path) -> None:
        write_visual_entry_shard("tokyo-rain", "slow pan", "s3://bucket/img.jpg", str(tmp_path))
        assert (tmp_path / "tokyo-rain.json").exists()

    def test_visual_entry_conformance(self, tmp_path) -> None:
        write_visual_entry_shard("test-id", "my video prompt", "s3://b/k.jpg", str(tmp_path))
        content = json.loads((tmp_path / "test-id.json").read_text())
        entry = VisualEntry.model_validate(content)
        assert entry.id == "test-id"
        assert entry.prompt == "my video prompt"
        assert entry.image == "s3://b/k.jpg"

    def test_field_mapping_video_prompt_to_prompt(self, tmp_path) -> None:
        """video_prompt from upstream becomes prompt in VisualEntry."""
        write_visual_entry_shard("x", "the video prompt text", "s3://b/k.png", str(tmp_path))
        content = json.loads((tmp_path / "x.json").read_text())
        assert content["prompt"] == "the video prompt text"

    def test_field_mapping_s3_uri_to_image(self, tmp_path) -> None:
        """S3 URI becomes image field in VisualEntry."""
        uri = "s3://my-bucket/images/city_skyline_004.jpg"
        write_visual_entry_shard("y", "prompt", uri, str(tmp_path))
        content = json.loads((tmp_path / "y.json").read_text())
        assert content["image"] == uri

    def test_overwrites_existing_file(self, tmp_path) -> None:
        write_visual_entry_shard("z", "old", "s3://b/old.jpg", str(tmp_path))
        write_visual_entry_shard("z", "new", "s3://b/new.jpg", str(tmp_path))
        content = json.loads((tmp_path / "z.json").read_text())
        assert content["prompt"] == "new"
        assert content["image"] == "s3://b/new.jpg"


# ---------------------------------------------------------------------------
# Unit tests: id-to-image-URI mapping preservation
# ---------------------------------------------------------------------------


class TestIdToImageUriMapping:
    """Verify that the id-to-image-URI mapping is preserved through the shard flow."""

    def test_id_preserved_in_output_shard(self, tmp_path) -> None:
        write_visual_entry_shard("my-entry-id", "prompt", "s3://b/img.jpg", str(tmp_path))
        content = json.loads((tmp_path / "my-entry-id.json").read_text())
        assert content["id"] == "my-entry-id"
        assert content["image"] == "s3://b/img.jpg"

    def test_multiple_entries_preserve_mapping(self, tmp_path) -> None:
        mappings = {
            "entry-1": "s3://bucket/images/img1.jpg",
            "entry-2": "s3://bucket/images/img2.png",
            "entry-3": "s3://bucket/images/img3.webp",
        }
        for entry_id, uri in mappings.items():
            write_visual_entry_shard(entry_id, f"prompt-{entry_id}", uri, str(tmp_path))

        for entry_id, uri in mappings.items():
            content = json.loads((tmp_path / f"{entry_id}.json").read_text())
            assert content["id"] == entry_id
            assert content["image"] == uri


# ---------------------------------------------------------------------------
# Unit tests: backward compatibility with old prompt-only input format
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """read_prompts still works with legacy formats for backward compatibility."""

    def test_read_prompts_from_json_with_prompt_field(self, tmp_path) -> None:
        data = [{"prompt": "hello world"}]
        (tmp_path / "input.json").write_text(json.dumps(data))

        result = read_prompts(str(tmp_path))
        assert result == ["hello world"]

    def test_read_prompts_from_json_with_retrieval_query(self, tmp_path) -> None:
        data = [{"retrieval_query": "tokyo neon", "prompt": "fallback"}]
        (tmp_path / "input.json").write_text(json.dumps(data))

        result = read_prompts(str(tmp_path))
        # retrieval_query takes precedence
        assert result == ["tokyo neon"]

    def test_read_prompts_from_txt_file(self, tmp_path) -> None:
        (tmp_path / "prompt.txt").write_text("a text prompt")

        result = read_prompts(str(tmp_path))
        assert result == ["a text prompt"]

    def test_read_prompts_empty_dir(self, tmp_path) -> None:
        result = read_prompts(str(tmp_path))
        assert result == []

    def test_read_input_shards_with_legacy_prompt_only(self, tmp_path) -> None:
        """read_input_shards handles legacy dicts that only have prompt (no retrieval_query)."""
        shard = {"id": "legacy", "prompt": "simple prompt"}
        (tmp_path / "legacy.json").write_text(json.dumps(shard))

        result = read_input_shards(str(tmp_path))
        assert len(result) == 1
        assert result[0]["prompt"] == "simple prompt"
        assert "retrieval_query" not in result[0]


# ---------------------------------------------------------------------------
# Unit tests: no image binary files in output directory
# ---------------------------------------------------------------------------


class TestNoImageBinaryFiles:
    """Verify that write_visual_entry_shard only produces JSON, no image binaries."""

    def test_output_contains_only_json(self, tmp_path) -> None:
        write_visual_entry_shard("a", "p", "s3://b/img.jpg", str(tmp_path))
        write_visual_entry_shard("b", "p", "s3://b/img.png", str(tmp_path))

        files = os.listdir(str(tmp_path))
        for f in files:
            assert f.endswith(".json"), f"Non-JSON file found: {f}"

    def test_no_image_extensions_in_output(self, tmp_path) -> None:
        write_visual_entry_shard("x", "p", "s3://b/photo.jpg", str(tmp_path))

        files = os.listdir(str(tmp_path))
        for f in files:
            _, ext = os.path.splitext(f)
            assert ext not in _IMAGE_EXTENSIONS, f"Image file found: {f}"


# ---------------------------------------------------------------------------
# Property 5: read_input_shards preserves all structured fields from VragOutputEntry shards
# ---------------------------------------------------------------------------


class TestReadInputShardsProperty:
    """Property-based tests for read_input_shards.

    **Validates: Requirements 6b.1**
    """

    # Feature: vrag-llm-container, Property 5: read_input_shards preserves all structured fields from VragOutputEntry shards
    @given(entries=st.lists(_vrag_output_entry_st, min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_preserves_all_fields(self, entries: list[dict]) -> None:
        """For any list of valid VragOutputEntry dicts written as individual
        {id}.json shards, read_input_shards returns dicts preserving id,
        retrieval_query, video_prompt, prompt, and image fields."""
        d = tempfile.mkdtemp()
        try:
            # Deduplicate by id to avoid file overwrites
            seen_ids: set[str] = set()
            unique_entries: list[dict] = []
            for entry in entries:
                if entry["id"] not in seen_ids:
                    seen_ids.add(entry["id"])
                    unique_entries.append(entry)

            # Write each entry as {id}.json
            for entry in unique_entries:
                path = os.path.join(d, f"{entry['id']}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(entry, f)

            result = read_input_shards(d)
            assert len(result) == len(unique_entries)

            # Build lookup by id
            result_by_id = {r["id"]: r for r in result}

            for entry in unique_entries:
                r = result_by_id[entry["id"]]
                assert r["id"] == entry["id"]
                assert r["prompt"] == entry["prompt"]
                assert r["image"] == entry["image"]
                assert r["retrieval_query"] == entry["retrieval_query"]
                assert r["video_prompt"] == entry["video_prompt"]
        finally:
            shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Property 6: Retrieval field mapping — video_prompt becomes prompt, S3 URI becomes image
# ---------------------------------------------------------------------------


class TestRetrievalFieldMappingProperty:
    """Property-based tests for retrieval field mapping.

    **Validates: Requirements 6b.3**
    """

    # Feature: vrag-llm-container, Property 6: Retrieval field mapping — video_prompt becomes prompt, S3 URI becomes image
    @given(
        entry_id=_safe_id_st,
        video_prompt=_str_st,
        s3_uri=_s3_uri_st,
    )
    @settings(max_examples=100)
    def test_video_prompt_becomes_prompt_s3_uri_becomes_image(
        self,
        entry_id: str,
        video_prompt: str,
        s3_uri: str,
    ) -> None:
        """For any entry with a video_prompt and S3 URI, write_visual_entry_shard
        produces a VisualEntry JSON shard where prompt equals video_prompt and
        image equals the S3 URI."""
        d = tempfile.mkdtemp()
        try:
            write_visual_entry_shard(entry_id, video_prompt, s3_uri, d)

            shard_path = os.path.join(d, f"{entry_id}.json")
            assert os.path.exists(shard_path)

            with open(shard_path) as f:
                content = json.load(f)

            # Validate as VisualEntry
            entry = VisualEntry.model_validate(content)

            # Field mapping assertions
            assert entry.prompt == video_prompt
            assert entry.image == s3_uri
            assert entry.id == entry_id
        finally:
            shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Property 7: Retrieval output contains only JSON shards, no image binary files
# ---------------------------------------------------------------------------


class TestRetrievalOutputNoImagesProperty:
    """Property-based tests for retrieval output file types.

    **Validates: Requirements 6b.4**
    """

    # Feature: vrag-llm-container, Property 7: Retrieval output contains only JSON shards, no image binary files
    @given(
        entries=st.lists(
            st.tuples(_safe_id_st, _str_st, _s3_uri_st),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100)
    def test_output_only_json_no_image_binaries(
        self,
        entries: list[tuple[str, str, str]],
    ) -> None:
        """For any set of retrieval results, the output directory contains only
        .json files and no image files (.jpg, .jpeg, .png, .webp, .bmp, .tiff)."""
        d = tempfile.mkdtemp()
        try:
            # Deduplicate by id
            seen_ids: set[str] = set()
            for entry_id, video_prompt, s3_uri in entries:
                if entry_id not in seen_ids:
                    seen_ids.add(entry_id)
                    write_visual_entry_shard(entry_id, video_prompt, s3_uri, d)

            files = os.listdir(d)
            assert len(files) > 0, "Output directory should not be empty"

            for filename in files:
                assert filename.endswith(".json"), f"Non-JSON file found: {filename}"
                _, ext = os.path.splitext(filename)
                assert ext not in _IMAGE_EXTENSIONS, f"Image binary found: {filename}"
        finally:
            shutil.rmtree(d)
