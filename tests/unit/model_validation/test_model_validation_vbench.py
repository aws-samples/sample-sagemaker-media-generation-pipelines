"""Unit tests for vbench container model validation logic.

Tests VBenchMetrics validation, strict mode behavior, and exclude_none serialization.
"""

import pytest
from pydantic import ValidationError

from processing_job.common.models import VBenchMetrics

pytestmark = pytest.mark.model_validation


ALL_VBENCH_FIELDS = {
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
    "temporal_style",
    "overall_consistency",
    "human_action",
}


class TestVBenchMetricsValidation:
    """VBenchMetrics validates parsed metric dicts."""

    def test_all_fields_float(self) -> None:
        data = {f: 0.85 for f in ALL_VBENCH_FIELDS}
        m = VBenchMetrics.model_validate(data)
        for field in ALL_VBENCH_FIELDS:
            assert getattr(m, field) == 0.85

    def test_partial_fields_valid(self) -> None:
        m = VBenchMetrics.model_validate({"subject_consistency": 0.9})
        assert m.subject_consistency == 0.9
        assert m.background_consistency is None

    def test_empty_dict_valid(self) -> None:
        m = VBenchMetrics.model_validate({})
        for field in ALL_VBENCH_FIELDS:
            assert getattr(m, field) is None

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VBenchMetrics.model_validate({"unknown_metric": 0.5})

    def test_string_value_rejected_strict(self) -> None:
        with pytest.raises(ValidationError):
            VBenchMetrics.model_validate({"subject_consistency": "high"})

    def test_int_coerced_to_float(self) -> None:
        """Pydantic v2 strict mode coerces int→float for float|None fields."""
        m = VBenchMetrics.model_validate({"dynamic_degree": 1})
        assert m.dynamic_degree == 1.0
        assert isinstance(m.dynamic_degree, float)

    def test_explicit_float_conversion_also_works(self) -> None:
        """VBench container converts int→float before validation (defensive)."""
        data = {"dynamic_degree": float(1)}
        m = VBenchMetrics.model_validate(data)
        assert m.dynamic_degree == 1.0
        assert isinstance(m.dynamic_degree, float)

    def test_invalid_entries_filtered(self) -> None:
        """Simulate vbench loop: valid metrics pass, invalid are skipped."""
        metrics_list = [
            {"subject_consistency": 0.9, "motion_smoothness": 0.8},
            {"unknown_field": 0.5},  # extra field → rejected
            {"aesthetic_quality": 0.7},
        ]
        validated = []
        for metrics in metrics_list:
            try:
                validated.append(VBenchMetrics.model_validate(metrics))
            except ValidationError:
                continue
        assert len(validated) == 2


class TestVBenchExcludeNoneSerialization:
    """model_dump(exclude_none=True) omits None fields."""

    def test_exclude_none_omits_unset_fields(self) -> None:
        m = VBenchMetrics(subject_consistency=0.9, motion_smoothness=0.8)
        d = m.model_dump(exclude_none=True)
        assert set(d.keys()) == {"subject_consistency", "motion_smoothness"}

    def test_exclude_none_all_set(self) -> None:
        data = {f: 0.5 for f in ALL_VBENCH_FIELDS}
        m = VBenchMetrics.model_validate(data)
        d = m.model_dump(exclude_none=True)
        assert set(d.keys()) == ALL_VBENCH_FIELDS

    def test_exclude_none_empty(self) -> None:
        m = VBenchMetrics()
        d = m.model_dump(exclude_none=True)
        assert d == {}

    def test_all_values_are_float(self) -> None:
        m = VBenchMetrics(aesthetic_quality=0.75, imaging_quality=0.6)
        d = m.model_dump(exclude_none=True)
        for v in d.values():
            assert isinstance(v, float)

    def test_without_exclude_none_includes_all(self) -> None:
        """Without exclude_none, all 10 fields appear (even as None)."""
        m = VBenchMetrics(subject_consistency=0.9)
        d = m.model_dump()
        assert len(d) == 10
        assert d["subject_consistency"] == 0.9
        assert d["background_consistency"] is None
