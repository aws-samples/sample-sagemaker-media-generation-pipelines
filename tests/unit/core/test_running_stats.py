"""Unit tests for processing_job.common.running_stats."""

import math
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.core


# ---------------------------------------------------------------------------
# _merge_stats unit tests
# ---------------------------------------------------------------------------


class TestMergeStats:
    """Pure-math tests for the batch merge function."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")

    @staticmethod
    def _merge(old_count, old_mean, old_ssd, old_min, old_max, new_values):
        from processing_job.common.running_stats import _merge_stats

        return _merge_stats(old_count, old_mean, old_ssd, old_min, old_max, new_values)

    def test_first_batch_single_value(self):
        result = self._merge(0, 0.0, 0.0, float("inf"), float("-inf"), [0.8])
        assert result["mean"] == pytest.approx(0.8)
        assert result["std"] == pytest.approx(0.0)
        assert result["min"] == pytest.approx(0.8)
        assert result["max"] == pytest.approx(0.8)

    def test_first_batch_multiple_values(self):
        values = [0.6, 0.8, 1.0]
        result = self._merge(0, 0.0, 0.0, float("inf"), float("-inf"), values)
        assert result["mean"] == pytest.approx(0.8)
        assert result["min"] == pytest.approx(0.6)
        assert result["max"] == pytest.approx(1.0)
        expected_std = (sum((v - 0.8) ** 2 for v in values) / 3) ** 0.5
        assert result["std"] == pytest.approx(expected_std)

    def test_merge_two_batches_equals_combined(self):
        """Merging batch A then batch B should equal computing stats on A+B."""
        batch_a = [0.9, 0.7, 0.8]
        batch_b = [0.6, 1.0]
        all_values = batch_a + batch_b

        r1 = self._merge(0, 0.0, 0.0, float("inf"), float("-inf"), batch_a)
        r2 = self._merge(len(batch_a), r1["mean"], r1["sum_sq_dev"], r1["min"], r1["max"], batch_b)

        expected_mean = sum(all_values) / len(all_values)
        expected_std = (sum((v - expected_mean) ** 2 for v in all_values) / len(all_values)) ** 0.5

        assert r2["mean"] == pytest.approx(expected_mean)
        assert r2["std"] == pytest.approx(expected_std)
        assert r2["min"] == pytest.approx(0.6)
        assert r2["max"] == pytest.approx(1.0)

    def test_merge_three_batches(self):
        b1, b2, b3 = [0.5, 0.6], [0.9], [0.7, 0.8, 1.0]
        all_v = b1 + b2 + b3

        r = self._merge(0, 0.0, 0.0, float("inf"), float("-inf"), b1)
        r = self._merge(len(b1), r["mean"], r["sum_sq_dev"], r["min"], r["max"], b2)
        r = self._merge(len(b1) + len(b2), r["mean"], r["sum_sq_dev"], r["min"], r["max"], b3)

        expected_mean = sum(all_v) / len(all_v)
        expected_std = math.sqrt(sum((v - expected_mean) ** 2 for v in all_v) / len(all_v))

        assert r["mean"] == pytest.approx(expected_mean)
        assert r["std"] == pytest.approx(expected_std)
        assert r["min"] == pytest.approx(0.5)
        assert r["max"] == pytest.approx(1.0)

    def test_identical_values_zero_std(self):
        r = self._merge(0, 0.0, 0.0, float("inf"), float("-inf"), [0.5, 0.5, 0.5])
        assert r["std"] == pytest.approx(0.0)

    def test_min_max_tracked_across_merges(self):
        r1 = self._merge(0, 0.0, 0.0, float("inf"), float("-inf"), [0.3, 0.9])
        r2 = self._merge(2, r1["mean"], r1["sum_sq_dev"], r1["min"], r1["max"], [0.5])
        assert r2["min"] == pytest.approx(0.3)
        assert r2["max"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# write_running_stats integration tests (mocked DynamoDB)
# ---------------------------------------------------------------------------


class TestWriteRunningStats:
    """Tests for write_running_stats with mocked DynamoDBOperations."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")

    @staticmethod
    def _make_mock_db(mean_row=None, min_row=None, max_row=None):
        mock = MagicMock()

        def _get_item(id, step):
            if id == "_mean":
                return mean_row or {}
            if id == "_min":
                return min_row or {}
            if id == "_max":
                return max_row or {}
            return {}

        mock.get_item.side_effect = _get_item
        mock.put_item.return_value = True
        return mock

    @staticmethod
    def _get_put_data(mock_db, pk):
        """Extract the data dict from the put_item call for a given partition key."""
        for c in mock_db.put_item.call_args_list:
            if c.kwargs["id"] == pk:
                return c.kwargs["data"]
        raise AssertionError(f"No put_item call with id={pk}")

    def test_first_execution_writes_four_rows(self):
        from processing_job.common.running_stats import write_running_stats

        mock_db = self._make_mock_db()
        accumulators = {"ltx23": {"aesthetic_quality": [0.8, 0.9]}}

        write_running_stats(mock_db, "vbench_t2v", accumulators)

        assert mock_db.put_item.call_count == 4
        pks = [c.kwargs["id"] for c in mock_db.put_item.call_args_list]
        assert sorted(pks) == ["_max", "_mean", "_min", "_std"]

        mean_data = self._get_put_data(mock_db, "_mean")
        assert mean_data["video_count"] == 2
        assert mean_data["aesthetic_quality"] == pytest.approx(0.85)

        min_data = self._get_put_data(mock_db, "_min")
        assert min_data["aesthetic_quality"] == pytest.approx(0.8)

        max_data = self._get_put_data(mock_db, "_max")
        assert max_data["aesthetic_quality"] == pytest.approx(0.9)

    def test_second_execution_merges_with_existing(self):
        from processing_job.common.running_stats import write_running_stats

        old_mean = 0.85
        old_ssd = sum((v - old_mean) ** 2 for v in [0.8, 0.9])
        mean_row = {
            "video_count": 2,
            "aesthetic_quality": old_mean,
            "aesthetic_quality_sum_sq_dev": old_ssd,
        }
        min_row = {"aesthetic_quality": 0.8}
        max_row = {"aesthetic_quality": 0.9}

        mock_db = self._make_mock_db(mean_row, min_row, max_row)
        accumulators = {"ltx23": {"aesthetic_quality": [0.7, 1.0, 0.6]}}

        write_running_stats(mock_db, "vbench_t2v", accumulators)

        all_values = [0.8, 0.9, 0.7, 1.0, 0.6]
        expected_mean = sum(all_values) / 5
        expected_std = math.sqrt(sum((v - expected_mean) ** 2 for v in all_values) / 5)

        mean_data = self._get_put_data(mock_db, "_mean")
        assert mean_data["video_count"] == 5
        assert mean_data["aesthetic_quality"] == pytest.approx(expected_mean)

        std_data = self._get_put_data(mock_db, "_std")
        assert std_data["aesthetic_quality"] == pytest.approx(expected_std)

        min_data = self._get_put_data(mock_db, "_min")
        assert min_data["aesthetic_quality"] == pytest.approx(0.6)

        max_data = self._get_put_data(mock_db, "_max")
        assert max_data["aesthetic_quality"] == pytest.approx(1.0)

    def test_multiple_models_written_separately(self):
        from processing_job.common.running_stats import write_running_stats

        mock_db = self._make_mock_db()
        accumulators = {
            "ltx23": {"motion_smoothness": [0.9]},
            "wan22": {"motion_smoothness": [0.7]},
        }

        write_running_stats(mock_db, "vbench_t2v", accumulators)

        # 4 rows per model = 8 total
        assert mock_db.put_item.call_count == 8
        steps = {c.kwargs["step"] for c in mock_db.put_item.call_args_list}
        assert steps == {"vbench_t2v#ltx23", "vbench_t2v#wan22"}

    def test_multiple_metrics_in_single_row(self):
        from processing_job.common.running_stats import write_running_stats

        mock_db = self._make_mock_db()
        accumulators = {
            "ltx23": {
                "aesthetic_quality": [0.8, 0.9],
                "dynamic_degree": [1.0, 0.0],
            },
        }

        write_running_stats(mock_db, "vbench_t2v", accumulators)

        mean_data = self._get_put_data(mock_db, "_mean")
        assert "aesthetic_quality" in mean_data
        assert "dynamic_degree" in mean_data

        min_data = self._get_put_data(mock_db, "_min")
        assert min_data["dynamic_degree"] == pytest.approx(0.0)

        max_data = self._get_put_data(mock_db, "_max")
        assert max_data["dynamic_degree"] == pytest.approx(1.0)

    def test_corrupted_existing_row_treated_as_fresh(self):
        from processing_job.common.running_stats import write_running_stats

        mean_row = {"video_count": "not_a_number", "aesthetic_quality": "corrupt"}
        mock_db = self._make_mock_db(mean_row)
        accumulators = {"ltx23": {"aesthetic_quality": [0.8]}}

        write_running_stats(mock_db, "vbench_t2v", accumulators)

        mean_data = self._get_put_data(mock_db, "_mean")
        assert mean_data["video_count"] == 1
        assert mean_data["aesthetic_quality"] == pytest.approx(0.8)

    def test_sort_key_format(self):
        from processing_job.common.running_stats import write_running_stats

        mock_db = self._make_mock_db()
        accumulators = {"wan22": {"imaging_quality": [0.75]}}

        write_running_stats(mock_db, "vbench_i2v", accumulators)

        for c in mock_db.put_item.call_args_list:
            assert c.kwargs["step"] == "vbench_i2v#wan22"
