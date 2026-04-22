"""Running statistics for DynamoDB aggregate rows.

Maintains per-metric running mean, std, min, max across pipeline executions
using only summary statistics (no individual values stored).

Std uses Welford's parallel/batch algorithm:
    combined_mean = (n_a * mean_a + n_b * mean_b) / (n_a + n_b)
    combined_S = S_a + S_b + n_a * n_b / (n_a + n_b) * (mean_a - mean_b)^2
    std = sqrt(S / n)
where S = sum of squared deviations from mean (n * variance).

DynamoDB rows written per model (sort_key = "{vbench_step}#{model}"):

    id="_mean"  — running mean per metric + video_count + {metric}_sum_sq_dev (internal)
    id="_std"   — running population standard deviation per metric
    id="_min"   — running minimum per metric
    id="_max"   — running maximum per metric

Attribute names on _std/_min/_max rows match the original per-video metric
names (e.g. aesthetic_quality, motion_smoothness) for easy comparison.
"""

import math

from loguru import logger

try:
    from common.dynamodb import DynamoDBOperations
except ImportError:
    from processing_job.common.dynamodb import DynamoDBOperations


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return default


def _merge_stats(
    old_count: int,
    old_mean: float,
    old_sum_sq_dev: float,
    old_min: float,
    old_max: float,
    new_values: list[float],
) -> dict[str, float]:
    """Merge existing summary stats with a new batch of values.

    Returns dict with keys: mean, std, min, max, sum_sq_dev.
    """
    new_count = len(new_values)
    new_mean = sum(new_values) / new_count
    new_sum_sq_dev = sum((v - new_mean) ** 2 for v in new_values)
    new_min = min(new_values)
    new_max = max(new_values)

    if old_count == 0:
        total = new_count
        combined_mean = new_mean
        combined_ssd = new_sum_sq_dev
        combined_min = new_min
        combined_max = new_max
    else:
        total = old_count + new_count
        delta = new_mean - old_mean
        combined_mean = (old_count * old_mean + new_count * new_mean) / total
        combined_ssd = old_sum_sq_dev + new_sum_sq_dev + (old_count * new_count / total) * delta**2
        combined_min = min(old_min, new_min)
        combined_max = max(old_max, new_max)

    std = math.sqrt(combined_ssd / total) if total > 0 else 0.0

    return {
        "mean": combined_mean,
        "std": std,
        "min": combined_min,
        "max": combined_max,
        "sum_sq_dev": combined_ssd,
    }


def write_running_stats(
    db_ops: DynamoDBOperations,
    step_name: str,
    model_accumulators: dict[str, dict[str, list[float]]],
) -> None:
    """Write running statistics as separate partition-key rows per stat type.

    For each model writes four rows sharing the same sort key:
        _mean  — metric means + video_count + internal sum_sq_dev fields
        _std   — metric standard deviations
        _min   — metric minimums
        _max   — metric maximums
    """
    for model, metric_lists in model_accumulators.items():
        new_count = len(next(iter(metric_lists.values())))
        sort_key = f"{step_name}#{model}"

        # Read existing _mean row for weighted merge (carries count + sum_sq_dev)
        existing = db_ops.get_item(id="_mean", step=sort_key)
        old_count = int(_safe_float(existing.get("video_count", 0)))

        existing_min = db_ops.get_item(id="_min", step=sort_key)
        existing_max = db_ops.get_item(id="_max", step=sort_key)

        total_count = old_count + new_count
        mean_row: dict[str, object] = {}
        std_row: dict[str, object] = {}
        min_row: dict[str, object] = {}
        max_row: dict[str, object] = {}

        for metric_name, values in metric_lists.items():
            old_mean = _safe_float(existing.get(metric_name))
            old_ssd = _safe_float(existing.get(f"{metric_name}_sum_sq_dev"))
            old_min = _safe_float(existing_min.get(metric_name), default=float("inf"))
            old_max = _safe_float(existing_max.get(metric_name), default=float("-inf"))

            stats = _merge_stats(old_count, old_mean, old_ssd, old_min, old_max, values)

            mean_row[metric_name] = stats["mean"]
            mean_row[f"{metric_name}_sum_sq_dev"] = stats["sum_sq_dev"]
            std_row[metric_name] = stats["std"]
            min_row[metric_name] = stats["min"]
            max_row[metric_name] = stats["max"]

        mean_row["video_count"] = total_count

        db_ops.put_item(id="_mean", step=sort_key, data=mean_row)
        db_ops.put_item(id="_std", step=sort_key, data=std_row)
        db_ops.put_item(id="_min", step=sort_key, data=min_row)
        db_ops.put_item(id="_max", step=sort_key, data=max_row)

        logger.info(
            "Updated running stats for model={} step={} (old={}, new={}, total={})",
            model,
            sort_key,
            old_count,
            new_count,
            total_count,
        )
