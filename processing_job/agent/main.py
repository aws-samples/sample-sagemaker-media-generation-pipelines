"""agent: Splits input entries into per-entry files for multi-instance sharding.

Reads inputs.json from the SageMaker input channel, writes each entry as a
separate JSON file (0.json, 1.json, ...) so SageMaker ShardedByS3Key can
distribute them across downstream instances. Images are served to downstream
steps via the DataStack input bucket (FullyReplicated).

Usage: python3 main.py --passthrough
"""

import argparse
import json
import os
import sys

from loguru import logger
from pydantic import ValidationError

try:
    from common.models import AudioEntry, VisualEntry
except ImportError:
    from processing_job.common.models import AudioEntry, VisualEntry

SM_INPUT_DIR = "/opt/ml/processing/input/input"
LOCAL_OUTPUT_DIR = os.environ.get("LOCAL_OUTPUT_DIR", "/opt/ml/processing/output/output")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent passthrough step")
    parser.add_argument("--passthrough", action="store_true", required=True)
    parser.parse_args()

    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(SM_INPUT_DIR):
        logger.error("Input directory not found: {}", SM_INPUT_DIR)
        sys.exit(1)

    inputs_path = os.path.join(SM_INPUT_DIR, "inputs_t2v.json")
    if not os.path.isfile(inputs_path):
        logger.error("inputs_t2v.json not found in {}", SM_INPUT_DIR)
        sys.exit(1)

    # Split inputs_t2v.json into individual files for ShardedByS3Key distribution
    with open(inputs_path) as f:
        entries = json.load(f)

    logger.info("Splitting inputs.json ({} entries) into individual files", len(entries))
    valid_count = 0
    for i, entry in enumerate(entries):
        try:
            if "tags" in entry:
                validated = AudioEntry.model_validate(entry)
            else:
                validated = VisualEntry.model_validate(entry)
        except ValidationError as e:
            logger.error("Validation failed for entry[{}]: {}", i, e)
            continue

        entry_id = validated.id
        out_path = os.path.join(LOCAL_OUTPUT_DIR, f"{entry_id}.json")
        with open(out_path, "w") as f:
            f.write(validated.model_dump_json())
        logger.info("Wrote {}.json", entry_id)
        valid_count += 1

    logger.info("Agent step complete: {}/{} valid JSON shards written", valid_count, len(entries))


if __name__ == "__main__":
    main()
    sys.exit(0)
