"""vrag_llm: Refines user prompts via a Strands Agent (Claude 3.5 Haiku).

Reads inputs.json from the SageMaker input channel, invokes a Strands
Agent to produce a retrieval_query and video_prompt per entry, and writes
VragOutputEntry JSON shards ({id}.json) for the downstream retrieval step.

Usage: python3 main.py --refine
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from loguru import logger
from pydantic import ValidationError

try:
    from common.dynamodb import DynamoDBOperations
    from common.models import VisualEntry, VragOutputEntry
except ImportError:
    from processing_job.common.dynamodb import DynamoDBOperations
    from processing_job.common.models import VisualEntry, VragOutputEntry

from schema.columns import COL

SM_INPUT_DIR = "/opt/ml/processing/input/input"
LOCAL_OUTPUT_DIR = os.environ.get("LOCAL_OUTPUT_DIR", "/opt/ml/processing/output/output/")
INPUT_FILENAME = "inputs_t2v.json"
DEFAULT_MODEL_ID = os.environ.get("VRAG_LLM_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "")
STEP_NAME = os.environ.get("STEP_NAME", "vrag_llm")
EXECUTION_ID = os.environ.get("EXECUTION_ID", "")
MAX_WORKERS = int(os.environ.get("VRAG_LLM_WORKERS", "1"))

SYSTEM_PROMPT = """\
You are a V-RAG prompt refinement agent. Given a user's description,
produce two outputs:
1. RETRIEVAL_QUERY: A short, natural image description (max 10 words) for
   semantic vector search against an image embedding index. Describe what the
   image looks like — the main subject, setting, and visual style. Do NOT use
   keyword lists, abstract concepts, or cinematic language. Think: how would
   you caption the ideal reference photo in one short phrase?
2. VIDEO_PROMPT: A descriptive prompt for video generation from the retrieved
   image. Focus on motion, transitions, camera movement, style, and mood.
   Be detailed and cinematic here.

Respond in this exact JSON format:
{"retrieval_query": "...", "video_prompt": "..."}\
"""


def load_entries(input_dir: str, filename: str = INPUT_FILENAME) -> list[VisualEntry]:
    """Read JSON array from file, validate each entry against VisualEntry.

    Logs and skips invalid entries. Returns list of validated VisualEntry objects.
    """
    filepath = os.path.join(input_dir, filename)
    if not os.path.isfile(filepath):
        logger.error("Input file not found: {}", filepath)
        return []

    with open(filepath) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        logger.error("Expected JSON array in {}, got {}", filepath, type(raw).__name__)
        return []

    entries: list[VisualEntry] = []
    for i, item in enumerate(raw):
        try:
            entry = VisualEntry.model_validate(item)
            entries.append(entry)
        except ValidationError as e:
            logger.warning("Skipping invalid entry[{}]: {}", i, e)

    logger.info("Loaded {}/{} valid entries from {}", len(entries), len(raw), filename)
    return entries


def create_agent(model_id: str = DEFAULT_MODEL_ID):
    """Create a Strands Agent with the V-RAG prompt refinement system prompt."""
    from strands import Agent
    from strands.models.bedrock import BedrockModel

    model = BedrockModel(model_id=model_id)
    agent = Agent(model=model, system_prompt=SYSTEM_PROMPT)
    logger.info("Created Strands Agent with model: {}", model_id)
    return agent


def extract_json(text: str) -> str:
    """Extract JSON string from agent response text.

    Handles:
    - ```json ... ``` code fences
    - ``` ... ``` code fences without language tag
    - Bare JSON object
    """
    # Try ```json ... ``` first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try bare JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    return text.strip()


def refine_prompt(agent, prompt: str) -> dict:
    """Invoke the agent with a prompt and parse the JSON response.

    Returns dict with 'retrieval_query' and 'video_prompt' keys.
    Raises ValueError on parse failure or missing fields.
    """
    response = agent(prompt)
    raw_text = str(response)

    json_str = extract_json(raw_text)
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse agent response as JSON: {e}. Raw: {raw_text[:200]}")

    if not isinstance(parsed, dict):
        raise ValueError(f"Agent response is not a JSON object (got {type(parsed).__name__}). Raw: {raw_text[:200]}")

    if "retrieval_query" not in parsed or "video_prompt" not in parsed:
        raise ValueError(
            f"Agent response missing required fields. Got keys: {list(parsed.keys())}. Raw: {raw_text[:200]}"
        )

    return parsed


def write_shard(
    entry_id: str,
    original_prompt: str,
    retrieval_query: str,
    video_prompt: str,
    output_dir: str,
) -> None:
    """Write a VragOutputEntry JSON shard as {id}.json."""
    entry = VragOutputEntry(
        id=entry_id,
        prompt=original_prompt,
        retrieval_query=retrieval_query,
        video_prompt=video_prompt,
    )
    out_path = os.path.join(output_dir, f"{entry_id}.json")
    with open(out_path, "w") as f:
        f.write(entry.model_dump_json())
    logger.info("Wrote shard: {}.json", entry_id)


def log_to_dynamodb(db_ops: DynamoDBOperations, entry_id: str, data: dict) -> None:
    """Log metadata for a processed entry to DynamoDB. Non-fatal on failure."""
    try:
        db_ops.put_item(id=entry_id, step=STEP_NAME, data=data)
    except Exception as e:
        logger.error("DynamoDB logging failed for entry {}: {}", entry_id, e)


def main() -> None:
    parser = argparse.ArgumentParser(description="V-RAG LLM prompt refinement step")
    parser.add_argument("--refine", action="store_true", required=True)
    parser.parse_args()

    logger.info(
        "vrag_llm step starting | model={} | table={} | execution={}",
        DEFAULT_MODEL_ID,
        DYNAMODB_TABLE_NAME,
        EXECUTION_ID,
    )

    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    # Load and validate entries
    entries = load_entries(SM_INPUT_DIR, INPUT_FILENAME)
    if not entries:
        logger.error("No valid entries found — exiting")
        sys.exit(1)

    # Create thread-local storage for agents (Strands Agent is not thread-safe)
    _thread_local = __import__("threading").local()

    def _get_agent():
        """Return a per-thread Agent instance."""
        if not hasattr(_thread_local, "agent"):
            _thread_local.agent = create_agent(DEFAULT_MODEL_ID)
        return _thread_local.agent

    # Initialize DynamoDB (if table configured)
    db_ops = None
    if DYNAMODB_TABLE_NAME:
        try:
            db_ops = DynamoDBOperations(DYNAMODB_TABLE_NAME)
        except Exception as e:
            logger.error("Failed to initialize DynamoDB: {}", e)

    # Process each entry (configurable concurrency via VRAG_LLM_WORKERS env var)
    written = 0
    skipped = 0
    lock = __import__("threading").Lock()

    def _process_entry(entry: VisualEntry) -> bool:
        """Process a single entry: refine prompt, write shard, log to DynamoDB. Returns True on success."""
        logger.info("Processing entry: {} | prompt: {}...", entry.id, entry.prompt[:80])
        try:
            result = refine_prompt(_get_agent(), entry.prompt)
        except Exception as e:
            logger.error("Failed to refine prompt for entry {}: {}", entry.id, e)
            return False

        retrieval_query = result["retrieval_query"]
        video_prompt = result["video_prompt"]

        write_shard(entry.id, entry.prompt, retrieval_query, video_prompt, LOCAL_OUTPUT_DIR)

        if db_ops:
            log_to_dynamodb(
                db_ops,
                entry.id,
                {
                    COL.PROMPT: entry.prompt,
                    COL.RETRIEVAL_QUERY: retrieval_query,
                    COL.VIDEO_PROMPT: video_prompt,
                    COL.LLM_MODEL_ID: DEFAULT_MODEL_ID,
                    COL.PIPELINE_EXECUTION_ID: EXECUTION_ID,
                    COL.TIMESTAMP: datetime.now(timezone.utc).isoformat(),
                },
            )
        return True

    logger.info("Processing {} entries with {} worker(s)", len(entries), MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_process_entry, entry): entry for entry in entries}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                success = future.result()
            except Exception as e:
                logger.error("Unexpected error for entry {}: {}", entry.id, e)
                success = False
            with lock:
                if success:
                    written += 1
                else:
                    skipped += 1
                if (written + skipped) % 50 == 0:
                    logger.info(
                        "Progress: {}/{} done ({} written, {} skipped)",
                        written + skipped,
                        len(entries),
                        written,
                        skipped,
                    )

    logger.info(
        "vrag_llm step complete: {}/{} shards written, {} skipped",
        written,
        len(entries),
        skipped,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
