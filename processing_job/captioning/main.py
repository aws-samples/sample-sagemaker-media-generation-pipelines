"""captioning: Generates captions for images using a configurable VL model.

Scans the SageMaker input channel for image files, generates a caption
for each one, writes a JSON sidecar per image to the output channel,
and logs results to DynamoDB.

The model is configured via the CAPTION_MODEL_NAME environment variable,
which corresponds to the models_prefix directory name in the pipeline config.

Usage: python3 main.py --caption
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

try:
    from common.models import CaptioningSidecarEntry, VisualEntry
except ImportError:
    from processing_job.common.models import CaptioningSidecarEntry, VisualEntry
from pydantic import ValidationError

SM_INPUT_DIR = "/opt/ml/processing/input/input"
LOCAL_OUTPUT_DIR = os.environ.get("LOCAL_OUTPUT_DIR", "/opt/ml/processing/output/output")
MODELS_DIR = "/opt/ml/processing/input"
CAPTION_MODEL_NAME = os.environ.get("CAPTION_MODEL_NAME", "qwen2_5_vl_7b")
_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompt.txt")
CAPTION_PROMPT = os.environ.get("CAPTION_PROMPT") or open(_PROMPT_FILE).read().strip()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def find_model_path() -> str:
    """Locate the model directory under /opt/ml/processing/input/models*.

    Uses CAPTION_MODEL_NAME to find the correct subdirectory.
    """
    for d in sorted(os.listdir(MODELS_DIR)):
        full = os.path.join(MODELS_DIR, d)
        if d.startswith("models") and os.path.isdir(full):
            model_dir = os.path.join(full, CAPTION_MODEL_NAME)
            if os.path.isdir(model_dir):
                return model_dir
            if os.path.isfile(os.path.join(full, "config.json")):
                return full
    raise FileNotFoundError(f"Could not find model '{CAPTION_MODEL_NAME}' under /opt/ml/processing/input/models*")


def load_model(model_path: str):
    """Load VL model and processor from local path using AutoModel."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info("Loading model '{}' from: {}", CAPTION_MODEL_NAME, model_path)
    logger.info("Available GPU memory: {:.1f} GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    logger.info("Model loaded successfully on device: {}", model.device)
    return model, processor


def generate_caption(model, processor, image_path: str, prompt: str) -> str:
    """Generate a caption for a single image.

    Uses qwen_vl_utils.process_vision_info to handle image preprocessing,
    following the official Qwen2.5-VL pattern.
    """

    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path, "resized_height": 280, "resized_width": 420},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    caption = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return caption


def scan_images(input_dir: str) -> list[Path]:
    """Recursively find all image files in the input directory."""
    images = []
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(Path(root) / f)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Captioning step")
    parser.add_argument("--caption", action="store_true", required=True)
    parser.parse_args()

    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(SM_INPUT_DIR):
        logger.error("Input directory not found: {}", SM_INPUT_DIR)
        sys.exit(1)

    images = scan_images(SM_INPUT_DIR)
    if not images:
        logger.error("No image files found in {}", SM_INPUT_DIR)
        sys.exit(1)

    logger.info("Found {} images in {}", len(images), SM_INPUT_DIR)
    logger.info("Caption prompt: {}", CAPTION_PROMPT[:120] if CAPTION_PROMPT else "(none)")

    if not CAPTION_PROMPT:
        logger.error("CAPTION_PROMPT environment variable is not set")
        sys.exit(1)

    # Load model
    model_path = find_model_path()
    model, processor = load_model(model_path)

    caption_metadata = {}

    for img_path in images:
        stem = img_path.stem  # filename without extension
        size_bytes = img_path.stat().st_size
        logger.info("Processing: {} ({:.1f} KB)", img_path.name, size_bytes / 1024)

        caption = ""
        try:
            caption = generate_caption(model, processor, str(img_path), CAPTION_PROMPT)
            logger.info("  Caption ({}ch): {}", len(caption), caption[:120])
        except Exception as e:
            logger.error("  Caption generation failed for {}: {}", img_path.name, e)

        # Write a per-image JSON shard for downstream i2v/flf2v consumption
        out_path = os.path.join(LOCAL_OUTPUT_DIR, f"{stem}.json")
        try:
            visual_entry = VisualEntry(id=stem, prompt=caption, image=img_path.name)
            with open(out_path, "w") as f:
                f.write(visual_entry.model_dump_json())
        except ValidationError as e:
            logger.error("VisualEntry validation failed for {}: {}", stem, e)
            continue
        logger.info("Wrote {}", f"{stem}.json")

        # Build sidecar metadata for DynamoDB logging
        file_prefix = f"{stem}"
        try:
            caption_metadata[file_prefix] = CaptioningSidecarEntry(
                input_id=stem,
                model=CAPTION_MODEL_NAME,
                mode="captioning",
                prompt=caption,
                source_filename=img_path.name,
                generation_index=0,
            ).model_dump()
        except ValidationError as e:
            logger.error("CaptioningSidecarEntry validation failed for {}: {}", stem, e)

    logger.info("Captioning complete: {} images processed", len(images))

    # Copy original images to output so downstream steps (e.g. i2v) can access them
    import shutil

    for img_path in images:
        dest = os.path.join(LOCAL_OUTPUT_DIR, img_path.name)
        if not os.path.exists(dest):
            shutil.copy2(str(img_path), dest)
            logger.info("Copied image to output: {}", img_path.name)

    # Write sidecar metadata for log_outputs
    sidecar_path = os.path.join(LOCAL_OUTPUT_DIR, "_caption_metadata.json")
    with open(sidecar_path, "w") as f:
        json.dump(caption_metadata, f)
    logger.info("Wrote caption metadata sidecar with {} entries", len(caption_metadata))

    # Log to DynamoDB
    logger.info("Logging captioned results to DynamoDB...")
    subprocess.run(
        [sys.executable, "-m", "common.log_outputs", "--extensions", ".json", "--sidecar-file", sidecar_path],
        check=True,
    )
    logger.info("Script completed successfully.")


if __name__ == "__main__":
    main()
    sys.exit(0)
