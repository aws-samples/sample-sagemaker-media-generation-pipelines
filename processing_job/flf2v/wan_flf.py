"""Wan 2.2 First-Last-Frame to Video workflow for ComfyUI.

Uses WanFirstLastFrameToVideo with high/low noise two-pass sampling
and LightX2V 4-step LoRAs for fast generation.

Import and call directly — do not run as a script.
"""

import json
import os

from comfy_script.runtime import *

load()
from comfy_script.runtime.nodes import *


def _resolve_input_dir(path: str) -> str:
    """Descend into the execution-ID subdirectory when JSON files are nested."""
    top_jsons = [f for f in os.listdir(path) if f.endswith(".json") and not f.startswith("_")]
    if top_jsons:
        return path
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not subdirs:
        return path
    exec_id = os.environ.get("EXECUTION_ID", "")
    if exec_id and exec_id in subdirs:
        return os.path.join(path, exec_id)
    if len(subdirs) == 1:
        return os.path.join(path, subdirs[0])
    return path


def load_inputs(path: str | None = None) -> list[dict]:
    """Load generation inputs from JSON file(s).

    Supports both a single inputs.json array and a directory of individual
    JSON files (0.json, 1.json, ...) produced by the agent sharding step.

    Returns:
        List of dicts with 'prompt' and 'image' keys.
    """
    if path is None:
        shards_dir = "/opt/ml/processing/input/shards"
        input_dir = os.environ.get("INPUTS_JSON", "/opt/ml/processing/input/input")
        path = shards_dir if os.path.isdir(shards_dir) else input_dir

    if os.path.isdir(path):
        path = _resolve_input_dir(path)
        single = os.path.join(path, "inputs.json")
        if os.path.isfile(single):
            with open(single, encoding="utf-8") as f:
                return json.load(f)
        entries = []
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".json") and not fname.startswith("_"):
                with open(os.path.join(path, fname), encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        entries.extend(data)
                    else:
                        entries.append(data)
        return entries

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_workflow(
    prompt: str,
    image: str,
    input_id: str = "output",
    file_prefix: str = "output",
    seed: int = 42,
    width: int = 640,
    height: int = 640,
    length: int = 81,
) -> None:
    """Run a Wan 2.2 First-Last-Frame to Video workflow.

    Uses the same image for both start and end frames to create a
    looping animation effect. Two-pass sampling with high-noise and
    low-noise models plus LightX2V 4-step LoRAs.

    Args:
        prompt: Text prompt for video generation.
        image: Filename of the input image (used as both first and last frame).
        input_id: Unique identifier for the input.
        file_prefix: Prefix used as the output filename.
        seed: Noise seed for reproducible generation.
        width: Output video width. Defaults to 640.
        height: Output video height. Defaults to 640.
        length: Number of frames. Defaults to 81.
    """
    with Workflow():
        # Low-noise model + LoRA
        model_low = UNETLoader("wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "default")
        model_low = LoraLoaderModelOnly(
            model=model_low,
            lora_name="wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            strength_model=1,
        )
        model_low = ModelSamplingSD3(model_low, 5)

        # High-noise model + LoRA
        model_high = UNETLoader("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "default")
        model_high = LoraLoaderModelOnly(
            model=model_high,
            lora_name="wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            strength_model=1,
        )
        model_high = ModelSamplingSD3(model_high, 5)

        # Text encoding
        clip = CLIPLoader(
            clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            type="wan",
            device="default",
        )
        conditioning = CLIPTextEncode(prompt, clip)
        conditioning2 = CLIPTextEncode(
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            clip,
        )

        # VAE + images (same image for first and last frame)
        vae = VAELoader("wan_2.1_vae.safetensors")
        start_image, _ = LoadImage(image)
        end_image, _ = LoadImage(image)

        # First-last-frame conditioning
        positive, negative, latent = WanFirstLastFrameToVideo(
            positive=conditioning,
            negative=conditioning2,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=1,
            start_image=start_image,
            end_image=end_image,
        )

        # Two-pass sampling: high-noise first 2 steps, low-noise remainder
        latent = KSamplerAdvanced(
            model=model_high,
            add_noise="enable",
            noise_seed=seed,
            steps=4,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent_image=latent,
            start_at_step=0,
            end_at_step=2,
            return_with_leftover_noise="enable",
        )
        latent = KSamplerAdvanced(
            model=model_low,
            add_noise="disable",
            noise_seed=0,
            steps=4,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent_image=latent,
            start_at_step=2,
            end_at_step=10000,
            return_with_leftover_noise="disable",
        )

        # Decode and save
        decoded = VAEDecode(latent, vae)
        video = CreateVideo(decoded, 16, None)
        prefix = f"video/{file_prefix}"
        SaveVideo(video=video, filename_prefix=prefix, format="auto", codec="auto")
