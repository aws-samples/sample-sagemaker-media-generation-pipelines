"""Wan 2.2 video generation workflow for ComfyUI (i2v and t2v modes).

Provides run_i2v() and run_t2v() for generating videos with Wan 2.2.
Import and call directly — do not run as a script.
"""

import json
import os

from comfy_script.runtime import *

load()
from comfy_script.runtime.nodes import *

NEGATIVE_PROMPT_CN = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

NEGATIVE_PROMPT_CN_T2V = NEGATIVE_PROMPT_CN + "，裸露，NSFW"


def _resolve_input_dir(path: str) -> str:
    """Descend into the execution-ID subdirectory when JSON files are nested.

    SageMaker S3 inputs preserve the S3 key structure, so sharded files
    often land inside an execution-ID subdirectory (e.g. shards/<exec-id>/*.json).
    Old executions may leave stale subdirectories alongside the current one.
    """
    top_jsons = [f for f in os.listdir(path) if f.endswith(".json")]
    if top_jsons:
        return path
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not subdirs:
        return path
    # Prefer the subdirectory matching the current execution ID
    exec_id = os.environ.get("EXECUTION_ID", "")
    if exec_id and exec_id in subdirs:
        return os.path.join(path, exec_id)
    # Fallback: single subdirectory
    if len(subdirs) == 1:
        return os.path.join(path, subdirs[0])
    return path


def load_inputs(path: str | None = None) -> list[dict]:
    """Load video generation inputs from JSON file(s).

    Supports both a single inputs.json array and a directory of individual
    JSON files produced by the agent sharding step.

    Args:
        path: Path to a JSON file or directory of JSON files. Defaults to
              the SageMaker input directory, or the INPUTS_JSON env var.

    Returns:
        List of dicts with 'prompt' and 'image' keys.
    """
    if path is None:
        # Prefer sharded input channel, fall back to standard input
        shards_dir = "/opt/ml/processing/input/shards"
        input_dir = os.environ.get("INPUTS_JSON", "/opt/ml/processing/input/input")
        path = shards_dir if os.path.isdir(shards_dir) else input_dir

    # If path is a directory, load all .json files from it
    if os.path.isdir(path):
        path = _resolve_input_dir(path)
        # Check for single inputs.json first (backward compat)
        single = os.path.join(path, "inputs.json")
        if os.path.isfile(single):
            with open(single, encoding="utf-8") as f:
                return json.load(f)
        # Otherwise load individual sharded files
        entries = []
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".json"):
                with open(os.path.join(path, fname), encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        entries.extend(data)
                    else:
                        entries.append(data)
        return entries

    # Path is a file
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_i2v(
    prompt: str, image: str, input_id: str = "output", file_prefix: str = "output", seed: int = 233767456947725
) -> None:
    """Run image-to-video workflow with distilled 4-step LoRA."""
    with Workflow():
        use_distilled = PrimitiveBoolean(True)
        distilled_steps = PrimitiveInt(4)
        full_steps = PrimitiveInt(20)
        total_steps = ComfySwitchNode(switch=use_distilled, on_false=full_steps, on_true=distilled_steps)

        distilled_split = PrimitiveInt(2)
        full_split = PrimitiveInt(10)
        split_step = ComfySwitchNode(switch=use_distilled, on_false=full_split, on_true=distilled_split)

        # Low-noise model (second pass)
        low_noise_model = UNETLoader("wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "default")
        low_noise_distilled = LoraLoaderModelOnly(
            model=low_noise_model,
            lora_name="wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            strength_model=1.0,
        )
        second_pass_model = ComfySwitchNode(switch=use_distilled, on_false=low_noise_model, on_true=low_noise_distilled)
        second_pass_model = ModelSamplingSD3(second_pass_model, 5.0)

        # Text encoding
        clip = CLIPLoader(
            clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            type="wan",
            device="default",
        )
        conditioning = CLIPTextEncode(prompt, clip)
        conditioning_neg = CLIPTextEncode(NEGATIVE_PROMPT_CN, clip)

        # Image-to-video setup
        vae = VAELoader("wan_2.1_vae.safetensors")
        loaded_image, _ = LoadImage(image)
        positive, negative, latent = WanImageToVideo(
            positive=conditioning,
            negative=conditioning_neg,
            vae=vae,
            width=1280,
            height=720,
            length=81,
            batch_size=1,
            start_image=loaded_image,
        )

        # High-noise model (first pass)
        high_noise_model = UNETLoader("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "default")
        high_noise_distilled = LoraLoaderModelOnly(
            model=high_noise_model,
            lora_name="wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            strength_model=1.0,
        )
        first_pass_model = ComfySwitchNode(
            switch=use_distilled, on_false=high_noise_model, on_true=high_noise_distilled
        )
        first_pass_model = ModelSamplingSD3(first_pass_model, 5.0)

        # Two-pass sampling
        latent = KSamplerAdvanced(
            model=first_pass_model,
            add_noise="enable",
            noise_seed=seed,
            steps=total_steps,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent_image=latent,
            start_at_step=0,
            end_at_step=split_step,
            return_with_leftover_noise="enable",
        )
        latent = KSamplerAdvanced(
            model=second_pass_model,
            add_noise="disable",
            noise_seed=0,
            steps=total_steps,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent_image=latent,
            start_at_step=split_step,
            end_at_step=total_steps,
            return_with_leftover_noise="disable",
        )

        # Decode and save
        decoded = VAEDecode(latent, vae)
        video = CreateVideo(decoded, 16, None)
        SaveVideo(
            video=video,
            filename_prefix=f"video/{file_prefix}",
            format="auto",
            codec="auto",
        )


def run_t2v(prompt: str, input_id: str = "output", file_prefix: str = "output", seed: int = 634896341486210) -> None:
    """Run text-to-video workflow with distilled 4-step LoRA."""
    with Workflow():
        # Low-noise model (second pass)
        low_noise_model = UNETLoader("wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", "default")
        low_noise_model = LoraLoaderModelOnly(
            model=low_noise_model,
            lora_name="wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
            strength_model=1.0,
        )
        low_noise_model = ModelSamplingSD3(low_noise_model, 5.0)

        # Text encoding
        clip = CLIPLoader(
            clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            type="wan",
            device="default",
        )
        conditioning = CLIPTextEncode(prompt, clip)
        conditioning_neg = CLIPTextEncode(NEGATIVE_PROMPT_CN_T2V, clip)

        # High-noise model (first pass)
        high_noise_model = UNETLoader("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "default")
        high_noise_model = LoraLoaderModelOnly(
            model=high_noise_model,
            lora_name="wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
            strength_model=1.0,
        )
        high_noise_model = ModelSamplingSD3(high_noise_model, 5.0)

        # Two-pass sampling
        latent = EmptyHunyuanLatentVideo(width=1280, height=720, length=81, batch_size=1)
        latent = KSamplerAdvanced(
            model=high_noise_model,
            add_noise="enable",
            noise_seed=seed,
            steps=4,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            positive=conditioning,
            negative=conditioning_neg,
            latent_image=latent,
            start_at_step=0,
            end_at_step=2,
            return_with_leftover_noise="enable",
        )
        latent = KSamplerAdvanced(
            model=low_noise_model,
            add_noise="disable",
            noise_seed=0,
            steps=4,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            positive=conditioning,
            negative=conditioning_neg,
            latent_image=latent,
            start_at_step=2,
            end_at_step=4,
            return_with_leftover_noise="disable",
        )

        # Decode and save
        vae = VAELoader("wan_2.1_vae.safetensors")
        decoded = VAEDecode(latent, vae)
        video = CreateVideo(decoded, 16, None)
        SaveVideo(
            video=video,
            filename_prefix=f"video/{file_prefix}",
            format="auto",
            codec="auto",
        )
