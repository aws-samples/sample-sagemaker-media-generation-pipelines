"""LTX 2.3 video generation workflow for ComfyUI.

Provides run_workflow() for generating videos with LTX 2.3.
Import and call directly — do not run as a script.
"""

import json
import os

from comfy_script.runtime import *

load()
from comfy_script.runtime.nodes import *


def _resolve_input_dir(path: str) -> str:
    """Descend into the execution-ID subdirectory when JSON files are nested.

    SageMaker S3 inputs preserve the S3 key structure, so sharded files
    often land inside an execution-ID subdirectory (e.g. shards/<exec-id>/*.json).
    Old executions may leave stale subdirectories alongside the current one.
    """
    top_jsons = [f for f in os.listdir(path) if f.endswith(".json") and not f.startswith("_")]
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
    JSON files (0.json, 1.json, ...) produced by the agent sharding step.

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
            if fname.endswith(".json") and not fname.startswith("_"):
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


def run_workflow(
    prompt: str, image: str, disable_i2v: bool, input_id: str = "output", file_prefix: str = "output", seed: int = 42
) -> None:
    """Run a single LTX 2.3 video generation workflow.

    Args:
        prompt: Text prompt for video generation.
        image: Filename of the input image.
        disable_i2v: When True, bypasses image conditioning (t2v mode).
        input_id: Unique identifier for the input (stored in metadata).
        file_prefix: Prefix used as the output filename (e.g. 'tokyo-rain-alley_ltx23').
        seed: Noise seed for reproducible generation. Defaults to 42.
    """
    with Workflow():
        height = PrimitiveInt(720)
        width = PrimitiveInt(1280)
        frame_rate = PrimitiveInt(24)

        frame_rate_float, frame_rate_int = ComfyMathExpression(expression="a", **{"values.a": frame_rate})
        noise = RandomNoise(seed)
        model, _, vae = CheckpointLoaderSimple("ltx-2.3-22b-dev-fp8.safetensors")
        model = LoraLoaderModelOnly(
            model=model,
            lora_name="ltx-2.3-22b-distilled-lora-384.safetensors",
            strength_model=0.5,
        )
        string = PrimitiveStringMultiline(prompt)
        clip = LTXAVTextEncoderLoader(
            text_encoder="gemma_3_12B_it_fp4_mixed.safetensors",
            ckpt_name="ltx-2.3-22b-dev-fp8.safetensors",
            device="default",
        )
        conditioning = CLIPTextEncode(string, clip)
        conditioning2 = CLIPTextEncode("pc game, console game, video game, cartoon, childish, ugly", clip)
        positive, negative = LTXVConditioning(
            positive=conditioning,
            negative=conditioning2,
            frame_rate=frame_rate_float,
        )
        noise2 = RandomNoise(seed + 1)
        guider = CFGGuider(model=model, positive=positive, negative=negative, cfg=1)
        sampler = KSamplerSelect("euler_ancestral_cfg_pp")
        sigmas = ManualSigmas("1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0")

        # Load and preprocess input image
        loaded_image, _ = LoadImage(image)
        loaded_image = ResizeImageMaskNode(
            input=loaded_image,
            scale_method="lanczos",
            **{
                "resize_type": "scale dimensions",
                "resize_type.crop": "center",
                "resize_type.width": width,
                "resize_type.height": height,
            },
        )
        loaded_image = ResizeImagesByLongerEdge(loaded_image, 1280)
        loaded_image = LTXVPreprocess(loaded_image, 18)

        _, half_width = ComfyMathExpression(expression="a/2", **{"values.a": width})
        _, half_height = ComfyMathExpression(expression="a/2", **{"values.a": height})
        num_frames = PrimitiveInt(121)
        latent = EmptyLTXVLatentVideo(width=half_width, height=half_height, length=num_frames, batch_size=1)
        latent = LTXVImgToVideoInplace(
            vae=vae,
            image=loaded_image,
            latent=latent,
            strength=0.7,
            bypass=disable_i2v,
        )

        # Audio latent
        audio_vae = LTXVAudioVAELoader("ltx-2.3-22b-dev-fp8.safetensors")
        audio_latent = LTXVEmptyLatentAudio(
            frames_number=num_frames,
            frame_rate=frame_rate_int,
            batch_size=1,
            audio_vae=audio_vae,
        )

        # First pass sampling
        combined_latent = LTXVConcatAVLatent(latent, audio_latent)
        combined_latent, _ = SamplerCustomAdvanced(
            noise=noise2,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=combined_latent,
        )
        combined_latent, audio_latent = LTXVSeparateAVLatent(combined_latent)

        # Second pass with upscaling
        positive2, negative2, _ = LTXVCropGuides(positive=positive, negative=negative, latent=combined_latent)
        guider2 = CFGGuider(model=model, positive=positive2, negative=negative2, cfg=1)
        sampler2 = KSamplerSelect("euler_cfg_pp")
        sigmas2 = ManualSigmas("0.85, 0.7250, 0.4219, 0.0")
        latent_upscale_model = LatentUpscaleModelLoader("ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
        upscaled_latent = LTXVLatentUpsampler(samples=combined_latent, upscale_model=latent_upscale_model, vae=vae)
        upscaled_latent = LTXVImgToVideoInplace(
            vae=vae,
            image=loaded_image,
            latent=upscaled_latent,
            strength=1,
            bypass=disable_i2v,
        )
        upscaled_latent = LTXVConcatAVLatent(upscaled_latent, audio_latent)
        upscaled_latent, final_audio_latent = SamplerCustomAdvanced(
            noise=noise,
            guider=guider2,
            sampler=sampler2,
            sigmas=sigmas2,
            latent_image=upscaled_latent,
        )
        upscaled_latent, final_audio_latent = LTXVSeparateAVLatent(upscaled_latent)

        # Decode and save
        decoded_image = VAEDecodeTiled(
            samples=upscaled_latent,
            vae=vae,
            tile_size=768,
            overlap=64,
            temporal_size=4096,
            temporal_overlap=4,
        )
        audio = LTXVAudioVAEDecode(final_audio_latent, audio_vae)
        prefix = f"video/{file_prefix}"
        video = CreateVideo(images=decoded_image, fps=frame_rate_float, audio=audio)
        SaveVideo(video=video, filename_prefix=prefix, format="auto", codec="auto")
