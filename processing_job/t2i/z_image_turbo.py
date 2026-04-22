"""Z-Image-Turbo image generation workflow for ComfyUI.

Provides run_workflow() for generating images with Z-Image-Turbo.
Import and call directly — do not run as a script.
"""

from comfy_script.runtime import *

load()
from comfy_script.runtime.nodes import *


def run_workflow(prompt: str, file_prefix: str = "output", seed: int = 0) -> None:
    """Run a single Z-Image-Turbo image generation workflow.

    Args:
        prompt: Text prompt for image generation.
        file_prefix: Prefix used as the output filename.
        seed: Noise seed for reproducible generation. Defaults to 0.
    """
    with Workflow():
        model = UNETLoader("z_image_turbo_bf16.safetensors", "default")
        model = ModelSamplingAuraFlow(model, 3)
        clip = CLIPLoader(
            clip_name="qwen_3_4b.safetensors",
            type="lumina2",
            device="default",
        )
        conditioning = CLIPTextEncode(prompt, clip)
        conditioning2 = ConditioningZeroOut(conditioning)
        latent = EmptySD3LatentImage(width=1024, height=1024, batch_size=1)
        latent = KSampler(
            model=model,
            seed=seed,
            steps=8,
            cfg=1,
            sampler_name="res_multistep",
            scheduler="simple",
            positive=conditioning,
            negative=conditioning2,
            latent_image=latent,
            denoise=1,
        )
        vae = VAELoader("ae.safetensors")
        image = VAEDecode(latent, vae)
        SaveImage(image, f"z-image-turbo/{file_prefix}")
