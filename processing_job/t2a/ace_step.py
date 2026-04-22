"""ACE Step 1.5 audio generation workflow for ComfyUI.

Provides run_workflow() for generating audio with ACE Step 1.5 Turbo AIO.
Import and call directly — do not run as a script.
"""

from comfy_script.runtime import *

load()
from comfy_script.runtime.nodes import *


def run_workflow(
    tags: str,
    lyrics: str,
    file_prefix: str = "output",
    seed: int = 31,
    bpm: int = 120,
    duration: int = 120,
    timesignature: str = "4",
    language: str = "en",
    keyscale: str = "C major",
    cfg_scale: float = 2,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0,
) -> None:
    """Run a single ACE Step 1.5 audio generation workflow.

    Args:
        tags: Genre/style tags for the audio generation.
        lyrics: Song lyrics with section markers.
        file_prefix: Prefix used as the output filename.
        seed: Noise seed for reproducible generation.
        bpm: Beats per minute.
        duration: Audio duration in seconds.
        timesignature: Time signature (e.g. "4" for 4/4).
        language: Lyrics language code.
        keyscale: Musical key and scale (e.g. "E minor").
        cfg_scale: Classifier-free guidance scale.
        temperature: Sampling temperature.
        top_p: Top-p (nucleus) sampling threshold.
        top_k: Top-k sampling threshold (0 to disable).
        min_p: Minimum probability threshold.
    """
    with Workflow():
        model, clip, vae = CheckpointLoaderSimple("ace_step_1.5_turbo_aio.safetensors")
        model = ModelSamplingAuraFlow(model, 3)
        conditioning = TextEncodeAceStepAudio1_5(
            clip=clip,
            tags=tags,
            lyrics=lyrics,
            seed=seed,
            bpm=bpm,
            duration=duration,
            timesignature=timesignature,
            language=language,
            keyscale=keyscale,
            generate_audio_codes=True,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        conditioning2 = ConditioningZeroOut(conditioning)
        latent = EmptyAceStep1_5LatentAudio(duration, 1)
        latent = KSampler(
            model=model,
            seed=seed,
            steps=8,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            positive=conditioning,
            negative=conditioning2,
            latent_image=latent,
            denoise=1,
        )
        audio = VAEDecodeAudio(latent, vae)
        SaveAudioMP3(audio=audio, filename_prefix=f"audio/{file_prefix}", quality="V0")
