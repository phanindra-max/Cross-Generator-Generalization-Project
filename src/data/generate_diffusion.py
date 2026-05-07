"""
Generate synthetic face images using Stable Diffusion v1.5.

Produces diverse portrait images that serve as the out-of-distribution
(diffusion-era) test set for cross-generator evaluation.
"""

import os
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm


PORTRAIT_PROMPTS = [
    "a portrait photo of a young woman with brown hair, natural lighting, neutral background",
    "a portrait photo of a middle-aged man with glasses, studio lighting, white background",
    "a candid photo of a person smiling, outdoor natural light",
    "a portrait of an elderly woman with gray hair, soft lighting",
    "a headshot of a young man with short hair, professional lighting",
    "a portrait photo of a woman with curly hair, warm lighting, bokeh background",
    "a close-up portrait of a man with a beard, dramatic side lighting",
    "a portrait of a young person with freckles, natural daylight",
    "a professional headshot of a woman, neutral expression, gray background",
    "a portrait photo of a man in his 30s, soft natural light from window",
    "a candid portrait of a smiling elderly man, outdoor setting",
    "a portrait of a young woman with straight black hair, even studio lighting",
    "a headshot of a person with short curly hair, clean background",
    "a portrait photo of a middle-aged woman, natural makeup, soft lighting",
    "a close-up portrait of a young man, shallow depth of field",
    "a portrait of a woman with red hair, golden hour lighting",
    "a professional portrait of a man in a suit, studio lighting",
    "a candid photo of a young person laughing, natural light",
    "a portrait of an older man with white hair, gentle expression",
    "a headshot of a woman with dark skin, bright even lighting",
]


def generate_diffusion_faces(
    output_dir: str,
    num_images: int = 1000,
    resolution: int = 128,
    batch_size: int = 4,
    seed: int = 42,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
):
    """
    Generate face images using Stable Diffusion v1.5.

    Args:
        output_dir: Directory to save generated images.
        num_images: Total number of images to generate.
        resolution: Output resolution (images generated at 512x512 then resized).
        batch_size: Number of images to generate per forward pass.
        seed: Base random seed for reproducibility.
        model_id: HuggingFace model ID for Stable Diffusion.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_path.glob("*.png"))
    if len(existing) >= num_images:
        print(
            f"Already have {len(existing)} diffusion images in {output_path}, "
            f"skipping generation."
        )
        return len(existing)
    if existing:
        # Partial state from an interrupted run. Restarting from index 0 keeps
        # the generator/seed contract intact, so the final 1000 images match
        # what a clean run would produce. Existing files at indices 0..N-1
        # are simply overwritten as we re-generate them.
        print(
            f"Found {len(existing)} partial diffusion images in {output_path}; "
            f"regenerating all {num_images} from scratch to preserve determinism."
        )

    from diffusers import StableDiffusionPipeline

    print(f"Loading Stable Diffusion pipeline: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    images_per_prompt = num_images // len(PORTRAIT_PROMPTS)
    remainder = num_images % len(PORTRAIT_PROMPTS)

    image_count = 0
    generator = torch.Generator(device="cuda").manual_seed(seed)

    print(f"Generating {num_images} images ({images_per_prompt} per prompt)...")

    for prompt_idx, prompt in enumerate(tqdm(PORTRAIT_PROMPTS, desc="Prompts")):
        n_for_this_prompt = images_per_prompt + (1 if prompt_idx < remainder else 0)
        generated_for_prompt = 0

        while generated_for_prompt < n_for_this_prompt:
            current_batch = min(batch_size, n_for_this_prompt - generated_for_prompt)

            results = pipe(
                prompt=[prompt] * current_batch,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=512,
                width=512,
            )

            for img in results.images:
                img_resized = img.resize((resolution, resolution), Image.LANCZOS)
                img_resized.save(output_path / f"{image_count:05d}.png")
                image_count += 1
                generated_for_prompt += 1

    print(f"Done. Generated {image_count} images in {output_path}")
    return image_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Stable Diffusion faces")
    parser.add_argument("--output-dir", type=str, default="data/diffusion_sd15")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()

    generate_diffusion_faces(
        output_dir=args.output_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
        seed=args.seed,
        resolution=args.resolution,
    )
