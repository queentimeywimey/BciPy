"""
forward_diffusion.py

Takes each image in a source directory and generates 30 steps of forward diffusion
(progressive noise addition) using Stable Diffusion's DDPM noise schedule. Timesteps
are evenly distributed between 0 and TOTAL_TIMESTEPS - 1. Because physical noise
variance v(t) is a nonlinear (saturating) function of raw timestep, this does not
mean noise variance itself increases by equal amounts between steps -- early steps
show larger visual jumps than the flat tail end of the schedule.

For each input image, a subfolder is created containing:
  step_00.png  →  original image (t=0, no noise)
  step_01.png  →  slightly noised
  ...
  step_29.png  →  heavily noised (near pure noise)

If an image's subfolder already contains every expected step file, that image
is skipped (no regeneration, no overwrite). Pass --force to regenerate anyway.

Usage:
    python forward_diffusion.py --input_dir ./images --output_dir ./diffusion_output

Requirements:
    pip install diffusers torch torchvision Pillow tqdm accelerate
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import DDPMScheduler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
NUM_STEPS = 10         # Number of forward diffusion steps to save
MAX_IMAGE_SIZE = 1024  # Longest-edge length in pixels; aspect ratio is preserved
TOTAL_TIMESTEPS = 1000  # Full DDPM schedule length


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def aspect_preserving_size(orig_size: tuple, max_size: int) -> tuple:
    """Scale (width, height) so the longer edge equals max_size, keeping aspect ratio.

    A 3:2 image stays 3:2 (e.g. 1024x683) instead of being squashed into a square.

    Args:
        orig_size: Original (width, height) of the image.
        max_size: Target length for the longer edge, in pixels.

    Returns:
        (width, height) tuple with the original aspect ratio preserved.
    """
    width, height = orig_size
    scale = max_size / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    return (new_width, new_height)


def load_image_as_tensor(image_path: Path, max_size: int) -> torch.Tensor:
    """Load an image file and convert it to a normalized float tensor.

    The image is resized so its longer edge is `max_size`, preserving the original
    aspect ratio (a 3:2 image stays 3:2 rather than being compressed into a square).
    The tensor is scaled to [-1, 1] as expected by Stable Diffusion's latent space.

    Args:
        image_path: Path to the image file.
        max_size: Target length for the longer edge, in pixels.

    Returns:
        Tensor of shape (1, 3, H, W) with values in [-1, 1].
    """
    img = Image.open(image_path).convert("RGB")
    target = aspect_preserving_size(img.size, max_size)
    img = img.resize(target, Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [0,255] → [-1,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, 3, H, W) tensor in [-1, 1] back to a PIL Image.

    Args:
        tensor: Float tensor with values in [-1, 1].

    Returns:
        PIL Image in RGB mode.
    """
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,3)
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def get_linear_timesteps(num_steps: int, total_timesteps: int) -> list:
    """Get timestep indices evenly distributed between 0 and total_timesteps - 1.

        t_i = round(i * (total_timesteps - 1) / (num_steps - 1))

    Step 0    → t=0   (original, no noise)
    Step N-1  → t=total_timesteps-1 (complete noise)

    Args:
        num_steps: Number of steps to generate.
        total_timesteps: Full DDPM schedule length.

    Returns:
        List of integer timestep values, strictly increasing.
    """
    return [int(round(t)) for t in np.linspace(0, total_timesteps - 1, num_steps)]


# ---------------------------------------------------------------------------
# Core diffusion logic
# ---------------------------------------------------------------------------

def forward_diffuse_image(
        image_tensor: torch.Tensor,
        scheduler: DDPMScheduler,
        timesteps: list,
        device: torch.device) -> list:
    """Apply forward diffusion at each requested timestep.

    At each timestep t, adds the appropriate amount of Gaussian noise to the
    original image according to the DDPM closed-form formula:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

    Args:
        image_tensor: Clean image tensor of shape (1, 3, H, W) in [-1, 1].
        scheduler: Hugging Face DDPMScheduler with pre-computed alphas.
        timesteps: List of integer timestep values to sample.
        device: Torch device to run on.

    Returns:
        List of (timestep, noisy_tensor) pairs in the same order as `timesteps`.
    """
    x0 = image_tensor.to(device)
    # Fix a single noise tensor so progression is smooth across steps
    noise = torch.randn_like(x0)

    results = []
    for t in timesteps:
        if t == 0:
            # Timestep 0 = original image, no noise added
            results.append((t, x0.clone()))
        else:
            t_tensor = torch.tensor([t], device=device)
            noisy = scheduler.add_noise(x0, noise, t_tensor)
            results.append((t, noisy))

    return results


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_library(input_dir: str, output_dir: str, force: bool = False) -> None:
    """Process all images in input_dir and write diffusion steps to output_dir.

    Args:
        input_dir: Directory containing source images.
        output_dir: Root directory where per-image subfolders will be created.
        force: If True, regenerate steps even if all output files already exist.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all supported image files
    image_files = sorted([
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        print(f"No supported images found in '{input_dir}'.")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        return

    print(f"Found {len(image_files)} image(s) in '{input_dir}'")

    # Set up device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    # Initialize the DDPM noise scheduler
    # Using the standard SD beta schedule for authentic SD forward diffusion
    scheduler = DDPMScheduler(
        num_train_timesteps=TOTAL_TIMESTEPS,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",  # Stable Diffusion default
        clip_sample=False,
    )

    # Compute which timesteps to save, evenly distributed between 0 and TOTAL_TIMESTEPS - 1
    timesteps = get_linear_timesteps(NUM_STEPS, TOTAL_TIMESTEPS)
    print(f"Diffusion timesteps: {timesteps}\n")

    # Process each image
    for image_file in tqdm(image_files, desc="Processing images", unit="img"):
        stem = image_file.stem  # filename without extension
        image_output_dir = output_path / stem

        expected_files = [
            image_output_dir / f"{stem}_t{t:03d}.png" for t in timesteps
        ]
        if not force and all(f.exists() for f in expected_files):
            tqdm.write(f"  [SKIP] All {NUM_STEPS} steps already exist → '{image_output_dir}'")
            continue

        image_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            image_tensor = load_image_as_tensor(image_file, MAX_IMAGE_SIZE)
        except Exception as e:
            print(f"\n  [SKIP] Could not load '{image_file.name}': {e}")
            continue

        # Run forward diffusion
        diffusion_steps = forward_diffuse_image(
            image_tensor, scheduler, timesteps, device
        )

        # Save each step
        for step_idx, (t, noisy_tensor) in enumerate(diffusion_steps):
            pil_image = tensor_to_image(noisy_tensor)
            filename = image_output_dir / f"{stem}_t{t:03d}.png"
            pil_image.save(filename)
            print(f"  Saved step {step_idx} (t={t}) → '{filename.name}'")

        tqdm.write(f"  Saved {NUM_STEPS} steps → '{image_output_dir}'")

    print(f"\nDone. Output written to '{output_path}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate forward diffusion steps for each image in a directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="images/raw",
        help="Path to folder containing source images (default: images)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="images/processed",
        help="Path to root output folder (default: diffusion_output)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=NUM_STEPS,
        help=f"Number of diffusion steps to save per image (default: {NUM_STEPS})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=MAX_IMAGE_SIZE,
        help="Longest-edge size in pixels; aspect ratio is preserved "
             f"(default: {MAX_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate steps even if all output files already exist "
             "(default: skip images whose steps are all already present)",
    )
    args = parser.parse_args()

    # Allow CLI overrides
    NUM_STEPS = args.steps
    MAX_IMAGE_SIZE = args.size

    process_library(args.input_dir, args.output_dir, args.force)