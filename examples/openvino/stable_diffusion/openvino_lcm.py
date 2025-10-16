# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code="union-attr,import-not-found"

import argparse
import logging
import os
import time
from typing import Any, Dict, Optional

import torch
from PIL import Image

try:
    from diffusers import LCMScheduler
    from transformers import CLIPTokenizer
except ImportError:
    raise ImportError(
        "Please install diffusers and transformers: pip install diffusers transformers"
    )

from executorch.runtime import Runtime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenVINOLCMPipeline:
    """OpenVINO optimized Latent Consistency Model pipeline for Intel hardware"""

    def __init__(self, device: str = "CPU", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self.models: Dict[str, Any] = {}
        self.tokenizer: Optional[CLIPTokenizer] = None
        self.scheduler: Optional[LCMScheduler] = None
        self.runtime = Runtime.get()
        self._initialized = False

        # Cumulative timing metrics
        self.models_load_time = 0.0
        self.exec_time = 0.0

    def load_tokenizer(self, vocab_path: str):
        """Load CLIP tokenizer"""
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            logger.info("✓ Tokenizer loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            return False

    def initialize_scheduler(
        self, original_model_id: str = "SimianLuo/LCM_Dreamshaper_v7"
    ):
        """Initialize the LCM scheduler"""
        try:
            self.scheduler = LCMScheduler.from_pretrained(
                original_model_id, subfolder="scheduler"
            )
            logger.info("✓ Scheduler loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load scheduler from {original_model_id}: {e}")
            return False

    def load_model_component(self, component_name: str, model_path: str):
        """Load a model component"""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False

            program = self.runtime.load_program(model_path)
            self.models[component_name] = program
            logger.info(f"✓ Loaded {component_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {component_name}: {e}")
            return False

    def encode_prompt(self, prompt: str):
        """Encode text prompt using the text encoder"""
        if "text_encoder" not in self.models or self.tokenizer is None:
            logger.error("Text encoder or tokenizer not loaded")
            return None

        try:
            inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )

            load_start = time.time()
            text_encoder_method = self.models["text_encoder"].load_method("forward")
            load_time = time.time() - load_start
            self.models_load_time += load_time

            exec_start = time.time()
            embeddings = text_encoder_method.execute([inputs.input_ids])[0]
            exec_time = time.time() - exec_start
            self.exec_time += exec_time

            logger.info(
                f"Text encoder - Load: {load_time:.3f}s, Execute: {exec_time:.3f}s"
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode prompt: {e}")
            return None

    def denoise_latents(
        self,
        text_embeddings: torch.Tensor,
        num_steps: int,
        guidance_scale: float,
        seed: Optional[int] = None,
    ):
        """Run the denoising process using the UNet model with LCM scheduler"""
        if "unet" not in self.models:
            logger.error("UNet model not loaded")
            return None

        try:
            # Initialize latents
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed)

            latents = torch.randn(
                (1, 4, 64, 64),  # Standard latent dimensions for SD
                generator=generator,
                dtype=self.dtype,
            )

            # Set timesteps for LCM
            self.scheduler.set_timesteps(num_steps)

            # Get UNet method
            load_start = time.time()
            unet_method = self.models["unet"].load_method("forward")
            load_time = time.time() - load_start
            self.models_load_time += load_time
            logger.info(f"UNet - Load: {load_time:.3f}s")

            # Denoising loop
            logger.info(f"Running LCM denoising with {num_steps} steps...")
            denoise_start = time.time()

            for step, timestep in enumerate(self.scheduler.timesteps):
                step_start = time.time()

                latent_model_input = self.scheduler.scale_model_input(latents, timestep)
                if latent_model_input.dtype != self.dtype:
                    latent_model_input = latent_model_input.to(self.dtype)

                timestep_tensor = torch.tensor(
                    timestep.item(), dtype=torch.long
                ).unsqueeze(0)
                noise_pred = unet_method.execute(
                    [latent_model_input, timestep_tensor, text_embeddings]
                )[0]

                if guidance_scale != 1.0:
                    noise_pred = noise_pred * guidance_scale

                latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
                logger.info(
                    f"  Step {step+1}/{num_steps} completed ({time.time() - step_start:.3f}s)"
                )

            exec_time = time.time() - denoise_start
            self.exec_time += exec_time
            logger.info(
                f"UNet - Execute: {exec_time:.3f}s, avg {exec_time/num_steps:.3f}s/step"
            )

            return latents
        except Exception as e:
            logger.error(f"Failed during denoising: {e}")
            return None

    def decode_image(self, latents: torch.Tensor):
        """Decode latents to final image using VAE decoder"""
        if "vae_decoder" not in self.models:
            logger.error("VAE decoder not loaded")
            return None

        try:
            load_start = time.time()
            vae_method = self.models["vae_decoder"].load_method("forward")
            load_time = time.time() - load_start
            self.models_load_time += load_time

            exec_start = time.time()
            decoded_image = vae_method.execute([latents])[0]
            exec_time = time.time() - exec_start
            self.exec_time += exec_time

            # Convert from (1, 3, 512, 512) CHW to (512, 512, 3) HWC
            conversion_start = time.time()
            decoded_image = decoded_image.squeeze(0).permute(1, 2, 0)
            decoded_image = (decoded_image * 255).clamp(0, 255).to(torch.uint8)
            image = Image.fromarray(decoded_image.numpy())
            postprocess_time = time.time() - conversion_start
            self.exec_time += postprocess_time

            logger.info(
                f"VAE decoder - Load: {load_time:.3f}s, "
                f"Execute: {exec_time:.3f}s, "
                f"Post-process: {postprocess_time:.3f}s"
            )

            return image
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return None

    def generate_image(
        self,
        prompt: str,
        num_steps: int = 4,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Complete image generation pipeline using LCM"""
        if not self._initialized:
            logger.error("Pipeline not initialized")
            return None

        logger.info("=" * 60)
        logger.info(f"Prompt: '{prompt}'")
        logger.info(f"Steps: {num_steps} | Guidance: {guidance_scale} | Seed: {seed}")
        logger.info("=" * 60)

        # Reset cumulative timers
        self.models_load_time = 0.0
        self.exec_time = 0.0

        total_start = time.time()

        text_embeddings = self.encode_prompt(prompt)
        if text_embeddings is None:
            return None

        latents = self.denoise_latents(text_embeddings, num_steps, guidance_scale, seed)
        if latents is None:
            return None

        image = self.decode_image(latents)
        if image is None:
            return None

        total_time = time.time() - total_start

        logger.info("=" * 60)
        logger.info("✓ Generation completed!")
        logger.info(f"  Total time: {total_time:.3f}s")
        logger.info(f"  Total load time: {self.models_load_time:.3f}s")
        logger.info(f"  Total Inference time: {self.exec_time:.3f}s")
        logger.info("=" * 60)
        return image

    def initialize(
        self,
        text_encoder_path: str,
        unet_path: str,
        vae_path: str,
        vocab_path: str,
        original_model_id: str = "SimianLuo/LCM_Dreamshaper_v7",
    ):
        """Initialize the LCM pipeline"""
        logger.info("Initializing pipeline...")

        if not self.load_tokenizer(vocab_path):
            return False

        if not self.initialize_scheduler(original_model_id):
            return False

        components = {
            "text_encoder": text_encoder_path,
            "unet": unet_path,
            "vae_decoder": vae_path,
        }

        for component, path in components.items():
            if not self.load_model_component(component, path):
                return False

        self._initialized = True
        logger.info("✓ Pipeline ready")
        return True


def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="OpenVINO LCM (Latent Consistency Model) Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python openvino_lcm.py --models_dir ./lcm_models --prompt "sunset over mountains" --steps 4
""",
    )

    parser.add_argument(
        "--models_dir", type=str, required=True, help="Directory containing PTE models"
    )
    parser.add_argument(
        "--prompt", type=str, default="a serene landscape", help="Text prompt"
    )
    parser.add_argument(
        "--steps", type=int, default=4, help="Denoising steps (default: 4)"
    )
    parser.add_argument(
        "--guidance", type=float, default=1.0, help="Guidance scale (default: 1.0)"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--device", choices=["CPU", "GPU"], default="CPU", help="Target device"
    )
    parser.add_argument(
        "--dtype", choices=["fp16", "fp32"], default="fp16", help="Model dtype"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./lcm_outputs", help="Output directory"
    )
    parser.add_argument("--filename", type=str, help="Custom output filename")
    parser.add_argument("--tokenizer_path", type=str, help="Tokenizer path (optional)")
    parser.add_argument(
        "--original_model_id",
        type=str,
        default="SimianLuo/LCM_Dreamshaper_v7",
        help="Model ID for scheduler",
    )

    return parser


def validate_model_files(models_dir: str):
    """Validate required model files exist"""
    for filename in ["text_encoder.pte", "unet.pte", "vae_decoder.pte"]:
        if not os.path.exists(os.path.join(models_dir, filename)):
            logger.error(f"Missing: {filename}")
            return False
    return True


def main():
    """Main execution function"""
    args = create_argument_parser().parse_args()

    if not validate_model_files(args.models_dir):
        return

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    pipeline = OpenVINOLCMPipeline(device=args.device, dtype=dtype)

    if not pipeline.initialize(
        text_encoder_path=os.path.join(args.models_dir, "text_encoder.pte"),
        unet_path=os.path.join(args.models_dir, "unet.pte"),
        vae_path=os.path.join(args.models_dir, "vae_decoder.pte"),
        vocab_path=args.tokenizer_path or "",
        original_model_id=args.original_model_id,
    ):
        return

    image = pipeline.generate_image(args.prompt, args.steps, args.guidance, args.seed)
    if image is None:
        return

    # Save image
    filename = args.filename or "output.jpg"
    if not filename.endswith(".jpg"):
        filename += ".jpg"

    output_path = os.path.join(args.output_dir, filename)
    image.save(output_path)
    logger.info(f"Image saved: {output_path}")


if __name__ == "__main__":
    main()
