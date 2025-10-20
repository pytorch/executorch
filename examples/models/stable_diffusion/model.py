# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

"""
Stable Diffusion / LCM model definitions.

This module provides reusable model wrappers that can be used with any backend
(OpenVINO, XNNPACK, etc.) for exporting Latent Consistency Models.
"""

import logging
from typing import Any, Optional

import torch

try:
    from diffusers import DiffusionPipeline
except ImportError:
    raise ImportError(
        "Please install diffusers and transformers: pip install diffusers transformers"
    )

logger = logging.getLogger(__name__)


class TextEncoderWrapper(torch.nn.Module):
    """Wrapper for CLIP text encoder that extracts last_hidden_state"""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        # Call text encoder and extract last_hidden_state
        output = self.text_encoder(input_ids, return_dict=True)
        return output.last_hidden_state


class UNetWrapper(torch.nn.Module):
    """Wrapper for UNet that extracts sample tensor from output"""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latents, timestep, encoder_hidden_states):
        # Call UNet and extract sample from the output
        output = self.unet(latents, timestep, encoder_hidden_states, return_dict=True)
        return output.sample


class VAEDecoder(torch.nn.Module):
    """Wrapper for VAE decoder with scaling and normalization"""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        # Scale latents
        latents = latents / self.vae.config.scaling_factor
        # Decode
        image = self.vae.decode(latents).sample
        # Scale to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image


class LCMModelLoader:
    """
    Backend-agnostic loader for Latent Consistency Model components.

    This class handles loading the LCM pipeline from HuggingFace and extracting
    individual components (text_encoder, unet, vae) as PyTorch modules ready
    for export to any backend.
    """

    def __init__(
        self,
        model_id: str = "SimianLuo/LCM_Dreamshaper_v7",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the LCM model loader.

        Args:
            model_id: HuggingFace model ID for the LCM model
            dtype: Target dtype for the models (fp16 or fp32)
        """
        self.model_id = model_id
        self.dtype = dtype
        self.pipeline: Optional[DiffusionPipeline] = None
        self.text_encoder: Any = None
        self.unet: Any = None
        self.vae: Any = None
        self.tokenizer: Any = None

    def load_models(self) -> bool:
        """
        Load the LCM pipeline and extract components.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading LCM pipeline: {self.model_id} (dtype: {self.dtype})")
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_id, use_safetensors=True
            )

            # Extract individual components and convert to desired dtype
            self.text_encoder = self.pipeline.text_encoder.to(dtype=self.dtype)
            self.unet = self.pipeline.unet.to(dtype=self.dtype)
            self.vae = self.pipeline.vae.to(dtype=self.dtype)
            self.tokenizer = self.pipeline.tokenizer

            # Set models to evaluation mode
            self.text_encoder.eval()
            self.unet.eval()
            self.vae.eval()

            logger.info("Successfully loaded all LCM model components")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_text_encoder_wrapper(self) -> TextEncoderWrapper:
        """Get wrapped text encoder ready for export"""
        if self.text_encoder is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        return TextEncoderWrapper(self.text_encoder)

    def get_unet_wrapper(self) -> UNetWrapper:
        """Get wrapped UNet ready for export"""
        if self.unet is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        return UNetWrapper(self.unet)

    def get_vae_decoder(self) -> VAEDecoder:
        """Get wrapped VAE decoder ready for export"""
        if self.vae is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        return VAEDecoder(self.vae)

    def get_dummy_inputs(self):
        """
        Get dummy inputs for each model component.

        Returns:
            Dictionary with dummy inputs for text_encoder, unet, and vae_decoder
        """
        if self.unet is None:
            raise ValueError("Models not loaded. Call load_models() first.")

        # Text encoder dummy input
        text_encoder_input = torch.ones(1, 77, dtype=torch.long)

        # UNet dummy inputs
        batch_size = 1
        latent_channels = 4
        latent_height = 64
        latent_width = 64
        text_embed_dim = self.unet.config.cross_attention_dim
        text_seq_len = 77

        unet_inputs = (
            torch.randn(
                batch_size,
                latent_channels,
                latent_height,
                latent_width,
                dtype=self.dtype,
            ),
            torch.tensor([981]),  # Random timestep
            torch.randn(batch_size, text_seq_len, text_embed_dim, dtype=self.dtype),
        )

        # VAE decoder dummy input
        vae_input = torch.randn(1, 4, 64, 64, dtype=self.dtype)

        return {
            "text_encoder": (text_encoder_input,),
            "unet": unet_inputs,
            "vae_decoder": (vae_input,),
        }
