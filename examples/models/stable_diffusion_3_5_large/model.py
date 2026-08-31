# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Iterable
from enum import Enum
from typing import Any, Optional

import torch
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel


logger = logging.getLogger(__name__)

MODEL_ID = "stabilityai/stable-diffusion-3.5-large"


def _subfolder_for_model_repo(
    repo_id: str, model_id: str, subfolder: str
) -> str | None:
    return subfolder if repo_id == model_id else None


def _optional_subfolder_kwargs(subfolder: str | None) -> dict[str, str]:
    return {"subfolder": subfolder} if subfolder else {}


class StableDiffusionComponent(Enum):
    """Stable Diffusion component names used by this exporter."""

    TEXT_ENCODER = "text_encoder"
    TEXT_ENCODER_2 = "text_encoder_2"
    TEXT_ENCODER_3 = "text_encoder_3"
    TRANSFORMER = "transformer"
    VAE_DECODER = "vae_decoder"


class SD3CLIPTextEncoderWrapper(torch.nn.Module):
    """Wrapper for SD3 CLIP text encoders."""

    def __init__(self, text_encoder, clip_skip: Optional[int] = None):
        super().__init__()
        self.text_encoder = text_encoder
        self.clip_skip = clip_skip

    def forward(self, input_ids):
        """Forward pass for CLIP text encoder."""
        output = self.text_encoder(
            input_ids, output_hidden_states=True, return_dict=True
        )
        hidden_state_index = -2 if self.clip_skip is None else -(self.clip_skip + 2)
        return output.hidden_states[hidden_state_index], output[0]


class SD3T5TextEncoderWrapper(torch.nn.Module):
    """Wrapper for SD3 T5 text encoder."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        """Forward pass for T5 text encoder."""
        output = self.text_encoder(input_ids, return_dict=True)
        return output.last_hidden_state


class SD3TransformerWrapper(torch.nn.Module):
    """Wrapper for SD3 transformer denoiser that extracts sample tensor."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        latents,
        timestep,
        encoder_hidden_states,
        pooled_projections,
    ):
        """Forward pass through the transformer denoiser.

        Args:
            latents: Input latent tensor
            timestep: Timestep for denoising
            encoder_hidden_states: Hidden states from text encoder
            pooled_projections: Pooled projection embeddings

        Returns:
            Sample output tensor from transformer
        """
        output = self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            return_dict=True,
        )
        return output.sample


class SD3VAEDecoderWrapper(torch.nn.Module):
    """Wrapper for SD3 VAE decoder with scaling, shift, and normalization."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        """Decode latents to image using VAE decoder with scaling and normalization."""
        latents = latents / self.vae.config.scaling_factor
        shift_factor = getattr(self.vae.config, "shift_factor", None)
        if shift_factor is not None:
            latents = latents + shift_factor
        image = self.vae.decode(latents, return_dict=True).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image


class StableDiffusion3ModelLoader:
    """Load SD3 components and construct export wrappers locally."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        text_encoder_id: Optional[str] = None,
        text_encoder_2_id: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.model_id = model_id
        self.text_encoder_id = text_encoder_id or model_id
        self.text_encoder_2_id = text_encoder_2_id or model_id
        self.dtype = dtype
        self.text_encoder: Any = None
        self.text_encoder_2: Any = None
        self.text_encoder_3: Any = None
        self.transformer: Any = None
        self.vae: Any = None
        self.tokenizer: Any = None
        self.tokenizer_2: Any = None

    def _load_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.text_encoder_id,
            **_optional_subfolder_kwargs(
                _subfolder_for_model_repo(
                    self.text_encoder_id,
                    self.model_id,
                    "tokenizer",
                )
            ),
        )

    def _load_text_encoder(self) -> None:
        if self.text_encoder is not None:
            return

        text_encoder_subfolder = _subfolder_for_model_repo(
            self.text_encoder_id,
            self.model_id,
            "text_encoder",
        )
        logger.info(
            "Loading CLIP text encoder: %s%s (dtype: %s)",
            self.text_encoder_id,
            f"/{text_encoder_subfolder}" if text_encoder_subfolder else "",
            self.dtype,
        )
        self._load_tokenizer()
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            self.text_encoder_id,
            torch_dtype=self.dtype,
            **_optional_subfolder_kwargs(text_encoder_subfolder),
        )

    def _load_text_encoder_2(self) -> None:
        if self.text_encoder_2 is not None:
            return

        tokenizer_2_subfolder = _subfolder_for_model_repo(
            self.text_encoder_2_id,
            self.model_id,
            "tokenizer_2",
        )
        text_encoder_2_subfolder = _subfolder_for_model_repo(
            self.text_encoder_2_id,
            self.model_id,
            "text_encoder_2",
        )
        logger.info(
            "Loading CLIP text encoder 2: %s%s (dtype: %s)",
            self.text_encoder_2_id,
            f"/{text_encoder_2_subfolder}" if text_encoder_2_subfolder else "",
            self.dtype,
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.text_encoder_2_id,
            **_optional_subfolder_kwargs(tokenizer_2_subfolder),
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.text_encoder_2_id,
            torch_dtype=self.dtype,
            **_optional_subfolder_kwargs(text_encoder_2_subfolder),
        )

    def _load_text_encoder_3(self) -> None:
        if self.text_encoder_3 is not None:
            return

        logger.info(
            "Loading T5 text encoder: %s/text_encoder_3 (dtype: %s)",
            self.model_id,
            self.dtype,
        )
        self.text_encoder_3 = T5EncoderModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder_3",
            torch_dtype=self.dtype,
        )

    def _load_transformer(self) -> None:
        if self.transformer is not None:
            return

        logger.info(
            "Loading SD3 transformer: %s/transformer (dtype: %s)",
            self.model_id,
            self.dtype,
        )
        self._load_tokenizer()
        self.transformer = SD3Transformer2DModel.from_pretrained(
            self.model_id,
            subfolder="transformer",
            torch_dtype=self.dtype,
        )

    def _load_vae(self) -> None:
        if self.vae is not None:
            return

        logger.info(
            "Loading VAE: %s/vae (dtype: %s)",
            self.model_id,
            self.dtype,
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=self.dtype,
        )

    def load_models(
        self, components: Optional[Iterable[StableDiffusionComponent]] = None
    ) -> bool:
        """Load the requested SD3 components."""
        try:
            requested_components = set(
                StableDiffusionComponent if components is None else components
            )
            if StableDiffusionComponent.TEXT_ENCODER in requested_components:
                self._load_text_encoder()
            if StableDiffusionComponent.TEXT_ENCODER_2 in requested_components:
                self._load_text_encoder_2()
            if StableDiffusionComponent.TEXT_ENCODER_3 in requested_components:
                self._load_text_encoder_3()
            if StableDiffusionComponent.TRANSFORMER in requested_components:
                self._load_transformer()
            if StableDiffusionComponent.VAE_DECODER in requested_components:
                self._load_vae()

            for model in (
                self.text_encoder,
                self.text_encoder_2,
                self.text_encoder_3,
                self.transformer,
                self.vae,
            ):
                if model is not None:
                    model.to(dtype=self.dtype)
                    model.eval()

            logger.info("Successfully loaded requested SD3 model components")
            return True
        except (OSError, ValueError, RuntimeError, ImportError) as e:
            logger.exception("Failed to load SD3 models: %s", e)
            return False

    def get_text_encoder_wrapper(
        self, clip_skip: Optional[int] = None
    ) -> SD3CLIPTextEncoderWrapper:
        """Get wrapped first CLIP text encoder ready for export."""
        if self.text_encoder is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        return SD3CLIPTextEncoderWrapper(self.text_encoder, clip_skip=clip_skip)

    def get_text_encoder_2_wrapper(
        self, clip_skip: Optional[int] = None
    ) -> SD3CLIPTextEncoderWrapper:
        """Get wrapped second CLIP text encoder ready for export."""
        if self.text_encoder_2 is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        return SD3CLIPTextEncoderWrapper(self.text_encoder_2, clip_skip=clip_skip)

    def get_text_encoder_3_wrapper(self) -> SD3T5TextEncoderWrapper:
        """Get wrapped T5 text encoder ready for export."""
        if self.text_encoder_3 is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        return SD3T5TextEncoderWrapper(self.text_encoder_3)

    def get_transformer_wrapper(self) -> SD3TransformerWrapper:
        """Get wrapped SD3 transformer ready for export."""
        if self.transformer is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        return SD3TransformerWrapper(self.transformer)

    def get_vae_decoder_wrapper(self) -> SD3VAEDecoderWrapper:
        """Get wrapped SD3 VAE decoder ready for export."""
        if self.vae is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        return SD3VAEDecoderWrapper(self.vae)

    def get_dummy_inputs(
        self,
        max_sequence_length: int = 256,
        latent_size: Optional[int] = None,
    ) -> dict[StableDiffusionComponent, tuple[Any, ...]]:
        """Get dummy inputs for each wrapped SD3 component."""
        if not any(
            (
                self.text_encoder,
                self.text_encoder_2,
                self.text_encoder_3,
                self.transformer,
                self.vae,
            )
        ):
            raise ValueError("Models not loaded. Call load_models() first.")

        batch_size = 1
        tokenizer_max_length = (
            self.tokenizer.model_max_length if self.tokenizer is not None else 77
        )
        text_seq_len = tokenizer_max_length + max_sequence_length

        dummy_inputs: dict[StableDiffusionComponent, tuple[Any, ...]] = {}
        if self.text_encoder is not None:
            dummy_inputs[StableDiffusionComponent.TEXT_ENCODER] = (
                torch.randn(batch_size, tokenizer_max_length)
                .abs()
                .round()
                .to(dtype=torch.long),
            )
        if self.text_encoder_2 is not None:
            dummy_inputs[StableDiffusionComponent.TEXT_ENCODER_2] = (
                torch.randn(batch_size, tokenizer_max_length)
                .abs()
                .round()
                .to(dtype=torch.long),
            )
        if self.text_encoder_3 is not None:
            dummy_inputs[StableDiffusionComponent.TEXT_ENCODER_3] = (
                torch.randn(batch_size, max_sequence_length)
                .abs()
                .round()
                .to(dtype=torch.long),
            )

        if self.transformer is not None:
            latent_channels = self.transformer.config.in_channels
            transformer_latent_size = latent_size or self.transformer.config.sample_size
            joint_attention_dim = self.transformer.config.joint_attention_dim
            pooled_projection_dim = self.transformer.config.pooled_projection_dim
            dummy_inputs[StableDiffusionComponent.TRANSFORMER] = (
                torch.randn(
                    batch_size,
                    latent_channels,
                    transformer_latent_size,
                    transformer_latent_size,
                    dtype=self.dtype,
                ),
                torch.tensor([1.0], dtype=torch.float32),
                torch.randn(
                    batch_size,
                    text_seq_len,
                    joint_attention_dim,
                    dtype=self.dtype,
                ),
                torch.randn(batch_size, pooled_projection_dim, dtype=self.dtype),
            )

        if self.vae is not None:
            vae_latent_size = latent_size or (
                self.transformer.config.sample_size
                if self.transformer is not None
                else 128
            )
            vae_latent_channels = getattr(self.vae.config, "latent_channels", 16)
            dummy_inputs[StableDiffusionComponent.VAE_DECODER] = (
                torch.randn(
                    batch_size,
                    vae_latent_channels,
                    vae_latent_size,
                    vae_latent_size,
                    dtype=self.dtype,
                ),
            )
        return dummy_inputs
