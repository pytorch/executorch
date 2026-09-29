# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig
from executorch.examples.qualcomm.oss_scripts.llama.dataset.loaders import (
    load_audio_file,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_config import (
    AudioModalityConfig,
    VisionModalityConfig,
)
from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import (
    ALM_SPECIAL_TOKENS,
    AUDIO_TOKEN,
    IMG_TOKEN,
    VLM_SPECIAL_TOKENS,
)

from transformers import AutoProcessor
from transformers.image_utils import load_image


class ModalityPreprocessor:
    """Wraps AutoProcessor for one model repo; caches the processor across calls."""

    def __init__(self, repo_id: str):
        self._processor = AutoProcessor.from_pretrained(repo_id)

    def preprocess_audio(
        self,
        config: AudioModalityConfig,
        prompt: str,
        wav: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # Pad to fixed length so input_features has shape [1, n_bins, input_dim]
        hop_length = self._processor.audio_processor.melspec_kwargs["hop_length"]
        target_len = (config.n_bins * 2 - 1) * hop_length
        pad = target_len - wav.shape[-1]
        if pad > 0:
            wav = F.pad(wav, (0, pad))
        elif pad < 0:
            suggested = (wav.shape[-1] // hop_length + 1) // 2
            raise ValueError(
                f"Audio length ({wav.shape[-1]} samples) exceeds target ({target_len} samples) "
                f"derived from n_bins={config.n_bins}. Set n_bins >= {suggested} in the config."
            )
        return (self._processor(prompt, wav, return_tensors="pt").input_features,)

    def preprocess_images(
        self,
        config: VisionModalityConfig,
        prompt: str,
        images: list,
    ) -> List[Tuple[torch.Tensor, ...]]:
        """Applies the vision processor to all images → list of (pixel_values,) per patch."""
        pixel_values = self._processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            crop_to_patches=False,
            size={"height": config.img_resized_h, "width": config.img_resized_w},
        ).pixel_values
        assert pixel_values.dim() in (
            4,
            5,
        ), f"Unsupported pixel_values dim={pixel_values.dim()}; expected 4D or 5D."
        # HTP Prepare failed when pixel_values has 5D dimension
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(0)  # (N, C, H, W)
        return [(pixel_values[i][None, ...],) for i in range(len(pixel_values))]


def preprocess_encoder_inputs(
    config: LLMModelConfig,
    decoder_model: str,
    audio_paths: List[str],
    image_paths: List[str],
) -> Dict[str, List[Tuple[torch.Tensor, ...]]]:
    """
    Preprocess raw encoder inputs for runtime inference.

    Separated from EncoderDatasetBuilder (calibration-only) because this is
    a runtime inference concern, not a calibration data concern.
    """
    preprocessor = ModalityPreprocessor(config.repo_id)
    result = {}
    for modality, paths in (
        (AUDIO_ENCODER, audio_paths),
        (VISION_ENCODER, image_paths),
    ):
        if not hasattr(config, modality) or not paths:
            continue
        modality_config = getattr(config, modality)
        if issubclass(modality_config, AudioModalityConfig):
            token = ALM_SPECIAL_TOKENS[decoder_model][AUDIO_TOKEN]
            normalized_prompt = token * len(paths)
            result[modality] = [
                preprocessor.preprocess_audio(
                    modality_config,
                    normalized_prompt,
                    load_audio_file(p, config.repo_id),
                )
                for p in paths
            ]
        elif issubclass(modality_config, VisionModalityConfig):
            token = VLM_SPECIAL_TOKENS[decoder_model][IMG_TOKEN]
            normalized_prompt = token * len(paths)
            result[modality] = preprocessor.preprocess_images(
                modality_config, normalized_prompt, [load_image(p) for p in paths]
            )
    return result
