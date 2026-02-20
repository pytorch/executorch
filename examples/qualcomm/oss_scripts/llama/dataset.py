# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import warnings
from typing import Callable, List, Optional

from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    TEXT_DECODER,
    TEXT_EMBEDDING,
    TEXT_ENCODER,
    VISION_ENCODER,
    VISION_ENCODER_INPUT_FILENAME,
)

from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_config import (
    MultiModalityConfig,
    VisionModalityConfig,
)
from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper

from transformers import AutoProcessor
from transformers.image_utils import load_image


class DatasetBuilder:
    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        tokenizer_wrapper: TokenizerWrapper,
    ):
        # Directory path where processed data artifacts will be saved
        self.control_args = control_args
        self.config = config
        self.tokenizer_wrapper = tokenizer_wrapper

        self.artifact = control_args.artifact
        self.repo_id = config.repo_id

    def _build_vision_dataset(self, config: VisionModalityConfig, prompt: str):
        """
        This will processes images using the HuggingFace processor and saves
        the processed pixel values for runtime evaluation.

        Args:
            config (VisionModalityConfig): containing image URL and resize parameters
            prompt (str): Text prompt to be processed alongside the image

        Returns:
            tuple of pixel values tensors
        """
        # Load image from user-specified path (URL or local file)
        # fall back to the default image URL if no image is provided.
        image_path = self.control_args.image_path or config.img_url
        if not self.control_args.image_path:
            warnings.warn(
                f"No image path/URL provided, using default image URL: {config.img_url}",
                UserWarning,
                stacklevel=1,
            )
        image = load_image(image_path)

        # Process image with text prompt using HuggingFace processor
        # Some HF processors (e.g. InternVL3) need to pass text arg or it will cause error and process failed
        processor = AutoProcessor.from_pretrained(self.repo_id)
        pixel_values = processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
            crop_to_patches=False,
            size={
                "height": config.img_resized_h,
                "width": config.img_resized_w,
            },
        ).pixel_values

        # save image file for runtime evaluation
        pixel_values.detach().numpy().tofile(
            f"{self.artifact}/{VISION_ENCODER_INPUT_FILENAME}.raw"
        )
        return (pixel_values,)

    def _build_dataset_for_encoder(
        self,
        config: MultiModalityConfig,
        prompt: str,
    ) -> Optional[tuple]:
        if issubclass(config, VisionModalityConfig):
            return self._build_vision_dataset(config, prompt)
        else:
            # Audio and text encoder dataset building are not yet implemented
            # TODO: Add support for AudioModalityConfig and TextModalityConfig
            raise NotImplementedError(
                f"Dataset building for {config} is not yet supported. "
                f"Currently only VisionModalityConfig is implemented."
            )

    def prepare_calibration_dataset(
        self,
        prompts: List[str],
        chat_template: Callable,
    ):
        calibration_data = {
            AUDIO_ENCODER: [],
            TEXT_ENCODER: [],
            VISION_ENCODER: [],
            TEXT_EMBEDDING: [],
            TEXT_DECODER: [],
        }

        is_multimodal = any(
            [
                hasattr(self.config, AUDIO_ENCODER),
                hasattr(self.config, VISION_ENCODER),
            ]
        )
        for prompt in prompts:
            # Apply chat template formatting if available (for instruction-tuned/reasoning models)
            prompt = (
                self.tokenizer_wrapper.apply_prompt_template(
                    chat_template, prompt, self.control_args.system_prompt
                )
                if chat_template is not None
                else prompt
            )

            # Build calibration datasets for each available encoder modality
            for modality in [AUDIO_ENCODER, TEXT_ENCODER, VISION_ENCODER]:
                if hasattr(self.config, modality):
                    data = self._build_dataset_for_encoder(
                        getattr(self.config, modality),
                        prompt,
                    )
                    calibration_data[modality].append(data)

            # Expand multimodal tokens in prompt for decoder
            prompt = (
                self.tokenizer_wrapper.prepare_multimodal_prompt(prompt)
                if is_multimodal
                else prompt
            )

            # Add prompt to decoder calibration data
            calibration_data[TEXT_DECODER].append(prompt)

        return calibration_data
