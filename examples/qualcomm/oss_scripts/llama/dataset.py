# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Callable, Dict, List, Optional

from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    TEXT_DECODER,
    TEXT_ENCODER,
    TOK_EMBEDDING,
    VISION_ENCODER,
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

    def _build_vision_dataset(
        self, config: VisionModalityConfig, prompt: str, files_path: List[str]
    ):
        """
        This will processes images using the HuggingFace processor and saves
        the processed pixel values for runtime evaluation.

        Args:
            config (VisionModalityConfig): containing image URL and resize parameters
            prompt (str): Text prompt
            files_path (List[str]): List of file paths for images. Each path can be either a URL or a local file path.

        Returns:
            tuple of pixel values tensors
        """

        images = [load_image(image_path) for image_path in files_path]

        # Process image with text prompt using HuggingFace processor
        # Some HF processors (e.g. InternVL3) need to pass text arg or it will cause error and process failed
        processor = AutoProcessor.from_pretrained(self.repo_id)
        pixel_values = processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            crop_to_patches=False,
            size={
                "height": config.img_resized_h,
                "width": config.img_resized_w,
            },
        ).pixel_values

        assert pixel_values.dim() in (4, 5), (
            f"Unsupported pixel_values dim={pixel_values.dim()}); "
            f"expected 5D (1,N,C,H,W) or 4D (N,C,H,W)."
        )

        # HTP Prepare failed when pixel_values has 5D dimension, so we squeeze the batch dimension here.
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(0)  # (N, C, H, W)

        # save image file for runtime evaluation
        return [(pixel_values[i][None, ...],) for i in range(len(pixel_values))]

    def _build_dataset_for_encoder(
        self,
        config: MultiModalityConfig,
        prompt: str,
        files_path: List[str],
    ) -> Optional[tuple]:
        if issubclass(config, VisionModalityConfig):
            return self._build_vision_dataset(config, prompt, files_path)
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
        # 1. Initialize data
        # Shape convention: (num_samples, num_turns).
        # Currently, user prompt calibration is one-shot per prompt (num_samples = 1).
        calibration_data: Dict[str, List[List]] = {
            # Encoders / embeddings: initialize an empty turn list for each prompt.
            AUDIO_ENCODER: [[] for _ in range(len(prompts))],
            TEXT_ENCODER: [[] for _ in range(len(prompts))],
            VISION_ENCODER: [[] for _ in range(len(prompts))],
            TOK_EMBEDDING: [[] for _ in range(len(prompts))],
            # Decoder targets: one string per prompt.
            TEXT_DECODER: ["" for _ in range(len(prompts))],
        }

        # 2. Prepare messages for multi-turn conversation
        messages = self.tokenizer_wrapper.prepare_messages(prompts)

        # 3. build dataset by modality
        is_multimodal = any(
            [
                hasattr(self.config, AUDIO_ENCODER),
                hasattr(self.config, VISION_ENCODER),
            ]
        )
        for turn_idx, message in enumerate(messages):
            prompt = message["text"]

            # 3.1. Apply chat template formatting if available (for instruction-tuned/reasoning models)
            prompt = (
                self.tokenizer_wrapper.apply_prompt_template(
                    chat_template, prompt, self.control_args.system_prompt
                )
                if chat_template is not None
                else prompt
            )

            # 3.2 Build calibration datasets for each available encoder modality
            for modality in [AUDIO_ENCODER, TEXT_ENCODER, VISION_ENCODER]:
                if not hasattr(self.config, modality) or not message["files_path"]:
                    continue

                data = self._build_dataset_for_encoder(
                    getattr(self.config, modality),
                    prompt,
                    message["files_path"],
                )
                calibration_data[modality][turn_idx] = data

            # 3.3. Expand multimodal tokens in prompt for decoder
            prompt = (
                self.tokenizer_wrapper.prepare_multimodal_prompt(prompt)
                if is_multimodal
                else prompt
            )

            # Add prompt to decoder calibration data
            calibration_data[TEXT_DECODER][turn_idx] = prompt

        return calibration_data
