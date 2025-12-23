# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass

from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_quant_recipe import (
    EncoderQuantRecipe,
    InternVL3_Encoder_QuantRecipe,
    SmolVLM_Encoder_QuantRecipe,
)

from executorch.examples.qualcomm.oss_scripts.llama.model.vision_encoder import (
    Idefics3VisionEncoder,
    InternVL3VisionEncoder,
)


@dataclass(init=False, frozen=True)
class LateFusionModalityConfig(ABC):
    """
    Base configuration class for late fusion modality encoders.

    Late fusion refers to the approach where different modalities (vision, audio, text)
    are processed separately by their respective encoders and then fused at a later stage.

    Attributes:
        encoder_class: The encoder class that implements the modality processing.
        quant_recipe: Quantization recipe for optimizing the encoder.
    """

    encoder_class: type
    quant_recipe: EncoderQuantRecipe

    @abstractmethod
    def create_encoder(self, config):
        pass


@dataclass(init=False, frozen=True)
class VisionModalityConfig(LateFusionModalityConfig):
    """
    Base config for vision modality encoders.

    Attributes:
        encoder_class: The vision encoder class implementation.
        quant_recipe: Quantization recipe for the vision encoder.
        img_seq_len: Number of image tokens/patches in the sequence.
        img_resized_h: Target height for image resizing (pixels).
        img_resized_w: Target width for image resizing (pixels).
        img_url: Default image URL for validation and calibration.
    """

    img_seq_len: int
    img_resized_h: int
    img_resized_w: int
    img_url: str

    def create_encoder(self, config):
        return self.encoder_class(
            config, img_resized_h=self.img_resized_h, img_resized_w=self.img_resized_w
        )


@dataclass(init=False, frozen=True)
class SmolVLMEncoder(VisionModalityConfig):
    """
    Config for SmolVLM vision encoder.
    """

    encoder_class = Idefics3VisionEncoder
    img_seq_len = 64
    img_resized_h = 512
    img_resized_w = 512
    img_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
    quant_recipe = SmolVLM_Encoder_QuantRecipe


@dataclass(init=False, frozen=True)
class InternVL3Encoder(VisionModalityConfig):
    """
    Config for InternVL3 vision encoder.
    """

    encoder_class = InternVL3VisionEncoder
    img_seq_len = 256
    img_resized_h = 448
    img_resized_w = 448
    img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    quant_recipe = InternVL3_Encoder_QuantRecipe
