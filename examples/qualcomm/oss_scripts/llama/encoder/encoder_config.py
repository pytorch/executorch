# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass

from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_quant_recipe import (
    EncoderQuantRecipe,
    GraniteSpeechEncoderQuantRecipe,
    InternVL3EncoderQuantRecipe,
    SmolVLMEncoderQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.audio_encoder import (
    GraniteSpeechCTCEncoderWrapper,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.vision_encoder import (
    Idefics3VisionEncoder,
    InternVL3VisionEncoder,
)


@dataclass(init=False, frozen=True)
class MultiModalityConfig(ABC):
    """
    Base config class for late fusion modality encoders.

    Attributes:
        encoder_class: The encoder class that implements the modality processing.
        quant_recipe: Quantization recipe for optimizing the encoder.
        num_sharding: Specify the number of splits by inserting the fallback custom op. The graph will be split evenly by layers. Only larger encoder needs sharding, e.g., GraniteSpeech with 24 layers. For smaller encoders like SmolVLM with 12 layers, sharding is not necessary.
    """

    encoder_class: type
    quant_recipe: EncoderQuantRecipe
    num_sharding: int = 1

    @abstractmethod
    def create_encoder(self, config):
        pass


@dataclass(init=False, frozen=True)
class AudioModalityConfig(MultiModalityConfig):
    """
    Base config for audio modality encoders.

    Attributes:
        audio_seq_len: Number of audio tokens in the sequence.
    """

    audio_seq_len: int
    n_bins: int

    def create_encoder(self, config):
        return self.encoder_class(config, n_bins=self.n_bins)


@dataclass(init=False, frozen=True)
class VisionModalityConfig(MultiModalityConfig):
    """
    Base config for vision modality encoders.

    Attributes:
        encoder_class: The vision encoder class implementation.
        quant_recipe: Quantization recipe for the vision encoder.
        img_seq_len: Number of image tokens/patches in the sequence.
        img_resized_h: Target height for image resizing (pixels).
        img_resized_w: Target width for image resizing (pixels).
    """

    img_seq_len: int
    img_resized_h: int
    img_resized_w: int

    def create_encoder(self, config):
        return self.encoder_class(
            config, img_resized_h=self.img_resized_h, img_resized_w=self.img_resized_w
        )


@dataclass(init=False, frozen=True)
class GraniteSpeechEncoder(AudioModalityConfig):
    """
    Config for GraniteSpeech audio encoder.
    """

    encoder_class = GraniteSpeechCTCEncoderWrapper
    audio_seq_len = 171
    n_bins = 844
    quant_recipe = GraniteSpeechEncoderQuantRecipe
    num_sharding = 8


@dataclass(init=False, frozen=True)
class SmolVLMEncoder(VisionModalityConfig):
    """
    Config for SmolVLM vision encoder.
    """

    encoder_class = Idefics3VisionEncoder
    img_seq_len = 64
    img_resized_h = 512
    img_resized_w = 512
    quant_recipe = SmolVLMEncoderQuantRecipe


@dataclass(init=False, frozen=True)
class Lfm3OmniVisionEncoder(VisionModalityConfig):
    """
    LFM3-Omni vision tower + projector at a fixed 448x448 input (196 image embeddings).
    Built from the HF checkpoint directly; exported in fp16 (no quant recipe).
    """

    encoder_class = None
    img_seq_len = 196
    img_resized_h = 448
    img_resized_w = 448
    quant_recipe = None
    repo_id = "LiquidAI/LFM3-Omni-2B"
    loads_own_weights = True
    use_fp16 = True

    @classmethod
    def preprocess_image(cls, image):
        from executorch.examples.qualcomm.oss_scripts.lfm3.preprocess import (
            preprocess_pil,
        )

        # [1, 1, patches, patch_dim], as the runner's image path feeds the encoder.
        return preprocess_pil(image, cls.img_resized_h)[None, None]

    def create_encoder(self, config):
        from executorch.examples.qualcomm.oss_scripts.lfm3.vision import (
            Lfm3OmniVisionEncoder as Encoder,
        )

        return Encoder(self.repo_id, self.img_resized_h, self.img_resized_w)


@dataclass(init=False, frozen=True)
class Lfm3OmniAudioEncoder(AudioModalityConfig):
    """
    LFM3-Omni speech encoder + projector on a fixed 10.24 s window (1024 mel frames,
    128 audio embeddings). Input rows: 128 log-mel bins plus a valid-frame row.
    Built from the HF checkpoint directly; exported in fp16 (no quant recipe).
    """

    encoder_class = None
    max_frames = 1024
    audio_seq_len = 128
    n_bins = 129
    quant_recipe = None
    repo_id = "LiquidAI/LFM3-Omni-2B"
    loads_own_weights = True
    use_fp16 = True

    @classmethod
    def preprocess_audio(cls, path):
        from executorch.examples.qualcomm.oss_scripts.lfm3.preprocess import (
            audio_features,
            load_wav,
        )

        return audio_features(load_wav(path), cls.max_frames)

    def create_encoder(self, config):
        from executorch.examples.qualcomm.oss_scripts.lfm3.audio import (
            Lfm3OmniAudioEncoder as Encoder,
        )

        return Encoder(self.repo_id, self.max_frames)


@dataclass(init=False, frozen=True)
class Lfm3VLVisionEncoder(Lfm3OmniVisionEncoder):
    """LFM3-VL vision tower + projector: same architecture as LFM3-Omni's."""

    repo_id = "LiquidAI/LFM3-VL-2B"


@dataclass(init=False, frozen=True)
class InternVL3Encoder(VisionModalityConfig):
    """
    Config for InternVL3 vision encoder.
    """

    encoder_class = InternVL3VisionEncoder
    img_seq_len = 256
    img_resized_h = 448
    img_resized_w = 448
    quant_recipe = InternVL3EncoderQuantRecipe
