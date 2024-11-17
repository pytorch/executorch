# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

import torch

from executorch.examples.models.model_base import EagerModelBase
from executorch.extension.llm.modules._position_embeddings import (
    replace_tile_positional_embedding,
    replace_tiled_token_positional_embedding,
)
from torchtune.models.llama3_2_vision._component_builders import llama3_2_vision_encoder


@dataclass
class VisionEncoderConfig:
    patch_size: int = 14
    num_heads: int = 16
    clip_embed_dim: int = 1280
    clip_num_layers: int = 32
    clip_hidden_states: list[int] = field(default_factory=lambda: [3, 7, 15, 23, 30])
    decoder_embed_dim: int = 4096
    num_layers_projection: int = 8
    tile_size: int = 560
    max_num_tiles: int = 4
    in_channels: int = 3


# 8 layers for CI testing purpose
demo_config: VisionEncoderConfig = VisionEncoderConfig(
    patch_size=14,
    num_heads=8,
    clip_embed_dim=768,
    clip_num_layers=6,
    clip_hidden_states=[1, 3, 5],
    decoder_embed_dim=1024,
    num_layers_projection=4,
    tile_size=224,
    max_num_tiles=4,
    in_channels=3,
)


class FlamingoVisionEncoderModel(EagerModelBase):
    def __init__(self, config: Optional[VisionEncoderConfig] = None):
        super().__init__()
        if config is None:
            config = demo_config
        self.config = config
        self.model = llama3_2_vision_encoder(
            patch_size=config.patch_size,
            num_heads=config.num_heads,
            clip_embed_dim=config.clip_embed_dim,
            clip_num_layers=config.clip_num_layers,
            clip_hidden_states=config.clip_hidden_states,
            decoder_embed_dim=config.decoder_embed_dim,
            num_layers_projection=config.num_layers_projection,
            tile_size=config.tile_size,
            max_num_tiles=config.max_num_tiles,
            in_channels=config.in_channels,
        )
        self.model = replace_tile_positional_embedding(self.model)
        self.model = replace_tiled_token_positional_embedding(self.model)
        self.image = torch.randn(
            1, 1, 4, 3, self.config.tile_size, self.config.tile_size
        )
        self.aspect_ratio = torch.tensor([[[1, 2]]])
        self.sample_inputs = (
            self.image,
            self.aspect_ratio,
        )

    def get_eager_model(self, **kwargs):
        return self.model

    def get_example_inputs(self):
        return self.sample_inputs

    def get_dynamic_shapes(self):
        dim = torch.export.Dim("num_tiles", min=1, max=self.config.max_num_tiles)
        image_dynamic_dim = {
            0: 1,
            1: 1,
            2: dim,
            3: 3,
            4: self.config.tile_size,
            5: self.config.tile_size,
        }
        return (image_dynamic_dim, None)
