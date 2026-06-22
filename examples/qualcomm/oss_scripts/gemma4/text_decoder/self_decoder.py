# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from executorch.examples.models.gemma4.text_decoder.gemma4_config import Gemma4Config
from executorch.examples.qualcomm.oss_scripts.gemma4.text_decoder.decoder_layer import (
    Gemma4DecoderLayer,
)
from executorch.examples.qualcomm.oss_scripts.llama.model import NORM_REGISTRY
from torch import nn


class Gemma4SelfDecoder(nn.Module):
    def __init__(
        self,
        config: Gemma4Config,
        output_new_cache_only: bool = True,
        enable_masked_softmax: bool = False,
    ):
        super().__init__()
        self.config = config
        self.num_layers = config.num_self_decoder_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = math.sqrt(config.hidden_size)

        self.embed_tokens_per_layer = nn.Embedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
        )
        self.embed_scale_per_layer = math.sqrt(config.hidden_size_per_layer_input)

        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection_norm = NORM_REGISTRY["rmsnorm"](
            config.hidden_size_per_layer_input, eps=config.rms_norm_eps
        )

        self._ple_input_scale = 1.0 / math.sqrt(2.0)
        self._ple_proj_scale = 1.0 / math.sqrt(config.hidden_size)

        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    config, i, output_new_cache_only, enable_masked_softmax
                )
                for i in range(self.num_layers)
            ]
        )

    def compute_per_layer_inputs(
        self,
        tokens: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        n_layers = self.config.num_hidden_layers
        hpl = self.config.hidden_size_per_layer_input

        per_layer_proj = (
            self.per_layer_model_projection(hidden_states) * self._ple_proj_scale
        )
        per_layer_proj = per_layer_proj.view(bsz, seq_len, n_layers, hpl)
        per_layer_proj = self.per_layer_projection_norm(per_layer_proj)
        per_layer_embed = (
            self.embed_tokens_per_layer(tokens) * self.embed_scale_per_layer
        )
        per_layer_embed = per_layer_embed.view(bsz, seq_len, n_layers, hpl)

        combined = (per_layer_proj + per_layer_embed) * self._ple_input_scale
        return combined.permute(2, 0, 1, 3)
