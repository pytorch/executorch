# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""

import numpy as np
import torch
from models.llm_models.configuration_llama import LlamaConfig

from models.llm_models.modeling_common import Attention, DecoderLayer, MLP, ModelChunk

np.random.seed(42)


class LlamaMLP(MLP):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)


class LlamaAttention(Attention):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)


class LlamaDecoderLayer(DecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        return_attn=False,
        jit_trace=False,
    ):
        super().__init__(
            config,
            return_attn,
            jit_trace,
            attn_class=LlamaAttention,
            mlp_class=LlamaMLP,
        )


class LlamaModelChunk(ModelChunk):
    def __init__(
        self,
        config: LlamaConfig,
        num_blocks,
        chunk_idx,
        dtype=torch.float32,
        include_tail=False,
        return_attn=False,
        jit_trace=False,
    ):
        super().__init__(
            config,
            num_blocks,
            chunk_idx,
            dtype,
            include_tail,
            return_attn,
            jit_trace,
            decoder_class=LlamaDecoderLayer,
        )
