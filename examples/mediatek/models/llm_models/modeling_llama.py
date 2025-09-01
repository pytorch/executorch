# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from models.llm_models.configuration_llama import LlamaConfig

from models.llm_models.modeling_common import Attention, DecoderLayer, MLP, ModelChunk

np.random.seed(42)


class LlamaMLP(MLP):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)


class LlamaAttention(Attention):
    def __init__(
        self,
        config: LlamaConfig,
        jit_trace=False,
    ):
        super().__init__(
            config,
            jit_trace,
        )


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
