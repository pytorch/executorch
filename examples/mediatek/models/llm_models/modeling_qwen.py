import numpy as np
import torch
from torch import nn
from models.llm_models.configuration_qwen import QwenConfig

from models.llm_models.modeling_common import Attention, DecoderLayer, MLP, ModelChunk, RMSNorm

np.random.seed(42)

class QwenMLP(MLP):
    def __init__(self, config: QwenConfig):
        super().__init__(config)

class QwenAttention(Attention):
    def __init__(self, config:QwenConfig, jit_trace=False):
        super().__init__(config, jit_trace)


class Qwen3Attention(Attention):
    def __init__(self, config: QwenConfig, jit_trace=False):
        super().__init__(config, jit_trace)


class QwenDecoderLayer(DecoderLayer):
    def __init__(
        self,
        config: QwenConfig,
        return_attn=False,
        jit_trace=False,
    ):
        super().__init__(
            config,
            return_attn,
            jit_trace,
            attn_class=QwenAttention,
            mlp_class=QwenMLP,
        )


class Qwen3DecoderLayer(DecoderLayer):
    def __init__(
        self,
        config: QwenConfig,
        return_attn=False,
        jit_trace=False,
    ):
        super().__init__(
            config,
            return_attn,
            jit_trace,
            attn_class=Qwen3Attention,
            mlp_class=QwenMLP,
        )

class Qwen2ModelChunk(ModelChunk):
    def __init__(
        self,
        config: QwenConfig,
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
            decoder_class=QwenDecoderLayer,
        )

class Qwen3ModelChunk(ModelChunk):
    def __init__(
        self,
        config: QwenConfig,
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
            decoder_class=Qwen3DecoderLayer,
        )

    