# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .attention_sink import AttentionSinkRope
from .blocks import (
    ATTENTION_REGISTRY,
    DECODER_LAYER_REGISTRY,
    FEED_FORWARD_REGISTRY,
    FeedForward,
    FeedForwardBase,
    Gemma4Attention,
    LayerNorm,
    LlamaAttention,
    LlamaDecoderLayer,
    Norm,
    NORM_REGISTRY,
    register_attention,
    register_decoder_layer,
    register_feed_forward,
    register_norm,
    repeat_kv,
    RMSNorm,
    YOCOCrossDecoderLayer,
)
from .embedding import TokenEmbedding
from .rope import (
    register_rope,
    register_rope_precompute,
    ROPE_PRECOMPUTE_REGISTRY,
    ROPE_REGISTRY,
    RopeFreqs,
)
from .static_llama import (
    LlamaModel,
    LlamaModelWithoutEmbedding,
    MultiScopeAwareLlamaModel,
)


__all__ = [
    "AttentionSinkRope",
    "FEED_FORWARD_REGISTRY",
    "FeedForward",
    "FeedForwardBase",
    "register_feed_forward",
    "NORM_REGISTRY",
    "Norm",
    "LayerNorm",
    "RMSNorm",
    "register_norm",
    "ROPE_REGISTRY",
    "ROPE_PRECOMPUTE_REGISTRY",
    "RopeFreqs",
    "register_rope",
    "register_rope_precompute",
    "repeat_kv",
    "LlamaAttention",
    "LlamaDecoderLayer",
    "TokenEmbedding",
    "LlamaModel",
    "LlamaModelWithoutEmbedding",
    "MultiScopeAwareLlamaModel",
    "Gemma4Attention",
    "ATTENTION_REGISTRY",
    "register_attention",
    "YOCOCrossDecoderLayer",
    "DECODER_LAYER_REGISTRY",
    "register_decoder_layer",
]
