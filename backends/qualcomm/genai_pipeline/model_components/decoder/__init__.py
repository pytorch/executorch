# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Transformer decoder architectures and their architectural primitives.

The decoder variants differ in their input representation or attention pattern:

* :class:`LlamaModel` consumes token IDs with a standard causal decoder.
* :class:`LlamaModelWithoutEmbedding` consumes precomputed token embeddings.
* :class:`MultiScopeAwareLlamaModel` interleaves local and global attention.

The registries parameterize the shared decoder with architecture-specific
normalization, feed-forward, and rotary-position-embedding implementations.
This captures model-family variation without duplicating the decoder stack.

"""


from executorch.examples.qualcomm.oss_scripts.llama.model.apply_rope import (
    apply_partial_rotary_emb_single,
    apply_rotary_emb_single,
    register_rotary_emb,
    ROTARY_EMB_REGISTRY,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.feed_forward import (
    CodegenFeedForward,
    FeedForward_REGISTRY,
    FeedForwardBase,
    GLMFeedForward,
    register_feed_forward,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.layernorm import (
    LayerNorm,
    Norm,
    NORM_REGISTRY,
    register_norm,
    RMSNorm,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    AttentionSinkRope,
    FeedForward,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaModelWithoutEmbedding,
    MultiScopeAwareLlamaModel,
    repeat_kv,
)

__all__ = [
    # Decoder architectures.
    "LlamaModel",
    "LlamaModelWithoutEmbedding",
    "MultiScopeAwareLlamaModel",
    # Attention masks.
    "AttentionMask",
    "BaseAttentionMask",
    "CausalAttentionMask",
    "SlidingWindowAttentionMask",
    # Attention, KV-head expansion, and rotary positional encoding.
    "AttentionSinkRope",
    "LlamaAttention",
    "repeat_kv",
    "apply_partial_rotary_emb_single",
    "apply_rotary_emb_single",
    "register_rotary_emb",
    "ROTARY_EMB_REGISTRY",
    # Feed-forward layers.
    "FeedForward",
    "CodegenFeedForward",
    "FeedForward_REGISTRY",
    "FeedForwardBase",
    "GLMFeedForward",
    "register_feed_forward",
    # Normalization layers.
    "LayerNorm",
    "Norm",
    "NORM_REGISTRY",
    "register_norm",
    "RMSNorm",
    # A decoder layers.
    "LlamaDecoderLayer",
]
