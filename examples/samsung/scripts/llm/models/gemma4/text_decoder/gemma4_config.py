# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""
Gemma 4 configuration for ExecuTorch export.

This configuration covers Gemma 4-specific parameters including
Per-Layer Embeddings, the Self/Cross decoder split, partial RoPE,
and per-layer head_dim (global_head_dim for full attention layers).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Default layer types pattern for Gemma 4 E2B (35 layers: 4 sliding + 1 full, repeated 7 times)
DEFAULT_LAYER_TYPES = [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
]


@dataclass
class Gemma4Config:
    """Configuration for Gemma 4 text decoder."""

    # Base transformer parameters
    hidden_size: int = 1536
    num_hidden_layers: int = 35
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    vocab_size: int = 262144
    intermediate_size: int = 6144
    use_double_wide_mlp: bool = True
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131072

    # RoPE parameters
    rope_theta: float = 1000000.0
    rope_local_base_freq: float = 10000.0
    partial_rotary_factor: float = 0.25

    # Attention configuration
    sliding_window: int = 512
    layer_types: List[str] = field(default_factory=lambda: DEFAULT_LAYER_TYPES.copy())
    sliding_window_pattern: int = 5

    # Gemma specific
    query_pre_attn_scalar: int = 256
    use_qk_norm: bool = True
    use_res_clamp: bool = True
    final_logit_softcapping: float = 30.0
    hidden_activation: str = "gelu_pytorch_tanh"

    # Per-Layer Embeddings
    hidden_size_per_layer_input: int = 256
    vocab_size_per_layer_input: int = 262144

    # YOCO (KV sharing)
    num_kv_shared_layers: int = 20

    # Tied embeddings
    tie_word_embeddings: bool = True

    # Audio embedding configuration
    audio_encoder_hidden_size: int = 1024
    audio_soft_tokens_per_input: int = 750
    audio_token_id: int = 258881
    image_token_id: int = 258880
    audio_vocab_offset: int = 258883  # eoa_token_id (legacy, unused)

    # KV cache settings
    use_kv_cache: bool = True
    max_seq_len: int = 2048
    max_context_len: int = 2048
    max_batch_size: int = 1
    enable_dynamic_shape: bool = False
    use_index_copy_for_kv_cache: bool = False

    @classmethod
    def from_json(cls, json_path: str) -> "Gemma4Config":
        """Load config from JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)

        if "text_config" in config_dict:
            text_config = config_dict["text_config"]
        else:
            text_config = config_dict

        intermediate_size = text_config.get("intermediate_size", 6144)
        if isinstance(intermediate_size, list):
            intermediate_size = intermediate_size[0]

        return cls(
            hidden_size=text_config.get("hidden_size", cls.hidden_size),
            num_hidden_layers=text_config.get(
                "num_hidden_layers", cls.num_hidden_layers
            ),
            num_attention_heads=text_config.get(
                "num_attention_heads", cls.num_attention_heads
            ),
            num_key_value_heads=text_config.get(
                "num_key_value_heads", cls.num_key_value_heads
            ),
            head_dim=text_config.get("head_dim", cls.head_dim),
            global_head_dim=text_config.get("global_head_dim", cls.global_head_dim),
            vocab_size=text_config.get("vocab_size", cls.vocab_size),
            intermediate_size=intermediate_size,
            rms_norm_eps=text_config.get("rms_norm_eps", cls.rms_norm_eps),
            max_position_embeddings=text_config.get(
                "max_position_embeddings", cls.max_position_embeddings
            ),
            rope_theta=text_config.get("rope_theta", cls.rope_theta),
            rope_local_base_freq=text_config.get(
                "rope_local_base_freq", cls.rope_local_base_freq
            ),
            partial_rotary_factor=text_config.get(
                "partial_rotary_factor", cls.partial_rotary_factor
            ),
            sliding_window=text_config.get("sliding_window", cls.sliding_window),
            layer_types=text_config.get("layer_types", DEFAULT_LAYER_TYPES),
            query_pre_attn_scalar=text_config.get(
                "query_pre_attn_scalar", cls.query_pre_attn_scalar
            ),
            use_qk_norm=text_config.get("use_qk_norm", cls.use_qk_norm),
            use_res_clamp=text_config.get("use_res_clamp", cls.use_res_clamp),
            final_logit_softcapping=text_config.get(
                "final_logit_softcapping", cls.final_logit_softcapping
            ),
            hidden_activation=text_config.get(
                "hidden_activation", cls.hidden_activation
            ),
            hidden_size_per_layer_input=text_config.get(
                "hidden_size_per_layer_input", cls.hidden_size_per_layer_input
            ),
            vocab_size_per_layer_input=text_config.get(
                "vocab_size_per_layer_input", cls.vocab_size_per_layer_input
            ),
            num_kv_shared_layers=text_config.get(
                "num_kv_shared_layers", cls.num_kv_shared_layers
            ),
            use_double_wide_mlp=text_config.get(
                "use_double_wide_mlp", cls.use_double_wide_mlp
            ),
            tie_word_embeddings=text_config.get(
                "tie_word_embeddings", cls.tie_word_embeddings
            ),
        )

    @classmethod
    def from_config(cls, variant: str = "e2b") -> "Gemma4Config":
        """Load a named config variant (e2b, e4b)."""
        config_path = Path(__file__).parent.parent / "config" / f"{variant}_config.json"
        return cls.from_json(str(config_path))

    @classmethod
    def from_e2b_config(cls) -> "Gemma4Config":
        """Load the default E2B config."""
        return cls.from_config("e2b")

    @property
    def num_self_decoder_layers(self) -> int:
        """Number of layers in self-decoder (before KV sharing)."""
        return self.num_hidden_layers - self.num_kv_shared_layers

    @property
    def num_cross_decoder_layers(self) -> int:
        """Number of layers in cross-decoder (with KV sharing)."""
        return self.num_kv_shared_layers

    def get_layer_type(self, layer_idx: int) -> str:
        """Get attention type for a specific layer."""
        if layer_idx < len(self.layer_types):
            return self.layer_types[layer_idx]
        pattern_idx = layer_idx % self.sliding_window_pattern
        if pattern_idx < len(self.layer_types):
            return self.layer_types[pattern_idx]
        return "sliding_attention" if pattern_idx % 2 == 0 else "full_attention"

    def is_sliding_attention(self, layer_idx: int) -> bool:
        """Check if layer uses sliding window attention."""
        return self.get_layer_type(layer_idx) == "sliding_attention"

    def get_rope_theta(self, layer_idx: int) -> float:
        """Get RoPE theta for a specific layer based on attention type."""
        if self.is_sliding_attention(layer_idx):
            return self.rope_local_base_freq
        return self.rope_theta

    def get_head_dim(self, layer_idx: int) -> int:
        """Get head_dim for a specific layer.

        Full attention layers use global_head_dim (512), sliding layers use head_dim (256).
        """
        if self.is_sliding_attention(layer_idx):
            return self.head_dim
        return self.global_head_dim

    def get_intermediate_size(self, layer_idx: int) -> int:
        """Get MLP intermediate_size for a specific layer.

        Cross-decoder layers (KV shared) use 2x intermediate_size when
        use_double_wide_mlp is enabled.
        """
        if self.use_double_wide_mlp and self.is_kv_shared_layer(layer_idx):
            return self.intermediate_size * 2
        return self.intermediate_size

    def is_kv_shared_layer(self, layer_idx: int) -> bool:
        """Check if layer uses KV sharing (YOCO)."""
        first_kv_shared_layer_idx = self.num_hidden_layers - self.num_kv_shared_layers
        return layer_idx >= first_kv_shared_layer_idx and first_kv_shared_layer_idx > 0

    def get_kv_shared_layer_index(self, layer_idx: int) -> Optional[int]:
        """Get the layer index to share K/V from for a given layer.

        For KV shared layers, returns the index of the last non-shared
        layer of the same attention type before sharing starts.

        Returns:
            Layer index to share K/V from, or None if not a KV shared layer.
        """
        if not self.is_kv_shared_layer(layer_idx):
            return None

        first_kv_shared_layer_idx = self.num_hidden_layers - self.num_kv_shared_layers
        current_layer_type = self.get_layer_type(layer_idx)

        for i in range(first_kv_shared_layer_idx - 1, -1, -1):
            if self.get_layer_type(i) == current_layer_type:
                return i

        return None

    def is_kv_donor_layer(self, layer_idx: int) -> bool:
        """Check if this layer donates K/V to later layers (YOCO).

        Returns:
            True if this layer should store K/V for later sharing.
        """
        if self.is_kv_shared_layer(layer_idx):
            return False

        first_kv_shared_layer_idx = self.num_hidden_layers - self.num_kv_shared_layers
        if first_kv_shared_layer_idx <= 0:
            return False

        current_layer_type = self.get_layer_type(layer_idx)

        for i in range(first_kv_shared_layer_idx - 1, -1, -1):
            if self.get_layer_type(i) == current_layer_type:
                return layer_idx == i

        return False
