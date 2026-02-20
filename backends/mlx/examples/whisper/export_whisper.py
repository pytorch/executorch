#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export Whisper model to MLX delegate using ExecuTorch.

This script exports three separate programs:
1. Encoder: Processes audio features -> encoder hidden states
2. Decoder: Token-by-token generation with static KV cache

The decoder uses:
- llama.update_cache for self-attention KV cache updates
- Pre-computed cross-attention K/V passed as inputs

Usage:
    python -m executorch.backends.mlx.examples.whisper.export_whisper \
        --model-id "openai/whisper-tiny" \
        --output-dir /tmp/whisper_mlx \
        --quantize int4

Requirements:
    pip install transformers torchao
"""

import argparse
import logging
import os
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import WhisperForConditionalGeneration

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import shared KV cache module
from executorch.backends.mlx.examples.cache import ETKVCache

# Import custom ops
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


# =============================================================================
# Whisper Encoder Wrapper
# =============================================================================


class WhisperEncoderExportable(nn.Module):
    """
    Wrapper around Whisper's encoder for export.

    forward(input_features) -> encoder_hidden_states
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features=input_features).last_hidden_state


# =============================================================================
# Whisper Decoder Self-Attention with KV Cache
# =============================================================================


class WhisperSelfAttentionWithCache(nn.Module):
    """
    Whisper self-attention layer with static KV cache.

    Uses llama.update_cache pattern for cache updates.
    """

    def __init__(
        self,
        attn_module: nn.Module,
        max_cache_len: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.q_proj = attn_module.q_proj
        self.k_proj = attn_module.k_proj
        self.v_proj = attn_module.v_proj
        self.out_proj = attn_module.out_proj

        self.num_heads = attn_module.num_heads
        self.head_dim = attn_module.head_dim
        self.scale = self.head_dim**-0.5
        self.max_cache_len = max_cache_len

        # Initialize KV cache module
        self.kv_cache = ETKVCache(
            max_batch_size=1,
            max_context_length=max_cache_len,
            n_heads=self.num_heads,
            head_dim=self.head_dim,
            enable_dynamic_shape=True,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, H*D]
        pos_int: int,  # Position as SymInt
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        # Linear projections
        q = self.q_proj(hidden_states)  # [B, T, H*D]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [B, H, T, D]
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Update KV cache
        k_cache, v_cache = self.kv_cache.update(pos_int, k, v)

        # Explicit windowing: slice cache to valid positions
        end_pos = pos_int + T
        k_win = k_cache[:, :, :end_pos, :]
        v_win = v_cache[:, :, :end_pos, :]

        # SDPA with causal mask
        attn_out = F.scaled_dot_product_attention(
            q, k_win, v_win, attn_mask=None, is_causal=True, scale=self.scale
        )

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.out_proj(attn_out)


# =============================================================================
# Whisper Cross-Attention (no cache update - K/V pre-computed)
# =============================================================================


class WhisperCrossAttention(nn.Module):
    """
    Whisper cross-attention layer.

    K/V are pre-computed from encoder output and passed as inputs.
    No cache update needed - just uses the pre-computed K/V directly.
    """

    def __init__(self, attn_module: nn.Module):
        super().__init__()
        self.q_proj = attn_module.q_proj
        self.out_proj = attn_module.out_proj

        self.num_heads = attn_module.num_heads
        self.head_dim = attn_module.head_dim
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T_dec, H*D]
        cross_k: torch.Tensor,  # [B, H, T_enc, D] - pre-computed
        cross_v: torch.Tensor,  # [B, H, T_enc, D] - pre-computed
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        # Query projection
        q = self.q_proj(hidden_states)
        q = q.view(B, T, H, D).transpose(1, 2)  # [B, H, T_dec, D]

        # SDPA with pre-computed K/V (no causal mask for cross-attention)
        attn_out = F.scaled_dot_product_attention(
            q, cross_k, cross_v, attn_mask=None, is_causal=False, scale=self.scale
        )

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.out_proj(attn_out)


# =============================================================================
# Whisper Decoder Layer Wrapper
# =============================================================================


class WhisperDecoderLayerWithCache(nn.Module):
    """
    Wrapper for a single Whisper decoder layer with KV cache.
    """

    def __init__(
        self,
        layer: nn.Module,
        max_cache_len: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        # Self-attention with cache
        self.self_attn = WhisperSelfAttentionWithCache(
            layer.self_attn, max_cache_len, dtype
        )
        self.self_attn_layer_norm = layer.self_attn_layer_norm

        # Cross-attention (K/V passed as inputs)
        self.encoder_attn = WhisperCrossAttention(layer.encoder_attn)
        self.encoder_attn_layer_norm = layer.encoder_attn_layer_norm

        # FFN
        self.fc1 = layer.fc1
        self.fc2 = layer.fc2
        self.final_layer_norm = layer.final_layer_norm
        self.activation_fn = layer.activation_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_int: int,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, pos_int)
        hidden_states = residual + hidden_states

        # Cross-attention
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(hidden_states, cross_k, cross_v)
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Whisper Decoder Wrapper
# =============================================================================


class WhisperDecoderWithCache(nn.Module):
    """
    Whisper decoder wrapper with static KV cache.

    Takes:
    - decoder_input_ids: [B, T_dec] token IDs
    - cache_position: [1] tensor with start position
    - cross_k_tuple: tuple of num_layers tensors [B, H, T_enc, D] - pre-computed cross K
    - cross_v_tuple: tuple of num_layers tensors [B, H, T_enc, D] - pre-computed cross V

    Returns:
    - logits: [B, T_dec, vocab_size]
    """

    def __init__(
        self,
        model: "WhisperForConditionalGeneration",
        max_decoder_seq_len: int,
    ):
        super().__init__()

        decoder = model.get_decoder()
        dtype = decoder.embed_tokens.weight.dtype

        self.embed_tokens = decoder.embed_tokens
        self.embed_positions = decoder.embed_positions
        self.layer_norm = decoder.layer_norm
        self.proj_out = model.proj_out

        # Wrap decoder layers with cache
        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayerWithCache(layer, max_decoder_seq_len, dtype)
                for layer in decoder.layers
            ]
        )

        self.num_layers = len(self.layers)
        self.max_decoder_seq_len = max_decoder_seq_len

    def forward(
        self,
        decoder_input_ids: torch.Tensor,  # [B, T_dec]
        cache_position: torch.Tensor,  # [1] tensor
        cross_k_tuple: Tuple[torch.Tensor, ...],  # num_layers x [B, H, T_enc, D]
        cross_v_tuple: Tuple[torch.Tensor, ...],  # num_layers x [B, H, T_enc, D]
    ) -> torch.Tensor:
        B, T = decoder_input_ids.shape

        # Get position as SymInt
        torch._check(cache_position.numel() == 1)
        pos_int = cache_position.item()
        torch._check_is_size(pos_int)
        torch._check(pos_int + T <= self.max_decoder_seq_len)

        # Token + positional embeddings
        # Whisper uses absolute positions [pos_int, pos_int + T)
        # Use F.embedding to ensure proper lowering (not aten.index.Tensor)
        positions = torch.arange(
            pos_int, pos_int + T, device=decoder_input_ids.device, dtype=torch.long
        )
        hidden_states = self.embed_tokens(decoder_input_ids)
        pos_embed = F.embedding(positions, self.embed_positions.weight)
        hidden_states = hidden_states + pos_embed

        # Decoder layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, pos_int, cross_k_tuple[i], cross_v_tuple[i]
            )

        hidden_states = self.layer_norm(hidden_states)
        logits = self.proj_out(hidden_states)
        return logits


# =============================================================================
# Cross-KV Projection Module
# =============================================================================


class WhisperCrossKVProjection(nn.Module):
    """
    Compute cross-attention K/V projections from encoder hidden states.

    forward(encoder_hidden_states) -> (k_tuple, v_tuple)
    """

    def __init__(self, model: "WhisperForConditionalGeneration"):
        super().__init__()
        decoder = model.get_decoder()

        # Store K/V projections for each layer
        self.k_projs = nn.ModuleList()
        self.v_projs = nn.ModuleList()
        self.num_heads_list = []
        self.head_dim_list = []

        for layer in decoder.layers:
            self.k_projs.append(layer.encoder_attn.k_proj)
            self.v_projs.append(layer.encoder_attn.v_proj)
            self.num_heads_list.append(layer.encoder_attn.num_heads)
            self.head_dim_list.append(layer.encoder_attn.head_dim)

    def forward(
        self, encoder_hidden_states: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Returns:
            (k_tuple, v_tuple) where each is a tuple of num_layers tensors,
            each with shape [B, H, T_enc, D]
        """
        B, T_enc, _ = encoder_hidden_states.shape

        k_list = []
        v_list = []

        for i, (k_proj, v_proj) in enumerate(zip(self.k_projs, self.v_projs)):
            H = self.num_heads_list[i]
            D = self.head_dim_list[i]

            k = k_proj(encoder_hidden_states)  # [B, T_enc, H*D]
            v = v_proj(encoder_hidden_states)

            # Reshape to [B, H, T_enc, D]
            k = k.view(B, T_enc, H, D).transpose(1, 2)
            v = v.view(B, T_enc, H, D).transpose(1, 2)

            k_list.append(k)
            v_list.append(v)

        return tuple(k_list), tuple(v_list)


# =============================================================================
# Export Functions
# =============================================================================


def export_whisper_to_mlx(
    model_id: str,
    output_dir: str,
    max_decoder_seq_len: int = 256,
    dtype: str = "bf16",
    quantize_linear: Optional[str] = None,
    quantize_embeddings: Optional[str] = None,
) -> None:
    """
    Export Whisper model components to MLX delegate.

    Exports:
    - encoder.pte: Audio encoder
    - cross_kv.pte: Cross-attention K/V projection
    - decoder.pte: Decoder with self-attention KV cache

    Args:
        model_id: HuggingFace model ID
        output_dir: Directory to save .pte files
        max_decoder_seq_len: Maximum decoder sequence length
        dtype: Model dtype ("fp32", "fp16", "bf16")
        quantize_linear: Quantization method for linear layers ("int4", "int8", or None)
        quantize_embeddings: Quantization method for embedding layers ("int4", "int8", or None)
    """
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    # Map dtype string to torch dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    logger.info(f"Loading model: {model_id} (dtype={dtype})")
    processor = AutoProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch_dtype
    )
    model.eval()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get feature extractor info
    fe = processor.feature_extractor
    batch_size = 1

    # Create example encoder input
    encoder_input = torch.zeros(
        (batch_size, fe.feature_size, fe.nb_max_frames), dtype=torch_dtype
    )

    # Create wrappers
    logger.info("Creating model wrappers...")
    encoder_wrapper = WhisperEncoderExportable(model.get_encoder()).eval()
    cross_kv_wrapper = WhisperCrossKVProjection(model).eval()

    # Get encoder output shape for decoder
    with torch.no_grad():
        encoder_hidden_states = encoder_wrapper(encoder_input)
    encoder_seq_len = encoder_hidden_states.shape[1]

    decoder_wrapper = WhisperDecoderWithCache(model, max_decoder_seq_len).eval()

    # Apply quantization if requested
    if quantize_linear or quantize_embeddings:
        logger.info("Applying quantization with TorchAO...")
        try:
            from torchao.quantization.granularity import PerGroup
            from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

            # Quantize embedding layers
            # Note: embed_positions is accessed via indexing which doesn't work with quantized tensors
            if quantize_embeddings:
                embed_dtype = (
                    torch.int4 if quantize_embeddings == "int4" else torch.int8
                )
                embed_group_size = 32 if quantize_embeddings == "int4" else 128
                logger.info(
                    f"Quantizing embedding layers with {quantize_embeddings} (group size {embed_group_size})..."
                )
                quantize_(
                    decoder_wrapper,
                    IntxWeightOnlyConfig(
                        weight_dtype=embed_dtype,
                        granularity=PerGroup(embed_group_size),
                    ),
                    lambda m, fqn: isinstance(m, nn.Embedding)
                    and "embed_tokens" in fqn,
                )

            # Quantize linear layers
            if quantize_linear:
                linear_dtype = torch.int4 if quantize_linear == "int4" else torch.int8
                linear_group_size = 32 if quantize_linear == "int4" else 128
                logger.info(
                    f"Quantizing linear layers with {quantize_linear} (group size {linear_group_size})..."
                )
                for module in [encoder_wrapper, cross_kv_wrapper, decoder_wrapper]:
                    quantize_(
                        module,
                        IntxWeightOnlyConfig(
                            weight_dtype=linear_dtype,
                            granularity=PerGroup(linear_group_size),
                        ),
                        filter_fn=lambda m, fqn: isinstance(m, nn.Linear),
                    )

            logger.info("Applied quantization successfully")
        except ImportError:
            logger.error("TorchAO not installed. Run: pip install torchao")
            raise

    # =========================================================================
    # Export Encoder
    # =========================================================================
    logger.info("Exporting encoder...")

    with torch.no_grad():
        encoder_ep = torch.export.export(
            encoder_wrapper, (encoder_input,), dynamic_shapes=None, strict=True
        )
        encoder_ep = encoder_ep.run_decompositions({})

    _save_to_pte(encoder_ep, os.path.join(output_dir, "encoder.pte"), "encoder")

    # =========================================================================
    # Export Cross-KV Projection
    # =========================================================================
    logger.info("Exporting cross-KV projection...")

    with torch.no_grad():
        example_cross_k, example_cross_v = cross_kv_wrapper(encoder_hidden_states)
        example_cross_k = tuple(k.contiguous() for k in example_cross_k)
        example_cross_v = tuple(v.contiguous() for v in example_cross_v)

        cross_kv_ep = torch.export.export(
            cross_kv_wrapper,
            (encoder_hidden_states,),
            dynamic_shapes=None,
            strict=True,
        )
        cross_kv_ep = cross_kv_ep.run_decompositions({})

    _save_to_pte(cross_kv_ep, os.path.join(output_dir, "cross_kv.pte"), "cross_kv")

    # =========================================================================
    # Export Decoder
    # =========================================================================
    logger.info("Exporting decoder...")

    # Example inputs for decoder
    start_id = getattr(model.config, "decoder_start_token_id", 0)
    decoder_input_ids = torch.tensor([[start_id]], dtype=torch.long)
    cache_position = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        # Build dynamic shapes for all inputs
        # decoder_input_ids: [B, T_dec] - T_dec is dynamic
        # cache_position: [1] - static
        # cross_k_tuple: tuple of num_layers tensors - static
        # cross_v_tuple: tuple of num_layers tensors - static
        seq_dim = torch.export.Dim.AUTO(min=1, max=max_decoder_seq_len)
        num_layers = decoder_wrapper.num_layers
        dynamic_shapes = (
            {1: seq_dim},  # decoder_input_ids
            None,  # cache_position
            tuple(None for _ in range(num_layers)),  # cross_k_tuple
            tuple(None for _ in range(num_layers)),  # cross_v_tuple
        )

        decoder_ep = torch.export.export(
            decoder_wrapper,
            (decoder_input_ids, cache_position, example_cross_k, example_cross_v),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
        decoder_ep = decoder_ep.run_decompositions({})

    _save_to_pte(decoder_ep, os.path.join(output_dir, "decoder.pte"), "decoder")

    # Save processor for inference
    processor_path = os.path.join(output_dir, "processor")
    processor.save_pretrained(processor_path)
    logger.info(f"Saved processor to: {processor_path}")

    # Save metadata
    metadata = {
        "model_id": model_id,
        "dtype": dtype,
        "quantize_linear": quantize_linear,
        "quantize_embeddings": quantize_embeddings,
        "max_decoder_seq_len": max_decoder_seq_len,
        "encoder_seq_len": encoder_seq_len,
        "num_decoder_layers": decoder_wrapper.num_layers,
    }
    import json

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {os.path.join(output_dir, 'metadata.json')}")


def _save_to_pte(ep, output_path: str, name: str) -> None:
    """Lower and save an ExportedProgram to a .pte file."""
    import executorch.exir as exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig

    # Allow repeat_interleave and sdpa ops
    edge_config = EdgeCompileConfig(
        _core_aten_ops_exception_list=[
            torch.ops.aten.repeat_interleave.self_int,
            torch.ops.aten.scaled_dot_product_attention.default,
        ]
    )

    edge_program = exir.to_edge_transform_and_lower(
        ep,
        partitioner=[MLXPartitioner()],
        compile_config=edge_config,
    )

    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )

    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    logger.info(
        f"Saved {name} to: {output_path} "
        f"({len(executorch_program.buffer) / 1024 / 1024:.2f} MB)"
    )


def main():
    parser = argparse.ArgumentParser(description="Export Whisper model to MLX delegate")
    parser.add_argument(
        "--model-id",
        type=str,
        default="openai/whisper-tiny",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="whisper_mlx",
        help="Output directory for .pte files",
    )
    parser.add_argument(
        "--max-decoder-seq-len",
        type=int,
        default=256,
        help="Maximum decoder sequence length",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Model dtype",
    )
    parser.add_argument(
        "--quantize-linear",
        type=str,
        choices=["int4", "int8"],
        default=None,
        help="Quantization method for linear layers",
    )
    parser.add_argument(
        "--quantize-embeddings",
        type=str,
        choices=["int4", "int8"],
        default=None,
        help="Quantization method for embedding layers",
    )

    args = parser.parse_args()

    export_whisper_to_mlx(
        model_id=args.model_id,
        output_dir=args.output_dir,
        max_decoder_seq_len=args.max_decoder_seq_len,
        dtype=args.dtype,
        quantize_linear=args.quantize_linear,
        quantize_embeddings=args.quantize_embeddings,
    )


if __name__ == "__main__":
    main()
