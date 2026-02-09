#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export Llama model to MLX delegate using ExecutorCh.

This script:
1. Loads a HuggingFace Llama model
2. Wraps it with functional KV cache and custom MLX ops
3. Optionally quantizes with TorchAO
4. Exports to .pte file using MLX delegate

Usage:
    python -m executorch.backends.apple.mlx.examples.llama.export_llama \
        --model-id "unsloth/Llama-3.2-1B-Instruct" \
        --output llama.pte \
        --quantize int4

Requirements:
    pip install transformers torchao
"""

import argparse
import logging
import os
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

# Import MLX ops to register handlers
import executorch.backends.apple.mlx.ops  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import shared KV cache module
from executorch.backends.apple.mlx.examples.cache import KVCache

# Import custom ops to register llama.update_cache
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


# =============================================================================
# Custom RMSNorm using aten op
# =============================================================================


class CustomRMSNorm(nn.Module):
    """RMSNorm using torch.nn.functional.rms_norm for efficient MLX execution.

    This replaces the HuggingFace LlamaRMSNorm (which uses manual variance + rsqrt
    computation) with the aten rms_norm op. The MLX backend has a handler that maps
    aten.rms_norm directly to MLX's fast::rms_norm, so this gives us fused execution
    without needing a custom op.
    """

    def __init__(self, base_rms: nn.Module):
        super().__init__()
        self.weight = base_rms.weight
        self.eps = float(
            getattr(base_rms, "eps", getattr(base_rms, "variance_epsilon", 1e-5))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.weight.shape[0],), self.weight, self.eps)


# =============================================================================
# Utility functions
# =============================================================================


def _get_attr_any(obj, *names, default=None):
    """Get first matching attribute from object."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _infer_heads_dims(
    attn_module: nn.Module,
    fallback_hidden_size: int,
    fallback_num_heads: int,
    fallback_num_kv_heads: int,
) -> Tuple[int, int, int, int]:
    """Infer attention head dimensions from module."""
    q_proj = _get_attr_any(attn_module, "q_proj")
    hidden_size = None
    if q_proj is not None and hasattr(q_proj, "out_features"):
        try:
            hidden_size = int(q_proj.out_features)
        except Exception:
            hidden_size = None
    if hidden_size is None:
        hidden_size = int(
            _get_attr_any(attn_module, "hidden_size", default=fallback_hidden_size)
        )

    num_heads = _get_attr_any(attn_module, "num_heads")
    if num_heads is None:
        num_heads = fallback_num_heads
    num_heads = int(num_heads)

    num_kv_heads = _get_attr_any(attn_module, "num_key_value_heads", "n_kv_heads")
    if num_kv_heads is None:
        num_kv_heads = fallback_num_kv_heads
    num_kv_heads = int(num_kv_heads)

    head_dim = _get_attr_any(attn_module, "head_dim")
    if head_dim is None:
        head_dim = hidden_size // max(1, num_heads)
    head_dim = int(head_dim)

    return hidden_size, num_heads, num_kv_heads, head_dim


# =============================================================================
# KV Cache Attention with RoPE
# =============================================================================


class KVCacheAttention(nn.Module):
    """
    Attention module with KV cache support and custom RoPE op.

    Uses:
    - mlx::apply_rope for efficient rotary position embedding
    - Functional KV cache updates that can be traced
    - Grouped query attention (GQA) support
    """

    def __init__(
        self,
        attn_module: nn.Module,
        *,
        fallback_hidden_size: int,
        fallback_num_heads: int,
        fallback_num_kv_heads: int,
        time_axis: int = 1,
        T_max: int = 4096,
        dtype: torch.dtype = torch.float32,
        rope_base: float = 500000.0,
    ):
        super().__init__()
        self.q_proj = _get_attr_any(attn_module, "q_proj")
        self.k_proj = _get_attr_any(attn_module, "k_proj")
        self.v_proj = _get_attr_any(attn_module, "v_proj")
        self.o_proj = _get_attr_any(attn_module, "o_proj", "out_proj", "o_proj_linear")

        if any(x is None for x in (self.q_proj, self.k_proj, self.v_proj, self.o_proj)):
            raise AttributeError(
                "Attention module missing q_proj/k_proj/v_proj/o_proj(out_proj)"
            )

        hidden_size, H, Hkv, Dh = _infer_heads_dims(
            attn_module,
            fallback_hidden_size,
            fallback_num_heads,
            fallback_num_kv_heads,
        )
        self.hidden_size = hidden_size
        self.num_heads = H  # Q heads
        self.num_key_value_heads = Hkv
        self.head_dim = Dh
        self.time_axis = int(time_axis)
        self.T_max = int(T_max)
        self.is_causal = True
        self.rope_base = rope_base

        # Initialize KV cache module
        self.kv_cache = KVCache(
            num_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_cache_len=self.T_max,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor, pos_int: int) -> torch.Tensor:
        """
        Forward pass with KV cache support.

        Args:
            hidden_states: [B, T, hidden_size]
            pos_int: Position as int (SymInt during tracing)
        """
        torch._check(hidden_states.size(0) == 1)
        B, T, _ = hidden_states.shape
        H, Hkv, Dh = self.num_heads, self.num_key_value_heads, self.head_dim

        # 1) Linear projections
        q_lin = self.q_proj(hidden_states)  # [B,T,H*D]
        k_lin = self.k_proj(hidden_states)  # [B,T,Hkv*D]
        v_lin = self.v_proj(hidden_states)  # [B,T,Hkv*D]

        # 2) Reshape to [B,T,H,D] / [B,T,Hkv,D]
        q_bthd = q_lin.view(B, T, H, Dh)
        k_bthd = k_lin.view(B, T, Hkv, Dh)
        v_bthd = v_lin.view(B, T, Hkv, Dh)

        # 3) Permute to B,H,T,D for rope + sdpa
        q_bhtd = q_bthd.permute(0, 2, 1, 3).contiguous()  # [B,H,T,D]
        k_bhtd = k_bthd.permute(0, 2, 1, 3).contiguous()  # [B,Hkv,T,D]
        v_bhtd = v_bthd.permute(0, 2, 1, 3).contiguous()  # [B,Hkv,T,D]

        # 4) Apply RoPE using custom mlx::rope op (once for Q, once for K)
        # This op is preserved through lowering and handled by MLX backend
        q_bhtd = torch.ops.mlx.rope(
            q_bhtd,  # [B,H,T,D]
            self.head_dim,
            pos_int,  # int from .item() at top level
            False,  # traditional
            self.rope_base,  # base
            1.0,  # scale
            None,  # freqs
        )
        k_bhtd = torch.ops.mlx.rope(
            k_bhtd,  # [B,Hkv,T,D]
            self.head_dim,
            pos_int,  # int from .item() at top level
            False,  # traditional
            self.rope_base,  # base
            1.0,  # scale
            None,  # freqs
        )

        # 5) Update KV cache using callable module (returns full cache)
        k_cache, v_cache = self.kv_cache(k_bhtd, v_bhtd, pos_int)

        # 6) Explicit windowing: slice cache to valid positions
        end_pos = pos_int + T
        k_ = k_cache[:, :, :end_pos, :]
        v_ = v_cache[:, :, :end_pos, :]

        # 7) Prepare for SDPA
        q_ = q_bhtd  # [B,H,T,D]

        B_, Hq_, T_, Dh_ = q_.shape
        _, Hkv_, Tk_, Dhk_ = k_.shape
        assert Dh_ == Dhk_

        # Assert that key sequence length is non-zero (required for SDPA)
        torch._check(Tk_ != 0)

        # Handle GQA by repeating K/V heads
        if Hq_ != Hkv_:
            torch._check(Hq_ >= Hkv_)
            torch._check(Hq_ % Hkv_ == 0)
            group = Hq_ // Hkv_
            k_ = k_.repeat_interleave(group, dim=1)
            v_ = v_.repeat_interleave(group, dim=1)

        # 8) Scaled dot-product attention
        # With windowed cache + is_causal=True:
        # - Prefill (pos=0, T>1): Q[B,H,T,D], K[B,H,T,D] - same length
        # - Decode (pos=N, T=1): Q[B,H,1,D], K[B,H,N+1,D] - different length
        #
        # NOTE: MLX's causal mask has different semantics than PyTorch!
        # - PyTorch is_causal=True: Q[i] attends to K[0:i+1] (based on Q's index)
        # - MLX mask='causal': Q[i] attends to K[0:T_kv-T_q+i+1] (allows all previous)
        #
        # For decode with Q[1] and K[N+1], MLX allows Q[0] to attend to all K[0:N+1].
        # This is the correct behavior for autoregressive generation, so no padding needed.
        attn_out = F.scaled_dot_product_attention(
            q_,  # [B,H,T,D]
            k_,  # [B,H,Tk,D] windowed cache
            v_,
            attn_mask=None,
            is_causal=True,
            scale=None,
        )  # → [B,H,T,D]

        # 9) Reshape back and output projection
        attn_out = (
            attn_out.permute(0, 2, 1, 3).contiguous().view(B, T, H * Dh)  # [B,T,H,D]
        )
        out = self.o_proj(attn_out)
        return out


# =============================================================================
# Llama Model Wrapper
# =============================================================================


class LlamaWithFunctionalKV(nn.Module):
    """
    Wrapper around HuggingFace Llama that:
    1. Replaces RMSNorm with aten rms_norm (mapped to MLX's fast::rms_norm)
    2. Replaces attention with KVCacheAttention (using mlx::apply_rope)
    3. Provides a trace-friendly forward that takes (token_ids, input_pos)
    """

    def __init__(
        self,
        base: "AutoModelForCausalLM",
        time_axis: int = 1,
        max_seq_len: int = 1024,
        rope_base: float = 500000.0,
    ):
        super().__init__()
        self.model = base

        # Swap RMSNorm modules with custom op version
        for layer in self.model.model.layers:
            layer.input_layernorm = CustomRMSNorm(layer.input_layernorm)
            layer.post_attention_layernorm = CustomRMSNorm(
                layer.post_attention_layernorm
            )
        self.model.model.norm = CustomRMSNorm(self.model.model.norm)

        # Get config for attention dimensions
        cfg = base.config
        fallback_hidden_size = int(cfg.hidden_size)
        fallback_num_heads = int(cfg.num_attention_heads)
        fallback_num_kv_heads = int(
            getattr(cfg, "num_key_value_heads", fallback_num_heads)
        )
        T_max = max_seq_len
        dtype = base.model.embed_tokens.weight.dtype

        # Get rope_theta from config if available
        if hasattr(cfg, "rope_theta"):
            rope_base = float(cfg.rope_theta)

        # Wrap attention modules with KVCacheAttention
        for layer in self.model.model.layers:
            layer.self_attn = KVCacheAttention(
                layer.self_attn,
                fallback_hidden_size=fallback_hidden_size,
                fallback_num_heads=fallback_num_heads,
                fallback_num_kv_heads=fallback_num_kv_heads,
                time_axis=time_axis,
                T_max=T_max,
                dtype=dtype,
                rope_base=rope_base,
            )

    def forward(self, token_ids: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with KV cache support.

        Args:
            token_ids: Input token IDs [B, T]
            input_pos: Starting position as rank-1 tensor [1]

        Returns:
            Logits tensor [B, T, vocab_size]
        """
        m = self.model
        hs = m.model.embed_tokens(token_ids)

        # Get position as int from tensor ONCE at top level
        # This ensures all layers share the same position value,
        # avoiding creation of 16 separate input tensors in the exported graph
        # Both MLX and Custom cache now accept int position directly
        pos_int = input_pos[0].item()
        torch._check_is_size(pos_int)

        for layer in m.model.layers:
            residual = hs
            hs = layer.input_layernorm(hs)
            hs = residual + layer.self_attn(hs, pos_int)
            residual = hs
            hs = layer.post_attention_layernorm(hs)
            hs = layer.mlp(hs)
            hs = residual + hs

        hs = m.model.norm(hs)
        logits = m.lm_head(hs)
        return logits


# =============================================================================
# Export Functions
# =============================================================================


def export_llama_to_mlx(
    model_id: str,
    output_path: str,
    max_seq_len: int = 1024,
    dtype: str = "fp32",
    quantize_linear: Optional[str] = None,
    quantize_embeddings: Optional[str] = None,
) -> None:
    """
    Export a Llama model to MLX delegate.

    Args:
        model_id: HuggingFace model ID
        output_path: Path to save the .pte file
        max_seq_len: Maximum sequence length for KV cache
        dtype: Model dtype ("fp32", "fp16", "bf16")
        quantize_linear: Quantization method for linear layers ("int4", "int8", or None)
        quantize_embeddings: Quantization method for embedding layers ("int4", "int8", or None)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Map dtype string to torch dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    logger.info(f"Loading model: {model_id} (dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)

    logger.info("Wrapping model with functional KV cache...")
    model = LlamaWithFunctionalKV(base, max_seq_len=max_seq_len)
    model.eval()

    # Apply quantization if requested
    if quantize_linear or quantize_embeddings:
        logger.info("Applying quantization with TorchAO...")
        try:
            from torchao.quantization.granularity import PerGroup
            from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

            # Quantize embedding layers
            if quantize_embeddings:
                embed_dtype = (
                    torch.int4 if quantize_embeddings == "int4" else torch.int8
                )
                embed_group_size = 32 if quantize_embeddings == "int4" else 128
                logger.info(
                    f"Quantizing embedding layers with {quantize_embeddings} (group size {embed_group_size})..."
                )
                quantize_(
                    model,
                    IntxWeightOnlyConfig(
                        weight_dtype=embed_dtype,
                        granularity=PerGroup(embed_group_size),
                    ),
                    filter_fn=lambda m, fqn: isinstance(m, torch.nn.Embedding),
                )

            # Quantize linear layers
            if quantize_linear:
                linear_dtype = torch.int4 if quantize_linear == "int4" else torch.int8
                linear_group_size = 32 if quantize_linear == "int4" else 128
                logger.info(
                    f"Quantizing linear layers with {quantize_linear} (group size {linear_group_size})..."
                )
                quantize_(
                    model,
                    IntxWeightOnlyConfig(
                        weight_dtype=linear_dtype,
                        granularity=PerGroup(linear_group_size),
                    ),
                    filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear),
                )

            # Tie lm_head weights to embedding after quantization
            if quantize_embeddings:
                model.model.lm_head.weight = model.model.model.embed_tokens.weight

            logger.info("Applied quantization successfully")
        except ImportError:
            logger.error("TorchAO not installed. Run: pip install torchao")
            raise

    # Prepare example inputs for export
    # input_pos is a [1] tensor, we call input_pos[0].item() to get the SymInt
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    input_pos = torch.tensor([0], dtype=torch.long)  # Rank-1 tensor [1]
    example_inputs = (token_ids, input_pos)

    # Set up dynamic shapes for variable sequence length
    dynamic_shapes = {
        "token_ids": {1: torch.export.Dim.AUTO(min=1, max=2048)},
        "input_pos": {
            0: torch.export.Dim.AUTO(min=1, max=max_seq_len)
        },  # Dynamic length
    }

    logger.info("Exporting model with torch.export...")
    with torch.no_grad():
        ep = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)
        ep = ep.run_decompositions({})

    logger.info("Delegating to MLX backend...")
    import executorch.exir as exir
    from executorch.backends.apple.mlx import MLXPartitioner
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig

    # Allow repeat_interleave and sdpa ops - they will be handled by MLX backend
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

    logger.info("Exporting to ExecuTorch...")
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
        )
    )

    # Save the program
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    logger.info(f"Saved model to: {output_path}")
    logger.info(f"Program size: {len(executorch_program.buffer) / 1024 / 1024:.2f} MB")

    # Save tokenizer alongside for inference
    tokenizer_path = output_path.replace(".pte", "_tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Saved tokenizer to: {tokenizer_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Llama model to MLX delegate")
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llama_mlx.pte",
        help="Output .pte file path",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Maximum sequence length for KV cache",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Model dtype (fp32, fp16, bf16)",
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

    export_llama_to_mlx(
        model_id=args.model_id,
        output_path=args.output,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        quantize_linear=args.quantize_linear,
        quantize_embeddings=args.quantize_embeddings,
    )


if __name__ == "__main__":
    main()
