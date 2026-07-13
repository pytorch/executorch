# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Load a GGUF file into a Gemma 4 31B model.

Streams tensors one at a time via the shared loader in
``extension/llm/export/gguf.py`` (each quantized weight arrives as an
``ExportableGGUFTensor`` wrapping the raw GGUF blob), remaps GGUF names to model
FQNs, handles the tied embed/lm_head, and converts each weight for the target
backend:

* **MLX**: every quantized weight stays an ``ExportableGGUFTensor`` and is lowered
  by the MLX GGUF pattern (Q6_K custom kernels, Q4_K native affine ops) for both
  linear and embedding. ``embed_tokens`` and ``lm_head`` stay tied -- they share
  the one quantized tensor.
* **CUDA**: Q4_K -> ``Int4Tensor``, Q6_K -> ``CudaDp4aPlanarInt6Tensor`` (a genuine
  6-bit packed weight, lossless, symmetric). ``embed_tokens`` and ``lm_head`` are
  untied: ``lm_head`` keeps a packed (int6/int4) matmul weight, while the token
  embedding becomes a gatherable ``IntxUnpackedToInt8Tensor`` (int8) -- the truly
  packed int4/int6 tensors can't gather. For the Q6_K tied weight the decode is
  done once and shared between the two, avoiding a whole-tensor bf16 dequant and
  a second decode (see ``_untie_embed_lm_head``).

Usage:
    model, config = load_gguf_model("model.gguf", backend="cuda")
    model, config = load_gguf_model("model.gguf", backend="mlx")
"""

from typing import Optional

import torch

# GGUF pattern → model FQN pattern. ``{}`` is the layer index.
_KEY_MAP = {
    "token_embd.weight": "embed_tokens.weight",
    "output_norm.weight": "norm.weight",
    # Per-layer attention
    "blk.{}.attn_q.weight": "layers.{}.self_attn.q_proj.weight",
    "blk.{}.attn_k.weight": "layers.{}.self_attn.k_proj.weight",
    "blk.{}.attn_v.weight": "layers.{}.self_attn.v_proj.weight",
    "blk.{}.attn_output.weight": "layers.{}.self_attn.o_proj.weight",
    "blk.{}.attn_q_norm.weight": "layers.{}.self_attn.q_norm.weight",
    "blk.{}.attn_k_norm.weight": "layers.{}.self_attn.k_norm.weight",
    # Per-layer norms
    "blk.{}.attn_norm.weight": "layers.{}.input_layernorm.weight",
    "blk.{}.post_attention_norm.weight": "layers.{}.post_attention_layernorm.weight",
    "blk.{}.ffn_norm.weight": "layers.{}.pre_feedforward_layernorm.weight",
    "blk.{}.post_ffw_norm.weight": "layers.{}.post_feedforward_layernorm.weight",
    # Per-layer MLP
    "blk.{}.ffn_gate.weight": "layers.{}.mlp.gate_proj.weight",
    "blk.{}.ffn_up.weight": "layers.{}.mlp.up_proj.weight",
    "blk.{}.ffn_down.weight": "layers.{}.mlp.down_proj.weight",
    # Per-layer scalar
    "blk.{}.layer_output_scale.weight": "layers.{}.layer_scalar",
}

_IGNORED_KEYS = {"rope_freqs.weight"}


def gguf_to_model_key(gguf_key: str) -> Optional[str]:
    """Map a GGUF tensor name to a model FQN, or ``None`` to skip."""
    if gguf_key in _IGNORED_KEYS:
        return None

    for gguf_pat, model_pat in _KEY_MAP.items():
        if "{}" not in gguf_pat:
            if gguf_key == gguf_pat:
                return model_pat
            continue
        prefix, suffix = gguf_pat.split("{}")
        if gguf_key.startswith(prefix) and gguf_key.endswith(suffix):
            layer_str = gguf_key[len(prefix) : len(gguf_key) - len(suffix)]
            if layer_str.isdigit():
                return model_pat.replace("{}", layer_str)

    return None


def _validate_no_meta(model):
    """Ensure all parameters have been loaded."""
    for fqn, p in model.named_parameters():
        if p.device.type == "meta":
            raise RuntimeError(
                f"Weight '{fqn}' not found in GGUF file "
                f"(model/checkpoint version mismatch?)"
            )
    for p in model.parameters():
        p.requires_grad_(False)


def _convert_weight(model, model_key: str, gtensor, backend: str):
    """Convert an ``ExportableGGUFTensor`` to the per-backend module weight."""
    if backend == "mlx":
        return gtensor
    # CUDA: Q4_K -> torchao Int4Tensor. Q6_K stays the raw ExportableGGUFTensor
    # (like MLX) -- the CUDA packer repacks it into CudaDp4aPlanarInt6Tensor via
    # CudaDp4aPlanarInt6Tensor.from_exportable_gguf, so the Q6_K block decode is
    # owned by gguf.py and reused, not duplicated here.
    if gtensor.ggml_type == "q4_k":
        return gtensor.to_int4_tensor()
    return gtensor


def _resolve_tied_lm_head(model, lm_head_weight, packers):
    """Assign a tied lm_head (GGUF ties it to the token embedding)."""
    from executorch.examples.models.gemma4_31b.quant import pack_one

    lm_head = getattr(model.lm_head, "weight", None)
    if lm_head is None or lm_head.device.type != "meta":
        return
    if lm_head_weight is not None:
        pack_one(model, "lm_head.weight", lm_head_weight, packers)
    else:
        pack_one(
            model, "lm_head.weight", model.embed_tokens.weight.data.clone(), packers
        )


def _untie_embed_lm_head(model, gtensor, weight, backend):
    """Untie the GGUF token-embed / lm_head weight, returning ``(embed, lm_head)``.

    GGUF ties ``embed_tokens`` and ``lm_head`` to one quantized weight. The
    returned ``lm_head`` is packed into ``model.lm_head`` after the streaming loop
    (``_resolve_tied_lm_head``), or is ``None`` when this function already
    assigned it.

    * **MLX**: keep both tied on the raw ``ExportableGGUFTensor``.
    * **CUDA** (Q6_K or Q4_K): untie so ``lm_head`` keeps a packed low-bit matmul
      weight while the token embedding becomes a gatherable int8
      ``IntxUnpackedToInt8Tensor`` -- the truly packed int4/int6 tensors can't
      gather. Instead of dequantizing the whole ~1.4 B-element weight to bf16
      (2 B/elem), decode it once to int8 (1 B/elem; the decode is lossless so the
      result is numerically identical), halving the embedding's host + GPU-constant
      footprint. The token embedding (Q4_K for the Gemma checkpoint) is the single
      biggest weight, so this is the dominant saving vs the bf16 path. ``lm_head``:
        - Q6_K -> ``CudaDp4aPlanarInt6Tensor`` from the *same* int8 decode and
          assigned here (``pack_linear_for_cuda`` would mis-route an int8 tensor to
          the int8 path), so the post-loop resolve is a no-op.
        - Q4_K -> kept as the native ``Int4Tensor`` and returned, so
          ``_resolve_tied_lm_head`` packs it to ``CudaCoalescedInt4Tensor`` (same
          as a regular Q4_K linear).
    * **CUDA, other types**: fall back to the bf16 embedding.
    """
    if backend == "mlx":
        return weight, gtensor

    if gtensor.ggml_type in ("q6_k", "q4_k"):
        intx = gtensor.to_intx_unpacked_to_int8_tensor()
        if gtensor.ggml_type == "q6_k":
            import torch.nn as nn
            from executorch.backends.cuda.dp4a_planar_int6_tensor import (
                CudaDp4aPlanarInt6Tensor,
            )

            model.lm_head.weight = nn.Parameter(
                CudaDp4aPlanarInt6Tensor._from_intx_int8(intx), requires_grad=False
            )
            return intx, None
        # Q4_K: ``weight`` is the native Int4Tensor; let _resolve_tied_lm_head
        # pack it to CudaCoalescedInt4Tensor. Only the embedding switches to int8.
        return intx, weight

    from executorch.examples.models.gemma4_31b.quant import dequantize_weight

    return dequantize_weight(weight, torch.bfloat16), weight


def load_gguf_model(
    gguf_path: str,
    max_seq_len: int = 4096,
    backend: str = "cuda",
    config=None,
) -> tuple:
    """Load a GGUF file, remap keys, and convert weights for the target backend.

    Streams tensors one at a time for low peak memory. GGUF ties ``embed_tokens``
    and ``lm_head``: on MLX they stay tied (one shared quantized tensor); on CUDA
    they are untied so the embedding can be dequantized for the gather while
    ``lm_head`` keeps its quantization. See the module docstring for the
    per-backend conversion details.

    ``config`` defaults to the full Gemma 4 31B config; pass a smaller
    ``Gemma4_31BConfig`` (e.g. in tests) to load a GGUF for a tiny model.

    Returns ``(model, config)``.
    """
    from executorch.examples.models.gemma4_31b.model import (
        Gemma4_31B,
        Gemma4_31BConfig,
        materialize_runtime_buffers,
    )
    from executorch.examples.models.gemma4_31b.quant import pack_one
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor, iter_gguf

    if backend == "cuda":
        from executorch.examples.models.gemma4_31b.quant import DEFAULT_CUDA_PACKERS

        packers = DEFAULT_CUDA_PACKERS
    elif backend == "mlx":
        from executorch.examples.models.gemma4_31b.quant import DEFAULT_MLX_PACKERS

        packers = DEFAULT_MLX_PACKERS
    else:
        raise ValueError(f"Unsupported backend: {backend!r}. Supported: 'cuda', 'mlx'.")

    if config is None:
        config = Gemma4_31BConfig(max_seq_len=max_seq_len)

    print("Building model on meta device...")
    with torch.device("meta"):
        model = Gemma4_31B(config)

    lm_head_weight = None  # tied weight resolved into lm_head after the loop
    n_processed = 0

    print(f"Streaming GGUF from {gguf_path}...")
    for gguf_name, value in iter_gguf(gguf_path):
        model_key = gguf_to_model_key(gguf_name)
        if model_key is None:
            continue

        if isinstance(value, ExportableGGUFTensor):
            weight = _convert_weight(model, model_key, value, backend)
            if model_key == "embed_tokens.weight":
                weight, lm_head_weight = _untie_embed_lm_head(
                    model, value, weight, backend
                )
            value = weight
        elif value.dtype == torch.float32:
            value = value.to(torch.bfloat16)

        pack_one(model, model_key, value, packers)

        n_processed += 1
        if n_processed % 100 == 0:
            print(f"  Processed {n_processed} tensors...")

    _resolve_tied_lm_head(model, lm_head_weight, packers)

    # Fill RoPE tables / KV caches / scalar constants (left on meta by the
    # streaming load), matching load_prequantized_model so the CUDA and eager
    # forward paths get bf16 runtime buffers instead of float32 defaults.
    materialize_runtime_buffers(model, dtype=torch.bfloat16)

    _validate_no_meta(model)
    model.eval()

    print(f"Model: {config.num_hidden_layers} layers, hidden={config.hidden_size}")
    return model, config
