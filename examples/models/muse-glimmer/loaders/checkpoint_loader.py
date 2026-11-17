# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Load Muse Glimmer checkpoints and pack weights for CUDA or MLX.

Input projections remain unfused until loading so mixed-precision GGUF,
prequantized safetensors, MLX-affine, and consolidated BF16 inputs share the
same backend-packing path.
"""

from __future__ import annotations

import os
import re
from typing import Callable

import torch

# GGUF tensor names to atomic Muse Glimmer FQNs.

_GGUF_KEY_MAP: dict[str, str] = {
    "token_embd.weight": "embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "output_norm.weight",
    "blk.{}.attn_q.weight": "layers.{}.self_attn.q_proj.weight",
    "blk.{}.attn_k.weight": "layers.{}.self_attn.k_proj.weight",
    "blk.{}.attn_v.weight": "layers.{}.self_attn.v_proj.weight",
    "blk.{}.attn_output.weight": "layers.{}.self_attn.o_proj.weight",
    "blk.{}.attn_output_gate.weight": "layers.{}.self_attn.og_proj.weight",
    # The 30B GGUFs renamed the output gate; older checkpoints still ship the
    # attn_output_gate spelling, so both names have to resolve.
    "blk.{}.attn_gate.weight": "layers.{}.self_attn.og_proj.weight",
    "blk.{}.attn_norm.weight": "layers.{}.self_attn.qkv_proj_norm.weight",
    "blk.{}.ffn_norm.weight": "layers.{}.mlp.norm.weight",
    "blk.{}.post_attention_norm.weight": "layers.{}.post_attn_norm.weight",
    "blk.{}.post_ffw_norm.weight": "layers.{}.post_ffn_norm.weight",
    "blk.{}.ffn_gate.weight": "layers.{}.mlp.gate_proj.weight",
    "blk.{}.ffn_up.weight": "layers.{}.mlp.up_proj.weight",
    "blk.{}.ffn_down.weight": "layers.{}.mlp.down_proj.weight",
}

_GGUF_IGNORED_KEYS = {"rope_freqs.weight"}
# Scaleless QK-norm weights (all-ones); model uses RMSNormNoWeight (no param).
_GGUF_QK_NORM_SUBSTRS = ("attn_q_norm", "attn_k_norm")


def _match_key_map(name: str, key_map: dict[str, str]) -> str | None:
    """Resolve a checkpoint tensor name through a ``{}``-templated key map."""
    for pat, target in key_map.items():
        if "{}" not in pat:
            if name == pat:
                return target
            continue
        prefix, suffix = pat.split("{}")
        if name.startswith(prefix) and name.endswith(suffix):
            layer = name[len(prefix) : len(name) - len(suffix)]
            if layer.isdigit():
                return target.replace("{}", layer)
    return None


def gguf_to_model_key(gguf_key: str) -> str | None:
    """Map a GGUF tensor name to an atomic (unfused) Muse Glimmer FQN, or ``None``."""
    if gguf_key in _GGUF_IGNORED_KEYS:
        return None
    if any(s in gguf_key for s in _GGUF_QK_NORM_SUBSTRS):
        return None
    return _match_key_map(gguf_key, _GGUF_KEY_MAP)


# MLX Hugging Face tensor names to atomic Muse Glimmer FQNs.
# Norm roles follow the checkpoint decoder; scaleless norms have no weights.
_MLX_HF_KEY_MAP: dict[str, str] = {
    "language_model.model.embed_tokens.weight": "embed_tokens.weight",
    "language_model.lm_head.weight": "lm_head.weight",
    "language_model.model.norm.weight": "output_norm.weight",
    "language_model.model.layers.{}.input_layernorm.weight": "layers.{}.self_attn.qkv_proj_norm.weight",
    "language_model.model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp.norm.weight",
    "language_model.model.layers.{}.post_attn_norm.weight": "layers.{}.post_attn_norm.weight",
    "language_model.model.layers.{}.post_ffn_norm.weight": "layers.{}.post_ffn_norm.weight",
    "language_model.model.layers.{}.self_attn.q_proj.weight": "layers.{}.self_attn.q_proj.weight",
    "language_model.model.layers.{}.self_attn.k_proj.weight": "layers.{}.self_attn.k_proj.weight",
    "language_model.model.layers.{}.self_attn.v_proj.weight": "layers.{}.self_attn.v_proj.weight",
    "language_model.model.layers.{}.self_attn.o_proj.weight": "layers.{}.self_attn.o_proj.weight",
    "language_model.model.layers.{}.self_attn.output_gate_proj.weight": "layers.{}.self_attn.og_proj.weight",
    "language_model.model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.gate_proj.weight",
    "language_model.model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.up_proj.weight",
    "language_model.model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.down_proj.weight",
}


def mlx_hf_to_model_key(hf_key: str) -> str | None:
    """Map an MLX (HF ``MuseGlimmerForCausalLM``) tensor name to an atomic MuseGlimmer FQN."""
    return _match_key_map(hf_key, _MLX_HF_KEY_MAP)


# MLX affine-quantized safetensors reader (injected into iter_checkpoint via
# its ``raw_iter`` hook, so the shared loader stays MLX-agnostic).


def _mlx_shard_paths(path: str) -> list[str]:
    """Resolve the safetensors shard file(s) for an MLX checkpoint dir."""
    import json

    if not os.path.isdir(path):
        return [path]
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, "r") as f:
            weight_map = json.load(f)["weight_map"]
        return [os.path.join(path, s) for s in sorted(set(weight_map.values()))]
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        return [single]
    raise FileNotFoundError(f"No safetensors checkpoint in {path}")


def _mlx_affine_intx(packed, scales, biases, bits, group_size):
    """Rebuild an MLX affine-quantized weight as a torchao ``IntxUnpackedToInt8Tensor``.

    ``packed`` is uint32 (``bits`` codes/word, little-endian bit-contiguous);
    ``scales`` + optional ``biases`` are per group. Codes are centered to signed
    and the affine bias folded into a float zero-point so ``scale * (qdata -
    zero_point)`` reproduces MLX exactly (same rewrite as gguf.py for Q4_K/Q5_K).

    NOTE: assumes MLX's little-endian bit-contiguous packing (``group_size *
    bits`` is a multiple of 32, so groups stay word-aligned).
    """
    import numpy as np
    from torchao.quantization import IntxUnpackedToInt8Tensor

    offset = 1 << (bits - 1)
    # 6-bit uses an int8 container (like gguf.py's Q6_K); 4/5/8 map directly.
    target_dtype = {4: torch.int4, 5: torch.int5, 6: torch.int8, 8: torch.int8}[bits]

    n = int(packed.shape[0])
    k = int(scales.shape[1]) * group_size

    # int32 view avoids torch's limited uint32->numpy support; bit pattern is
    # identical.
    words = packed.contiguous().cpu()
    if words.dtype == torch.uint32:
        words = words.view(torch.int32)
    byte_view = torch.from_numpy(words.numpy().view(np.uint8))  # [N, 4W]

    # Extract each code by ``bits`` vectorized bit-plane gathers (byte index +
    # intra-byte shift precomputed per code) in torch int8.
    pos = torch.arange(k, dtype=torch.int64) * bits
    pos = pos.unsqueeze(0) + torch.arange(bits, dtype=torch.int64).unsqueeze(1)
    byte_idx = pos >> 3  # [bits, k]
    byte_sh = (pos & 7).to(torch.uint8)  # [bits, k]
    qdata = torch.empty((n, k), dtype=torch.int8)
    for i in range(0, n, 65536):
        blk = byte_view[i : i + 65536]
        codes = torch.zeros((blk.shape[0], k), dtype=torch.int8)
        for t in range(bits):
            codes += (((blk[:, byte_idx[t]] >> byte_sh[t]) & 1) << t).to(torch.int8)
        qdata[i : i + 65536] = codes - int(offset)

    scales_f = scales.to(torch.float32)
    if biases is not None:
        zero = torch.where(
            scales_f != 0,
            -biases.to(torch.float32) / scales_f,
            torch.zeros_like(scales_f),
        )
    else:
        zero = torch.full_like(scales_f, float(offset))

    return IntxUnpackedToInt8Tensor(
        qdata=qdata,
        scale=scales,
        zero_point=(zero - offset).to(scales.dtype),
        target_dtype=target_dtype,
        block_size=(1, group_size),
        dtype=scales.dtype,
        activation_quantization=None,
    )


def _iter_mlx(path: str):
    """Yield raw ``(name, tensor)`` pairs from an MLX quantized dir (any format).

    One reader for every mlx_lm quant format. Each weight's *type* is resolved
    per-tensor from ``config.json["quantization"]``: a default ``mode`` (absent
    => ``affine``) / ``bits`` / ``group_size`` plus per-module override dicts
    keyed by the HF module name (no ``.weight`` suffix), e.g.
    ``"language_model.model.layers.0.self_attn.v_proj": {"mode": "mxfp8", ...}``.
    The safetensors carry no per-tensor metadata, so these config entries are the
    per-tensor type source.

    Each quantized weight is a ``X.weight`` (uint32 bit-packed codes) + a
    ``X.scales`` pair, plus ``X.biases`` for affine. Dispatch by mode:

      * ``affine`` -> torchao ``IntxUnpackedToInt8Tensor`` (bit-unpacked; affine
        bias folded into a float zero-point).
      * ``nvfp4``  -> ``ExportableNVFP4Tensor`` (per-tensor scale 1.0 -- these
        checkpoints carry no global scale).
      * ``mxfp8``  -> ``ExportableMXTensor``.

    nvfp4/mxfp8 keep the MLX-native packed uint32/uint8 verbatim (the MLX kernel
    consumes them directly); only affine is bit-unpacked. Unquantized tensors
    (norms) pass through. Mixed-format checkpoints are handled per weight.
    """
    import contextlib
    import json

    from executorch.extension.llm.export.mx import ExportableMXTensor
    from executorch.extension.llm.export.nvfp4 import ExportableNVFP4Tensor
    from safetensors import safe_open

    with open(os.path.join(path, "config.json"), "r") as f:
        quant = json.load(f)["quantization"]
    default_mode = quant.get("mode", "affine")
    # Per-weight overrides are the dict-valued entries, keyed by HF module name.
    overrides = {k: v for k, v in quant.items() if isinstance(v, dict)}

    def _params_for(base: str):
        cfg = overrides.get(base, {})
        return (
            cfg.get("mode", default_mode),
            int(cfg.get("bits", quant["bits"])),
            int(cfg.get("group_size", quant["group_size"])),
        )

    with contextlib.ExitStack() as stack:
        handles = [
            stack.enter_context(safe_open(p, framework="pt", device="cpu"))
            for p in _mlx_shard_paths(path)
        ]
        key_to_handle: dict = {}
        for handle in handles:
            for key in handle.keys():
                key_to_handle[key] = handle

        n_weights = sum(1 for key in key_to_handle if key.endswith(".weight"))
        n_done = 0
        for key, handle in key_to_handle.items():
            if key.endswith(".scales") or key.endswith(".biases"):
                continue  # consumed alongside the paired ``.weight``
            if not key.endswith(".weight"):
                yield key, handle.get_tensor(key)
                continue
            n_done += 1
            if n_done % 25 == 0 or n_done == n_weights:
                print(f"  MLX unpack: {n_done}/{n_weights} weights...", flush=True)

            base = key[: -len(".weight")]
            scales_key = f"{base}.scales"
            if scales_key not in key_to_handle:
                yield key, handle.get_tensor(key)  # unquantized (e.g. a norm)
                continue

            qdata = handle.get_tensor(key)  # uint32 [N, K*bits/32]
            scales = key_to_handle[scales_key].get_tensor(scales_key)  # [N, groups]
            mode, bits, group_size = _params_for(base)

            if mode == "nvfp4":
                pts = torch.tensor(1.0, dtype=torch.float32)
                yield (
                    key,
                    ExportableNVFP4Tensor(
                        qdata,
                        scales,
                        pts,
                        block_size=group_size,
                        orig_dtype=torch.bfloat16,
                    ),
                )
            elif mode == "mxfp8":
                # On disk MLX packs 4 E4M3 codes per uint32 and stores E8M0
                # scales as uint8; reinterpret to the backend-agnostic native
                # dtypes ExportableMXTensor holds (the MLX handler repacks).
                qdata_fp8 = qdata.view(torch.uint8).view(torch.float8_e4m3fn)
                scale_e8m0 = scales.view(torch.uint8).view(torch.float8_e8m0fnu)
                yield (
                    key,
                    ExportableMXTensor(
                        qdata_fp8,
                        scale_e8m0,
                        elem_dtype=torch.float8_e4m3fn,
                        block_size=group_size,
                        orig_dtype=torch.bfloat16,
                    ),
                )
            elif mode == "affine":
                biases_key = f"{base}.biases"
                biases = (
                    key_to_handle[biases_key].get_tensor(biases_key)
                    if biases_key in key_to_handle
                    else None
                )
                yield key, _mlx_affine_intx(qdata, scales, biases, bits, group_size)
            else:
                raise ValueError(
                    f"Unsupported MLX quantization mode {mode!r} for {key!r}"
                )


# Backend fusion groups: fused FQN -> ordered atomic shard FQNs.
# Slice order matches the fused forward in model.py (qko = [Q, K, OG];
# gate_up = [gate, up]). v_proj stays standalone on CUDA for its own bit-width.

_FUSION_GROUPS: dict[str, dict[str, list[str]]] = {
    "cuda": {
        "layers.{}.self_attn.qko_proj.weight": [
            "layers.{}.self_attn.q_proj.weight",
            "layers.{}.self_attn.k_proj.weight",
            "layers.{}.self_attn.og_proj.weight",
        ],
        "layers.{}.mlp.gate_up_proj.weight": [
            "layers.{}.mlp.gate_proj.weight",
            "layers.{}.mlp.up_proj.weight",
        ],
    },
    "mlx": {
        "layers.{}.mlp.gate_up_proj.weight": [
            "layers.{}.mlp.gate_proj.weight",
            "layers.{}.mlp.up_proj.weight",
        ],
    },
}


def _fuse_state_dict(
    atomic_sd: dict, backend: str, n_layers: int
) -> tuple[dict, set[str]]:
    """Fuse the atomic shards into the backend's fused projections.

    Returns a state dict keyed by the fused model's FQNs, plus the group templates
    that fused successfully. Fused entries use ``fuse_along_output``
    (subclass-aware output-channel concat). A group stays atomic across every
    layer if any layer contains incompatible quantization types.
    """
    from executorch.extension.llm.export.quant import fuse_along_output

    groups = _FUSION_GROUPS[backend]
    consumed: set[str] = set()
    fused_sd: dict[str, torch.Tensor] = {}
    fused_groups: set[str] = set()
    for fused_tmpl, shard_tmpls in groups.items():
        group_sd = {}
        group_consumed = set()
        try:
            for layer in range(n_layers):
                fused_fqn = fused_tmpl.format(layer)
                shard_fqns = [t.format(layer) for t in shard_tmpls]
                if fused_fqn in atomic_sd:
                    group_sd[fused_fqn] = atomic_sd[fused_fqn]
                    group_consumed.add(fused_fqn)
                    continue
                group_sd[fused_fqn] = fuse_along_output(
                    [atomic_sd[f] for f in shard_fqns]
                )
                group_consumed.update(shard_fqns)
        except ValueError as exc:
            print(f"Keeping {fused_tmpl} projections separate: {exc}")
            continue
        fused_sd.update(group_sd)
        consumed.update(group_consumed)
        fused_groups.add(fused_tmpl)
    for fqn, value in atomic_sd.items():
        if fqn not in consumed:
            fused_sd[fqn] = value
    return fused_sd, fused_groups


# Source heads: produce an atomic (unfused) quantized state dict.


def _atomic_sd_from_gguf(gguf_path: str, activation_dtype: torch.dtype) -> dict:
    from executorch.extension.llm.export.load import iter_checkpoint
    from executorch.extension.llm.export.quant import to_default

    return dict(
        iter_checkpoint(
            gguf_path,
            key_map=gguf_to_model_key,
            convert=to_default,
            dtype=activation_dtype,
        )
    )


def _atomic_sd_from_safetensors(
    safetensors_path: str, activation_dtype: torch.dtype
) -> dict:
    from executorch.extension.llm.export.load import iter_checkpoint
    from executorch.extension.llm.export.quant import to_default

    return dict(
        iter_checkpoint(safetensors_path, convert=to_default, dtype=activation_dtype)
    )


def _atomic_sd_from_bf16(
    checkpoint_dir: str, recipe, max_seq_len: int | None, activation_dtype: torch.dtype
) -> tuple[dict, object]:
    from executorch.examples.models.muse_glimmer.model.model import (
        load_unfused_bf16_state_dict,
    )
    from executorch.extension.llm.export.quant import quantize_stream, to_default

    bf16_sd, config = load_unfused_bf16_state_dict(checkpoint_dir, max_seq_len)
    # Quantize on the fly, then normalize to the portable form (Int4Tensor ->
    # ExportableInt4Tensor) so the downstream fuse/backend pass matches the
    # safetensors/GGUF paths exactly.
    atomic_sd = {
        fqn: to_default(fqn, value)
        for fqn, value in quantize_stream(
            bf16_sd.items(), recipe, dtype=activation_dtype
        )
    }
    return atomic_sd, config


# Shared tail: fuse -> build fused model -> assign -> backend pass.


def _finalize(
    atomic_sd: dict,
    backend: str,
    config,
    activation_dtype: torch.dtype,
    *,
    defer_runtime_buffers: bool = False,
):
    from executorch.examples.models.muse_glimmer.model.model import (
        materialize_runtime_buffers,
        MuseGlimmerModel,
    )
    from executorch.extension.llm.export.load import assign_state_dict
    from executorch.extension.llm.export.quant import identity

    if backend == "cuda" and any(
        fqn.endswith(".self_attn.qkv_proj.weight") for fqn in atomic_sd
    ):
        from executorch.extension.llm.export.int4 import ExportableInt4Tensor

        q_dim = config.n_heads * config.head_dim
        kv_dim = config.n_kv_heads * config.head_dim
        split_sd = {}
        for fqn, value in atomic_sd.items():
            if not fqn.endswith(".self_attn.qkv_proj.weight"):
                split_sd[fqn] = value
                continue
            prefix = fqn[: -len("qkv_proj.weight")]
            if isinstance(value, ExportableInt4Tensor):

                def rows(start, end, value=value):
                    return ExportableInt4Tensor(
                        value.qdata[start:end],
                        value.scale[:, start:end],
                        value.zero_point[:, start:end],
                        value.group_size,
                        value.orig_dtype,
                    )

                q = rows(0, q_dim)
                k = rows(q_dim, q_dim + kv_dim)
                v = rows(q_dim + kv_dim, q_dim + 2 * kv_dim)
                og = rows(q_dim + 2 * kv_dim, value.shape[0])
                from executorch.extension.llm.export.quant import fuse_along_output

                split_sd[f"{prefix}qko_proj.weight"] = fuse_along_output([q, k, og])
                split_sd[f"{prefix}v_proj.weight"] = v
            else:
                from executorch.examples.models.muse_glimmer.model.model import (
                    _split_fused_qkv_to_qko,
                )

                split_sd.update(_split_fused_qkv_to_qko({fqn: value}, config))
        atomic_sd = split_sd

    fused_sd, fused_groups = _fuse_state_dict(atomic_sd, backend, config.n_layers)

    # Fused runtime layout for the target backend.
    config.fuse_qkv = False
    config.fuse_qko = "layers.{}.self_attn.qko_proj.weight" in fused_groups
    config.fuse_gate_up = "layers.{}.mlp.gate_up_proj.weight" in fused_groups
    print("Building fused model on meta device...")
    with torch.device("meta"):
        model = MuseGlimmerModel(config)

    # Weights are already in portable form; assign directly (convert=identity).
    assign_state_dict(model, fused_sd, convert=identity)

    if backend == "cuda":
        from executorch.examples.models.gemma4_31b.cuda_packers import (
            convert_quantized_tensors_for_cuda,
        )

        print("Converting quantized tensors for CUDA...")
        convert_quantized_tensors_for_cuda(model)

    # The off-graph MLX export replaces attention before materialization, so
    # the original meta KV buffers must not allocate their full context here.
    if not defer_runtime_buffers:
        materialize_runtime_buffers(model, dtype=activation_dtype)
    model.eval()
    print(f"Model: {config.n_layers} layers, dim={config.dim}")
    return model


def _validate_backend(backend: str) -> None:
    if backend not in ("cuda", "mlx"):
        raise ValueError(f"Unsupported backend: {backend!r}. Supported: 'cuda', 'mlx'.")


# GGUF requantization (MLX): decoder linears and lm_head are dequantized in
# row chunks and requantized, one weight (or fusion group) at a time.

_REQUANT_FUSION_GROUPS: dict[str, list[str]] = {
    "layers.{}.self_attn.qkv_proj": [
        "layers.{}.self_attn.q_proj",
        "layers.{}.self_attn.k_proj",
        "layers.{}.self_attn.v_proj",
        "layers.{}.self_attn.og_proj",
    ],
    "layers.{}.mlp.gate_up_proj": [
        "layers.{}.mlp.gate_proj",
        "layers.{}.mlp.up_proj",
    ],
}


def _is_requant_linear(fqn: str) -> bool:
    return fqn == "lm_head.weight" or (
        fqn.startswith("layers.") and fqn.endswith("_proj.weight")
    )


def _float_rows(value: torch.Tensor) -> Callable[[int, int], torch.Tensor]:
    """``rows(start, end)`` dequantizes rows ``[start, end)`` of a GGUF (or
    float) ``[N, K]`` weight to fp32 without materializing the rest."""
    from executorch.extension.llm.export.gguf import (
        _dequantize_gguf,
        ExportableGGUFTensor,
    )

    if isinstance(value, ExportableGGUFTensor):

        def rows(start, end):
            return _dequantize_gguf(
                value.raw[start:end], value.ggml_type, torch.float32
            )

    elif type(value) is torch.Tensor and value.dtype.is_floating_point:

        def rows(start, end):
            return value[start:end].float()

    else:
        raise TypeError(f"Cannot requantize {type(value).__name__}")
    return rows


# Per-channel MLX requant kinds: symmetric int4 weights (a zero point of 0) run
# by mlx::{int8_sym,fp}_int4_affine_gemm.
_PER_CHANNEL_ACTS = {"int8_int4": "int8_sym", "fp_int4": "fp"}


def _per_channel_linear(kind: str, w_packed: torch.Tensor, w_scale: torch.Tensor):
    from executorch.backends.mlx.custom_kernel_ops.int4_affine_gemm import (
        Int4AffineLinear,
    )

    scale = w_scale[:, None]
    return Int4AffineLinear(
        w_packed, scale, torch.zeros_like(scale), _PER_CHANNEL_ACTS[kind]
    )


def _per_channel_sd_from_gguf(
    gguf_path: str, activation_dtype: torch.dtype, kind: str
) -> tuple[dict, dict]:
    """Stream a GGUF into ``(other_sd, packed)``: ``packed`` maps each linear
    module FQN to ``(w_packed, w_scale)``, requantized to symmetric per-channel
    int4; everything else lands in ``other_sd`` cast to ``activation_dtype``."""
    from executorch.backends.mlx.custom_kernel_ops.int4_affine_gemm import (
        quantize_int4_sym_per_channel,
    )
    from executorch.extension.llm.export.load import iter_checkpoint
    from executorch.extension.llm.export.quant import to_default
    from executorch.extension.llm.export.quant.convert import maybe_cast

    other_sd, packed = {}, {}
    for fqn, value in iter_checkpoint(
        gguf_path, key_map=gguf_to_model_key, convert=to_default
    ):
        if _is_requant_linear(fqn):
            packed[fqn[: -len(".weight")]] = quantize_int4_sym_per_channel(
                _float_rows(value), value.shape[0]
            )
            print(f"  {kind}: {fqn} {tuple(value.shape)}", flush=True)
        else:
            other_sd[fqn] = maybe_cast(value, activation_dtype)
        del value
    return other_sd, packed


def _fuse_packed(packed: dict, n_layers: int) -> None:
    """Concatenate packed shards (tuples of tensors whose rows are output
    channels, e.g. ``(w_packed, w_scale)``) into fused projections, in place.
    Per-row quantization parameters make output-row concatenation exact."""
    for fused_tmpl, shard_tmpls in _REQUANT_FUSION_GROUPS.items():
        for layer in range(n_layers):
            shards = [t.format(layer) for t in shard_tmpls]
            shards = [s for s in shards if s in packed]
            if not shards:
                continue
            parts = [packed.pop(s) for s in shards]
            packed[fused_tmpl.format(layer)] = tuple(
                torch.cat(field) for field in zip(*parts)
            )


def _fuse_weights(sd: dict, n_layers: int) -> None:
    """Fuse requantized shard weights (``ExportableInt4Tensor`` etc.) left in
    ``sd`` into the same fused projections, in place."""
    from executorch.extension.llm.export.quant import fuse_along_output

    for fused_tmpl, shard_tmpls in _REQUANT_FUSION_GROUPS.items():
        for layer in range(n_layers):
            shards = [f"{t.format(layer)}.weight" for t in shard_tmpls]
            shards = [s for s in shards if s in sd]
            if shards:
                sd[f"{fused_tmpl.format(layer)}.weight"] = fuse_along_output(
                    [sd.pop(s) for s in shards]
                )


def _finalize_requant(
    other_sd: dict,
    packed: dict,
    config,
    activation_dtype: torch.dtype,
    make_linear: Callable[..., torch.nn.Module],
    in_per_byte: int,
    label: str,
    *,
    defer_runtime_buffers: bool = False,
):
    """Build the fused model with each ``packed`` module replaced by
    ``make_linear(*packed[fqn])`` (``in_per_byte`` input features per packed
    byte) and ``other_sd`` assigned to the rest."""
    from executorch.examples.models.muse_glimmer.model.model import (
        materialize_runtime_buffers,
        MuseGlimmerModel,
    )
    from executorch.extension.llm.export.load import assign_state_dict
    from executorch.extension.llm.export.quant import identity

    _fuse_packed(packed, config.n_layers)
    _fuse_weights(other_sd, config.n_layers)

    def fused(module: str) -> bool:
        return module in packed or f"{module}.weight" in other_sd

    config.fuse_qko = False
    config.fuse_qkv = fused("layers.0.self_attn.qkv_proj")
    config.fuse_gate_up = fused("layers.0.mlp.gate_up_proj")
    print("Building fused model on meta device...")
    with torch.device("meta"):
        model = MuseGlimmerModel(config)

    for fqn, fields in packed.items():
        parent_fqn, _, attr = fqn.rpartition(".")
        parent = model.get_submodule(parent_fqn) if parent_fqn else model
        old = getattr(parent, attr)
        w_packed = fields[0]
        shape = (w_packed.shape[0], in_per_byte * w_packed.shape[1])
        if (old.out_features, old.in_features) != shape:
            raise ValueError(
                f"{fqn}: checkpoint shape {shape} does not match model "
                f"{(old.out_features, old.in_features)}"
            )
        setattr(parent, attr, make_linear(*fields))
    packed.clear()

    assign_state_dict(model, other_sd, convert=identity)
    if not defer_runtime_buffers:
        materialize_runtime_buffers(model, dtype=activation_dtype)
    model.eval()
    print(f"Model: {config.n_layers} layers, dim={config.dim} ({label} linears)")
    return model


def _finalize_per_channel(
    other_sd: dict,
    packed: dict,
    config,
    activation_dtype: torch.dtype,
    kind: str,
    *,
    defer_runtime_buffers: bool = False,
):
    return _finalize_requant(
        other_sd,
        packed,
        config,
        activation_dtype,
        lambda w_packed, w_scale: _per_channel_linear(kind, w_packed, w_scale),
        2,
        kind,
        defer_runtime_buffers=defer_runtime_buffers,
    )


# Asymmetric int4 affine MLX path: each layer group (attention, FFN, lm_head)
# gets its own scheme. "pc" / "gs256" requantize to int4 with a float scale and
# zero point per channel / per 256 weights, run by mlx::int8_asym_int4_affine_gemm
# or mlx::fp_int4_affine_gemm (int8 or fp activations); "mlx" requantizes to
# 4-bit affine gs=32, run by MLX's built-in quantized matmul.

AFFINE_GROUPS = ("attn", "ffn", "lm_head")
AFFINE_SCHEMES = ("pc", "gs256", "mlx")
_AFFINE_GROUP_SIZE = {"pc": None, "gs256": 256}
_REQUANT_CHUNK_ROWS = 4096


def _affine_group(fqn: str) -> str:
    if fqn == "lm_head.weight":
        return "lm_head"
    if ".self_attn." in fqn:
        return "attn"
    if ".mlp." in fqn:
        return "ffn"
    raise ValueError(f"{fqn}: not an attention, FFN, or lm_head linear")


def _mlx_q4_gs32(
    rows: Callable[[int, int], torch.Tensor], n: int, activation_dtype: torch.dtype
) -> torch.Tensor:
    """4-bit asymmetric min / max gs=32 ``ExportableInt4Tensor`` (MLX's affine
    quantized matmul), quantized ``_REQUANT_CHUNK_ROWS`` rows at a time."""
    from executorch.extension.llm.export.quant import (
        fuse_along_output,
        QuantConfig,
        to_default,
    )
    from executorch.extension.llm.export.quant.convert import maybe_cast
    from executorch.extension.llm.export.quant.quantize import quantize_weight

    config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
    chunks = [
        maybe_cast(
            to_default(
                "",
                quantize_weight(
                    rows(start, min(start + _REQUANT_CHUNK_ROWS, n)), config
                ),
            ),
            activation_dtype,
        )
        for start in range(0, n, _REQUANT_CHUNK_ROWS)
    ]
    return chunks[0] if len(chunks) == 1 else fuse_along_output(chunks)


def _affine_sd_from_gguf(
    gguf_path: str, activation_dtype: torch.dtype, schemes: dict[str, str]
) -> tuple[dict, dict]:
    """Stream a GGUF into ``(other_sd, packed)``: ``packed`` maps each "pc" /
    "gs256" linear module FQN to ``(w_packed, scale, zero)`` from
    ``quantize_int4_affine``; "mlx" linears land in ``other_sd`` as
    ``ExportableInt4Tensor`` weights, everything else cast to
    ``activation_dtype``."""
    from executorch.backends.mlx.custom_kernel_ops.int4_affine_gemm import (
        quantize_int4_affine,
    )
    from executorch.extension.llm.export.load import iter_checkpoint
    from executorch.extension.llm.export.quant import to_default
    from executorch.extension.llm.export.quant.convert import maybe_cast

    other_sd, packed = {}, {}
    for fqn, value in iter_checkpoint(
        gguf_path, key_map=gguf_to_model_key, convert=to_default
    ):
        if _is_requant_linear(fqn):
            scheme = schemes[_affine_group(fqn)]
            rows, n = _float_rows(value), value.shape[0]
            if scheme == "mlx":
                other_sd[fqn] = _mlx_q4_gs32(rows, n, activation_dtype)
            else:
                packed[fqn[: -len(".weight")]] = quantize_int4_affine(
                    rows, n, _AFFINE_GROUP_SIZE[scheme], _REQUANT_CHUNK_ROWS
                )
            print(f"  int4 affine {scheme}: {fqn} {tuple(value.shape)}", flush=True)
        else:
            other_sd[fqn] = maybe_cast(value, activation_dtype)
        del value
    return other_sd, packed


def _finalize_affine(
    other_sd: dict,
    packed: dict,
    config,
    activation_dtype: torch.dtype,
    act: str,
    schemes: dict[str, str],
    *,
    defer_runtime_buffers: bool = False,
):
    from executorch.backends.mlx.custom_kernel_ops.int4_affine_gemm import (
        Int4AffineLinear,
    )

    label = f"int4 affine {act}: " + ", ".join(
        f"{g} {schemes[g]}" for g in AFFINE_GROUPS
    )
    return _finalize_requant(
        other_sd,
        packed,
        config,
        activation_dtype,
        lambda w_packed, scale, zero: Int4AffineLinear(w_packed, scale, zero, act),
        2,
        label,
        defer_runtime_buffers=defer_runtime_buffers,
    )


# NVFP4 MLX path: E2M1 values with E4M3 scales per 16 weights and an fp32
# per-tensor scale. Shards of one fusion group share the per-tensor scale so they
# fuse exactly.
#
# MLX runs the nvfp4 matmul without the per-tensor scale and multiplies it into
# the output afterwards, so the fp16 matmul output is inflated by 1 / scale.
# torchao's choice (largest block scale -> E4M3 max 448) inflates by ~10^4 and
# overflows fp16 for MG's K. A scale _NVFP4_HEADROOM times larger, rounded up to
# a power of two, keeps block scales out of E4M3 subnormals (same accuracy on MG
# weights) while inflating the output only ~16-128x.

_NVFP4_BLOCK = 16
_NVFP4_CHUNK_ROWS = 4096
_NVFP4_HEADROOM = 256


def _quantize_nvfp4_group(
    values: list[torch.Tensor], activation_dtype: torch.dtype
) -> list[torch.Tensor]:
    """Requantize weights to ``ExportableNVFP4Tensor`` with one shared
    per-tensor scale, dequantizing ``_NVFP4_CHUNK_ROWS`` rows at a time."""
    from executorch.extension.llm.export.nvfp4 import ExportableNVFP4Tensor
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        nvfp4_quantize,
        per_tensor_amax_to_scale,
    )

    sources = [(_float_rows(v), v.shape[0]) for v in values]

    def chunks(rows, n):
        for start in range(0, n, _NVFP4_CHUNK_ROWS):
            yield rows(start, min(start + _NVFP4_CHUNK_ROWS, n))

    amax = max(w.abs().max() for rows, n in sources for w in chunks(rows, n))
    scale = per_tensor_amax_to_scale(amax).float() * _NVFP4_HEADROOM
    per_tensor_scale = torch.exp2(torch.ceil(torch.log2(scale)))
    out = []
    for rows, n in sources:
        qdata, scales = [], []
        for w in chunks(rows, n):
            s, q = nvfp4_quantize(w, _NVFP4_BLOCK, per_tensor_scale)
            scales.append(s.view(torch.uint8))
            qdata.append(q)
        out.append(
            ExportableNVFP4Tensor(
                torch.cat(qdata).view(torch.uint32),
                torch.cat(scales),
                # MLX multiplies this into the matmul output; fp32 would
                # promote it. Exact: the scale is a power of two.
                per_tensor_scale.to(activation_dtype),
                _NVFP4_BLOCK,
                activation_dtype,
            )
        )
    return out


def _requant_group(fqn: str) -> tuple[str, int] | None:
    """``(fused template, layer)`` if ``fqn`` is a shard of a fusion group."""
    if not fqn.startswith("layers."):
        return None
    layer = int(fqn.split(".")[1])
    module = fqn[: -len(".weight")]
    for tmpl, shard_tmpls in _REQUANT_FUSION_GROUPS.items():
        if any(module == t.format(layer) for t in shard_tmpls):
            return tmpl, layer
    return None


def _nvfp4_sd_from_gguf(gguf_path: str, activation_dtype: torch.dtype) -> dict:
    """Stream a GGUF into a state dict whose decoder linears and ``lm_head``
    are ``ExportableNVFP4Tensor``s, with q/k/v/og and gate/up already fused.

    Fusion-group shards are held (still GGUF-compressed) until their layer is
    complete, then requantized together with a shared per-tensor scale."""
    from executorch.extension.llm.export.load import iter_checkpoint
    from executorch.extension.llm.export.quant import fuse_along_output, to_default
    from executorch.extension.llm.export.quant.convert import maybe_cast

    sd: dict[str, torch.Tensor] = {}
    pending: dict[tuple[str, int], dict[str, torch.Tensor]] = {}

    def flush(keep_layer: int | None = None) -> None:
        for key in [k for k in pending if k[1] != keep_layer]:
            tmpl, layer = key
            shards = pending.pop(key)
            order = [
                f"{t.format(layer)}.weight"
                for t in _REQUANT_FUSION_GROUPS[tmpl]
                if f"{t.format(layer)}.weight" in shards
            ]
            fused = fuse_along_output(
                _quantize_nvfp4_group([shards[f] for f in order], activation_dtype)
            )
            sd[f"{tmpl.format(layer)}.weight"] = fused
            print(f"  nvfp4: {tmpl.format(layer)} {tuple(fused.shape)}", flush=True)

    for fqn, value in iter_checkpoint(
        gguf_path, key_map=gguf_to_model_key, convert=to_default
    ):
        layer = int(fqn.split(".")[1]) if fqn.startswith("layers.") else None
        flush(keep_layer=layer)
        group = _requant_group(fqn) if _is_requant_linear(fqn) else None
        if group is not None:
            pending.setdefault(group, {})[fqn] = value
        elif _is_requant_linear(fqn):
            (sd[fqn],) = _quantize_nvfp4_group([value], activation_dtype)
            print(f"  nvfp4: {fqn} {tuple(value.shape)}", flush=True)
        else:
            sd[fqn] = maybe_cast(value, activation_dtype)
        del value
    flush()
    return sd


def _finalize_nvfp4(
    sd: dict,
    config,
    activation_dtype: torch.dtype,
    *,
    defer_runtime_buffers: bool = False,
):
    from executorch.examples.models.muse_glimmer.model.model import (
        materialize_runtime_buffers,
        MuseGlimmerModel,
    )
    from executorch.extension.llm.export.load import assign_state_dict
    from executorch.extension.llm.export.quant import identity

    config.fuse_qko = False
    config.fuse_qkv = "layers.0.self_attn.qkv_proj.weight" in sd
    config.fuse_gate_up = "layers.0.mlp.gate_up_proj.weight" in sd
    print("Building fused model on meta device...")
    with torch.device("meta"):
        model = MuseGlimmerModel(config)
    assign_state_dict(model, sd, convert=identity)
    sd.clear()
    if not defer_runtime_buffers:
        materialize_runtime_buffers(model, dtype=activation_dtype)
    model.eval()
    print(f"Model: {config.n_layers} layers, dim={config.dim} (nvfp4 linears)")
    return model


# mmproj (vision encoder) loading.

_VISION_LAYER_MAP: dict[str, str] = {
    "attn_q": "blocks.{}.attn.q_proj",
    "attn_k": "blocks.{}.attn.k_proj",
    "attn_v": "blocks.{}.attn.v_proj",
    "attn_out": "blocks.{}.attn.o_proj",
    "ffn_up": "blocks.{}.mlp.c_fc",
    "ffn_down": "blocks.{}.mlp.c_proj",
    "ln1": "blocks.{}.ln_1",
    "ln2": "blocks.{}.ln_2",
}

_VISION_TOP_MAP: dict[str, str] = {
    "v.patch_embd.weight": "conv1_linear.weight",
    "v.pre_ln.weight": "ln_pre.weight",
    "v.pre_ln.bias": "ln_pre.bias",
    "v.post_ln.weight": "ln_post.weight",
    "v.post_ln.bias": "ln_post.bias",
    "mm.adapter_fc.weight": "adapter_fc.weight",
    "mm.adapter_proj.weight": "adapter_proj.weight",
    "mm.vision_proj.weight": "vision_proj.weight",
    # The 30B mmproj uses numeric projector layer names.
    "mm.0.weight": "adapter_fc.weight",
    "mm.1.weight": "adapter_proj.weight",
    "mm.2.weight": "vision_proj.weight",
}

_VISION_POS_EMBD_KEY = "v.position_embd.weight"
_VISION_LAYER_RE = re.compile(r"^v\.blk\.(\d+)\.(.+)\.(weight|bias)$")


def load_mmproj_vision_model(  # noqa: C901
    mmproj_path: str,
    config=None,
    backend: str = "cuda",
    activation_dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Load an mmproj GGUF into a backend-ready ``MuseGlimmerVisionEncoder``.

    Quantized tensors remain in their native GGUF representation for MLX. CUDA
    applies its shared terminal layout conversion, preserving source bit widths
    on both backends.
    """
    from executorch.examples.models.muse_glimmer.model.vision_tower import (
        MuseGlimmerVisionEncoder,
    )
    from executorch.examples.models.muse_glimmer.vision.precompute import (
        MuseGlimmerVisionConfig,
    )
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor, iter_gguf
    from executorch.extension.llm.export.load import assign_state_dict
    from executorch.extension.llm.export.quant import identity

    _validate_backend(backend)
    if config is None:
        config = MuseGlimmerVisionConfig()

    print("Building vision encoder on meta device...")
    with torch.device("meta"):
        model = MuseGlimmerVisionEncoder(config)

    pos_embed_table = None
    state_dict = {}

    print(f"Streaming mmproj GGUF from {mmproj_path}...")
    for name, value in iter_gguf(mmproj_path):
        if name == _VISION_POS_EMBD_KEY:
            table = (
                value.dequantize(torch.bfloat16)
                if isinstance(value, ExportableGGUFTensor)
                else value.to(torch.bfloat16)
            )
            pos_embed_table = table.contiguous()
            continue

        match = _VISION_LAYER_RE.match(name)
        if match is not None:
            layer, leaf, kind = int(match.group(1)), match.group(2), match.group(3)
            fqn_base = _VISION_LAYER_MAP.get(leaf)
            if fqn_base is None:
                continue
            fqn = f"{fqn_base.format(layer)}.{kind}"
        else:
            fqn = _VISION_TOP_MAP.get(name)
            if fqn is None:
                continue

        if name == "v.patch_embd.weight" and not isinstance(
            value, ExportableGGUFTensor
        ):
            if value.ndim == 4:
                # The 30B GGUF folds the image-only temporal axis into a
                # Conv2d weight: W_folded = sum_t W_t. The runner supplies the
                # same image patch for every temporal slot, so [W_folded, 0]
                # is exactly equivalent while retaining the shared flattened
                # patch contract used by both CUDA and MLX.
                folded = value.flatten(1)
                temporal_padding = config.patch_dim - folded.shape[1]
                if temporal_padding < 0:
                    raise RuntimeError(
                        f"mmproj patch embedding has {folded.shape[1]} input "
                        f"features, expected at most {config.patch_dim}"
                    )
                value = torch.nn.functional.pad(
                    folded, (0, temporal_padding)
                ).contiguous()
            if tuple(value.shape) != (config.latent_dim, config.patch_dim):
                raise RuntimeError(
                    f"mmproj patch embedding shape {tuple(value.shape)} does not "
                    f"match ({config.latent_dim}, {config.patch_dim})"
                )

        if isinstance(value, ExportableGGUFTensor):
            if backend == "mlx" and value.orig_dtype != activation_dtype:
                value = ExportableGGUFTensor(
                    value.raw, value.ggml_type, activation_dtype
                )
        elif value.dtype.is_floating_point:
            value = value.to(activation_dtype)
        state_dict[fqn] = value

    if pos_embed_table is None:
        raise RuntimeError(f"mmproj {mmproj_path} missing {_VISION_POS_EMBD_KEY!r}")

    assign_state_dict(model, state_dict, convert=identity)
    if backend == "cuda":
        from executorch.examples.models.gemma4_31b.cuda_packers import (
            convert_quantized_tensors_for_cuda,
        )

        convert_quantized_tensors_for_cuda(model)
    model.eval()

    print(f"Vision encoder: {config.n_layers} blocks, latent={config.latent_dim}")
    return model, pos_embed_table, config


def _gguf_native_context(gguf_path: str) -> int:
    """Read the context field qualified by the GGUF's declared architecture."""
    from gguf import GGUFReader, GGUFValueType

    reader = GGUFReader(gguf_path)

    def read_scalar(key):
        field = reader.get_field(key)
        if field is None or field.types[0] == GGUFValueType.ARRAY:
            return None
        data = field.parts[field.data[0]]
        if field.types[0] == GGUFValueType.STRING:
            return data.tobytes().decode("utf-8")
        return data.item()

    architecture = read_scalar("general.architecture")
    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError(
            f"Native context requires a nonempty 'general.architecture' in {gguf_path}"
        )
    key = f"{architecture}.context_length"
    value = read_scalar(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"Native context requires a positive non-boolean integer "
            f"{key!r} in {gguf_path}; got {value!r}"
        )
    return value


def _mlx_native_context(mlx_dir: str) -> int:
    """Read HF text context, preferring the nested text configuration."""
    import json

    path = os.path.join(mlx_dir, "config.json")
    with open(path) as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Native context requires a JSON object in {path}")
    text_config = config.get("text_config", {})
    if not isinstance(text_config, dict):
        raise ValueError(
            f"Native context requires 'text_config' to be an object in {path}"
        )
    value = text_config.get(
        "max_position_embeddings", config.get("max_position_embeddings")
    )
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"Native context requires a positive non-boolean integer "
            f"'max_position_embeddings' in {path}; got {value!r}"
        )
    return value


# Public entry points: max_seq_len=None requires checkpoint-native context.


def load_gguf_model(
    gguf_path: str,
    backend: str = "cuda",
    max_seq_len: int | None = 131072,
    config=None,
    activation_dtype: torch.dtype = torch.bfloat16,
    *,
    defer_runtime_buffers: bool = False,
    int8_int4: bool = False,
    fp_int4: bool = False,
    nvfp4: bool = False,
    int4_affine_act: str | None = None,
    int4_affine_schemes: dict[str, str] | None = None,
) -> tuple:
    """Load a GGUF checkpoint into a Muse Glimmer model packed for ``backend``.

    At most one of these (MLX only) requantizes every decoder linear and
    ``lm_head``:
      * ``int8_int4``: symmetric per-channel int4 with symmetric int8 dynamic
        per-token activations (``mlx::int8_sym_int4_affine_gemm`` with a zero
        point of 0).
      * ``fp_int4``: symmetric per-channel int4 with fp16 / bf16 activations
        (``mlx::fp_int4_affine_gemm`` with a zero point of 0).
      * ``nvfp4``: NVFP4 run by MLX's ``nvfp4`` quantized matmul.
      * ``int4_affine_act`` ("int8_asym" / "int8_sym" / "fp" / "runtime"): asymmetric
        int4 with a scheme per layer group from ``int4_affine_schemes``
        (``AFFINE_GROUPS`` -> ``AFFINE_SCHEMES``, default "pc"): "pc" / "gs256"
        run by ``mlx::{int8_asym,int8_sym,fp}_int4_affine_gemm``, or for "runtime"
        ``mlx::int4_affine_gemm`` (activations picked by MLX backend options at
        load), "mlx" by MLX's 4-bit gs=32 affine quantized matmul.
    """
    _validate_backend(backend)
    flags = {
        "int8_int4": int8_int4,
        "fp_int4": fp_int4,
        "nvfp4": nvfp4,
        "int4_affine": int4_affine_act is not None,
    }
    chosen = [name for name, on in flags.items() if on]
    if chosen and backend != "mlx":
        raise ValueError(f"{chosen[0]} linears require backend='mlx'")
    if len(chosen) > 1:
        raise ValueError(f"{' and '.join(chosen)} are mutually exclusive")
    from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerConfig

    if config is None:
        config = MuseGlimmerConfig(
            max_seq_len=(
                _gguf_native_context(gguf_path) if max_seq_len is None else max_seq_len
            )
        )
    print(f"Loading GGUF from {gguf_path}...")
    kind = next((c for c in chosen if c in _PER_CHANNEL_ACTS), None)
    if kind is not None:
        other_sd, packed = _per_channel_sd_from_gguf(gguf_path, activation_dtype, kind)
        model = _finalize_per_channel(
            other_sd,
            packed,
            config,
            activation_dtype,
            kind,
            defer_runtime_buffers=defer_runtime_buffers,
        )
        return model, config
    if int4_affine_act is not None:
        schemes = {g: "pc" for g in AFFINE_GROUPS} | (int4_affine_schemes or {})
        other_sd, packed = _affine_sd_from_gguf(gguf_path, activation_dtype, schemes)
        model = _finalize_affine(
            other_sd,
            packed,
            config,
            activation_dtype,
            int4_affine_act,
            schemes,
            defer_runtime_buffers=defer_runtime_buffers,
        )
        return model, config
    if nvfp4:
        model = _finalize_nvfp4(
            _nvfp4_sd_from_gguf(gguf_path, activation_dtype),
            config,
            activation_dtype,
            defer_runtime_buffers=defer_runtime_buffers,
        )
        return model, config
    atomic_sd = _atomic_sd_from_gguf(gguf_path, activation_dtype)
    return (
        _finalize(
            atomic_sd,
            backend,
            config,
            activation_dtype,
            defer_runtime_buffers=defer_runtime_buffers,
        ),
        config,
    )


def load_prequantized_model(
    prequantized_dir: str,
    max_seq_len: int | None = 16384,
    backend: str = "cuda",
    activation_dtype: torch.dtype = torch.bfloat16,
    *,
    defer_runtime_buffers: bool = False,
) -> tuple:
    """Load an atomic quantized safetensors checkpoint, packed for ``backend``."""
    _validate_backend(backend)
    from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerConfig

    config = MuseGlimmerConfig.from_json(
        os.path.join(prequantized_dir, "params.json"),
        require_native_context=max_seq_len is None,
    )
    if max_seq_len is not None:
        config.max_seq_len = max_seq_len
    safetensors_path = os.path.join(prequantized_dir, "model.safetensors")
    print(f"Loading quantized checkpoint from {safetensors_path}...")
    atomic_sd = _atomic_sd_from_safetensors(safetensors_path, activation_dtype)
    return (
        _finalize(
            atomic_sd,
            backend,
            config,
            activation_dtype,
            defer_runtime_buffers=defer_runtime_buffers,
        ),
        config,
    )


def load_and_quantize(
    checkpoint_dir: str,
    recipe,
    max_seq_len: int | None = 16384,
    backend: str = "cuda",
    activation_dtype: torch.dtype = torch.bfloat16,
    *,
    defer_runtime_buffers: bool = False,
) -> tuple:
    """Load a bf16 consolidated checkpoint, quantize, and pack for ``backend``."""
    _validate_backend(backend)
    print(f"Loading + quantizing checkpoint from {checkpoint_dir}...")
    atomic_sd, config = _atomic_sd_from_bf16(
        checkpoint_dir, recipe, max_seq_len, activation_dtype
    )
    return (
        _finalize(
            atomic_sd,
            backend,
            config,
            activation_dtype,
            defer_runtime_buffers=defer_runtime_buffers,
        ),
        config,
    )


# Muse Glimmer ET ``RMSNorm`` applies its weight as an absolute gain, but the HF
# ``MuseGlimmerRMSNorm`` stores these norms as a delta from 1.0 (``rms_norm(x) * (1 +
# weight)``). Fold the delta by adding 1.0 on load -- same suffixes/offset as the
# consolidated ``.pt`` loader. ``output_norm`` is a direct-gain norm (excluded).
_MLX_DELTA_NORM_SUFFIXES = (
    ".qkv_proj_norm.weight",
    ".mlp.norm.weight",
    ".post_attn_norm.weight",
    ".post_ffn_norm.weight",
)


def _mlx_target_convert(fqn: str, w: torch.Tensor) -> torch.Tensor:
    """Weight convert for the MLX *target* load.

    Applies the standard portable conversion, then folds the HF ``MuseGlimmerRMSNorm``
    delta convention into muse_glimmer's ET ``RMSNorm``: the HF model stores those norm
    weights as ``gain - 1`` (forward does ``rms_norm(x) * (1 + weight)``) while ET
    applies the weight as an absolute gain, so add 1.0 -- the same suffixes and
    offset the consolidated ``.pt`` loader uses (``model._load_and_remap_checkpoint``).
    The final ``output_norm`` is a direct-gain norm and is intentionally excluded.
    """
    from executorch.extension.llm.export.quant import to_default

    w = to_default(fqn, w)
    if fqn.endswith(_MLX_DELTA_NORM_SUFFIXES):
        w = w + 1.0
    return w


def load_mlx_model(
    mlx_dir: str,
    backend: str = "mlx",
    max_seq_len: int | None = 131072,
    config=None,
    activation_dtype: torch.dtype = torch.bfloat16,
    *,
    defer_runtime_buffers: bool = False,
) -> tuple:
    """Load an MLX-quantized safetensors checkpoint (e.g. mlx_lm output).

    The directory carries HF ``MuseGlimmerForCausalLM`` tensor names and a ``config.json``
    ``quantization`` block. ``_iter_mlx`` reads that block per weight (default +
    per-module overrides) and builds the right representation for each tensor's
    ``mode``: ``affine`` -> torchao intx, ``nvfp4`` -> ``ExportableNVFP4Tensor``,
    ``mxfp8`` -> ``ExportableMXTensor`` (mixed-format checkpoints are handled
    per weight). ``mlx_hf_to_model_key`` remaps the HF names to atomic Muse Glimmer FQNs
    and ``_mlx_target_convert`` folds the HF RMSNorm delta into an absolute gain;
    the standard fuse (gate/up) + backend pass then finalizes the model.
    """
    _validate_backend(backend)
    from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerConfig
    from executorch.extension.llm.export.load import iter_checkpoint

    if config is None:
        config = MuseGlimmerConfig(
            max_seq_len=(
                _mlx_native_context(mlx_dir) if max_seq_len is None else max_seq_len
            )
        )
    print(f"Loading MLX checkpoint from {mlx_dir}...")
    atomic_sd = dict(
        iter_checkpoint(
            mlx_dir,
            raw_iter=_iter_mlx,
            key_map=mlx_hf_to_model_key,
            convert=_mlx_target_convert,
            dtype=activation_dtype,
        )
    )
    return (
        _finalize(
            atomic_sd,
            backend,
            config,
            activation_dtype,
            defer_runtime_buffers=defer_runtime_buffers,
        ),
        config,
    )
