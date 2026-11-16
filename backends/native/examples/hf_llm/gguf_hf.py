# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Map GGUF tensors onto Hugging Face modules.

Use with ``iter_checkpoint`` / ``load_checkpoint``: :func:`gguf_to_hf` returns the
``key_map`` and ``convert`` that load a GGUF file of a supported architecture into
a Hugging Face model. Names come from gguf-py's standard tensor map, the one
llama.cpp's ``convert_hf_to_gguf.py`` writes with, and ``convert`` undoes the
weight changes that converter makes for the architecture. Those changes move whole
rows or touch only unquantized norms, so quantized GGUF blocks stay byte-exact.
"""

from __future__ import annotations

from typing import Callable

import gguf
import torch
from executorch.extension.llm.export.gguf import ExportableGGUFTensor
from executorch.extension.llm.export.quant.convert import Convert

KeyMap = Callable[[str], str | None]

# GGUF tensors that Hugging Face computes from the model config instead.
_DERIVED = frozenset({"rope_freqs.weight"})


def gguf_architecture(path: str) -> str:
    """Return a GGUF file's ``general.architecture``."""
    field = gguf.GGUFReader(path).get_field("general.architecture")
    if field is None:
        raise ValueError(f"{path} has no general.architecture")
    return field.contents()


def _restore_rows(rows: torch.Tensor, heads: int) -> torch.Tensor:
    # GGUF row (h, 2 * i + j) holds HF row (h, j * head_dim / 2 + i).
    return (
        rows.reshape(heads, rows.shape[0] // heads // 2, 2, *rows.shape[1:])
        .transpose(1, 2)
        .reshape(rows.shape)
    )


def _restore_qk_rows(config) -> Convert:
    """Undo the Q/K row permutation llama.cpp applies for its interleaved RoPE."""
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    if head_dim % 2:
        raise ValueError(f"interleaved RoPE needs an even head_dim, got {head_dim}")
    heads_by_module = {
        "self_attn.q_proj": config.num_attention_heads,
        "self_attn.k_proj": config.num_key_value_heads,
    }

    def convert(fqn: str, value: torch.Tensor) -> torch.Tensor:
        module = fqn.rpartition(".")[0]
        heads = next(
            (n for m, n in heads_by_module.items() if module.endswith(m)), None
        )
        if heads is None:
            return value
        if value.dim() > 2 or value.shape[0] != heads * head_dim:
            raise ValueError(
                f"{fqn}: expected {heads * head_dim} rows, "
                f"got shape {tuple(value.shape)}"
            )
        if isinstance(value, ExportableGGUFTensor):
            return ExportableGGUFTensor.from_raw(
                _restore_rows(value.raw, heads), value.ggml_type, value.orig_dtype
            )
        return _restore_rows(value, heads).contiguous()

    return convert


def _remove_norm_offset(config) -> Convert:
    """Subtract the 1 llama.cpp adds to Gemma norm weights; HF Gemma adds it in
    its forward instead."""

    def convert(fqn: str, value: torch.Tensor) -> torch.Tensor:
        if not fqn.endswith("norm.weight"):
            return value
        if isinstance(value, ExportableGGUFTensor):
            raise ValueError(f"{fqn}: expected an unquantized norm weight")
        return value.float() - 1

    return convert


# GGUF architecture -> (the Hugging Face model types it holds, what undoes
# llama.cpp's weight changes). llama.cpp stores Mistral as "llama".
_ARCHITECTURES: dict[str, tuple[frozenset[str], Callable[..., Convert] | None]] = {
    "llama": (frozenset({"llama", "mistral"}), _restore_qk_rows),
    "granite": (frozenset({"granite"}), _restore_qk_rows),
    "qwen2": (frozenset({"qwen2"}), None),
    "qwen3": (frozenset({"qwen3"}), None),
    "gemma": (frozenset({"gemma"}), _remove_norm_offset),
    "gemma2": (frozenset({"gemma2"}), _remove_norm_offset),
    "gemma3": (frozenset({"gemma3_text"}), _remove_norm_offset),
}


def gguf_to_hf(model: torch.nn.Module, architecture: str) -> tuple[KeyMap, Convert]:
    """Return the ``key_map`` and ``convert`` that load a GGUF file of
    ``architecture`` into ``model``, a Hugging Face model.

    The key map raises for a GGUF tensor that names no parameter of ``model``.
    """
    if architecture not in _ARCHITECTURES:
        raise ValueError(
            f"unsupported GGUF architecture {architecture!r}; supported: "
            f"{sorted(_ARCHITECTURES)}"
        )
    model_types, make_convert = _ARCHITECTURES[architecture]
    config = model.config.get_text_config()
    if config.model_type not in model_types:
        raise ValueError(
            f"a {architecture!r} GGUF cannot load a {config.model_type!r} model"
        )
    arch = next(a for a, n in gguf.MODEL_ARCH_NAMES.items() if n == architecture)
    tensor_map = gguf.get_tensor_name_map(arch, config.num_hidden_layers)
    names = {}
    for fqn in model.state_dict():
        name = tensor_map.get_name(fqn, try_suffixes=(".weight", ".bias"))
        if name is not None:
            names[name] = fqn

    def key_map(name: str) -> str | None:
        if name in names:
            return names[name]
        if name in _DERIVED:
            return None
        raise ValueError(
            f"GGUF tensor {name!r} maps to no {config.model_type} parameter"
        )

    convert = make_convert(config) if make_convert else lambda _, value: value
    return key_map, convert
