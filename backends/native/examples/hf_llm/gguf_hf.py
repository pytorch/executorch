# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Map GGUF tensors onto Hugging Face modules.

Use with ``iter_checkpoint`` / ``load_checkpoint``: :func:`gguf_to_hf` returns the
``key_map`` and ``convert`` that load a GGUF file of a supported architecture into
a Hugging Face model. Names come from gguf-py's standard tensor map, the one
llama.cpp's ``convert_hf_to_gguf.py`` writes with, or a table here for an
architecture gguf-py does not know; ``convert`` undoes the
weight changes that converter makes for the architecture. Those changes move whole
rows or touch only unquantized norms, so quantized GGUF blocks stay byte-exact.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
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


def _norm_offset_remover(suffix: str) -> Convert:
    """Subtract the 1 llama.cpp adds to the norm weights ending in ``suffix``,
    for HF norms that add it in their forward instead."""

    def convert(fqn: str, value: torch.Tensor) -> torch.Tensor:
        if not fqn.endswith(suffix):
            return value
        if isinstance(value, ExportableGGUFTensor):
            raise ValueError(f"{fqn}: expected an unquantized norm weight")
        return value.float() - 1

    return convert


def _remove_norm_offset(config) -> Convert:
    """Gemma adds the 1 in every norm."""
    return _norm_offset_remover("norm.weight")


def _muse_glimmer_convert(config) -> Convert:
    """Muse Glimmer rotates like Llama, and adds the 1 in its decoder layer
    norms but not its final norm."""
    restore_qk = _restore_qk_rows(config)
    remove_offset = _norm_offset_remover("layernorm.weight")
    return lambda fqn, value: remove_offset(fqn, restore_qk(fqn, value))


_MG_TEXT = "model.language_model"
# GGUF name -> HF module, for the blocks and the rest; None skips a tensor.
_MUSE_GLIMMER_BLOCK = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_gate": "self_attn.gate_proj",
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_norm": "pre_feedforward_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    # The QK norm has no weight in HF; see _muse_glimmer_config_tensors.
    "attn_q_norm": None,
    "attn_k_norm": None,
}
_MUSE_GLIMMER_TOP = {
    "token_embd": f"{_MG_TEXT}.embed_tokens",
    "output_norm": f"{_MG_TEXT}.norm",
    "output": "lm_head",
}


def _muse_glimmer_names(layers: int) -> dict[str, str | None]:
    names = {f"{g}.weight": f"{h}.weight" for g, h in _MUSE_GLIMMER_TOP.items()}
    for i in range(layers):
        for g, h in _MUSE_GLIMMER_BLOCK.items():
            names[f"blk.{i}.{g}.weight"] = (
                f"{_MG_TEXT}.layers.{i}.{h}.weight" if h else None
            )
    return names


def _muse_glimmer_config_tensors(config) -> dict[str, float]:
    """llama.cpp's converter stores the query scale HF reads from the config as
    the Q norm weight, and ones as the K norm weight."""
    return {"attn_q_norm.weight": config.qk_scale_factor, "attn_k_norm.weight": 1.0}


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
    "muse-glimmer": (frozenset({"muse_glimmer_text"}), _muse_glimmer_convert),
}

# Architectures gguf-py has no tensor map for: layer count -> GGUF name -> HF
# parameter, or None to skip.
_NAMES: dict[str, Callable[[int], dict[str, str | None]]] = {
    "muse-glimmer": _muse_glimmer_names,
}

# Skipped GGUF block tensors that hold a value Hugging Face reads from its
# config: config -> tensor -> that value.
_CONFIG_TENSORS: dict[str, Callable[..., dict[str, float]]] = {
    "muse-glimmer": _muse_glimmer_config_tensors,
}


def check_config_tensors(
    model: torch.nn.Module,
    architecture: str,
    tensors: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pass a GGUF file's ``(name, tensor)`` pairs through, raising for a
    skipped block tensor that disagrees with the value ``model``'s config gives
    it, which loading would otherwise ignore."""
    make = _CONFIG_TENSORS.get(architecture)
    want = make(model.config.get_text_config()) if make else {}
    for name, value in tensors:
        block = re.match(r"blk\.\d+\.(.+)", name)
        if block and block[1] in want:
            expected = torch.tensor(want[block[1]])
            if isinstance(value, ExportableGGUFTensor) or not torch.allclose(
                value.float(), expected, rtol=1e-3, atol=0
            ):
                raise ValueError(
                    f"{name} must hold {want[block[1]]}, as the config gives"
                )
        yield name, value


def gguf_to_hf(
    model: torch.nn.Module, architecture: str, truncated: bool = False
) -> tuple[KeyMap, Convert]:
    """Return the ``key_map`` and ``convert`` that load a GGUF file of
    ``architecture`` into ``model``, a Hugging Face model.

    The key map raises for a GGUF tensor that names no parameter of ``model``,
    unless ``truncated``: a model built with fewer layers than the file holds
    then skips the blocks past its last layer.
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
    if architecture in _NAMES:
        names = _NAMES[architecture](config.num_hidden_layers)
    else:
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
        block = re.match(r"blk\.(\d+)\.", name)
        if truncated and block and int(block[1]) >= config.num_hidden_layers:
            return None
        raise ValueError(
            f"GGUF tensor {name!r} maps to no {config.model_type} parameter"
        )

    convert = make_convert(config) if make_convert else lambda _, value: value
    return key_map, convert
