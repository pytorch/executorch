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
Export covers the text decoder only: a GGUF's vision and audio encoder tensors,
if it has any, are skipped with a warning.

For the architectures transformers' own GGUF loader converts, names and
conversions come from its tables instead, applied to quantized tensors' packed
rows where transformers marks every conversion safe on packed bytes; other
quantized tensors are dequantized first. A weight whose columns transformers
reorders can keep its blocks instead, its module's input reordered.
"""

from __future__ import annotations

import copy
import re
import warnings
from collections.abc import Iterable, Iterator
from typing import Callable, NamedTuple

import gguf
import torch
from executorch.extension.llm.export.gguf import ExportableGGUFTensor
from executorch.extension.llm.export.quant.convert import Convert
from transformers.core_model_loading import (
    rename_source_key,
    WeightConverter,
    WeightRenaming,
    WeightTransform,
)
from transformers.integrations.gguf.gguf_conversion_mapping import (
    ConcatenateRows,
    LogNegate,
    SubtractOne,
    TiledToGroupedInputs,
    TiledToGroupedRows,
    Unsqueeze,
)
from transformers.integrations.gguf.utils import (
    get_gguf_conversion_mapping,
    get_gguf_plan,
)

KeyMap = Callable[[str], str | None]

# GGUF tensors that Hugging Face computes from the model config instead.
_DERIVED = frozenset({"rope_freqs.weight"})


def _encoder_prefixes() -> tuple[str, ...]:
    """Name prefixes of gguf-py's vision and audio encoder tensors."""
    return tuple(
        sorted(
            {
                name.split(".", 1)[0] + "."
                for tensor, name in gguf.TENSOR_NAMES.items()
                if tensor.name.startswith(("V_", "A_"))
            }
        )
    )


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
_EXPLICIT_NAME_MAPS: dict[str, Callable[[int], dict[str, str | None]]] = {
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


# GGUF architecture -> the Hugging Face model types it holds, for the
# architectures whose names and conversions come from transformers' GGUF tables.
_TRANSFORMERS_ARCHITECTURES: dict[str, frozenset[str]] = {
    "qwen35": frozenset({"qwen3_5_text"}),
    "qwen35moe": frozenset({"qwen3_5_moe_text"}),
}
# The conversion ops those tables use. Any other fails loudly, so a transformers
# update cannot change a mapping unnoticed.
_TRANSFORMERS_OPS = (
    ConcatenateRows,
    LogNegate,
    SubtractOne,
    TiledToGroupedInputs,
    TiledToGroupedRows,
    Unsqueeze,
)


class _GgmlTypes(NamedTuple):
    """What ``get_gguf_plan`` reads of transformers' ``GgufHeader``."""

    ggml_types: dict[str, int]


def _transformers_mapping(
    architecture: str, config: object
) -> tuple[list[WeightTransform], list[WeightRenaming], list[WeightConverter]]:
    mapping = get_gguf_conversion_mapping(architecture, config)
    renamings = [t for t in mapping if isinstance(t, WeightRenaming)]
    converters = [t for t in mapping if isinstance(t, WeightConverter)]
    if len(renamings) + len(converters) != len(mapping):
        raise RuntimeError(
            f"transformers' GGUF mapping for {architecture!r} holds transforms "
            "other than WeightRenaming and WeightConverter"
        )
    for converter in converters:
        for op in converter.operations:
            if not isinstance(op, _TRANSFORMERS_OPS):
                raise RuntimeError(
                    f"transformers' GGUF mapping for {architecture!r} uses "
                    f"{type(op).__name__}, which export does not know"
                )
    return mapping, renamings, converters


def _keeps_blocks(
    mapping: list[WeightTransform],
    name: str,
    fqn: str,
    value: ExportableGGUFTensor,
    input_permutations: dict[str, torch.Tensor] | None,
) -> bool:
    """Whether transformers' plan keeps `value`'s blocks: every conversion on
    its way to `fqn` is safe on packed bytes, and a reordering of its module's
    input that keeping them needs goes into `input_permutations`; without
    that dict, such a weight is dequantized. An identity reordering, e.g. of
    one value head per key head, needs none."""
    ggml_type = gguf.GGMLQuantizationType[value.ggml_type.upper()].value
    plan = get_gguf_plan(_GgmlTypes({name: ggml_type}), mapping)
    if not (
        isinstance(plan, tuple)
        and len(plan) == 3
        and all(isinstance(p, dict) for p in plan)
    ):
        raise RuntimeError("transformers' get_gguf_plan changed what it returns")
    _, packable, permutations = plan
    permutation = permutations.get(fqn)
    if permutation is not None and torch.equal(
        permutation, torch.arange(permutation.numel())
    ):
        permutation = None
    if fqn not in packable or (permutation is not None and input_permutations is None):
        return False
    if permutation is not None:
        input_permutations[fqn] = permutation
    return True


def _convert_with(
    ops: list, fqn: str, pattern: str, value: torch.Tensor
) -> torch.Tensor:
    for op in ops:
        (value,) = op.convert(
            {fqn: value}, source_patterns=[pattern], target_patterns=[fqn]
        ).values()
    return value


def _join(
    mapping: list[WeightTransform],
    converter: WeightConverter,
    fqn: str,
    parts: dict[str, tuple[str, torch.Tensor]],
) -> torch.Tensor:
    """One weight from the GGUF tensors `parts` holds by source pattern, such
    as an MoE layer's gate and up experts. Their blocks stay packed when they
    share a ggml type and transformers marks the join safe on packed bytes."""
    patterns = converter.source_patterns
    values = [parts[p][1] for p in patterns]
    quantized = [v for v in values if isinstance(v, ExportableGGUFTensor)]
    if len(quantized) == len(values) and len({v.ggml_type for v in values}) == 1:
        types = {
            name: gguf.GGMLQuantizationType[value.ggml_type.upper()].value
            for name, value in parts.values()
        }
        _, packable, permutations = get_gguf_plan(_GgmlTypes(types), mapping)
        if fqn in packable and fqn not in permutations:
            raws = dict(zip(patterns, (v.raw for v in quantized)))
            for op in converter.operations:
                (raw,) = op.convert(
                    raws, source_patterns=patterns, target_patterns=[fqn]
                ).values()
                raws = {fqn: raw}
            return ExportableGGUFTensor.from_raw(
                raw, values[0].ggml_type, values[0].orig_dtype
            )
    tensors = {
        p: v.dequantize(torch.float32) if isinstance(v, ExportableGGUFTensor) else v
        for p, v in zip(patterns, values)
    }
    for op in converter.operations:
        (value,) = op.convert(
            tensors, source_patterns=patterns, target_patterns=[fqn]
        ).values()
        tensors = {fqn: value}
    return value


def _transformers_to_hf(  # noqa: C901
    model: torch.nn.Module,
    architecture: str,
    truncated: bool,
    input_permutations: dict[str, torch.Tensor] | None,
) -> tuple[KeyMap, Convert]:
    """``gguf_to_hf`` through transformers' GGUF tables. A multi-token
    prediction block, which transformers names ``mtp.*``, is skipped with a
    warning. A weight transformers builds from several GGUF tensors comes out
    of ``convert`` with the last of them; ``convert`` returns None for the
    others."""
    config = model.config.get_text_config()
    mapping, _, converters = _transformers_mapping(architecture, model.config)
    # rename_source_key records matches on the transforms it tries (_was_used, a
    # GroupWeightRename's _active), so names match on a copy and convert keeps the
    # converters it collects tensors into separate.
    names = copy.deepcopy(mapping)
    name_renamings = [t for t in names if isinstance(t, WeightRenaming)]
    name_converters = [t for t in names if isinstance(t, WeightConverter)]
    converter_of = {p: c for c in converters for p in c.source_patterns}
    pending: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    params = set(model.state_dict())
    sources: dict[str, tuple[str, str | None]] = {}
    encoder = _encoder_prefixes()
    warned: set[str] = set()

    def skip(kind: str) -> None:
        if kind not in warned:
            warnings.warn(
                f"skipping the GGUF's {kind} tensors: export covers the text "
                "decoder only",
                stacklevel=3,
            )
            warned.add(kind)

    def key_map(name: str) -> str | None:
        if name in _DERIVED:
            return None
        if name.startswith(encoder):
            skip("vision and audio encoder")
            return None
        block = re.match(r"blk\.(\d+)\.", name)
        if truncated and block and int(block[1]) >= config.num_hidden_layers:
            return None
        fqn, pattern = rename_source_key(name, name_renamings, name_converters)
        if fqn.startswith("mtp."):
            skip("multi-token prediction")
            return None
        if fqn not in params:
            raise ValueError(
                f"GGUF tensor {name!r} maps to no {config.model_type} parameter"
            )
        sources[fqn] = (name, pattern)
        return fqn

    def convert(fqn: str, value: torch.Tensor) -> torch.Tensor | None:
        name, pattern = sources[fqn]
        converter = converter_of.get(pattern)
        if converter is None:
            return value
        if len(converter.source_patterns) > 1:
            parts = pending.setdefault(fqn, {})
            parts[pattern] = (name, value)
            if len(parts) < len(converter.source_patterns):
                return None
            return _join(mapping, converter, fqn, pending.pop(fqn))
        ops = list(converter.operations)
        if isinstance(value, ExportableGGUFTensor):
            if _keeps_blocks(mapping, name, fqn, value, input_permutations):
                raw = _convert_with(ops, fqn, pattern, value.raw)
                return ExportableGGUFTensor.from_raw(
                    raw, value.ggml_type, value.orig_dtype
                )
            value = value.dequantize(torch.float32)
        return _convert_with(ops, fqn, pattern, value)

    return key_map, convert


def gguf_to_hf(  # noqa: C901
    model: torch.nn.Module,
    architecture: str,
    truncated: bool = False,
    input_permutations: dict[str, torch.Tensor] | None = None,
) -> tuple[KeyMap, Convert]:
    """Return the ``key_map`` and ``convert`` that load a GGUF file of
    ``architecture`` into ``model``, a Hugging Face model.

    The key map skips vision and audio encoder tensors, warning once, and the
    tensors Hugging Face derives from the config, e.g. ``rope_freqs.weight``. It
    raises for any other GGUF tensor that names no parameter of ``model``,
    unless ``truncated``: a model built with fewer layers than the file holds
    then skips the blocks past its last layer.

    A quantized weight whose columns the conversion reorders is dequantized,
    unless ``input_permutations`` is given: ``convert`` then keeps its blocks
    as stored and maps its FQN to ``p``, the module computing ``x[..., p]``
    with it as it would ``x`` with the reordered weight.
    """
    supported = {
        **_TRANSFORMERS_ARCHITECTURES,
        **{a: t for a, (t, _) in _ARCHITECTURES.items()},
    }
    if architecture not in supported:
        raise ValueError(
            f"unsupported GGUF architecture {architecture!r}; supported: "
            f"{sorted(supported)}"
        )
    config = model.config.get_text_config()
    if config.model_type not in supported[architecture]:
        raise ValueError(
            f"a {architecture!r} GGUF cannot load a {config.model_type!r} model"
        )
    if architecture in _TRANSFORMERS_ARCHITECTURES:
        return _transformers_to_hf(model, architecture, truncated, input_permutations)
    _, make_convert = _ARCHITECTURES[architecture]
    if architecture in _EXPLICIT_NAME_MAPS:
        names = _EXPLICIT_NAME_MAPS[architecture](config.num_hidden_layers)
    else:
        arch = next(a for a, n in gguf.MODEL_ARCH_NAMES.items() if n == architecture)
        tensor_map = gguf.get_tensor_name_map(arch, config.num_hidden_layers)
        names = {}
        for fqn in model.state_dict():
            name = tensor_map.get_name(fqn, try_suffixes=(".weight", ".bias"))
            if name is not None:
                names[name] = fqn

    encoder = _encoder_prefixes()
    warned = False

    def key_map(name: str) -> str | None:
        nonlocal warned
        if name in names:
            return names[name]
        if name in _DERIVED:
            return None
        if name.startswith(encoder):
            if not warned:
                warnings.warn(
                    "skipping the GGUF's vision and audio encoder tensors: export "
                    "covers the text decoder only",
                    stacklevel=2,
                )
                warned = True
            return None
        block = re.match(r"blk\.(\d+)\.", name)
        if truncated and block and int(block[1]) >= config.num_hidden_layers:
            return None
        raise ValueError(
            f"GGUF tensor {name!r} maps to no {config.model_type} parameter"
        )

    convert = make_convert(config) if make_convert else lambda _, value: value
    return key_map, convert
