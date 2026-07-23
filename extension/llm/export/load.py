# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Load a checkpoint (quantized or unquantized) into a meta-device model.

``iter_checkpoint`` is the model-free reader: it picks a reader by ``path``
(``.gguf`` -> raw GGUF blobs; a ``.safetensors`` file or a directory of them ->
safetensors, quantized or plain), remaps each tensor name to a model FQN via
``key_map`` (returning ``None`` skips it), then applies ``convert`` -- a
``(fqn, tensor) -> tensor`` step (default ``to_default``) that wraps quantized
subclasses in their portable ``Exportable*`` form. It yields ``(fqn, tensor)``
pairs and never touches a model, so ``dict(iter_checkpoint(path, convert=identity))``
is a plain state dict.

``load_checkpoint`` is the model-aware wrapper: it streams ``iter_checkpoint``
into ``model`` via ``assign_one`` (param-vs-buffer, no packer dispatch), fans a
weight out to a second FQN via ``tie_map`` (e.g. a tied lm_head, keyed by the
*remapped* FQN and applied only when the destination is still on meta), and
asserts every *parameter* is materialized once the stream is exhausted (runtime
buffers may still be on meta -- callers fill those afterward, e.g. via
``materialize_runtime_buffers``). Backend-specific layout conversion (CUDA
coalesced int4, MLX gather buffers) is a separate terminal pass over the loaded
model, not part of loading.
"""

import contextlib
import json
import os
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
from executorch.extension.llm.export.quant.convert import (
    Convert,
    maybe_cast,
    to_default,
)

# Remap a checkpoint tensor name to a model FQN, or ``None`` to skip it.
KeyMap = Callable[[str], Optional[str]]

# Fan a source tensor out to a second module: remapped source FQN ->
# (destination FQN, clone). ``clone=True`` packs an independent copy of the
# tensor into the destination (untie); ``clone=False`` packs the same tensor
# object, so a pass-through packer leaves both modules sharing one storage (tie).
# The fan-out is applied only when the destination is still on meta, so a real
# weight already present in the checkpoint always wins.
TieMap = dict[str, tuple[str, bool]]

Pair = tuple[str, torch.Tensor]

# Custom raw reader: ``path -> (name, tensor)`` pairs (before key_map/convert/
# dtype). When given to ``iter_checkpoint`` / ``load_checkpoint`` it overrides the
# built-in ``.gguf`` / ``.safetensors`` reader selection, so callers can inject a
# format-specific reader (e.g. guac's MLX affine reader) without this module
# needing to know about it.
RawIter = Callable[[str], Iterable[Pair]]


def _split_parent(model: nn.Module, fqn: str) -> tuple[nn.Module, str]:
    """Resolve ``fqn`` to ``(parent_module, attr_name)``."""
    parts = fqn.rsplit(".", 1)
    parent_fqn = parts[0] if len(parts) > 1 else ""
    attr = parts[-1]
    parent = model.get_submodule(parent_fqn) if parent_fqn else model
    return parent, attr


def assert_all_materialized(model: nn.Module) -> None:
    """Raise if any parameter is still on the meta device after loading."""
    for fqn, p in model.named_parameters():
        if p.device.type == "meta":
            raise RuntimeError(
                f"Weight '{fqn}' not found in checkpoint "
                f"(model/checkpoint version mismatch?)"
            )


def assign_one(model: nn.Module, fqn: str, value: torch.Tensor) -> None:
    """Assign one weight into ``model`` as a parameter or buffer, in place.

    Chooses parameter vs buffer by inspecting the existing attribute on the
    parent module (a slot that is currently an ``nn.Parameter`` -- typically a
    meta tensor -- becomes a non-grad ``nn.Parameter``; anything else is
    registered as a buffer). No conversion or packer dispatch: ``value`` is
    assigned as given (already in its final portable/quantized form).
    """
    parent, attr = _split_parent(model, fqn)
    if isinstance(getattr(parent, attr, None), nn.Parameter):
        setattr(parent, attr, nn.Parameter(value, requires_grad=False))
    else:
        parent.register_buffer(attr, value)


def _resolve_shard_paths(path: str) -> list[str]:
    """Return the safetensors shard file paths for ``path``.

    A file path yields itself; a directory is resolved via
    ``model.safetensors.index.json`` (sharded) or a single ``model.safetensors``.
    """
    if not os.path.isdir(path):
        return [path]

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        return [
            os.path.join(path, shard)
            for shard in sorted(set(index["weight_map"].values()))
        ]
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        return [single]
    raise FileNotFoundError(f"No safetensors checkpoint in {path}")


def _iter_safetensors(path: str) -> Iterable[Pair]:
    """Yield ``(name, tensor)`` pairs from a safetensors checkpoint.

    ``path`` may be a single ``.safetensors`` file or a directory (sharded via
    ``model.safetensors.index.json``, or a single ``model.safetensors``). All
    shards are opened together (mmap, so no tensor data is read up front) and a
    ``key -> shard`` map is built, so a logical weight whose sub-tensors are
    split across shards can still be reassembled.

    A torchao-quantized checkpoint (identified by a ``tensor_names`` metadata
    key) has each logical weight's subclass sub-tensors gathered -- from whatever
    shard holds them -- and reassembled via ``unflatten_tensor_state_dict`` into
    a single quantized tensor. A plain checkpoint yields its tensors as-is.
    Either way only one logical weight is materialized at a time. Casting to a
    target dtype is the caller's job (``iter_checkpoint``).
    """
    from safetensors import safe_open

    with contextlib.ExitStack() as stack:
        handles = [
            stack.enter_context(safe_open(p, framework="pt", device="cpu"))
            for p in _resolve_shard_paths(path)
        ]
        key_to_handle = {}
        metadata: dict = {}
        tensor_names: list = []
        for handle in handles:
            shard_meta = handle.metadata() or {}
            metadata.update(shard_meta)
            tensor_names.extend(
                name
                for name in json.loads(shard_meta.get("tensor_names", "[]"))
                if name not in tensor_names
            )
            for key in handle.keys():
                key_to_handle[key] = handle

        if not tensor_names:
            for key, handle in key_to_handle.items():
                yield key, handle.get_tensor(key)
            return

        from torchao.prototype.safetensors.safetensors_support import (
            unflatten_tensor_state_dict,
        )

        for name in tensor_names:
            parts = name.rsplit(".", 1)
            module_fqn = parts[0] if len(parts) > 1 else ""
            weight_name = parts[-1]
            prefix = (
                f"{module_fqn}._{weight_name}_" if module_fqn else f"_{weight_name}_"
            )
            partial = {
                key: handle.get_tensor(key)
                for key, handle in key_to_handle.items()
                if key == name or key.startswith(prefix)
            }
            result, _ = unflatten_tensor_state_dict(partial, metadata)
            for fqn, value in result.items():
                yield fqn, value


def _iter_gguf(path: str) -> Iterable[Pair]:
    """Yield ``(gguf_name, tensor)`` pairs from a GGUF file.

    Quantized weights arrive as ``ExportableGGUFTensor`` (the raw GGUF blob).
    Names are the raw GGUF tensor names -- pass a ``key_map`` to
    ``iter_checkpoint`` to remap them to model FQNs. Casting to a target dtype is
    the caller's job (``iter_checkpoint``).
    """
    from executorch.extension.llm.export.gguf import iter_gguf

    yield from iter_gguf(path)


def iter_checkpoint(
    path: str,
    *,
    key_map: Optional[KeyMap] = None,
    convert: Convert = to_default,
    tie_map: Optional[TieMap] = None,
    dtype: Optional[torch.dtype] = None,
    raw_iter: Optional[RawIter] = None,
) -> Iterable[Pair]:
    """Yield ``(fqn, tensor)`` pairs from a checkpoint, model-free.

    ``raw_iter`` (when given) supplies the raw ``(name, tensor)`` reader and
    overrides the selection below; otherwise the reader is picked by ``path``:

    * ``*.gguf``        -- raw GGUF blobs; quantized weights become
                           ``ExportableGGUFTensor``.
    * ``*.safetensors`` -- a single safetensors file.
    * a directory       -- a safetensors checkpoint, sharded via
                           ``model.safetensors.index.json`` or a single
                           ``model.safetensors``.

    For each weight, ``key_map`` remaps the stream name to a model FQN first
    (identity when ``None``; return ``None`` to skip), then ``convert`` maps the
    value (default ``to_default``: ``Int4Tensor -> ExportableInt4Tensor``, others
    unchanged), then ``dtype`` casts the *converted* weight (plain tensors
    normally; quantized subclasses keep their payload and only re-stamp their
    dequantized output dtype). Casting runs after ``convert`` so a raw torchao
    ``Int4Tensor`` -- whose own ``.to`` ignores dtype -- is first wrapped as an
    ``ExportableInt4Tensor`` (which ``maybe_cast`` *can* re-stamp, e.g. to fp16
    for MLX). ``convert`` therefore sees the *remapped* FQN and the *uncast*
    tensor. Only one logical weight is materialized at a time.

    ``tie_map`` fans a source weight out to a second FQN (e.g. a tied lm_head,
    keyed by the *remapped* source FQN). The tied copy is emitted after the stream
    is exhausted and only when the destination did not appear on its own, so an
    untied checkpoint always wins; ``clone=True`` emits an independent copy.
    """
    if raw_iter is not None:
        raw = raw_iter(path)
    elif path.endswith(".gguf"):
        raw = _iter_gguf(path)
    elif path.endswith(".safetensors") or os.path.isdir(path):
        raw = _iter_safetensors(path)
    else:
        raise ValueError(
            f"Cannot load checkpoint from '{path}': expected a .gguf file, a "
            ".safetensors file, or a directory with a safetensors checkpoint."
        )
    seen: set[str] = set()
    owed: dict[str, Pair] = {}  # dst FQN -> (source value, clone) for tie fan-out
    for name, value in raw:
        fqn = key_map(name) if key_map is not None else name
        if fqn is None:
            continue
        out = maybe_cast(convert(fqn, value), dtype)
        seen.add(fqn)
        yield fqn, out
        if tie_map is not None and fqn in tie_map:
            dst, clone = tie_map[fqn]
            owed[dst] = (out, clone)
    for dst, (value, clone) in owed.items():
        if dst not in seen:
            yield dst, value.clone() if clone else value


def load_checkpoint(
    path: str,
    model: nn.Module,
    *,
    key_map: Optional[KeyMap] = None,
    convert: Convert = to_default,
    tie_map: Optional[TieMap] = None,
    dtype: Optional[torch.dtype] = None,
    raw_iter: Optional[RawIter] = None,
) -> None:
    """Load a checkpoint and assign it into ``model`` in place.

    Streams ``iter_checkpoint(path, key_map=key_map, convert=convert,
    tie_map=tie_map, dtype=dtype)`` (see it for reader selection and the
    ``key_map`` / ``convert`` / ``tie_map`` / ``dtype`` semantics) and assigns each
    ``(fqn, tensor)`` via ``assign_one`` (param-vs-buffer, no packer dispatch --
    the tensor is already in its final portable form).

    Asserts every parameter is materialized afterward; runtime buffers left on
    meta are the caller's responsibility (e.g. ``materialize_runtime_buffers``).
    Backend-specific layout conversion (CUDA coalesced int4, MLX gather buffers)
    is a separate terminal pass over the loaded model.
    """
    for fqn, value in iter_checkpoint(
        path,
        key_map=key_map,
        convert=convert,
        tie_map=tie_map,
        dtype=dtype,
        raw_iter=raw_iter,
    ):
        assign_one(model, fqn, value)
    assert_all_materialized(model)


def assign_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    convert: Convert = to_default,
) -> None:
    """Convert + assign an in-memory state dict into ``model`` in place.

    The in-memory dual of :func:`load_checkpoint`: applies ``convert`` (default
    ``to_default``) to each weight and assigns it via :func:`assign_one`. Use
    this when weights are already materialized in memory (e.g. the output of
    ``quantize_model``) rather than streamed from disk. Backend layout conversion
    (if any) is a separate terminal pass over the assigned model. Asserts every
    parameter is materialized afterward.
    """
    for fqn, value in state_dict.items():
        assign_one(model, fqn, convert(fqn, value))
    assert_all_materialized(model)
    for p in model.parameters():
        p.requires_grad_(False)
