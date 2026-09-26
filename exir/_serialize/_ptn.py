# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
The .ptn package format.

A .ptn is an uncompressed zip holding a native Program flatbuffer and, when the
program references constants, those tensors as safetensors. Member names are fixed,
so renaming the package does not break it:

    program.ptg           native Program flatbuffer (file identifier "NPTG")
    program.safetensors   referenced constants, content-deduped (owners only)
    aliases.json          duplicate data_key -> owner key, only when duplicates exist

Byte-identical *immutable* tensors (same dtype, shape, and content) are stored
once: the first key owns the safetensors entry and the rest alias to it. An owner
is always a real safetensors key and never appears in the alias map, so resolution
is ``owner = aliases.get(key, key)``.

An alias means only "these keys had identical bytes at save time, so one copy was
stored". It never asserts that two keys share runtime state, which is why mutable
keys are excluded from dedup entirely -- see _dedup. Do not overload aliases to
express mutable alias topology; that would need explicit storage groups in the
format.

Sits alongside the PTE and PTD serializers here because the format is independent
of any backend: write_ptn treats the graph blob as opaque bytes. Its producer is
``exir.native.to_native``, whose NativeProgramManager.save calls it.

The safetensors payload is written a tensor at a time, straight into the zip entry,
so the writer never materializes the whole model as Python bytes. Owner tensors are
still retained until the write completes, and normalizing non-CPU or non-contiguous
inputs can allocate tensor copies. The header is emitted here by hand because
safetensors.torch.save_file goes through _flatten, which converts every tensor to
bytes before writing anything. Reading goes through the real library, so the tests
validate this writer against it.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from typing import IO

import torch

PTG_ENTRY = "program.ptg"
SAFETENSORS_ENTRY = "program.safetensors"
ALIASES_ENTRY = "aliases.json"
_SAFETENSORS_METADATA_KEY = "__metadata__"

# safetensors dtype codes; mirrors safetensors.torch._TYPES inverted.
_DTYPE_CODES: dict[torch.dtype, str] = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
    torch.complex64: "C64",
}
for _name, _code in (("uint16", "U16"), ("uint32", "U32"), ("uint64", "U64")):
    _dtype = getattr(torch, _name, None)
    if _dtype is not None:
        _DTYPE_CODES[_dtype] = _code


def _raw(tensor: torch.Tensor) -> memoryview:
    """A zero-copy byte view of a contiguous CPU tensor.

    Reinterpreting as uint8 keeps dtypes numpy cannot express, such as bfloat16,
    working; neither the view nor .numpy() copies.
    """
    return tensor.flatten().view(torch.uint8).numpy().data


def _content_signature(tensor: torch.Tensor) -> tuple[str, tuple[int, ...], str]:
    digest = hashlib.sha256()
    digest.update(_raw(tensor))
    return (str(tensor.dtype), tuple(tensor.shape), digest.hexdigest())


def _check_mutable_storage_aliases(
    constants: dict[str, torch.Tensor], mutable_keys: frozenset[str]
) -> None:
    """Reject alias topology that the current PTN data model cannot express.

    Immutable views may be materialized independently because their storage
    identity is not observable. If either of two distinct keys is mutable,
    separating shared source storage would change program semantics. The current
    format has no storage/view manifest, so fail rather than silently sever that
    alias.
    """
    keys_by_storage: dict[tuple[str, int, int], list[str]] = {}
    for key, tensor in constants.items():
        # All empty tensors have an unobservable zero-byte range and commonly
        # report data_ptr() == 0 even when their storages are unrelated.
        if tensor.numel() == 0:
            continue
        device_index = tensor.device.index if tensor.device.index is not None else -1
        storage = (
            tensor.device.type,
            device_index,
            tensor.untyped_storage().data_ptr(),
        )
        keys_by_storage.setdefault(storage, []).append(key)

    for keys in keys_by_storage.values():
        if len(keys) > 1 and any(key in mutable_keys for key in keys):
            names = sorted(keys)
            raise ValueError(
                "write_ptn: distinct data keys share source storage and at least "
                f"one is mutable: {names}. The current PTN format cannot preserve "
                "mutable alias topology; use one data key for one logical state "
                "object."
            )


def _dedup(
    constants: dict[str, torch.Tensor],
    mutable_keys: frozenset[str],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Split constants into safetensors owners and a duplicate -> owner alias map.

    Only immutable keys are deduplicated. Equal bytes are not evidence that two
    mutable keys are the same runtime state: two independently mutable buffers can
    be zero-initialized to identical bytes and still need separate storage, so
    aliasing them would make a write to one visible through the other. A mutable
    key is therefore always its own owner, and never becomes an owner some other
    key aliases to.

    Immutable tensors sharing storage need no special handling: each owner's bytes
    are written independently. Distinct keys sharing storage when either is
    mutable are rejected, because writing them independently would silently break
    runtime aliasing and the current PTN format cannot encode their storage
    topology.
    """
    unknown_mutable = sorted(mutable_keys.difference(constants))
    if unknown_mutable:
        raise ValueError(
            "write_ptn: mutable_keys contains keys with no tensor data: "
            f"{unknown_mutable}"
        )

    for key, tensor in constants.items():
        if not isinstance(key, str):
            raise TypeError(
                f"write_ptn: tensor data key must be str, got {type(key).__name__}"
            )
        if key == _SAFETENSORS_METADATA_KEY:
            raise ValueError(
                f"write_ptn: {_SAFETENSORS_METADATA_KEY!r} is reserved by "
                "safetensors and cannot be a tensor data key."
            )
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"write_ptn: value for {key!r} must be a Tensor, got "
                f"{type(tensor).__name__}"
            )
        if tensor.dtype not in _DTYPE_CODES:
            # Preflight before opening the destination. This does not make the
            # complete save atomic, but predictable schema errors must not destroy
            # an existing valid artifact.
            raise KeyError(
                f"write_ptn: safetensors cannot represent dtype {tensor.dtype} "
                f"for data key {key!r}"
            )

    _check_mutable_storage_aliases(constants, mutable_keys)
    owners: dict[str, torch.Tensor] = {}
    aliases: dict[str, str] = {}
    owner_by_content: dict[tuple[str, tuple[int, ...], str], str] = {}
    for key in sorted(constants):
        tensor = constants[key].detach().cpu().contiguous()
        if key in mutable_keys:
            owners[key] = tensor
            continue
        signature = _content_signature(tensor)
        owner = owner_by_content.get(signature)
        if owner is None:
            owner_by_content[signature] = key
            owners[key] = tensor
        else:
            aliases[key] = owner
    return owners, aliases


def _write_safetensors(stream: IO[bytes], owners: dict[str, torch.Tensor]) -> None:
    """Emit the safetensors format: header length, JSON header, then tensor data.

    Offsets in the header are relative to the start of the data section, and the
    header is space-padded so that section begins 8-byte aligned.
    """
    header: dict[str, object] = {}
    offset = 0
    for key, tensor in owners.items():
        size = tensor.numel() * tensor.element_size()
        header[key] = {
            "dtype": _DTYPE_CODES[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size

    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)

    stream.write(len(encoded).to_bytes(8, "little"))
    stream.write(encoded)
    for tensor in owners.values():
        stream.write(_raw(tensor))


def write_ptn(
    path: str,
    ptg: bytes,
    constants: dict[str, torch.Tensor],
    mutable_keys: frozenset[str] = frozenset(),
) -> None:
    """Write a .ptn package, streaming the constants a tensor at a time.

    Args:
        path: Destination .ptn path. Member names inside are fixed, so the
            filename carries no meaning and the package survives a rename.
        ptg: Serialized native Program flatbuffer.
        constants: Every data key the program references, mapped to its tensor.
        mutable_keys: Keys whose runtime state is mutable. Excluded from content
            dedup, since equal bytes do not make two mutable buffers the same
            state. Passed in rather than derived here, so this stays independent of
            the graph format. Every mutable key must occur in ``constants``.

    Raises:
        KeyError: If a constant has a dtype safetensors cannot represent.
        TypeError: If a data key is not a string or a value is not a tensor.
        ValueError: If mutable-key policy is inconsistent or a data key is reserved.
    """
    # TODO: save atomically. This opens the destination directly, so a failure
    # part way through truncates an existing valid package. Write to a temporary
    # file beside the destination and os.replace it once complete.
    owners, aliases = _dedup(constants, mutable_keys)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # ZIP_STORED keeps entries byte-addressable; allowZip64 lifts the format's
    # 4GiB per-entry and total-size limits.
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True) as pkg:
        pkg.writestr(PTG_ENTRY, ptg)
        if owners:
            with pkg.open(SAFETENSORS_ENTRY, "w") as stream:
                _write_safetensors(stream, owners)
            if aliases:
                pkg.writestr(
                    ALIASES_ENTRY, json.dumps(aliases, indent=2, sort_keys=True)
                )


def read_ptn(path: str) -> tuple[bytes, dict[str, torch.Tensor]]:
    """Inverse of write_ptn, with aliases resolved back to their owner's tensor.

    Returns every data key the program references, owners and aliases alike. Used
    by the tests to verify the round-trip; the lowering path only ever writes.

    Raises:
        ValueError: If an alias names a key with no safetensors entry, or a key is
            both an owner and an alias.
    """
    from safetensors.torch import load as safetensors_load

    with zipfile.ZipFile(path) as pkg:
        entries = set(pkg.namelist())
        ptg = pkg.read(PTG_ENTRY)
        owners: dict[str, torch.Tensor] = (
            safetensors_load(pkg.read(SAFETENSORS_ENTRY))
            if SAFETENSORS_ENTRY in entries
            else {}
        )
        aliases: dict[str, str] = {}
        if ALIASES_ENTRY in entries:
            raw_aliases = json.loads(pkg.read(ALIASES_ENTRY))
            if not isinstance(raw_aliases, dict):
                raise ValueError("read_ptn: aliases.json must contain a JSON object.")
            for key, owner in raw_aliases.items():
                if not isinstance(key, str) or not isinstance(owner, str):
                    raise ValueError(
                        "read_ptn: every alias and owner name must be a string."
                    )
                if key == _SAFETENSORS_METADATA_KEY:
                    raise ValueError(
                        f"read_ptn: alias key {_SAFETENSORS_METADATA_KEY!r} is "
                        "reserved by safetensors."
                    )
                aliases[key] = owner

    constants = dict(owners)
    for key, owner in aliases.items():
        if owner not in owners:
            raise ValueError(
                f"read_ptn: alias {key!r} names owner {owner!r}, which has no "
                f"safetensors entry."
            )
        if key in owners:
            raise ValueError(
                f"read_ptn: {key!r} is both a safetensors owner and an alias."
            )
        constants[key] = owners[owner]
    return ptg, constants


__all__ = ["ALIASES_ENTRY", "PTG_ENTRY", "SAFETENSORS_ENTRY", "read_ptn", "write_ptn"]
