# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Optional recorded method calls stored inside a .ptn package.

Each set lives under its own prefix, so adding one only appends members:

    samples/<set>/index.json         each call's method, tensors and scalars
    samples/<set>/data.safetensors   those tensors, aligned like constants

Program loaders ignore these members. The format is specified in
backends/native/README.md; read_samples enforces it.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import tempfile
import warnings
import zipfile
from dataclasses import dataclass

import torch
from executorch.exir._serialize._ptn import (
    _DATA_ALIGNMENT,
    _DTYPE_CODES,
    _LOCAL_HEADER_SIZE,
    _write_safetensors,
    PTG_ENTRY,
)

_SAMPLES_DIR = "samples/"
_INDEX_ENTRY = "index.json"
_DATA_ENTRY = "data.safetensors"
_VERSION = "1.0"

_NAME = re.compile(r"[A-Za-z0-9_-]+")
_VERSION_PATTERN = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
_SSA_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

Value = torch.Tensor | int | float | bool | list["Value"]
Tolerance = tuple[float, float]
_LOOSE_TOL = 1e-3


@dataclass
class Call:
    """One call of a method: positional inputs, and optionally what it produced."""

    method: str
    inputs: list[Value]
    outputs: list[Value] | None = None
    intermediates: dict[str, torch.Tensor] | None = None
    tol: Tolerance | None = None


@dataclass
class Sample:
    """Calls run in order from the program's initial state, with no reset between."""

    calls: list[Call]
    tol: Tolerance | None = None


@dataclass
class SampleSet:
    samples: list[Sample]
    description: str | None = None
    source: str | None = None
    tol: Tolerance | None = None


def _check_tol(tol: object, where: str) -> Tolerance:
    if (
        not isinstance(tol, (list, tuple))
        or len(tol) != 2
        or not all(
            isinstance(t, (int, float))
            and not isinstance(t, bool)
            and math.isfinite(t)
            and t >= 0
            for t in tol
        )
    ):
        raise ValueError(f"{where}: tol must be two finite numbers >= 0, got {tol!r}")
    return (float(tol[0]), float(tol[1]))


def _encode_tol(tol: object, where: str, stacklevel: int) -> list[float]:
    checked = _check_tol(tol, f"write_samples: {where}")
    if max(checked) > _LOOSE_TOL:
        warnings.warn(
            f"write_samples: {where} tol (atol, rtol) = {checked} is above "
            f"{_LOOSE_TOL} and may hide numerical bugs",
            stacklevel=stacklevel,
        )
    return list(checked)


def _encode(value: object, key: str, tensors: dict[str, torch.Tensor]) -> object:
    """The index.json form of `value`; tensors are collected under their path key."""
    if isinstance(value, torch.Tensor):
        if value.dtype not in _DTYPE_CODES:
            raise KeyError(
                f"write_samples: safetensors cannot represent dtype {value.dtype} "
                f"at {key}"
            )
        tensors[key] = value.detach().resolve_conj().resolve_neg().cpu().contiguous()
        return key
    if isinstance(value, (list, tuple)):
        return [_encode(v, f"{key}[{i}]", tensors) for i, v in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"write_samples: scalar at {key} is not finite")
    if isinstance(value, (bool, int, float)):
        return value
    raise TypeError(
        f"write_samples: unsupported value at {key}: {type(value).__name__}"
    )


def _encode_call(
    call: Call, at: str, tensors: dict[str, torch.Tensor]
) -> dict[str, object]:
    for field, values in (("inputs", call.inputs), ("outputs", call.outputs)):
        if values is not None and not isinstance(values, (list, tuple)):
            raise TypeError(
                f"write_samples: {at} {field} must be a list, "
                f"got {type(values).__name__}"
            )
    if not isinstance(call.method, str) or not call.method:
        raise ValueError(f"write_samples: {at} method must be a non-empty string")
    entry: dict[str, object] = {
        "method": call.method,
        "in": _encode(call.inputs, f"{at}.in", tensors),
    }
    if call.outputs is not None:
        entry["out"] = _encode(call.outputs, f"{at}.out", tensors)
    if call.intermediates is not None:
        intermediates = {}
        for ssa, tensor in call.intermediates.items():
            if not _SSA_NAME.fullmatch(ssa):
                raise ValueError(
                    f"write_samples: intermediate name {ssa!r} must match "
                    f"{_SSA_NAME.pattern}"
                )
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"write_samples: intermediate {ssa!r} must be a Tensor")
            intermediates[ssa] = _encode(tensor, f"{at}.intermediates.{ssa}", tensors)
        entry["intermediates"] = intermediates
    if call.tol is not None:
        entry["tol"] = _encode_tol(call.tol, at, stacklevel=5)
    return entry


def _encode_set(
    samples: SampleSet, tensors: dict[str, torch.Tensor]
) -> dict[str, object]:
    index: dict[str, object] = {"version": _VERSION}
    for field, text in (
        ("description", samples.description),
        ("source", samples.source),
    ):
        if text is None:
            continue
        if not isinstance(text, str):
            raise TypeError(f"write_samples: {field} must be a string")
        index[field] = text
    if samples.tol is not None:
        index["tol"] = _encode_tol(samples.tol, "set", stacklevel=4)
    entries = []
    for i, sample in enumerate(samples.samples):
        if not sample.calls:
            raise ValueError(f"write_samples: samples[{i}] needs at least one call")
        calls = []
        for j, call in enumerate(sample.calls):
            calls.append(_encode_call(call, f"samples[{i}].calls[{j}]", tensors))
        entry: dict[str, object] = {"calls": calls}
        if sample.tol is not None:
            entry["tol"] = _encode_tol(sample.tol, f"samples[{i}]", stacklevel=4)
        entries.append(entry)
    index["samples"] = entries
    return index


def write_samples(path: str, name: str, samples: SampleSet) -> None:
    """Append a sample set to an existing .ptn without touching its other members.

    The package is replaced only once the new set is complete, so a failure
    leaves it as it was. This costs a copy of the package.

    Warns when a tolerance's atol or rtol is above 1e-3.

    Raises:
        ValueError: If the set name or a method name is invalid, a set or sample
            is empty, the set already exists, the package has no program, or a
            scalar or tolerance is out of range.
        KeyError: If a tensor has a dtype safetensors cannot represent.
        TypeError: If a value is not a tensor, scalar or list, or a call's inputs
            or outputs are not a list.
    """
    if not _NAME.fullmatch(name):
        raise ValueError(f"write_samples: set name {name!r} must match {_NAME.pattern}")
    if not samples.samples:
        raise ValueError("write_samples: a set needs at least one sample")
    path = os.path.realpath(path)

    tensors: dict[str, torch.Tensor] = {}
    encoded = json.dumps(_encode_set(samples, tensors), indent=1, allow_nan=False)

    prefix = f"{_SAMPLES_DIR}{name}/"
    with zipfile.ZipFile(path) as pkg:
        names = pkg.namelist()
    if PTG_ENTRY not in names:
        raise ValueError(f"write_samples: {path} has no {PTG_ENTRY}")
    if any(n.startswith(prefix) for n in names):
        raise ValueError(f"write_samples: {prefix} already exists in {path}")
    fd, staging = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".ptn.tmp")
    os.close(fd)
    try:
        shutil.copyfile(path, staging)
        shutil.copymode(path, staging)
        with zipfile.ZipFile(staging, "a", zipfile.ZIP_STORED, allowZip64=True) as pkg:
            _write_safetensors(pkg, tensors, prefix + _DATA_ENTRY)
            pkg.writestr(prefix + _INDEX_ENTRY, encoded)
        os.replace(staging, path)
    except BaseException:
        os.unlink(staging)
        raise


def _data_offset(path: str, info: zipfile.ZipInfo) -> int:
    """File offset of a stored member's data, read from its local header."""
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(_LOCAL_HEADER_SIZE)
    name_size, extra_size = struct.unpack("<HH", local[26:30])
    return info.header_offset + _LOCAL_HEADER_SIZE + name_size + extra_size


class _SetDecoder:
    def __init__(self, tensors: dict[str, torch.Tensor], where: str) -> None:
        self.tensors = tensors
        self.where = where
        self.unreferenced: set[str] = set(tensors)

    def fail(self, message: str) -> ValueError:
        return ValueError(f"{self.where}: {message}")

    def tol(self, entry: dict[str, object]) -> Tolerance | None:
        return _check_tol(entry["tol"], self.where) if "tol" in entry else None

    def tensor(self, value: str, key: str) -> torch.Tensor:
        if value != key:
            raise self.fail(f"key {value!r} must be {key!r}")
        if value not in self.tensors:
            raise self.fail(f"no tensor named {value!r}")
        self.unreferenced.discard(value)
        return self.tensors[value]

    def value(self, value: object, key: str) -> Value:
        if isinstance(value, str):
            return self.tensor(value, key)
        if isinstance(value, list):
            return self.values(value, key)
        if isinstance(value, (bool, int)) or (
            isinstance(value, float) and math.isfinite(value)
        ):
            return value
        raise self.fail(f"unsupported value at {key}: {value!r}")

    def values(self, values: list[object], key: str) -> list[Value]:
        return [self.value(v, f"{key}[{i}]") for i, v in enumerate(values)]

    def intermediates(self, raw: object, at: str) -> dict[str, torch.Tensor]:
        if not isinstance(raw, dict):
            raise self.fail(f"{at}.intermediates must be an object")
        result = {}
        for ssa, key in raw.items():
            if not _SSA_NAME.fullmatch(ssa) or not isinstance(key, str):
                raise self.fail(f"bad intermediate {ssa!r} in {at}")
            result[ssa] = self.tensor(key, f"{at}.intermediates.{ssa}")
        return result

    def call(self, entry: object, at: str) -> Call:
        if not isinstance(entry, dict) or not isinstance(entry.get("in"), list):
            raise self.fail(f"{at} must be an object with an 'in' list")
        method = entry.get("method")
        if not isinstance(method, str) or not method:
            raise self.fail(f"{at}.method must be a non-empty string")
        if "out" in entry and not isinstance(entry["out"], list):
            raise self.fail(f"{at}.out must be a list")
        return Call(
            method=method,
            inputs=self.values(entry["in"], f"{at}.in"),
            outputs=self.values(entry["out"], f"{at}.out") if "out" in entry else None,
            intermediates=(
                self.intermediates(entry["intermediates"], at)
                if "intermediates" in entry
                else None
            ),
            tol=self.tol(entry),
        )

    def sample(self, entry: object, at: str) -> Sample:
        calls = entry.get("calls") if isinstance(entry, dict) else None
        if not isinstance(calls, list) or not calls:
            raise self.fail(f"{at} must be an object with a non-empty 'calls' list")
        assert isinstance(entry, dict)
        return Sample(
            calls=[self.call(c, f"{at}.calls[{j}]") for j, c in enumerate(calls)],
            tol=self.tol(entry),
        )


def _decode_set(
    index: object, tensors: dict[str, torch.Tensor], where: str
) -> SampleSet:
    decoder = _SetDecoder(tensors, where)
    if not isinstance(index, dict):
        raise decoder.fail("index.json must contain a JSON object")
    version = index.get("version")
    match = _VERSION_PATTERN.fullmatch(version) if isinstance(version, str) else None
    if match is None or match[1] != _VERSION.split(".")[0]:
        raise decoder.fail(f"unsupported version {version!r}, expected {_VERSION}")
    entries = index.get("samples")
    if not isinstance(entries, list) or not entries:
        raise decoder.fail("samples must be a non-empty list")
    for field in ("description", "source"):
        if field in index and not isinstance(index[field], str):
            raise decoder.fail(f"{field} must be a string")

    samples = [decoder.sample(e, f"samples[{i}]") for i, e in enumerate(entries)]
    newer_minor = int(match[2]) > int(_VERSION.split(".")[1])
    if decoder.unreferenced and not newer_minor:
        raise decoder.fail(f"unreferenced tensors {sorted(decoder.unreferenced)}")
    return SampleSet(
        samples=samples,
        description=index.get("description"),
        source=index.get("source"),
        tol=decoder.tol(index),
    )


def read_samples(path: str) -> dict[str, SampleSet]:
    """Every sample set in a .ptn, keyed by set name, validated as read.

    A package without samples returns an empty dict.

    Raises:
        ValueError: If any samples/ member, index or tensor violates the format.
    """
    from safetensors import SafetensorError
    from safetensors.torch import load as safetensors_load

    sets: dict[str, SampleSet] = {}
    with zipfile.ZipFile(path) as pkg:
        found: dict[str, dict[str, zipfile.ZipInfo]] = {}
        for info in pkg.infolist():
            if not info.filename.startswith(_SAMPLES_DIR):
                continue
            parts = info.filename[len(_SAMPLES_DIR) :].split("/")
            if (
                len(parts) != 2
                or not _NAME.fullmatch(parts[0])
                or parts[1] not in (_INDEX_ENTRY, _DATA_ENTRY)
            ):
                raise ValueError(f"read_samples: unexpected member {info.filename!r}")
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"read_samples: {info.filename!r} is compressed")
            members = found.setdefault(parts[0], {})
            if parts[1] in members:
                raise ValueError(f"read_samples: duplicate member {info.filename!r}")
            members[parts[1]] = info

        for name, members in sorted(found.items()):
            where = f"read_samples: {_SAMPLES_DIR}{name}"
            for entry in (_INDEX_ENTRY, _DATA_ENTRY):
                if entry not in members:
                    raise ValueError(f"{where}: missing {entry}")
            try:
                data = pkg.read(members[_DATA_ENTRY])
                index = json.loads(pkg.read(members[_INDEX_ENTRY]))
                tensors = safetensors_load(data)
            except (
                ValueError,
                KeyError,
                RecursionError,
                SafetensorError,
                zipfile.BadZipFile,
            ) as e:
                raise ValueError(f"{where}: {e}") from e
            header_size = int.from_bytes(data[:8], "little")
            start = _data_offset(path, members[_DATA_ENTRY]) + 8 + header_size
            if start % _DATA_ALIGNMENT:
                raise ValueError(
                    f"{where}: tensor data is not {_DATA_ALIGNMENT}-byte aligned"
                )
            try:
                sets[name] = _decode_set(index, tensors, where)
            except (RecursionError, OverflowError) as e:
                raise ValueError(f"{where}: {e}") from e
    return sets


__all__ = [
    "Call",
    "Sample",
    "SampleSet",
    "Value",
    "read_samples",
    "write_samples",
]
