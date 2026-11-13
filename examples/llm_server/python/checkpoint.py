# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit CPU checkpoints and durable files; no cache discovery or eviction."""

import hashlib
import hmac
import json
import os
import stat
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_MAGIC = b"ETCP0002"
_MAX_HEADER = 1024 * 1024
_DEFAULT_MAX_BYTES = 16 * 1024**3


def checkpoint_identity(
    files: Sequence[os.PathLike], configuration: Mapping[str, Any]
) -> str:
    """Hash explicitly supplied model/tokenizer artifacts and execution settings.

    Include the PTE, external weights, tokenizer, adapters, and settings that
    affect cached activations (e.g. RoPE and draft configuration). Paths are not
    identity: moving identical files does not invalidate a checkpoint.
    """
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    for path in files:
        artifact = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                artifact.update(block)
        digest.update(artifact.digest())
    return digest.hexdigest()


def _integer(value, name, maximum=2**31 - 1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"invalid checkpoint {name}")
    return value


def _validate_cache(cache, position, is_target, buffer):
    if not isinstance(cache, dict):
        raise ValueError("invalid checkpoint cache record")
    end = _integer(cache.get("position"), "cache position")
    if end > position or (is_target and end != position):
        raise ValueError("inconsistent checkpoint cache positions")
    dtype = _integer(cache.get("dtype"), "dtype")
    itemsize = {5: 2, 6: 4, 15: 2}.get(dtype)
    if itemsize is None:
        raise ValueError("unsupported checkpoint dtype")
    if "dflash_block_size" not in cache:
        raise ValueError("invalid checkpoint cache record")
    block = cache["dflash_block_size"]
    if block is not None:
        _integer(block, "draft block size")
    layers = cache.get("layers")
    if not isinstance(layers, list) or not 1 <= len(layers) <= 1024:
        raise ValueError("invalid checkpoint layers")
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("invalid checkpoint layer record")
        begin = _integer(layer.get("begin"), "retained start")
        heads = _integer(layer.get("heads"), "heads")
        dim = _integer(layer.get("head_dim"), "head dimension")
        window = _integer(layer.get("window"), "window")
        if begin > end or not heads or not dim or (not window and begin):
            raise ValueError("invalid checkpoint layer geometry")
        size = (end - begin) * heads * dim * itemsize
        buffer(layer.get("key"), size)
        buffer(layer.get("value"), size)


def _validate_manifest(manifest, data):
    data_bytes = len(data)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("format"), str):
        raise ValueError("invalid checkpoint manifest")
    if _integer(manifest.get("version"), "manifest version") != 2:
        raise ValueError("unsupported checkpoint manifest version")
    position = _integer(manifest.get("position"), "position")
    if manifest.get("data_bytes") != data_bytes:
        raise ValueError("checkpoint payload size mismatch")
    caches = manifest.get("caches")
    if not isinstance(caches, list) or not 1 <= len(caches) <= 2:
        raise ValueError("invalid checkpoint cache count")
    offset = 0

    def buffer(descriptor, expected):
        nonlocal offset
        if (
            not isinstance(descriptor, list)
            or len(descriptor) != 2
            or any(type(n) is not int for n in descriptor)
            or descriptor != [offset, expected]
            or expected > data_bytes - offset
        ):
            raise ValueError("invalid checkpoint buffer extent")
        offset += expected

    buffer(manifest.get("tokens"), position * 4)
    token_data = memoryview(data)[: position * 4]
    if any(token < 0 for (token,) in struct.iter_unpack("<i", token_data)):
        raise ValueError("invalid checkpoint token")
    for i, cache in enumerate(caches):
        _validate_cache(cache, position, i == 0, buffer)
    row = _integer(manifest.get("hidden_row_bytes"), "hidden row bytes")
    if len(caches) == 1 and row:
        raise ValueError("unexpected draft hidden state")
    hidden = (position - caches[1]["position"]) * row if len(caches) == 2 else 0
    if len(caches) == 2 and not row:
        raise ValueError("missing draft hidden geometry")
    buffer(manifest.get("hidden"), hidden)
    if offset != data_bytes:
        raise ValueError("checkpoint contains unreferenced bytes")


@dataclass(frozen=True)
class CPUCheckpoint:
    identity: str
    manifest: dict
    data: bytes

    def validate(self, expected_identity: str) -> None:
        if (
            not isinstance(expected_identity, str)
            or not expected_identity
            or self.identity != expected_identity
        ):
            raise ValueError("checkpoint model/configuration identity mismatch")
        if not isinstance(self.data, bytes):
            raise TypeError("checkpoint data must be owned CPU bytes")
        _validate_manifest(self.manifest, self.data)

    @property
    def tokens(self) -> tuple[int, ...]:
        offset, size = self.manifest["tokens"]
        return tuple(
            token
            for (token,) in struct.iter_unpack(
                "<i", memoryview(self.data)[offset : offset + size]
            )
        )

    def common_prefix_length(self, tokens: Sequence[int]) -> int:
        """Token match only; this does not assert that the state can rewind there."""
        for i, (saved, requested) in enumerate(zip(self.tokens, tokens)):
            if saved != requested:
                return i
        return min(len(self.tokens), len(tokens))

    def validate_restore_position(self, position: int) -> None:
        self.validate(self.identity)
        _integer(position, "restore position")
        if not 0 < position <= self.manifest["position"]:
            raise ValueError("restore position is outside the checkpoint")
        for cache in self.manifest["caches"]:
            end = min(position, cache["position"])
            for layer in cache["layers"]:
                required = (
                    max(0, end - layer["window"] + (cache["dflash_block_size"] is None))
                    if layer["window"]
                    else 0
                )
                if end and layer["begin"] > required:
                    raise ValueError(
                        "restore position requires history this checkpoint no longer retains"
                    )


def _metadata(checkpoint):
    try:
        return json.dumps(
            {"identity": checkpoint.identity, "manifest": checkpoint.manifest},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except RecursionError as error:
        raise ValueError("invalid checkpoint header") from error


def save_checkpoint(checkpoint: CPUCheckpoint, path: os.PathLike) -> None:
    """Synchronously publish a complete file atomically; I/O errors propagate."""
    checkpoint.validate(checkpoint.identity)
    metadata = _metadata(checkpoint)
    digest = hashlib.sha256(metadata)
    digest.update(checkpoint.data)
    header = json.dumps(
        {"metadata": json.loads(metadata), "sha256": digest.hexdigest()},
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    if len(header) > _MAX_HEADER:
        raise ValueError("checkpoint metadata exceeds format limit")
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_MAGIC)
            stream.write(struct.pack("<Q", len(header)))
            stream.write(header)
            stream.write(checkpoint.data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _open_checkpoint(path):
    flags = os.O_RDONLY
    if os.name == "posix":
        flags |= os.O_NOFOLLOW | os.O_NONBLOCK
    elif os.name == "nt":
        flags |= os.O_BINARY
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("checkpoint must be a regular file")
        stream = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    return stream


def load_checkpoint(
    path: os.PathLike, *, expected_identity: str, max_bytes: int = _DEFAULT_MAX_BYTES
) -> CPUCheckpoint:
    """Read into CPU memory, verify integrity and identity; never restore a device."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    with _open_checkpoint(path) as stream:
        if stream.read(8) != _MAGIC:
            raise ValueError("unsupported checkpoint file version")
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("truncated checkpoint header")
        length = struct.unpack("<Q", raw_length)[0]
        if length > _MAX_HEADER:
            raise ValueError("checkpoint header exceeds limit")
        raw_header = stream.read(length)
        if len(raw_header) != length:
            raise ValueError("truncated checkpoint header")
        try:
            header = json.loads(raw_header)
        except RecursionError as error:
            raise ValueError("invalid checkpoint header") from error
        if (
            not isinstance(header, dict)
            or set(header) != {"metadata", "sha256"}
            or not isinstance(header["sha256"], str)
            or len(header["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in header["sha256"])
            or not isinstance(header["metadata"], dict)
            or set(header["metadata"]) != {"identity", "manifest"}
            or not isinstance(header["metadata"]["manifest"], dict)
        ):
            raise ValueError("invalid checkpoint header")
        metadata = header["metadata"]
        if metadata["identity"] != expected_identity:
            raise ValueError("checkpoint model/configuration identity mismatch")
        size = os.fstat(stream.fileno()).st_size - 16 - length
        if (
            size < 0
            or size > max_bytes
            or metadata["manifest"].get("data_bytes") != size
        ):
            raise ValueError("checkpoint payload is truncated or exceeds limit")
        data = stream.read(size + 1)
    checkpoint = CPUCheckpoint(metadata["identity"], metadata["manifest"], data)
    checkpoint.validate(expected_identity)
    digest = hashlib.sha256(_metadata(checkpoint))
    digest.update(data)
    if not hmac.compare_digest(digest.hexdigest(), header["sha256"]):
        raise ValueError("checkpoint checksum mismatch")
    return checkpoint
