# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import base64
import binascii
import json
from typing import Any, Dict, List, Sequence, Union

# Functions here are used to take the raw delegate_debug_metadata bytes stored in ETDump,
# check whether those bytes contain VGF neural statistics JSON,
# validate the schema/version, decode base64-encoded binary blobs,
# and return a normal Python dictionary that tooling can consume.

SCHEMA = "executorch.vgf.neural_statistics"
SCHEMA_VERSION = 1

DelegateMetadataBytes = Union[bytes, bytearray, str]


def _to_bytes(metadata: DelegateMetadataBytes) -> bytes:
    if isinstance(metadata, bytes):
        return metadata
    if isinstance(metadata, bytearray):
        return bytes(metadata)
    if isinstance(metadata, str):
        return metadata.encode("utf-8")
    raise TypeError(f"Unsupported delegate metadata type: {type(metadata)}")


def _decode_blob(blob: Dict[str, Any]) -> Dict[str, Any]:
    decoded = dict(blob)

    if not decoded.get("available", False):
        decoded.setdefault("raw_data", b"")
        return decoded

    if decoded.get("encoding") != "base64":
        raise ValueError(
            f"Unsupported VGF neural statistics blob encoding: {decoded.get('encoding')}"
        )

    encoded_data = decoded.get("data", "")
    try:
        decoded["raw_data"] = base64.b64decode(encoded_data, validate=True)
    except (binascii.Error, TypeError) as exc:
        raise ValueError("Malformed base64 data in VGF neural statistics blob") from exc

    return decoded


def parse_vgf_neural_statistics_metadata(
    metadata: DelegateMetadataBytes,
) -> Dict[str, Any]:
    payload = json.loads(_to_bytes(metadata).decode("utf-8"))

    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Not VGF neural statistics metadata: {payload.get('schema')}")

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported VGF neural statistics metadata schema version: "
            f"{payload.get('schema_version')}"
        )

    payload = dict(payload)
    decoded_segments = []

    for segment in payload.get("segments", []):
        decoded_segment = dict(segment)
        for key in ("debug_database", "statistics_info", "statistics_memory"):
            blob = decoded_segment.get(key)
            if isinstance(blob, dict):
                decoded_segment[key] = _decode_blob(blob)
        decoded_segments.append(decoded_segment)

    payload["segments"] = decoded_segments
    return payload


def parse_vgf_neural_statistics_delegate_metadata(
    delegate_metadata_list: Sequence[DelegateMetadataBytes],
) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []

    for metadata in delegate_metadata_list:
        if metadata is None or metadata == b"" or metadata == "":
            continue

        try:
            metadata_bytes = _to_bytes(metadata)
        except TypeError:
            # Not a valid delegate metadata representation.
            # Treat it as unrelated metadata from another source.
            continue

        try:
            payload = json.loads(metadata_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # If the blob appears to be VGF neural statistics metadata but is
            # malformed, surface the error instead of silently dropping it.
            if SCHEMA.encode("utf-8") in metadata_bytes:
                raise ValueError(
                    "Malformed VGF neural statistics delegate metadata"
                ) from exc

            # Otherwise this is generic delegate metadata from another backend.
            continue

        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            # Inspector events can contain delegate metadata from other backends.
            # Ignore only records that are clearly not VGF neural statistics.
            continue

        # From this point onward the record claims to be VGF neural statistics.
        # Do not swallow parse errors: malformed VGF records should be visible.
        parsed.append(parse_vgf_neural_statistics_metadata(metadata_bytes))

    return parsed
