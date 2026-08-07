# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Receipt contract for the Gemma 4 target-only prefill oracle."""

from __future__ import annotations

import hashlib
import json
import math

from pathlib import Path
from typing import Mapping, Sequence, TypeGuard


TARGET_PREFILL_SCHEMA_VERSION = 2
TARGET_PREFILL_ENVELOPE_KIND = "target_prefill_v2"
TARGET_PREFILL_AUTHORITY = "target_only_eager"
TARGET_PREFILL_CHUNK_SIZE = 512
TARGET_PREFILL_ATOL = 1e-4
TARGET_PREFILL_RTOL = 1e-3
TARGET_PREFILL_CONTEXTS = (128, 511, 512, 513, 514, 1024, 2048, 4096, 4097, 8192)
TARGET_PREFILL_VOCAB_SIZE = 262144

_TOP_LEVEL_KEYS = {
    "authority",
    "checkpoint_acquisition",
    "contexts",
    "envelope_kind",
    "producer",
    "run",
    "schema_version",
}
_CONTEXT_KEYS = {
    "arm_configs",
    "cache_reset_counts",
    "chunk_size",
    "context",
    "final_chunk_length",
    "final_chunk_start",
    "layer0_manual_unfused_vs_custom_sdpa_fused",
    "logits_post_softcap",
    "logits_pre_softcap",
    "prefill_token_post_softcap",
    "prefill_token_raw",
    "prompt_plan_sha256",
}
_ARM_CONFIG_KEYS = {
    "dtype",
    "enable_dynamic_shape",
    "group_size",
    "max_seq_len",
    "text_quantize",
    "use_custom_sdpa",
    "use_kv_cache",
    "variant",
}
_TENSOR_KEYS = {"byte_order", "dtype", "layout", "sha256", "shape"}


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys mismatch: expected {sorted(expected)}, got {sorted(value)}"
        )


def canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"target-prefill input is not a regular file: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return {"bytes": byte_count, "sha256": digest.hexdigest()}


def prompt_tokens(context: int) -> list[int]:
    if not _is_int(context) or context <= 0:
        raise ValueError("target-prefill context must be a positive integer")
    return [(index % (TARGET_PREFILL_VOCAB_SIZE - 1)) + 1 for index in range(context)]


def prompt_plan_sha256(context: int) -> str:
    encoded = json.dumps(prompt_tokens(context), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def final_chunk_range(context: int) -> tuple[int, int]:
    if not _is_int(context) or context <= 0:
        raise ValueError("target-prefill context must be a positive integer")
    remainder = context % TARGET_PREFILL_CHUNK_SIZE
    length = remainder if remainder else TARGET_PREFILL_CHUNK_SIZE
    return context - length, length


def reviewed_producer_source_path() -> Path:
    path = Path(__file__).with_name("generate_target_prefill_oracle.py")
    if not path.is_file():
        raise ValueError("reviewed target-prefill producer resource is missing")
    return path


def _validate_tensor_envelope(
    value: object,
    *,
    label: str,
    expected_shape: list[int],
) -> None:
    envelope = _mapping(value, label)
    _require_exact_keys(envelope, _TENSOR_KEYS, label)
    if envelope.get("byte_order") != "little":
        raise ValueError(f"{label} byte order must be little")
    if envelope.get("dtype") != "float32":
        raise ValueError(f"{label} dtype must be float32")
    if envelope.get("layout") != "row_major_contiguous":
        raise ValueError(f"{label} layout must be row_major_contiguous")
    if not _is_hex(envelope.get("sha256"), 64):
        raise ValueError(f"{label} has an invalid sha256")
    shape = _sequence(envelope.get("shape"), f"{label} shape")
    if list(shape) != expected_shape:
        raise ValueError(f"{label} shape must be {expected_shape}")


def _validate_arm_configs(value: object) -> None:
    configs = _mapping(value, "target-prefill arm configurations")
    expected_names = {"custom_sdpa_fused", "manual_unfused"}
    _require_exact_keys(configs, expected_names, "target-prefill arm configurations")
    fused = _mapping(configs["custom_sdpa_fused"], "custom SDPA configuration")
    manual = _mapping(configs["manual_unfused"], "manual SDPA configuration")
    _require_exact_keys(fused, _ARM_CONFIG_KEYS, "custom SDPA configuration")
    _require_exact_keys(manual, _ARM_CONFIG_KEYS, "manual SDPA configuration")
    if fused.get("use_custom_sdpa") is not True:
        raise ValueError("custom SDPA arm must enable use_custom_sdpa")
    if manual.get("use_custom_sdpa") is not False:
        raise ValueError("manual SDPA arm must disable use_custom_sdpa")
    fused_without_route = dict(fused)
    manual_without_route = dict(manual)
    del fused_without_route["use_custom_sdpa"]
    del manual_without_route["use_custom_sdpa"]
    if fused_without_route != manual_without_route:
        raise ValueError("target-prefill arm configurations differ beyond SDPA")
    required = {
        "dtype": "float32",
        "enable_dynamic_shape": True,
        "group_size": 128,
        "max_seq_len": 8960,
        "text_quantize": "8da4w+emb4",
        "use_kv_cache": True,
        "variant": "e2b",
    }
    if fused_without_route != required:
        raise ValueError("target-prefill arm configuration is not production-shaped")


def _validate_context(value: object, expected_context: int) -> None:  # noqa: C901
    context = _mapping(value, f"target-prefill context {expected_context}")
    _require_exact_keys(
        context, _CONTEXT_KEYS, f"target-prefill context {expected_context}"
    )
    if context.get("context") != expected_context:
        raise ValueError("target-prefill context value does not match its key")
    if context.get("chunk_size") != TARGET_PREFILL_CHUNK_SIZE:
        raise ValueError("target-prefill chunk size must be 512")
    final_start, final_length = final_chunk_range(expected_context)
    if (
        context.get("final_chunk_start") != final_start
        or context.get("final_chunk_length") != final_length
    ):
        raise ValueError("target-prefill final chunk range mismatch")
    if context.get("prompt_plan_sha256") != prompt_plan_sha256(expected_context):
        raise ValueError("target-prefill prompt plan mismatch")

    _validate_arm_configs(context.get("arm_configs"))
    reset_counts = _mapping(
        context.get("cache_reset_counts"), "target-prefill cache reset counts"
    )
    _require_exact_keys(
        reset_counts,
        {"custom_sdpa_fused", "manual_unfused"},
        "target-prefill cache reset counts",
    )
    if any(not _is_int(count) or count <= 0 for count in reset_counts.values()):
        raise ValueError("target-prefill cache reset counts must be positive")

    _validate_tensor_envelope(
        context.get("logits_pre_softcap"),
        label="pre-softcap logits",
        expected_shape=[1, 1, TARGET_PREFILL_VOCAB_SIZE],
    )
    _validate_tensor_envelope(
        context.get("logits_post_softcap"),
        label="post-softcap logits",
        expected_shape=[1, 1, TARGET_PREFILL_VOCAB_SIZE],
    )
    raw_token = context.get("prefill_token_raw")
    post_token = context.get("prefill_token_post_softcap")
    if (
        not _is_int(raw_token)
        or raw_token < 0
        or raw_token >= TARGET_PREFILL_VOCAB_SIZE
    ):
        raise ValueError("target-prefill raw token is invalid")
    if post_token != raw_token:
        raise ValueError("target-prefill raw/post-softcap tokens differ")

    witness = _mapping(
        context.get("layer0_manual_unfused_vs_custom_sdpa_fused"),
        "target-prefill AV witness",
    )
    _require_exact_keys(
        witness,
        {"agreement", "custom_sdpa_fused", "manual_unfused"},
        "target-prefill AV witness",
    )
    av_shape = [1, final_length, 8, 256]
    _validate_tensor_envelope(
        witness.get("custom_sdpa_fused"),
        label="custom SDPA AV tensor",
        expected_shape=av_shape,
    )
    _validate_tensor_envelope(
        witness.get("manual_unfused"),
        label="manual SDPA AV tensor",
        expected_shape=av_shape,
    )
    agreement = _mapping(witness.get("agreement"), "target-prefill AV agreement")
    _require_exact_keys(
        agreement,
        {"atol", "max_abs", "passed", "rel_rms", "rtol"},
        "target-prefill AV agreement",
    )
    if agreement.get("atol") != TARGET_PREFILL_ATOL:
        raise ValueError("target-prefill AV agreement atol mismatch")
    if agreement.get("rtol") != TARGET_PREFILL_RTOL:
        raise ValueError("target-prefill AV agreement rtol mismatch")
    if agreement.get("passed") is not True:
        raise ValueError("target-prefill AV agreement must pass")
    for metric in ("max_abs", "rel_rms"):
        observed = agreement.get(metric)
        if not _is_number(observed) or not math.isfinite(float(observed)):
            raise ValueError(f"target-prefill AV agreement {metric} is not finite")
        if float(observed) < 0:
            raise ValueError(f"target-prefill AV agreement {metric} is negative")


def validate_target_prefill_receipt(  # noqa: C901
    receipt: Mapping[str, object],
    *,
    expected_checkpoint_acquisition: Mapping[str, object],
    expected_producer_path: Path,
    expected_producer_sha256: str,
    expected_runtime_source_identity: Mapping[str, object],
    expected_fbsource_commit: str,
) -> None:
    _require_exact_keys(receipt, _TOP_LEVEL_KEYS, "target-prefill receipt")
    if receipt.get("schema_version") != TARGET_PREFILL_SCHEMA_VERSION:
        raise ValueError("target-prefill schema version mismatch")
    if receipt.get("envelope_kind") != TARGET_PREFILL_ENVELOPE_KIND:
        raise ValueError("target-prefill envelope kind mismatch")
    if receipt.get("authority") != TARGET_PREFILL_AUTHORITY:
        raise ValueError("target-prefill authority mismatch")
    if receipt.get("checkpoint_acquisition") != expected_checkpoint_acquisition:
        raise ValueError("target-prefill checkpoint acquisition mismatch")

    producer = _mapping(receipt.get("producer"), "target-prefill producer")
    _require_exact_keys(
        producer,
        {
            "fbsource_commit",
            "runtime_source_receipt",
            "source_path",
            "source_sha256",
        },
        "target-prefill producer",
    )
    if producer.get("source_path") != expected_producer_path.name:
        raise ValueError("target-prefill producer source path mismatch")
    if producer.get("source_sha256") != expected_producer_sha256:
        raise ValueError("target-prefill producer source hash mismatch")
    actual_identity = file_identity(expected_producer_path)
    if actual_identity.get("sha256") != expected_producer_sha256:
        raise ValueError("reviewed target-prefill producer bytes changed")
    if producer.get("runtime_source_receipt") != expected_runtime_source_identity:
        raise ValueError("target-prefill runtime source identity mismatch")
    if producer.get("fbsource_commit") != expected_fbsource_commit:
        raise ValueError("target-prefill fbsource commit mismatch")
    if not _is_hex(expected_fbsource_commit, 40):
        raise ValueError("expected target-prefill fbsource commit is invalid")

    run = _mapping(receipt.get("run"), "target-prefill run")
    _require_exact_keys(
        run,
        {"command", "finished_at_utc", "host", "started_at_utc"},
        "target-prefill run",
    )
    command = _sequence(run.get("command"), "target-prefill run command")
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError("target-prefill run command must be nonempty strings")
    for key in ("host", "started_at_utc", "finished_at_utc"):
        if not isinstance(run.get(key), str) or not run[key]:
            raise ValueError(f"target-prefill run {key} must be a nonempty string")

    contexts = _mapping(receipt.get("contexts"), "target-prefill contexts")
    expected_keys = {str(context) for context in TARGET_PREFILL_CONTEXTS}
    if set(contexts) != expected_keys:
        raise ValueError("target-prefill receipt must contain the exact contexts")
    for expected_context in TARGET_PREFILL_CONTEXTS:
        _validate_context(contexts[str(expected_context)], expected_context)
