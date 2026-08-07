# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Portable/eager replay bound to target-only Gemma 4 prefill evidence.

The replay shares the K=2 model definition with the exporter. The staged
target-only receipt independently covers the eager target prefill path, not the
shared Gemma model implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math

from pathlib import Path
from typing import Any, Mapping, Sequence

from executorch.examples.models.gemma4.target_prefill_contract import (
    canonical_json_bytes,
    prompt_plan_sha256,
    reviewed_producer_source_path,
    TARGET_PREFILL_AUTHORITY,
    TARGET_PREFILL_CONTEXTS,
    TARGET_PREFILL_ENVELOPE_KIND,
    TARGET_PREFILL_SCHEMA_VERSION,
    validate_target_prefill_receipt as validate_target_prefill_v2_receipt,
)
from executorch.examples.models.gemma4.webgpu_artifact_manifest import (
    CHECKPOINT_ACQUISITION,
    MTP_SOURCE_VERIFIED_PROVENANCE,
    validate_combined_runtime_envelope,
)

ORACLE_SCHEMA_VERSION = 2
LEGACY_ORACLE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
SUPPORTED_AUTHORITIES = ("portable_eager",)
TARGET_PREFILL_AUTHORITIES: tuple[str, ...] = (TARGET_PREFILL_AUTHORITY,)
CLOSURE_STATES = ("absent", "full")
TARGET_PREFILL_BINDING_STATES = ("bound", "legacy_unbound")

K2_METHOD_NAME = "k2_round"
K2_DRAFT_COUNT = 2
K2_GREEDY_COUNT = 3
K2_MIN_START_POSITION = 2
K2_VOCAB_SIZE = 262144
K2_MAX_SEQ_LEN = 8960
K2_MAX_INPUT_LEN = 512
K2_PTD_COUNT = 3
ORACLE_TOKEN_BUDGET = 32

K2_ROUND_ABI: dict[str, Any] = {
    "buffer_mutation_count": 31,
    "operator_counts": {
        "aten.argmax.default": 3,
        "aten.scatter.src": 2,
        "aten.topk.default": 2,
        "llama.custom_sdpa.default": 43,
    },
    "seed_mutation_count": 1,
    "user_inputs": ["input_ids", "input_pos", "is_round", "donor_length"],
    "user_outputs": [
        {"dtype": "int64", "name": "candidates", "shape": [1, 2]},
        {"dtype": "int64", "name": "target_greedy", "shape": [1, 3]},
        {"dtype": "int64", "name": "output_matches", "shape": [1]},
        {"dtype": "int64", "name": "output_bonus", "shape": [1, 1]},
        {"dtype": "float32", "name": "state_probe", "shape": [1, 1]},
    ],
}

CHECKPOINT_ROLES = ("assistant", "target")

TARGET_PREFILL_WITNESS_KEYS = (
    "layer0_av_sha256",
    "layer0_qk_sha256",
    "logits_sha256",
    "prefill_token",
)

ORACLE_CONTEXT_KEYS = (
    "accepted_prefix",
    "bonus_accounting",
    "decoded_text",
    "kv_witnesses",
    "reset_replay",
    "rounds",
    "selected_logits",
    "stop_handling",
    "target_prefill",
    "useful_tokens",
)

_ORACLE_TOP_LEVEL_KEYS = {
    "abi",
    "authority",
    "closure_state",
    "contexts",
    "method",
    "mtp_manifest_sha256",
    "production_binding",
    "records",
    "replay_independence",
    "schema_version",
    "stop_tokens",
    "target_prefill_authority",
    "target_prefill_oracle_sha256",
    "token_budget",
}
_PRODUCTION_BINDING_KEYS = {
    "checkpoint_acquisition",
    "combined_runtime_sha256",
    "mtp_manifest_sha256",
    "mtp_provenance",
    "producer",
    "run",
    "target_prefill_receipt_sha256",
}
_RAW_ROUND_KEYS = {
    "bonus",
    "candidates",
    "match_count",
    "state_probe",
    "target_greedy",
}
_DECISION_KEYS = {
    "accepted_drafts",
    "committed",
    "discarded",
    "next_position",
    "next_seed",
    "selected",
    "stop_token",
    "stopped",
    "valid",
}
_ROUND_KEYS = _RAW_ROUND_KEYS | _DECISION_KEYS | {"kv_witness"}


class OracleError(Exception):
    """Fail-closed rejection raised before any oracle bytes are written."""


def _is_exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex_digest(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise OracleError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise OracleError(f"{label} must be a list")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise OracleError(f"{label} has an unexpected key set")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise OracleError(f"unreadable {label}: {path}") from error
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise OracleError(f"malformed {label}: {path}") from error
    if not isinstance(document, dict):
        raise OracleError(f"{label} must be a JSON object: {path}")
    return document


def require_schema_version(
    document: Mapping[str, Any], label: str, expected: int
) -> None:
    version = document.get("schema_version")
    if not _is_exact_int(version) or version != expected:
        raise OracleError(f"{label} schema_version must be {expected}, got {version!r}")


def parse_contexts(value: str, max_context_length: int) -> list[int]:
    fields = [field.strip() for field in value.split(",")]
    if not fields or any(not field for field in fields):
        raise OracleError(f"--contexts must be a non-empty integer list: {value!r}")
    contexts: list[int] = []
    for field in fields:
        if not field.isdigit():
            raise OracleError(f"--contexts entry is not a decimal integer: {field!r}")
        context = int(field)
        if context <= 0 or context > max_context_length:
            raise OracleError(
                f"--contexts entry {context} is outside 1..{max_context_length}"
            )
        if context in contexts:
            raise OracleError(f"--contexts entry is duplicated: {context}")
        contexts.append(context)
    return contexts


def _validate_checkpoints(manifest: Mapping[str, Any], root: Path) -> dict[str, Path]:
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise OracleError("MTP manifest is missing the checkpoints binding")
    if tuple(sorted(checkpoints)) != CHECKPOINT_ROLES:
        raise OracleError(
            f"MTP manifest checkpoints must name exactly {list(CHECKPOINT_ROLES)}"
        )
    resolved: dict[str, Path] = {}
    for role in CHECKPOINT_ROLES:
        value = checkpoints[role]
        if not isinstance(value, str) or not value:
            raise OracleError(f"MTP manifest {role} checkpoint must be a path")
        path = Path(value)
        candidate = path if path.is_absolute() else root / path
        if not candidate.is_dir():
            raise OracleError(f"MTP manifest {role} checkpoint is not a directory")
        resolved[role] = candidate
    return resolved


def _validate_artifact_roles(manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    ptd_order = manifest.get("ptd_order")
    if not isinstance(artifacts, list) or not isinstance(ptd_order, list):
        raise OracleError("MTP manifest artifacts/PTD order must be lists")
    roles = [entry.get("role") for entry in artifacts if isinstance(entry, dict)]
    if len(roles) != len(artifacts):
        raise OracleError("MTP manifest artifact entries must be objects")
    if roles.count("pte") != 1:
        raise OracleError("MTP manifest requires exactly one PTE artifact")
    if roles.count("ptd") != K2_PTD_COUNT or len(ptd_order) != K2_PTD_COUNT:
        raise OracleError(f"MTP manifest requires exactly {K2_PTD_COUNT} ordered PTDs")


def load_stop_tokens(target_checkpoint: Path) -> list[int]:
    document = load_json_object(
        target_checkpoint / "generation_config.json", "generation config"
    )
    value = document.get("eos_token_id")
    tokens = value if isinstance(value, list) else [value]
    if not tokens or any(
        not _is_exact_int(token) or token < 0 or token >= K2_VOCAB_SIZE
        for token in tokens
    ):
        raise OracleError(f"generation config eos_token_id is invalid: {value!r}")
    if len(set(tokens)) != len(tokens):
        raise OracleError("generation config eos_token_id contains duplicates")
    return list(tokens)


def validate_mtp_manifest(
    manifest: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any]:
    require_schema_version(manifest, "MTP manifest", MANIFEST_SCHEMA_VERSION)
    if manifest.get("method") != K2_METHOD_NAME:
        raise OracleError(
            f"MTP manifest method must be {K2_METHOD_NAME!r}, "
            f"got {manifest.get('method')!r}"
        )
    if manifest.get("abi") != K2_ROUND_ABI:
        raise OracleError("MTP manifest does not match the D8 K=2 graph/ABI contract")
    _validate_artifact_roles(manifest)
    max_context_length = manifest.get("max_context_length")
    if (
        not _is_exact_int(max_context_length)
        or max_context_length <= 0
        or max_context_length > K2_MAX_SEQ_LEN
    ):
        raise OracleError(
            f"MTP manifest max_context_length must be 1..{K2_MAX_SEQ_LEN}, "
            f"got {max_context_length!r}"
        )
    checkpoints = _validate_checkpoints(manifest, manifest_path.parent)
    return {
        "checkpoints": checkpoints,
        "max_context_length": max_context_length,
        "stop_tokens": load_stop_tokens(checkpoints["target"]),
    }


def validate_target_prefill_receipt(
    receipt: Mapping[str, Any], authority: str, contexts: Sequence[int]
) -> dict[str, Mapping[str, Any]]:
    require_schema_version(receipt, "target-prefill receipt", RECEIPT_SCHEMA_VERSION)
    if receipt.get("authority") != authority:
        raise OracleError(
            f"target-prefill receipt authority is {receipt.get('authority')!r}, "
            f"not {authority!r}"
        )
    entries = receipt.get("contexts")
    if not isinstance(entries, dict):
        raise OracleError("target-prefill receipt contexts must be an object")
    witnesses: dict[str, Mapping[str, Any]] = {}
    for context in contexts:
        key = str(context)
        entry = entries.get(key)
        if not isinstance(entry, dict):
            raise OracleError(f"target-prefill receipt lacks context {context}")
        if tuple(sorted(entry)) != TARGET_PREFILL_WITNESS_KEYS:
            raise OracleError(
                f"target-prefill witness {context} must name exactly "
                f"{list(TARGET_PREFILL_WITNESS_KEYS)}"
            )
        token = entry["prefill_token"]
        if not _is_exact_int(token) or token < 0 or token >= K2_VOCAB_SIZE:
            raise OracleError(f"target-prefill witness {context} token is invalid")
        for name in TARGET_PREFILL_WITNESS_KEYS:
            if name != "prefill_token" and not _is_hex_digest(entry[name], 64):
                raise OracleError(
                    f"target-prefill witness {context} {name} is not a digest"
                )
        witnesses[key] = entry
    return witnesses


def _combined_receipt_path(root: Path, envelope: Mapping[str, Any], role: str) -> Path:
    receipts = envelope.get("receipts")
    if not isinstance(receipts, dict):
        raise OracleError("combined runtime receipts must be an object")
    identity = receipts.get(role)
    if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
        raise OracleError(f"combined runtime lacks the {role} receipt identity")
    resolved_root = root.resolve(strict=True)
    path = (root / identity["path"]).resolve(strict=True)
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise OracleError(
            f"combined runtime {role} receipt escapes its root"
        ) from error
    if not path.is_file():
        raise OracleError(f"combined runtime {role} receipt is not a regular file")
    return path


def _validate_production_target_prefill_receipt(
    receipt: Mapping[str, Any], contexts: Sequence[int]
) -> dict[str, Mapping[str, Any]]:
    if receipt.get("schema_version") != TARGET_PREFILL_SCHEMA_VERSION:
        raise OracleError("production target-prefill receipt requires schema version 2")
    if receipt.get("envelope_kind") != TARGET_PREFILL_ENVELOPE_KIND:
        raise OracleError("production target-prefill envelope kind mismatch")
    if receipt.get("authority") not in TARGET_PREFILL_AUTHORITIES:
        raise OracleError("production target-prefill authority mismatch")
    if tuple(contexts) != TARGET_PREFILL_CONTEXTS:
        raise OracleError("production binding requires the exact ten contexts")
    entries = receipt.get("contexts")
    if not isinstance(entries, dict) or set(entries) != {
        str(context) for context in TARGET_PREFILL_CONTEXTS
    }:
        raise OracleError("production receipt requires the exact ten contexts")
    witnesses: dict[str, Mapping[str, Any]] = {}
    for context in TARGET_PREFILL_CONTEXTS:
        entry = entries.get(str(context))
        if not isinstance(entry, dict):
            raise OracleError(
                f"production target-prefill receipt lacks context {context}"
            )
        raw = entry.get("prefill_token_raw")
        post = entry.get("prefill_token_post_softcap")
        if not _is_exact_int(raw) or raw < 0 or raw >= K2_VOCAB_SIZE or post != raw:
            raise OracleError(
                f"target-prefill context {context} raw/post-softcap token mismatch"
            )
        if entry.get("prompt_plan_sha256") != prompt_plan_sha256(context):
            raise OracleError(f"target-prefill context {context} prompt plan mismatch")
        witnesses[str(context)] = entry
    return witnesses


def validate_k2_abi_edge_census(edge_census: Mapping[str, int]) -> None:
    expected = {
        "custom_scatter": K2_ROUND_ABI["operator_counts"]["aten.scatter.src"],
        "gemma_sdpa": K2_ROUND_ABI["operator_counts"]["llama.custom_sdpa.default"],
        "topk": K2_ROUND_ABI["operator_counts"]["aten.topk.default"],
    }
    if any(edge_census.get(name) != count for name, count in expected.items()):
        raise OracleError("production edge census drifted from the K=2 ABI")


def require_prefill_token_match(
    context: int, actual_token: int, witness: Mapping[str, Any]
) -> None:
    expected = witness.get("prefill_token_raw", witness.get("prefill_token"))
    if actual_token != expected:
        raise OracleError(
            f"context {context} prefill token disagrees with the target-only receipt"
        )


def reconcile_k2_round(
    candidates: Sequence[int],
    target_greedy: Sequence[int],
    match_count: int,
    bonus: int,
    state_probe: float,
    start_position: int,
    token_budget: int,
    stop_tokens: Sequence[int],
    vocab_size: int = K2_VOCAB_SIZE,
) -> dict[str, Any]:
    """Portable mirror of `reconcile_gemma4_k2` in `gemma4_spec_runner.h`."""
    decision: dict[str, Any] = {
        "accepted_drafts": 0,
        "committed": [],
        "discarded": [],
        "next_position": -1,
        "next_seed": -1,
        "selected": [],
        "stop_token": -1,
        "stopped": False,
        "valid": False,
    }
    if (
        len(candidates) != K2_DRAFT_COUNT
        or len(target_greedy) != K2_GREEDY_COUNT
        or start_position < K2_MIN_START_POSITION
        or token_budget <= 0
        or vocab_size <= 0
        or match_count < 0
        or match_count > K2_DRAFT_COUNT
        or not math.isfinite(state_probe)
    ):
        return decision
    if any(
        token < 0 or token >= vocab_size
        for token in list(candidates) + list(target_greedy)
    ):
        return decision

    expected_matches = 0
    if candidates[0] == target_greedy[0]:
        expected_matches = 2 if candidates[1] == target_greedy[1] else 1
    if (
        match_count != expected_matches
        or bonus < 0
        or bonus >= vocab_size
        or bonus != target_greedy[match_count]
    ):
        return decision

    selected = list(candidates[:match_count]) + [bonus]
    committed: list[int] = []
    discarded: list[int] = []
    for index, token in enumerate(selected):
        if token in stop_tokens:
            decision["stopped"] = True
            decision["stop_token"] = token
            discarded.extend(selected[index + 1 :])
            break
        if len(committed) == token_budget:
            discarded.extend(selected[index:])
            break
        committed.append(token)

    decision.update(
        {
            "accepted_drafts": match_count,
            "committed": committed,
            "discarded": discarded,
            "next_position": start_position + match_count + 1,
            "next_seed": bonus,
            "selected": selected,
            "valid": True,
        }
    )
    return decision


def _bind_combined_runtime(
    binding: dict[str, Any],
    combined_runtime_root: Path,
    *,
    expected_mtp_sha256: str | None = None,
    expected_checkpoint_acquisition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    envelope_path = combined_runtime_root / "gemma4_webgpu_combined_runtime.json"
    envelope = load_json_object(envelope_path, "combined runtime envelope")
    try:
        validate_combined_runtime_envelope(combined_runtime_root, envelope)
    except ValueError as error:
        raise OracleError(str(error)) from error
    if envelope.get("schema_version") != 3:
        raise OracleError("production binding requires combined runtime schema 3")

    mtp_receipt_path = _combined_receipt_path(combined_runtime_root, envelope, "mtp")
    target_prefill_path = _combined_receipt_path(
        combined_runtime_root, envelope, "target_prefill"
    )
    mtp_sha256 = _sha256(mtp_receipt_path)
    if expected_mtp_sha256 is not None and mtp_sha256 != expected_mtp_sha256:
        raise OracleError("production MTP manifest is not the staged MTP receipt")
    production_mtp = load_json_object(mtp_receipt_path, "production MTP manifest")
    if production_mtp.get("provenance") != MTP_SOURCE_VERIFIED_PROVENANCE:
        raise OracleError("production MTP manifest is not source verified")
    target_receipt = load_json_object(
        target_prefill_path, "production target-prefill receipt"
    )
    if (
        expected_checkpoint_acquisition is not None
        and target_receipt.get("checkpoint_acquisition")
        != expected_checkpoint_acquisition
    ):
        raise OracleError("target-prefill checkpoint acquisition mismatch")
    binding.update(
        {
            "closure_state": "full",
            "mtp_manifest_sha256": mtp_sha256,
            "production_binding": {
                "checkpoint_acquisition": target_receipt.get(
                    "checkpoint_acquisition"
                ),
                "combined_runtime_sha256": _sha256(envelope_path),
                "mtp_manifest_sha256": mtp_sha256,
                "mtp_provenance": production_mtp["provenance"],
                "producer": target_receipt.get("producer"),
                "run": target_receipt.get("run"),
                "target_prefill_receipt_sha256": _sha256(target_prefill_path),
            },
            "target_prefill": _validate_production_target_prefill_receipt(
                target_receipt, binding["contexts"]
            ),
            "target_prefill_authority": "bound",
            "target_prefill_oracle_sha256": _sha256(target_prefill_path),
        }
    )
    return binding


def build_production_oracle_binding(
    production_mtp_manifest: Path,
    target_checkpoint: Path,
    assistant_checkpoint: Path,
    authority: str,
    contexts: str,
    output: Path,
    *,
    combined_runtime_root: Path,
) -> dict[str, Any]:
    if authority not in SUPPORTED_AUTHORITIES:
        raise OracleError(
            f"unknown --authority {authority!r}; expected one of "
            f"{list(SUPPORTED_AUTHORITIES)}"
        )
    if output.exists() or output.is_symlink():
        raise OracleError(f"refusing to overwrite existing artifact: {output}")
    manifest = load_json_object(production_mtp_manifest, "production MTP manifest")
    if manifest.get("provenance") != MTP_SOURCE_VERIFIED_PROVENANCE:
        raise OracleError("production MTP manifest is not source verified")
    export = _mapping(manifest.get("export"), "production MTP export")
    if export.get("methods") != [K2_METHOD_NAME]:
        raise OracleError("production MTP manifest does not expose k2_round")
    max_context_length = export.get("max_seq_len")
    if (
        not _is_exact_int(max_context_length)
        or max_context_length <= 0
        or max_context_length > K2_MAX_SEQ_LEN
    ):
        raise OracleError("production MTP max_seq_len is invalid")
    _validate_artifact_roles(manifest)
    acquisition = _mapping(
        manifest.get("acquisition"), "production MTP acquisition"
    )
    if acquisition.get("target") != CHECKPOINT_ACQUISITION:
        raise OracleError("production MTP target acquisition mismatch")
    checkpoints = _validate_checkpoints(
        {
            "checkpoints": {
                "assistant": str(assistant_checkpoint),
                "target": str(target_checkpoint),
            }
        },
        production_mtp_manifest.parent,
    )
    resolved = parse_contexts(contexts, max_context_length)
    mtp_sha256 = _sha256(production_mtp_manifest)
    binding: dict[str, Any] = {
        "authority": authority,
        "checkpoints": checkpoints,
        "contexts": resolved,
        "closure_state": "absent",
        "max_context_length": max_context_length,
        "mtp_manifest_sha256": mtp_sha256,
        "production_binding": None,
        "stop_tokens": load_stop_tokens(checkpoints["target"]),
        "target_prefill_authority": "legacy_unbound",
        "token_budget": ORACLE_TOKEN_BUDGET,
    }
    return _bind_combined_runtime(
        binding,
        combined_runtime_root,
        expected_mtp_sha256=mtp_sha256,
        expected_checkpoint_acquisition=CHECKPOINT_ACQUISITION,
    )


def build_oracle_binding(
    mtp_manifest: Path,
    target_prefill_oracle: Path | None,
    authority: str,
    contexts: str,
    output: Path,
    *,
    combined_runtime_root: Path | None = None,
) -> dict[str, Any]:
    if authority not in SUPPORTED_AUTHORITIES:
        raise OracleError(
            f"unknown --authority {authority!r}; expected one of "
            f"{list(SUPPORTED_AUTHORITIES)}"
        )
    if output.exists() or output.is_symlink():
        raise OracleError(f"refusing to overwrite existing artifact: {output}")
    manifest = load_json_object(mtp_manifest, "MTP manifest")
    bound = validate_mtp_manifest(manifest, mtp_manifest)
    resolved = parse_contexts(contexts, bound["max_context_length"])
    binding: dict[str, Any] = {
        "authority": authority,
        "checkpoints": bound["checkpoints"],
        "contexts": resolved,
        "closure_state": "absent",
        "max_context_length": bound["max_context_length"],
        "mtp_manifest_sha256": _sha256(mtp_manifest),
        "production_binding": None,
        "stop_tokens": bound["stop_tokens"],
        "target_prefill_authority": "legacy_unbound",
        "token_budget": ORACLE_TOKEN_BUDGET,
    }
    if combined_runtime_root is None:
        if target_prefill_oracle is None:
            raise OracleError("legacy binding requires a target-prefill receipt")
        receipt = load_json_object(target_prefill_oracle, "target-prefill receipt")
        binding["target_prefill"] = validate_target_prefill_receipt(
            receipt, authority, resolved
        )
        binding["target_prefill_oracle_sha256"] = _sha256(target_prefill_oracle)
        return binding

    if target_prefill_oracle is not None:
        raise OracleError(
            "production binding derives the target-prefill receipt from the "
            "combined runtime root"
        )
    return _bind_combined_runtime(binding, combined_runtime_root)


def assemble_oracle_document(
    binding: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [str(context) for context in binding["contexts"]]
    if sorted(records) != sorted(expected):
        raise OracleError("authority records do not cover the requested contexts")
    for key in expected:
        record = records[key]
        if tuple(sorted(record)) != ORACLE_CONTEXT_KEYS:
            raise OracleError(f"authority record {key} has an unexpected key set")
        if record["target_prefill"] != binding["target_prefill"][key]:
            raise OracleError(f"authority record {key} reinterprets the D6 receipt")
    production = binding.get("closure_state") == "full"
    if production and tuple(binding["contexts"]) != TARGET_PREFILL_CONTEXTS:
        raise OracleError("production oracle requires the exact ten contexts")
    document: dict[str, Any] = {
        "abi": K2_ROUND_ABI,
        "authority": binding["authority"],
        "closure_state": binding.get("closure_state", "absent"),
        "contexts": list(binding["contexts"]),
        "method": K2_METHOD_NAME,
        "mtp_manifest_sha256": binding["mtp_manifest_sha256"],
        "records": {key: dict(records[key]) for key in expected},
        "replay_independence": "eager_vs_lowered_only",
        "schema_version": (
            ORACLE_SCHEMA_VERSION if production else LEGACY_ORACLE_SCHEMA_VERSION
        ),
        "stop_tokens": list(binding["stop_tokens"]),
        "target_prefill_authority": binding.get(
            "target_prefill_authority", "legacy_unbound"
        ),
        "target_prefill_oracle_sha256": binding["target_prefill_oracle_sha256"],
        "token_budget": binding["token_budget"],
    }
    if production:
        document["production_binding"] = binding["production_binding"]
    return document


def _validate_production_binding(
    document: Mapping[str, Any], records: Mapping[str, Any]
) -> None:
    binding = _mapping(document.get("production_binding"), "production binding")
    _require_exact_keys(binding, _PRODUCTION_BINDING_KEYS, "production binding")
    for key in (
        "combined_runtime_sha256",
        "mtp_manifest_sha256",
        "target_prefill_receipt_sha256",
    ):
        if not _is_hex_digest(binding.get(key), 64):
            raise OracleError(f"production binding {key} is invalid")
    if binding.get("mtp_manifest_sha256") != document.get("mtp_manifest_sha256"):
        raise OracleError("production MTP manifest identity is inconsistent")
    if binding.get("target_prefill_receipt_sha256") != document.get(
        "target_prefill_oracle_sha256"
    ):
        raise OracleError("target-prefill receipt identity is inconsistent")
    if binding.get("mtp_provenance") != MTP_SOURCE_VERIFIED_PROVENANCE:
        raise OracleError("production binding MTP provenance is not source verified")
    if binding.get("checkpoint_acquisition") != CHECKPOINT_ACQUISITION:
        raise OracleError("production binding checkpoint acquisition mismatch")

    producer = _mapping(binding.get("producer"), "target-prefill producer")
    runtime_source_identity = _mapping(
        producer.get("runtime_source_receipt"),
        "target-prefill runtime source identity",
    )
    producer_path = reviewed_producer_source_path()
    receipt = {
        "authority": TARGET_PREFILL_AUTHORITY,
        "checkpoint_acquisition": binding["checkpoint_acquisition"],
        "contexts": {
            key: _mapping(record, f"oracle record {key}").get("target_prefill")
            for key, record in records.items()
        },
        "envelope_kind": TARGET_PREFILL_ENVELOPE_KIND,
        "producer": producer,
        "run": binding.get("run"),
        "schema_version": TARGET_PREFILL_SCHEMA_VERSION,
    }
    receipt_sha256 = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    if receipt_sha256 != binding.get("target_prefill_receipt_sha256"):
        raise OracleError("target-prefill receipt content does not match its identity")
    try:
        validate_target_prefill_v2_receipt(
            receipt,
            expected_checkpoint_acquisition=CHECKPOINT_ACQUISITION,
            expected_producer_path=producer_path,
            expected_producer_sha256=str(producer.get("source_sha256")),
            expected_runtime_source_identity=runtime_source_identity,
            expected_fbsource_commit=str(producer.get("fbsource_commit")),
        )
    except ValueError as error:
        raise OracleError(str(error)) from error


def _int_sequence(value: object, label: str) -> list[int]:
    sequence = _sequence(value, label)
    if any(not _is_exact_int(item) for item in sequence):
        raise OracleError(f"{label} must contain only integers")
    return list(sequence)


def _validate_oracle_record(
    record: object,
    *,
    context: int,
    stop_tokens: Sequence[int],
    token_budget: int,
) -> None:
    value = _mapping(record, f"oracle record {context}")
    _require_exact_keys(value, set(ORACLE_CONTEXT_KEYS), f"oracle record {context}")
    target_prefill = _mapping(
        value.get("target_prefill"), f"oracle target-prefill witness {context}"
    )
    prefill_token = target_prefill.get("prefill_token_raw")
    if not _is_exact_int(prefill_token):
        raise OracleError(f"oracle target-prefill token {context} is invalid")
    useful_tokens = _int_sequence(value.get("useful_tokens"), "oracle useful tokens")
    if not useful_tokens or useful_tokens[0] != prefill_token:
        raise OracleError("oracle useful tokens do not start with the prefill token")
    if len(useful_tokens) > token_budget:
        raise OracleError("oracle useful tokens exceed the token budget")

    rounds = _sequence(value.get("rounds"), "oracle rounds")
    if not rounds:
        raise OracleError("production oracle requires at least one K=2 round")
    accepted_prefix: list[int] = []
    bonus_accounting: list[int] = []
    kv_witnesses: list[str] = []
    selected: list[list[int]] = []
    discarded: list[list[int]] = []
    reconstructed = [prefill_token]
    position = context
    reset_replay: dict[str, Any] | None = None
    stop_token: int | None = None
    for index, round_value in enumerate(rounds):
        round_record = _mapping(round_value, f"oracle round {context}:{index}")
        _require_exact_keys(
            round_record, _ROUND_KEYS, f"oracle round {context}:{index}"
        )
        candidates = _int_sequence(
            round_record.get("candidates"), "oracle round candidates"
        )
        target_greedy = _int_sequence(
            round_record.get("target_greedy"), "oracle round target_greedy"
        )
        bonus = round_record.get("bonus")
        match_count = round_record.get("match_count")
        state_probe = round_record.get("state_probe")
        if (
            not _is_exact_int(bonus)
            or not _is_exact_int(match_count)
            or not isinstance(state_probe, (int, float))
            or isinstance(state_probe, bool)
            or not math.isfinite(float(state_probe))
        ):
            raise OracleError("oracle round scalar fields are invalid")
        remaining = token_budget - len(reconstructed)
        if remaining <= 0:
            raise OracleError("oracle carries rounds after exhausting its token budget")
        decision = reconcile_k2_round(
            candidates,
            target_greedy,
            match_count,
            bonus,
            float(state_probe),
            position,
            remaining,
            stop_tokens,
        )
        if not decision["valid"] or any(
            round_record.get(key) != decision[key] for key in _DECISION_KEYS
        ):
            raise OracleError("oracle round disagrees with K=2 reconciliation")
        kv_witness = round_record.get("kv_witness")
        if not _is_hex_digest(kv_witness, 64):
            raise OracleError("oracle round KV witness is invalid")
        raw = {key: round_record[key] for key in _RAW_ROUND_KEYS}
        if reset_replay is None:
            reset_replay = raw
        accepted_prefix.append(decision["accepted_drafts"])
        bonus_accounting.append(decision["next_seed"])
        kv_witnesses.append(kv_witness)
        selected.append(decision["selected"])
        discarded.append(decision["discarded"])
        reconstructed.extend(decision["committed"])
        position = decision["next_position"]
        if decision["stopped"]:
            stop_token = decision["stop_token"]
            if index != len(rounds) - 1:
                raise OracleError("oracle carries rounds after a stop token")

    if value.get("accepted_prefix") != accepted_prefix:
        raise OracleError("oracle accepted-prefix summary is inconsistent")
    if value.get("bonus_accounting") != bonus_accounting:
        raise OracleError("oracle bonus summary is inconsistent")
    if value.get("kv_witnesses") != kv_witnesses:
        raise OracleError("oracle KV-witness summary is inconsistent")
    if value.get("selected_logits") != selected:
        raise OracleError("oracle selected-token summary is inconsistent")
    if value.get("reset_replay") != reset_replay:
        raise OracleError("oracle reset replay is inconsistent")
    if useful_tokens != reconstructed:
        raise OracleError("oracle useful-token summary is inconsistent")
    stop_handling = _mapping(value.get("stop_handling"), "oracle stop handling")
    _require_exact_keys(stop_handling, {"discarded", "stop_token"}, "stop handling")
    if stop_handling != {"discarded": discarded, "stop_token": stop_token}:
        raise OracleError("oracle stop handling is inconsistent")
    if not isinstance(value.get("decoded_text"), str):
        raise OracleError("oracle decoded text must be a string")


def _validate_production_oracle_document(document: Mapping[str, Any]) -> None:
    _require_exact_keys(document, _ORACLE_TOP_LEVEL_KEYS, "production oracle")
    if document.get("schema_version") != ORACLE_SCHEMA_VERSION:
        raise OracleError("production oracle schema version mismatch")
    if document.get("method") != K2_METHOD_NAME or document.get("abi") != K2_ROUND_ABI:
        raise OracleError("production oracle K=2 method/ABI mismatch")
    if document.get("authority") not in SUPPORTED_AUTHORITIES:
        raise OracleError("production oracle replay authority mismatch")
    if document.get("replay_independence") != "eager_vs_lowered_only":
        raise OracleError("production oracle independence claim mismatch")
    if document.get("token_budget") != ORACLE_TOKEN_BUDGET:
        raise OracleError("production oracle token budget mismatch")
    for key in ("mtp_manifest_sha256", "target_prefill_oracle_sha256"):
        if not _is_hex_digest(document.get(key), 64):
            raise OracleError(f"production oracle {key} is invalid")
    contexts = _int_sequence(document.get("contexts"), "oracle contexts")
    if tuple(contexts) != TARGET_PREFILL_CONTEXTS:
        raise OracleError("production oracle requires the exact ten contexts")
    stop_tokens = _int_sequence(document.get("stop_tokens"), "oracle stop tokens")
    if (
        not stop_tokens
        or len(stop_tokens) != len(set(stop_tokens))
        or any(token < 0 or token >= K2_VOCAB_SIZE for token in stop_tokens)
    ):
        raise OracleError("production oracle stop tokens are invalid")
    records = _mapping(document.get("records"), "oracle records")
    if set(records) != {str(context) for context in TARGET_PREFILL_CONTEXTS}:
        raise OracleError("production oracle records do not cover all contexts")
    _validate_production_binding(document, records)
    for context in TARGET_PREFILL_CONTEXTS:
        _validate_oracle_record(
            records[str(context)],
            context=context,
            stop_tokens=stop_tokens,
            token_budget=ORACLE_TOKEN_BUDGET,
        )


def production_oracle_is_acceptable(document: Mapping[str, Any]) -> bool:
    closure_state = document.get("closure_state")
    if closure_state not in CLOSURE_STATES:
        raise OracleError(f"unknown oracle closure_state: {closure_state!r}")
    target_authority = document.get("target_prefill_authority")
    if target_authority not in TARGET_PREFILL_BINDING_STATES:
        raise OracleError(f"unknown target_prefill_authority: {target_authority!r}")
    if (
        document.get("schema_version") != ORACLE_SCHEMA_VERSION
        or closure_state != "full"
        or target_authority != "bound"
        or tuple(document.get("contexts", ())) != TARGET_PREFILL_CONTEXTS
        or not isinstance(document.get("production_binding"), dict)
    ):
        return False
    try:
        _validate_production_oracle_document(document)
    except (KeyError, OracleError, TypeError, ValueError):
        return False
    return True


def _round_inputs(
    torch: Any, seed_token: int, start_position: int
) -> tuple[Any, Any, Any, Any]:
    return (
        torch.tensor([[seed_token, 0, 0]], dtype=torch.long),
        torch.arange(start_position, start_position + 3, dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([[start_position]], dtype=torch.long),
    )


def _kv_witness(torch: Any, module: Any) -> str:
    digest = hashlib.sha256()
    for name, buffer in sorted(module.state_dict().items()):
        if name.endswith("k_cache") or name.endswith("v_cache"):
            digest.update(name.encode("utf-8"))
            digest.update(buffer.detach().to(torch.float32).cpu().numpy().tobytes())
    return digest.hexdigest()


def _read_round(module: Any, inputs: tuple[Any, ...]) -> dict[str, Any]:
    candidates, target_greedy, matches, bonus, state_probe = module(*inputs)
    return {
        "bonus": int(bonus.reshape(-1)[0].item()),
        "candidates": [int(value) for value in candidates.reshape(-1).tolist()],
        "match_count": int(matches.reshape(-1)[0].item()),
        "state_probe": float(state_probe.reshape(-1)[0].item()),
        "target_greedy": [int(value) for value in target_greedy.reshape(-1).tolist()],
    }


def _prefill(module: Any, torch: Any, prompt: Sequence[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    start = 0
    while start < len(prompt):
        count = min(K2_MAX_INPUT_LEN, len(prompt) - start)
        output = _read_round(
            module,
            (
                torch.tensor([list(prompt[start : start + count])], dtype=torch.long),
                torch.arange(start, start + count, dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([[2 if start == 0 else start]], dtype=torch.long),
            ),
        )
        start += count
    return output


def _context_record(
    module: Any,
    torch: Any,
    tokenizer: Any,
    binding: Mapping[str, Any],
    context: int,
    prompt: Sequence[int],
    witness: Mapping[str, Any],
) -> dict[str, Any]:
    prefill = _prefill(module, torch, prompt)
    require_prefill_token_match(context, prefill["bonus"], witness)
    tokens: list[int] = [prefill["bonus"]]
    rounds: list[dict[str, Any]] = []
    seed = prefill["bonus"]
    position = len(prompt)
    stopped: int | None = None
    replay: dict[str, Any] | None = None
    while stopped is None and len(tokens) < binding["token_budget"]:
        raw = _read_round(module, _round_inputs(torch, seed, position))
        if replay is None:
            replay = raw
        decision = reconcile_k2_round(
            raw["candidates"],
            raw["target_greedy"],
            raw["match_count"],
            raw["bonus"],
            raw["state_probe"],
            position,
            binding["token_budget"] - len(tokens),
            binding["stop_tokens"],
        )
        if not decision["valid"]:
            raise OracleError(f"context {context} produced an invalid K=2 round")
        tokens.extend(decision["committed"])
        rounds.append({"kv_witness": _kv_witness(torch, module), **raw, **decision})
        stopped = decision["stop_token"] if decision["stopped"] else None
        seed = decision["next_seed"]
        position = decision["next_position"]
    return {
        "accepted_prefix": [entry["accepted_drafts"] for entry in rounds],
        "bonus_accounting": [entry["next_seed"] for entry in rounds],
        "decoded_text": tokenizer.decode(tokens),
        "kv_witnesses": [entry["kv_witness"] for entry in rounds],
        "reset_replay": replay,
        "rounds": rounds,
        "selected_logits": [entry["selected"] for entry in rounds],
        "stop_handling": {
            "discarded": [entry["discarded"] for entry in rounds],
            "stop_token": stopped,
        },
        "target_prefill": witness,
        "useful_tokens": tokens,
    }


def run_portable_eager_authority(
    binding: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Replay the D8 K=2 graph in eager mode; requires torch and the D8 module."""
    import torch

    from executorch.examples.models.gemma4.export_speculative import (
        build_k2_round_program,
    )
    from transformers import AutoTokenizer

    checkpoints = binding["checkpoints"]
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoints["target"]))
    program = build_k2_round_program(
        checkpoints["target"],
        checkpoints["assistant"],
        max_seq_len=binding["max_context_length"],
        max_input_len=K2_MAX_INPUT_LEN,
    )
    records: dict[str, dict[str, Any]] = {}
    for context in binding["contexts"]:
        prompt = [(index % (K2_VOCAB_SIZE - 1)) + 1 for index in range(context)]
        module = program.module()
        record = _context_record(
            module,
            torch,
            tokenizer,
            binding,
            context,
            prompt,
            binding["target_prefill"][str(context)],
        )
        replayed = _read_round(
            program.module(), _round_inputs(torch, record["useful_tokens"][0], context)
        )
        if replayed != record["reset_replay"]:
            raise OracleError(f"context {context} reset replay is not deterministic")
        records[str(context)] = record
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Gemma 4 MTP/spec oracle bound to target-only eager "
            "prefill evidence; it does not independently validate the shared model"
        )
    )
    parser.add_argument("--oracle-binding-manifest", type=Path)
    parser.add_argument("--production-mtp-manifest", type=Path)
    parser.add_argument("--target-checkpoint", type=Path)
    parser.add_argument("--assistant-checkpoint", type=Path)
    parser.add_argument("--combined-runtime-root", type=Path, required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    direct_inputs = (
        args.production_mtp_manifest,
        args.target_checkpoint,
        args.assistant_checkpoint,
    )
    if args.oracle_binding_manifest is not None:
        if any(value is not None for value in direct_inputs):
            parser.error(
                "--oracle-binding-manifest cannot be combined with direct production inputs"
            )
        binding = build_oracle_binding(
            args.oracle_binding_manifest,
            None,
            args.authority,
            args.contexts,
            args.output,
            combined_runtime_root=args.combined_runtime_root,
        )
    else:
        if any(value is None for value in direct_inputs):
            parser.error(
                "direct production mode requires --production-mtp-manifest, "
                "--target-checkpoint, and --assistant-checkpoint"
            )
        binding = build_production_oracle_binding(
            args.production_mtp_manifest,
            args.target_checkpoint,
            args.assistant_checkpoint,
            args.authority,
            args.contexts,
            args.output,
            combined_runtime_root=args.combined_runtime_root,
        )
    document = assemble_oracle_document(binding, run_portable_eager_authority(binding))
    if binding.get("closure_state") == "full" and not production_oracle_is_acceptable(
        document
    ):
        raise OracleError("generated oracle failed production validation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
