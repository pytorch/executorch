# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Gemma 4 E2B acquisition and WebGPU artifact manifest contract."""

import argparse
import copy
import dataclasses
import errno
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import threading

from pathlib import Path
from types import FrameType
from typing import Any, Mapping, Sequence

from executorch.backends.webgpu.scripts.webgpu_artifact_manifest import (
    create_manifest,
    validate_manifest,
)
from executorch.examples.models.gemma4.target_prefill_contract import (
    file_identity as target_prefill_file_identity,
    reviewed_producer_source_path,
    validate_target_prefill_receipt,
)


SOURCE_CONFIG_SHA256 = (
    "526e9fd34a8a489c35952535335a4b8556e9169d851187a895f80286e7466206"
)
SOURCE_CONFIG_BYTES = 2214
WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES = 1_500_000_000
EXPECTED_LAYER_TYPES: list[str] = [
    attention_type
    for _ in range(7)
    for attention_type in [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
]
ARCHITECTURE_FINGERPRINT: dict[str, object] = {
    "model_type": "gemma4",
    "num_hidden_layers": 35,
    "hidden_size": 1536,
    "intermediate_size": 6144,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_kv_shared_layers": 20,
    "vocab_size": 262144,
    "layer_types": EXPECTED_LAYER_TYPES,
}
CHECKPOINT_ACQUISITION: dict[str, object] = {
    "repo_id": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
    "revision": "6befbaca7398925921802abd1f277b495b78b738",
    "files": {
        "model.safetensors": {
            "bytes": 10208852878,
            "sha256": "33fe0cece08fb527ffefbd1a3a9ce73bd71073727993a283506293e5c6bf0137",
        },
        "config.json": {
            "bytes": 4946,
            "sha256": "bbeff1e2fd3fe282536e7ace02309d43e0dbd9b6ac4b6a149b97e3ab6942a878",
        },
        "tokenizer.json": {
            "bytes": 32169626,
            "sha256": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        },
        "tokenizer_config.json": {
            "bytes": 3729,
            "sha256": "3ab5c7b94dc97d65ca7064496fa69b88ff875378e1cb7ee3e43070c3a8170999",
        },
        "generation_config.json": {
            "bytes": 203,
            "sha256": "b69207f9be617e982d13cc273cce6fd88c98dda99a4bdc5e2d52ffe0a0d9f0a9",
        },
        "processor_config.json": {
            "bytes": 1689,
            "sha256": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
        },
        "chat_template.jinja": {
            "bytes": 18569,
            "sha256": "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
        },
        "README.md": {
            "bytes": 29351,
            "sha256": "aaab87052837925e0fb400bb20700553b11088fa5b3ae21fa0c1ec5da53637a4",
        },
        ".gitattributes": {
            "bytes": 1570,
            "sha256": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
        },
    },
}
ASSISTANT_MODEL_CONTRACT: dict[str, object] = {
    "architecture": "Gemma4AssistantForCausalLM",
    "backboneHiddenSize": 1536,
    "hiddenSize": 256,
    "modelType": "gemma4_assistant",
    "numHiddenLayers": 4,
    "vocabSize": 262144,
}
ASSISTANT_CHECKPOINT_ACQUISITION: dict[str, object] = {
    "repo_id": "google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant",
    "revision": "ebc7e1a211354561464cb82ed6d886792138dcb6",
    "files": {
        "config.json": {
            "bytes": 2356,
            "sha256": "5d01e9f3f8e969aa8147201a26e849c05446c7c746fa918101ed0622b201db15",
        },
        "model.safetensors": {
            "bytes": 157565344,
            "sha256": "28b11aa1fef73e655107984e0024ed1b149df4b8b36dcb95f27cca603eabc960",
        },
    },
}
EXPORT_CONTRACT: dict[str, object] = {
    "backend": "webgpu",
    "max_input_len": 512,
    "max_seq_len": 8960,
    "methods": ["text_decoder"],
    "output": {
        "dtype": "int64",
        "semantic": "greedy_token",
        "shape": [1, 1],
    },
    "quantization": "8da4w+emb4",
}
_SOURCE_CLOSURE_RECEIPT_SCHEMA_VERSION = 3
_SOURCE_MANIFEST_SCHEMA_VERSION = 1
_WGSL_MANIFEST_SCHEMA_VERSION = 1
_GEMMA_PRODUCTION_DIFF_SUMMARIES: tuple[str, ...] = (
    "[ExecuTorch][WebGPU] Add shared model runtime prerequisites",
    "[ExecuTorch][Vulkan] Support scoped Gemma symbolic partitioning",
    "[ExecuTorch][WebGPU] Add Gemma 4 plain runtime and guarded routes",
    "[ExecuTorch][WebGPU] Add Gemma 4 plain export and artifact contract",
    "[ExecuTorch][WebGPU] Add plain Gemma 4 source-closure tests",
    "[ExecuTorch][WebGPU] Add Gemma 4 MTP operator and route support",
    "[ExecuTorch][WebGPU] Add Gemma 4 MTP export path",
    "[ExecuTorch][WebGPU] Add Gemma 4 speculative decode runtime",
    "[ExecuTorch][WebGPU] Add Gemma 4 MTP and speculative-decode source-closure tests",
)
MTP_EXPORT_CONTRACT: dict[str, object] = {
    "assistant_calls_per_round": 2,
    "assistant_lm_head_bits": 4,
    "assistant_quantization": "8da4w",
    "backend": "webgpu",
    "donor_length": {"min": 2, "max": 8960},
    "max_input_len": 512,
    "max_seq_len": 8960,
    "methods": ["k2_round"],
    "selection": {
        "centroid_top_k": 32,
        "num_centroids": 2048,
        "raw_token_ordering_shape": [262144],
        "logical_token_ordering_shape": [2048, 128],
        "selected_token_count": 4096,
        "tokens_per_centroid": 128,
    },
    "speculation_k": 2,
    "target_quantization": "8da4w+emb4",
}
MTP_SOURCE_CONFIG: dict[str, object] = {
    "path": "config/e2b_config.json",
    "sha256": SOURCE_CONFIG_SHA256,
}
MTP_ACCEPTED_PROVENANCE: dict[str, object] = {
    "artifact_status": "accepted_behavior_oracle",
    "source_closure": "pending_final_source_rebuild",
}
MTP_PENDING_SOURCE_PROVENANCE: dict[str, object] = {
    "artifact_status": "generated_from_current_source",
    "source_closure": "pending_final_source_receipt",
}
MTP_SOURCE_VERIFIED_PROVENANCE: dict[str, object] = {
    "artifact_status": "generated_from_current_source",
    "source_closure": "source_verified",
}
MTP_PENDING_SOURCE_CLOSURES: frozenset[str] = frozenset(
    {
        str(MTP_ACCEPTED_PROVENANCE["source_closure"]),
        str(MTP_PENDING_SOURCE_PROVENANCE["source_closure"]),
    }
)
MTP_EDGE_CENSUS: dict[str, int] = {
    "custom_scatter": 2,
    "gemma_sdpa": 43,
    "generic_scatter": 0,
    "legacy_custom_sdpa": 0,
    "topk": 2,
}
_MTP_K2_DONOR_VIEW_ORDER: list[dict[str, object]] = [
    {"role": "fullK", "layer": 14, "cacheKind": "k_cache", "layout": "BHKD"},
    {"role": "fullV", "layer": 14, "cacheKind": "v_cache", "layout": "BHKD"},
    {
        "role": "slidingK",
        "layer": 13,
        "cacheKind": "k_cache",
        "layout": "BHKD",
    },
    {
        "role": "slidingV",
        "layer": 13,
        "cacheKind": "v_cache",
        "layout": "BHKD",
    },
]
_MTP_K2_INPUT_ORDER = ["input_ids", "input_pos", "is_round", "donor_length"]
_MTP_K2_OUTPUT_ORDER = [
    "candidates",
    "target_greedy",
    "output_matches",
    "output_bonus",
    "state_probe",
]
_MTP_K2_OPERATOR_COUNTS = {
    "aten.argmax.default": 3,
    "aten.scatter.src": 2,
    "aten.topk.default": 2,
    "llama.custom_sdpa.default": 43,
    "llama.update_cache.default": 31,
}
_MTP_K2_STATE_ALIAS = {
    "logicalSource": "nextFeature[1,1,1536]",
    "physicalDestination": "seed_feature[1,1,1,1536]",
    "mutation": "llama.update_cache.default",
}
_MTP_QAT_DONOR_SEQUENCE = [2, 16, 511, 512, 513, 514, 1024, 8960, 2]
_MTP_QAT_SELECTION_CONTRACT = {
    "centroidTopK": 32,
    "numCentroids": 2048,
    "selectedTokenCount": 4096,
    "tokensPerCentroid": 128,
}
COMBINED_RUNTIME_CONTRACT: dict[str, object] = {
    "capacities": {
        "mtp": {
            "donor": 8960,
            "max_input": 512,
            "target": 8960,
        },
        "plain": {"max_context": 8960, "max_input": 512},
    },
    "context": {
        "collision": "fail_closed",
        "lifetime": "runner_owned",
        "registration": "compare_and_set_default_webgpu_context",
        "release": "compare_and_set_before_destroy",
    },
    "methods": {"mtp": ["k2_round"], "plain": ["text_decoder"]},
    "profile": {
        "builds": {
            "mtp": {
                "profile": "compile_time_enabled",
                "wall": "compile_time_disabled",
            },
            "plain": {"wall": "compile_time_disabled"},
        },
        "fields": [
            "schemaVersion",
            "supported",
            "fresh",
            "valid",
            "context_generation",
            "querypool_generation",
            "execute_generation",
            "total_kernel_ms",
            "pass_span_ms",
            "interpass_gap_ms",
            "perop",
        ],
        "schema_version": 1,
    },
    "reset": {
        "mtp": {
            "clears": ["accepted_drafts", "buffered_tokens", "execute_count"],
            "failure": "fail_closed",
            "method": "unload_then_reload",
        },
        "plain": {
            "failure": "fail_closed",
            "method": "unload_then_reload",
        },
    },
    "tensor_data_files": {"mtp": 3, "plain": 3},
}
COMBINED_RUNTIME_VIEWS: dict[str, object] = {
    "combined": {
        "sequence": ["plain", "mtp", "plain"],
        "status": "pending_cross_view_gpu_execution_validation",
    },
    "mtp": {
        "receipt": "mtp",
        "runtime": "mtp",
        "status": "pending_gpu_execution_validation",
    },
    "plain": {
        "receipt": "plain",
        "runtime": "plain",
        "status": "pending_gpu_execution_validation",
    },
}
RUNTIME_BUILD_TARGETS: dict[str, str] = {
    "mtp": "gemma4_spec_browser",
    "plain": "gemma4_plain_wasm",
}
RUNTIME_PROFILING_MODES: dict[str, dict[str, bool]] = {
    "mtp": {"profile": True, "wall": False},
    "plain": {"profile": True, "wall": False},
}
RUNTIME_FACTORY_NAMES: dict[str, dict[str, str]] = {
    "mtp": {
        "profile": "createGemma4MtpProfile",
        "wall": "createGemma4Mtp",
    },
    "plain": {
        "profile": "createWebGPULlama",
        "wall": "createWebGPULlama",
    },
}
RUNTIME_OUTPUT_STEMS: dict[str, dict[str, str]] = {
    "mtp": {
        "profile": "gemma4_mtp_profile",
        "wall": "gemma4_mtp",
    },
    "plain": {
        "profile": "webgpu_llama",
        "wall": "webgpu_llama",
    },
}
CLOSURE_SOURCE_PATHS: dict[str, str] = {
    "source_manifest": "closure/source_manifest.json",
    "wgsl_manifest": "closure/wgsl_manifest.json",
}
CLOSURE_RECIPE_PATHS: dict[str, dict[str, str]] = {
    "mtp": {
        "profile": "closure/recipes/mtp-profile.json",
        "wall": "closure/recipes/mtp-wall.json",
    },
    "plain": {
        "profile": "closure/recipes/plain-profile.json",
        "wall": "closure/recipes/plain-wall.json",
    },
}
RUNTIME_SOURCE_SCHEMA_VERSION = 4
_BUILD_RECIPE_SCHEMA_VERSION = 2
_COMBINED_RUNTIME_SCHEMA_VERSION = 3
_COMMON_WEBGPU_CMAKE_ARGS = [
    "-S",
    ".",
    "-GNinja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DPYTHON_EXECUTABLE=.venv/bin/python",
    "-DEXECUTORCH_BUILD_WEBGPU=ON",
    "-DEXECUTORCH_BUILD_WEBGPU_TEST=OFF",
    "-DEXECUTORCH_BUILD_WASM=ON",
    "-DEXECUTORCH_BUILD_XNNPACK=OFF",
    "-DEXECUTORCH_BUILD_CPUINFO=ON",
    "-DEXECUTORCH_BUILD_PTHREADPOOL=ON",
    "-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON",
    "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON",
    "-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON",
    "-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON",
    "-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON",
]


def _source_config_path() -> Path:
    return Path(__file__).parent / "config" / "e2b_config.json"


def _single_file_manifest(path: str, byte_count: int, sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifacts": [
            {
                "bytes": byte_count,
                "path": path,
                "role": "source",
                "sha256": sha256,
            }
        ],
        "ptd_order": [],
    }


def _load_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _validate_architecture(config: Mapping[str, object], label: str) -> None:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError(f"{label} is missing text_config")
    observed: dict[str, object] = {"model_type": config.get("model_type")}
    for key in ARCHITECTURE_FINGERPRINT:
        if key != "model_type":
            observed[key] = text_config.get(key)
    if observed != ARCHITECTURE_FINGERPRINT:
        raise ValueError(f"{label} architecture fingerprint mismatch")


def _is_hex_digest(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} schema mismatch: {sorted(value)}")


def _canonical_set_digest(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _regular_file_identity(path: Path, root: Path) -> dict[str, object]:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source closure path escapes its root: {path}") from error
    cursor = root
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"source closure root must be a regular directory: {root}")
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"source closure rejects symlink traversal: {path}")
    if not path.is_file():
        raise ValueError(f"source closure requires a regular non-symlink file: {path}")
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _canonical_owned_path(value: str) -> str:
    path = Path(value)
    normalized = path.as_posix()
    if (
        path.is_absolute()
        or normalized != value
        or not path.parts
        or ".." in path.parts
        or path.parts[0] in {"fbcode", "xplat"}
    ):
        raise ValueError(f"non-canonical owned source path: {value}")
    return normalized


def _run_source_control(argv: Sequence[str], label: str) -> str:
    try:
        result = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ValueError(f"cannot inspect {label} checkout: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"cannot inspect {label} checkout: {detail}")
    return result.stdout


def _checkout_snapshot(root: Path, kind: str) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{kind} checkout root must be a regular directory")
    if kind == "fbsource":
        head = _run_source_control(
            [
                "sl",
                "--cwd",
                str(root),
                "log",
                "-r",
                ".",
                "-T",
                "{node}",
                "--reason",
                "derive Gemma source receipt - sl help log",
            ],
            kind,
        ).strip()
        status = _run_source_control(
            [
                "sl",
                "--cwd",
                str(root),
                "status",
                "--reason",
                "verify Gemma source checkout - sl help status",
            ],
            kind,
        )
    elif kind == "oss":
        head = _run_source_control(
            ["git", "-C", str(root), "rev-parse", "HEAD"], kind
        ).strip()
        status = _run_source_control(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            kind,
        )
    else:
        raise ValueError(f"unsupported source checkout kind: {kind}")
    if not _is_hex_digest(head, 40):
        raise ValueError(f"{kind} checkout did not report a 40-character head")
    if status.strip():
        raise ValueError(f"{kind} checkout is not clean")
    return {"clean": True, "head": head}


def _derive_owned_paths(
    fbsource_root: Path,
    reviewed_summaries: Sequence[str] | None = None,
) -> list[str]:
    summaries = tuple(
        _GEMMA_PRODUCTION_DIFF_SUMMARIES
        if reviewed_summaries is None
        else reviewed_summaries
    )
    if not summaries:
        raise ValueError("reviewed Gemma production summaries must not be empty")
    expected_reverse = tuple(reversed(summaries))
    changed: set[str] = set()
    for offset, expected_summary in enumerate(expected_reverse):
        revision = "." if offset == 0 else f".~{offset}"
        identity = _run_source_control(
            [
                "sl",
                "--cwd",
                str(fbsource_root),
                "log",
                "-r",
                revision,
                "-T",
                "{node}\\n{desc|firstline}\\n",
                "--reason",
                "derive Gemma production ownership - sl help log",
            ],
            "fbsource",
        ).splitlines()
        if (
            len(identity) != 2
            or not _is_hex_digest(identity[0], 40)
            or identity[1] != expected_summary
        ):
            raise ValueError("fbsource head is not the reviewed Gemma production stack")
        paths = _run_source_control(
            [
                "sl",
                "--cwd",
                str(fbsource_root),
                "status",
                "--change",
                identity[0],
                "--no-status",
                "--root-relative",
                "--reason",
                "derive Gemma production file union - sl help status",
            ],
            "fbsource",
        ).splitlines()
        changed.update(path for path in paths if path)
    xplat_prefix = "xplat/executorch/"
    fbcode_prefix = "fbcode/executorch/"
    unsupported = sorted(
        path
        for path in changed
        if not path.startswith(xplat_prefix) and not path.startswith(fbcode_prefix)
    )
    if unsupported:
        raise ValueError(
            "Gemma production diffs contain files outside the mirrored ExecuTorch roots"
        )
    xplat = {
        path.removeprefix(xplat_prefix)
        for path in changed
        if path.startswith(xplat_prefix)
    }
    fbcode = {
        path.removeprefix(fbcode_prefix)
        for path in changed
        if path.startswith(fbcode_prefix)
    }
    if not xplat or xplat != fbcode:
        raise ValueError(
            "Gemma production xplat/fbcode ownership is not byte-mirror complete"
        )
    return sorted(_canonical_owned_path(path) for path in xplat)


def _source_copy_paths(
    fbsource_root: Path, oss_root: Path, logical_path: str
) -> dict[str, Path]:
    return {
        "fbcode": fbsource_root / "fbcode/executorch" / logical_path,
        "oss": oss_root / logical_path,
        "xplat": fbsource_root / "xplat/executorch" / logical_path,
    }


def create_source_manifest(
    fbsource_root: Path,
    oss_root: Path,
) -> dict[str, object]:
    before = {
        "fbsource": _checkout_snapshot(fbsource_root, "fbsource"),
        "oss": _checkout_snapshot(oss_root, "oss"),
    }
    paths = _derive_owned_paths(fbsource_root)
    files: list[dict[str, object]] = []
    for logical_path in paths:
        copy_paths = _source_copy_paths(fbsource_root, oss_root, logical_path)
        copies = {
            label: {
                "path": path.relative_to(root).as_posix(),
                **_regular_file_identity(path, root),
            }
            for label, path, root in (
                ("fbcode", copy_paths["fbcode"], fbsource_root),
                ("oss", copy_paths["oss"], oss_root),
                ("xplat", copy_paths["xplat"], fbsource_root),
            )
        }
        identities = {(copy["bytes"], copy["sha256"]) for copy in copies.values()}
        if len(identities) != 1:
            raise ValueError(f"source mirror/OSS identity mismatch: {logical_path}")
        files.append({"copies": copies, "path": logical_path})
    after = {
        "fbsource": _checkout_snapshot(fbsource_root, "fbsource"),
        "oss": _checkout_snapshot(oss_root, "oss"),
    }
    if after != before:
        raise ValueError("source checkout changed while creating the manifest")
    return {
        "checkouts": after,
        "file_set_sha256": _canonical_set_digest(paths),
        "files": files,
        "schema_version": _SOURCE_MANIFEST_SCHEMA_VERSION,
    }


def _validate_checkout_identity(checkout: object, label: str) -> None:
    if not isinstance(checkout, dict):
        raise ValueError(f"Gemma4 {label} checkout must be an object")
    expected = {"clean": True, "head": checkout.get("head")}
    if checkout != expected or not _is_hex_digest(checkout.get("head"), 40):
        raise ValueError(f"Gemma4 {label} checkout identity is invalid")


def _validated_source_manifest_entry(entry: object) -> str:
    if not isinstance(entry, dict):
        raise ValueError("Gemma4 source manifest entry must be an object")
    _require_exact_keys(entry, {"copies", "path"}, "source manifest entry")
    path = entry.get("path")
    if not isinstance(path, str):
        raise ValueError("Gemma4 source manifest path must be a string")
    path = _canonical_owned_path(path)
    copies = entry.get("copies")
    if not isinstance(copies, dict):
        raise ValueError("Gemma4 source manifest copies must be an object")
    _require_exact_keys(copies, {"fbcode", "oss", "xplat"}, "source copies")
    expected_paths = {
        "fbcode": f"fbcode/executorch/{path}",
        "oss": path,
        "xplat": f"xplat/executorch/{path}",
    }
    identities: set[tuple[object, object]] = set()
    for label, expected_path in expected_paths.items():
        copy_identity = copies[label]
        if not isinstance(copy_identity, dict):
            raise ValueError("Gemma4 source manifest copy must be an object")
        _require_exact_keys(
            copy_identity, {"bytes", "path", "sha256"}, "source copy identity"
        )
        valid = (
            copy_identity.get("path") == expected_path
            and isinstance(copy_identity.get("bytes"), int)
            and int(copy_identity["bytes"]) >= 0
            and _is_hex_digest(copy_identity.get("sha256"), 64)
        )
        if not valid:
            raise ValueError("Gemma4 source manifest copy identity is invalid")
        identities.add((copy_identity["bytes"], copy_identity["sha256"]))
    if len(identities) != 1:
        raise ValueError("Gemma4 source manifest mirror/OSS identity mismatch")
    return path


def validate_source_manifest(manifest: Mapping[str, object]) -> None:
    _require_exact_keys(
        manifest,
        {"checkouts", "file_set_sha256", "files", "schema_version"},
        "Gemma4 source manifest",
    )
    if manifest.get("schema_version") != _SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Gemma4 source manifest schema mismatch")
    checkouts = manifest.get("checkouts")
    if not isinstance(checkouts, dict):
        raise ValueError("Gemma4 source manifest checkouts must be an object")
    _require_exact_keys(checkouts, {"fbsource", "oss"}, "source checkouts")
    for label in ("fbsource", "oss"):
        _validate_checkout_identity(checkouts[label], label)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Gemma4 source manifest files must be a non-empty list")
    paths = [_validated_source_manifest_entry(entry) for entry in files]
    if paths != sorted(set(paths)):
        raise ValueError("Gemma4 source manifest paths are not sorted and unique")
    if manifest.get("file_set_sha256") != _canonical_set_digest(paths):
        raise ValueError("Gemma4 source manifest file-set identity mismatch")


def _load_wgsl_generator(backend_root: Path) -> Any:
    generator_path = backend_root / "scripts/gen_wgsl_headers.py"
    _regular_file_identity(generator_path, backend_root)
    spec = importlib.util.spec_from_file_location(
        "_gemma4_wgsl_generator", generator_path
    )
    if spec is None:
        raise ValueError("cannot load the WGSL generator")
    loader = spec.loader
    if loader is None:
        raise ValueError("cannot load the WGSL generator")
    generator = importlib.util.module_from_spec(spec)
    loader.exec_module(generator)
    generator.BACKEND_ROOT = backend_root.resolve(strict=True)
    return generator


def create_wgsl_manifest(backend_root: Path) -> dict[str, object]:
    backend_root = backend_root.resolve(strict=True)
    if backend_root.parts[-4:] != ("xplat", "executorch", "backends", "webgpu"):
        raise ValueError("WGSL backend root is not inside an fbsource xplat checkout")
    fbsource_root = backend_root.parents[3]
    before = _checkout_snapshot(fbsource_root, "fbsource")
    generator = _load_wgsl_generator(backend_root)
    shaders = list(generator.discover())
    outputs, orphans = generator.collect_outputs()
    if orphans:
        raise ValueError("WGSL manifest rejects orphan generated headers")
    yaml_paths = sorted((backend_root / "runtime/ops").glob("**/*.yaml"))
    expected_yaml = sorted(
        shader.with_suffix(".yaml")
        for shader in shaders
        if shader.with_suffix(".yaml").exists()
    )
    if yaml_paths != expected_yaml:
        raise ValueError("WGSL manifest rejects orphan YAML specifications")
    roles: dict[Path, str] = {
        backend_root / "scripts/gen_wgsl_headers.py": "generator",
        **{shader: "wgsl" for shader in shaders},
        **{path: "yaml" for path in yaml_paths},
    }
    for output, generated in outputs.items():
        identity = _regular_file_identity(output, backend_root)
        if output.read_bytes() != generated:
            raise ValueError(f"WGSL generated output is stale: {output}")
        roles[output] = (
            "global_registry"
            if output == generator.registry_path()
            else "generated_header"
        )
        if identity["bytes"] != len(generated):
            raise ValueError(f"WGSL generated output size mismatch: {output}")
    files = [
        {
            "path": path.relative_to(backend_root).as_posix(),
            "role": roles[path],
            **_regular_file_identity(path, backend_root),
        }
        for path in sorted(roles)
    ]
    after = _checkout_snapshot(fbsource_root, "fbsource")
    if after != before:
        raise ValueError("fbsource checkout changed while creating the WGSL manifest")
    path_roles = [{"path": entry["path"], "role": entry["role"]} for entry in files]
    return {
        "fbsource_commit": after["head"],
        "file_set_sha256": _canonical_set_digest(path_roles),
        "files": files,
        "orphans": [],
        "schema_version": _WGSL_MANIFEST_SCHEMA_VERSION,
    }


def _validated_wgsl_manifest_entry(entry: object) -> tuple[str, str]:
    if not isinstance(entry, dict):
        raise ValueError("Gemma4 WGSL manifest entry must be an object")
    _require_exact_keys(
        entry, {"bytes", "path", "role", "sha256"}, "WGSL file identity"
    )
    path = entry.get("path")
    role = entry.get("role")
    valid_roles = {"generated_header", "generator", "global_registry", "wgsl", "yaml"}
    valid = (
        isinstance(path, str)
        and _canonical_owned_path(path) == path
        and role in valid_roles
        and isinstance(entry.get("bytes"), int)
        and int(entry["bytes"]) >= 0
        and _is_hex_digest(entry.get("sha256"), 64)
    )
    if not valid:
        raise ValueError("Gemma4 WGSL manifest file identity is invalid")
    assert isinstance(path, str)
    return path, str(role)


def validate_wgsl_manifest(manifest: Mapping[str, object]) -> None:
    _require_exact_keys(
        manifest,
        {
            "fbsource_commit",
            "file_set_sha256",
            "files",
            "orphans",
            "schema_version",
        },
        "Gemma4 WGSL manifest",
    )
    if manifest.get("schema_version") != _WGSL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Gemma4 WGSL manifest schema mismatch")
    if manifest.get("orphans") != []:
        raise ValueError("Gemma4 WGSL manifest contains orphan outputs")
    if not _is_hex_digest(manifest.get("fbsource_commit"), 40):
        raise ValueError("Gemma4 WGSL manifest fbsource identity is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Gemma4 WGSL manifest files must be a non-empty list")
    path_roles: list[dict[str, str]] = []
    role_counts: dict[str, int] = {}
    for entry in files:
        path, role = _validated_wgsl_manifest_entry(entry)
        path_roles.append({"path": path, "role": role})
        role_counts[role] = role_counts.get(role, 0) + 1
    if path_roles != sorted(path_roles, key=lambda item: item["path"]):
        raise ValueError("Gemma4 WGSL manifest paths are not sorted")
    if len({item["path"] for item in path_roles}) != len(path_roles):
        raise ValueError("Gemma4 WGSL manifest contains duplicate paths")
    if role_counts.get("generator") != 1 or role_counts.get("global_registry") != 1:
        raise ValueError("Gemma4 WGSL manifest requires one generator and registry")
    if not role_counts.get("wgsl") or not role_counts.get("generated_header"):
        raise ValueError("Gemma4 WGSL manifest has incomplete source/output closure")
    if manifest.get("file_set_sha256") != _canonical_set_digest(path_roles):
        raise ValueError("Gemma4 WGSL manifest file-set identity mismatch")


def create_source_closure_receipt(
    fbsource_root: Path, oss_root: Path, backend_root: Path
) -> dict[str, object]:
    source_manifest = create_source_manifest(fbsource_root, oss_root)
    wgsl_manifest = create_wgsl_manifest(backend_root)
    checkouts = source_manifest["checkouts"]
    assert isinstance(checkouts, dict)
    fbsource = checkouts["fbsource"]
    oss = checkouts["oss"]
    assert isinstance(fbsource, dict) and isinstance(oss, dict)
    if wgsl_manifest["fbsource_commit"] != fbsource["head"]:
        raise ValueError("source and WGSL manifests describe different fbsource heads")
    return {
        "fbsource_commit": fbsource["head"],
        "oss_commit": oss["head"],
        "schema_version": _SOURCE_CLOSURE_RECEIPT_SCHEMA_VERSION,
        "source_current": True,
        "source_manifest": copy.deepcopy(source_manifest),
        "verification": {
            "source_checkout": "verified",
            "wgsl_codegen": "verified",
        },
        "wgsl_manifest": copy.deepcopy(wgsl_manifest),
    }


def canonical_build_recipe(model: str, flavor: str) -> dict[str, object]:
    if model not in RUNTIME_BUILD_TARGETS or flavor not in RUNTIME_PROFILING_MODES.get(
        model, {}
    ):
        raise ValueError(f"unsupported Gemma4 build recipe: {model} {flavor}")
    profiling_enabled = RUNTIME_PROFILING_MODES[model][flavor]
    build_directory = (
        "cmake-out-gemma4-webgpu-profile"
        if profiling_enabled
        else "cmake-out-gemma4-webgpu-wall"
    )
    output_directory = (
        "backends/webgpu/browser_gemma4_plain"
        if model == "plain"
        else "backends/webgpu/browser_gemma4_mtp"
    )
    output_stem = RUNTIME_OUTPUT_STEMS[model][flavor]
    factory = RUNTIME_FACTORY_NAMES[model][flavor]
    target = RUNTIME_BUILD_TARGETS[model]
    return {
        "build_argv": [
            "cmake",
            "--build",
            build_directory,
            "--target",
            target,
            "-j",
        ],
        "configure_argv": [
            "emcmake",
            "cmake",
            *_COMMON_WEBGPU_CMAKE_ARGS,
            f"-DEXECUTORCH_BUILD_WEBGPU_PROFILING={'ON' if profiling_enabled else 'OFF'}",
            f"-DGEMMA4_SPEC_WASM_EXPORT_NAME={RUNTIME_FACTORY_NAMES['mtp'][flavor]}",
            f"-DGEMMA4_SPEC_WASM_OUTPUT_NAME={RUNTIME_OUTPUT_STEMS['mtp'][flavor]}",
            "-B",
            build_directory,
        ],
        "cwd": ".",
        "factory": factory,
        "flavor": flavor,
        "model": model,
        "output_stem": output_stem,
        "outputs": {
            "javascript": f"{build_directory}/{output_directory}/{output_stem}.js",
            "wasm": f"{build_directory}/{output_directory}/{output_stem}.wasm",
        },
        "profiling_enabled": profiling_enabled,
        "schema_version": _BUILD_RECIPE_SCHEMA_VERSION,
        "target": target,
    }


def _validate_build_recipe(path: Path, model: str, flavor: str) -> Mapping[str, object]:
    label = f"{'MTP' if model == 'mtp' else 'plain'} {flavor} recipe"
    try:
        recipe = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Gemma4 {label} is not canonical JSON") from error
    if recipe != canonical_build_recipe(model, flavor):
        raise ValueError(f"Gemma4 {label} does not match the canonical contract")
    return recipe


def _validate_source_receipt(root: Path, artifacts: Sequence[object]) -> None:
    source_paths = [
        artifact.get("path")
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("role") == "source"
    ]
    if len(source_paths) != 1 or not isinstance(source_paths[0], str):
        raise ValueError("Gemma4 plain manifest requires one source receipt")
    receipt = _load_json(root / source_paths[0])
    _require_exact_keys(
        receipt,
        {
            "fbsource_commit",
            "oss_commit",
            "schema_version",
            "source_current",
            "source_manifest",
            "verification",
            "wgsl_manifest",
        },
        "Gemma4 source receipt",
    )
    if (
        receipt.get("schema_version") != _SOURCE_CLOSURE_RECEIPT_SCHEMA_VERSION
        or receipt.get("source_current") is not True
    ):
        raise ValueError("Gemma4 source receipt is not source-current")
    source_manifest = receipt.get("source_manifest")
    wgsl_manifest = receipt.get("wgsl_manifest")
    if not isinstance(source_manifest, dict) or not isinstance(wgsl_manifest, dict):
        raise ValueError("Gemma4 source receipt lacks semantic manifests")
    validate_source_manifest(source_manifest)
    validate_wgsl_manifest(wgsl_manifest)
    checkouts = source_manifest["checkouts"]
    assert isinstance(checkouts, dict)
    fbsource = checkouts["fbsource"]
    oss = checkouts["oss"]
    assert isinstance(fbsource, dict) and isinstance(oss, dict)
    if wgsl_manifest.get("fbsource_commit") != fbsource.get("head"):
        raise ValueError("Gemma4 source receipt has a mismatched WGSL checkout")
    if receipt.get("fbsource_commit") != fbsource.get("head"):
        raise ValueError("Gemma4 source receipt has an invalid fbsource commit")
    if receipt.get("oss_commit") != oss.get("head"):
        raise ValueError("Gemma4 source receipt has an invalid OSS commit")
    if receipt.get("verification") != {
        "source_checkout": "verified",
        "wgsl_codegen": "verified",
    }:
        raise ValueError("Gemma4 source receipt verification is incomplete")


def _model_source_receipt(
    root: Path, manifest: Mapping[str, object], label: str
) -> Mapping[str, object]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"Gemma4 {label} manifest has no artifacts")
    source_paths = [
        artifact.get("path")
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("role") == "source"
    ]
    if len(source_paths) != 1 or not isinstance(source_paths[0], str):
        raise ValueError(f"Gemma4 {label} manifest requires one source receipt")
    return _load_json(root / source_paths[0])


def validate_export_identity(checkpoint_root: Path) -> Mapping[str, object]:
    source_config = _source_config_path()
    validate_manifest(
        source_config.parent,
        _single_file_manifest(
            source_config.name, SOURCE_CONFIG_BYTES, SOURCE_CONFIG_SHA256
        ),
    )
    _validate_architecture(_load_json(source_config), "source config")

    files = CHECKPOINT_ACQUISITION["files"]
    assert isinstance(files, dict)
    for name, identity in files.items():
        assert isinstance(name, str)
        assert isinstance(identity, dict)
        validate_manifest(
            checkpoint_root,
            _single_file_manifest(
                name,
                int(identity["bytes"]),
                str(identity["sha256"]),
            ),
        )
    _validate_architecture(
        _load_json(checkpoint_root / "config.json"), "checkpoint config"
    )
    return CHECKPOINT_ACQUISITION


def validate_assistant_export_identity(
    checkpoint_root: Path,
) -> Mapping[str, object]:
    files = ASSISTANT_CHECKPOINT_ACQUISITION["files"]
    assert isinstance(files, dict)
    for name, identity in files.items():
        assert isinstance(name, str)
        assert isinstance(identity, dict)
        validate_manifest(
            checkpoint_root,
            _single_file_manifest(
                name,
                int(identity["bytes"]),
                str(identity["sha256"]),
            ),
        )
    config = _load_json(checkpoint_root / "config.json")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("assistant config is missing text_config")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("assistant config is missing architectures")
    observed_contract = {
        "architecture": architectures[0],
        "backboneHiddenSize": config.get("backbone_hidden_size"),
        "hiddenSize": text_config.get("hidden_size"),
        "modelType": config.get("model_type"),
        "numHiddenLayers": text_config.get("num_hidden_layers"),
        "vocabSize": text_config.get("vocab_size"),
    }
    if observed_contract != ASSISTANT_MODEL_CONTRACT:
        raise ValueError("assistant checkpoint model contract mismatch")
    return ASSISTANT_CHECKPOINT_ACQUISITION


def create_plain_manifest(
    root: Path,
    role_paths: Mapping[str, Path],
    ptd_paths: Sequence[Path],
) -> dict[str, object]:
    if "pte" not in role_paths or "source" not in role_paths:
        raise ValueError("Gemma4 plain manifest requires PTE and source receipt roles")
    if len(ptd_paths) != 3:
        raise ValueError("Gemma4 plain manifest requires exactly three ordered PTDs")
    manifest = create_manifest(root, role_paths, ptd_paths)
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, list)
    for artifact in artifacts:
        if (
            isinstance(artifact, dict)
            and artifact.get("role") == "ptd"
            and int(artifact["bytes"]) >= WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES
        ):
            raise ValueError("Gemma4 PTD exceeds the WebGPU external-constant limit")
    manifest.update(
        {
            "acquisition": CHECKPOINT_ACQUISITION,
            "export": EXPORT_CONTRACT,
            "model": {
                "architecture": ARCHITECTURE_FINGERPRINT,
                "source_config": {
                    "path": "config/e2b_config.json",
                    "sha256": SOURCE_CONFIG_SHA256,
                },
            },
        }
    )
    _validate_source_receipt(root, artifacts)
    return manifest


def _require_unique_mtp_artifact_paths(artifacts: Sequence[object]) -> None:
    seen: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            continue
        normalized = os.path.normpath(str(artifact["path"]))
        if normalized in seen:
            raise ValueError(
                f"Gemma4 MTP duplicate normalized artifact path: {normalized}"
            )
        seen.add(normalized)


def create_mtp_manifest(
    root: Path,
    role_paths: Mapping[str, Path],
    ptd_paths: Sequence[Path],
) -> dict[str, object]:
    if set(role_paths) not in ({"pte"}, {"pte", "source"}):
        raise ValueError(
            "Gemma4 MTP manifest requires one K=2 PTE role and an optional "
            "source receipt"
        )
    if len(ptd_paths) != 3:
        raise ValueError("Gemma4 MTP manifest requires exactly three ordered PTDs")
    _require_unique_mtp_artifact_paths(
        [{"path": str(path)} for path in list(role_paths.values()) + list(ptd_paths)]
    )
    manifest = create_manifest(root, role_paths, ptd_paths)
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, list)
    _require_unique_mtp_artifact_paths(artifacts)
    for artifact in artifacts:
        if (
            isinstance(artifact, dict)
            and artifact.get("role") == "ptd"
            and int(artifact["bytes"]) >= WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES
        ):
            raise ValueError("Gemma4 MTP PTD exceeds the WebGPU constant limit")
    paths = [
        str(artifact["path"]) for artifact in artifacts if isinstance(artifact, dict)
    ]
    if len(set(paths)) != len(paths):
        raise ValueError("Gemma4 MTP artifact paths must be distinct")
    if any(len(Path(path).parts) != 1 for path in paths):
        raise ValueError("Gemma4 MTP artifact paths must be flat")
    manifest.update(
        {
            "acquisition": {
                "assistant": ASSISTANT_CHECKPOINT_ACQUISITION,
                "target": CHECKPOINT_ACQUISITION,
            },
            "export": MTP_EXPORT_CONTRACT,
            "model": {
                "architecture": ARCHITECTURE_FINGERPRINT,
                "assistant": ASSISTANT_MODEL_CONTRACT,
                "source_config": MTP_SOURCE_CONFIG,
            },
        }
    )
    if "source" in role_paths:
        _validate_source_receipt(root, artifacts)
        manifest["provenance"] = dict(MTP_SOURCE_VERIFIED_PROVENANCE)
    else:
        manifest["provenance"] = dict(MTP_PENDING_SOURCE_PROVENANCE)
    return manifest


def validate_plain_manifest(
    root: Path,
    manifest: Mapping[str, object],
    require_source_receipt: bool = True,
) -> None:
    validate_manifest(root, manifest)
    if require_source_receipt and manifest.get("provenance") is not None:
        raise ValueError("Gemma4 plain production provenance must be absent")
    if manifest.get("acquisition") != CHECKPOINT_ACQUISITION:
        raise ValueError("Gemma4 checkpoint acquisition identity mismatch")
    model = manifest.get("model")
    if (
        not isinstance(model, dict)
        or model.get("architecture") != ARCHITECTURE_FINGERPRINT
    ):
        raise ValueError("Gemma4 architecture identity mismatch")
    source_config = model.get("source_config")
    if source_config != {
        "path": "config/e2b_config.json",
        "sha256": SOURCE_CONFIG_SHA256,
    }:
        raise ValueError("Gemma4 source config identity mismatch")
    if manifest.get("export") != EXPORT_CONTRACT:
        raise ValueError("Gemma4 WebGPU export contract mismatch")

    artifacts = manifest.get("artifacts")
    ptd_order = manifest.get("ptd_order")
    assert isinstance(artifacts, list)
    assert isinstance(ptd_order, list)
    if len(ptd_order) != 3:
        raise ValueError("Gemma4 plain manifest requires exactly three ordered PTDs")
    for artifact in artifacts:
        if (
            isinstance(artifact, dict)
            and artifact.get("role") == "ptd"
            and int(artifact["bytes"]) >= WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES
        ):
            raise ValueError("Gemma4 PTD exceeds the WebGPU external-constant limit")
    roles = {
        artifact.get("role") for artifact in artifacts if isinstance(artifact, dict)
    }
    if "pte" not in roles or (require_source_receipt and "source" not in roles):
        raise ValueError("Gemma4 plain manifest is missing PTE/source receipt roles")
    if require_source_receipt:
        _validate_source_receipt(root, artifacts)

    expected_paths = {
        str(artifact["path"]) for artifact in artifacts if isinstance(artifact, dict)
    }
    if any(len(Path(path).parts) != 1 for path in expected_paths):
        raise ValueError("Gemma4 artifact staging directory must be flat")
    actual_paths = {entry.name for entry in root.iterdir()}
    if actual_paths != expected_paths:
        raise ValueError("Gemma4 artifact staging contains missing or extra entries")


def _expected_mtp_mutation_order() -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "logicalTarget": "seed_feature",
            "role": "nextFeatureSeed",
            "shape": [1, 1, 1, 1536],
            "logicalLayout": "BSHD",
            "logicalDimOrder": [0, 1, 2, 3],
            "vulkanSourceStorage": "BUFFER",
            "vulkanDestinationStorage": "TEXTURE_3D",
        }
    ]
    for layer in range(15):
        head_dim = 512 if layer in {4, 9, 14} else 256
        for cache_kind in ("k_cache", "v_cache"):
            records.append(
                {
                    "logicalTarget": (
                        f"self_decoder.layers.{layer}.self_attn.kv_cache."
                        f"{cache_kind}"
                    ),
                    "role": "targetKvCache",
                    "layer": layer,
                    "cacheKind": cache_kind,
                    "shape": [1, 8960, 1, head_dim],
                    "logicalLayout": "BSHD",
                    "logicalDimOrder": [0, 1, 2, 3],
                    "vulkanSourceStorage": "BUFFER",
                    "vulkanDestinationStorage": "BUFFER",
                }
            )
    return records


def _validate_mtp_k2_evidence(k2_abi: object) -> None:
    if not isinstance(k2_abi, dict):
        raise ValueError("Gemma4 MTP K=2 ABI evidence must be an object")
    _require_exact_keys(
        k2_abi,
        {
            "bufferMutationCount",
            "donorViewOrder",
            "inputOrder",
            "mutationOrder",
            "operatorCounts",
            "outputOrder",
            "seedMutationCount",
            "stateAlias",
        },
        "Gemma4 MTP K=2 ABI evidence",
    )
    for key in ("bufferMutationCount", "seedMutationCount"):
        value = k2_abi[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Gemma4 MTP {key} must be a nonnegative integer")
    expected_mutations = _expected_mtp_mutation_order()
    if (
        k2_abi["bufferMutationCount"] != len(expected_mutations)
        or k2_abi["seedMutationCount"] != 1
        or k2_abi["donorViewOrder"] != _MTP_K2_DONOR_VIEW_ORDER
        or k2_abi["inputOrder"] != _MTP_K2_INPUT_ORDER
        or k2_abi["mutationOrder"] != expected_mutations
        or k2_abi["outputOrder"] != _MTP_K2_OUTPUT_ORDER
        or k2_abi["stateAlias"] != _MTP_K2_STATE_ALIAS
    ):
        raise ValueError("Gemma4 MTP K=2 ABI semantic evidence mismatch")
    operator_counts = k2_abi["operatorCounts"]
    if not isinstance(operator_counts, dict) or not operator_counts:
        raise ValueError("Gemma4 MTP operator counts must be a non-empty object")
    if any(
        not isinstance(key, str)
        or not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        for key, value in operator_counts.items()
    ):
        raise ValueError("Gemma4 MTP operator counts must be nonnegative integers")
    if operator_counts != _MTP_K2_OPERATOR_COUNTS:
        raise ValueError("Gemma4 MTP operator-count evidence mismatch")


def _validate_mtp_token_record(record: object, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Gemma4 MTP {label} token ordering must be an object")
    _require_exact_keys(
        record,
        {
            "max",
            "min",
            "numel",
            "permutationExact",
            "rawShape",
            "sha256",
            "shape",
            "uniqueCount",
        },
        f"Gemma4 MTP {label} token ordering",
    )
    expected = {
        "max": 262143,
        "min": 0,
        "numel": 262144,
        "permutationExact": True,
        "rawShape": [262144],
        "shape": [2048, 128],
        "uniqueCount": 262144,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Gemma4 MTP {label} token-ordering proof mismatch")
    if not _is_hex_digest(record.get("sha256"), 64):
        raise ValueError(f"Gemma4 MTP {label} token-ordering SHA-256 is invalid")


def _validate_mtp_qat_case(case: object, index: int, donor_length: int) -> None:
    if not isinstance(case, dict):
        raise ValueError("Gemma4 MTP QAT case must be an object")
    _require_exact_keys(
        case,
        {
            "caseIndex",
            "donorLength",
            "greedyTokenExact",
            "inputSha256",
            "outputs",
            "topk",
        },
        "Gemma4 MTP QAT case",
    )
    input_digests = case["inputSha256"]
    outputs = case["outputs"]
    if (
        case["caseIndex"] != index
        or case["donorLength"] != donor_length
        or case["greedyTokenExact"] is not True
        or not isinstance(input_digests, list)
        or not input_digests
        or not all(_is_hex_digest(value, 64) for value in input_digests)
        or not isinstance(outputs, list)
        or len(outputs) != 2
    ):
        raise ValueError("Gemma4 MTP QAT case semantic evidence mismatch")
    for output, expected_name in zip(outputs, ("logits", "last_hidden_state")):
        if not isinstance(output, dict):
            raise ValueError("Gemma4 MTP QAT output evidence must be an object")
        _require_exact_keys(
            output,
            {
                "actualSha256",
                "bitExact",
                "close",
                "maxAbsError",
                "name",
                "referenceSha256",
                "shape",
            },
            "Gemma4 MTP QAT output evidence",
        )
        max_error = output["maxAbsError"]
        shape = output["shape"]
        valid = (
            output["name"] == expected_name
            and isinstance(output["bitExact"], bool)
            and output["close"] is True
            and isinstance(max_error, (int, float))
            and not isinstance(max_error, bool)
            and math.isfinite(float(max_error))
            and float(max_error) >= 0.0
            and isinstance(shape, list)
            and bool(shape)
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in shape
            )
            and _is_hex_digest(output["actualSha256"], 64)
            and _is_hex_digest(output["referenceSha256"], 64)
        )
        if not valid:
            raise ValueError("Gemma4 MTP QAT output semantic evidence mismatch")
    topk = case["topk"]
    if not isinstance(topk, dict):
        raise ValueError("Gemma4 MTP QAT top-k evidence must be an object")
    _require_exact_keys(
        topk,
        {
            "allFinite",
            "boundaryGap",
            "indicesSha256",
            "stableReferenceExact",
            "top32PairwiseDistinct",
            "top33IndicesSha256",
            "top33ValuesSha256",
            "valuesSha256",
        },
        "Gemma4 MTP QAT top-k evidence",
    )
    boundary_gap = topk["boundaryGap"]
    valid_topk = (
        topk["allFinite"] is True
        and topk["stableReferenceExact"] is True
        and topk["top32PairwiseDistinct"] is True
        and isinstance(boundary_gap, (int, float))
        and not isinstance(boundary_gap, bool)
        and math.isfinite(float(boundary_gap))
        and float(boundary_gap) > 0.0
        and all(
            _is_hex_digest(topk[key], 64)
            for key in (
                "indicesSha256",
                "top33IndicesSha256",
                "top33ValuesSha256",
                "valuesSha256",
            )
        )
    )
    if not valid_topk:
        raise ValueError("Gemma4 MTP QAT top-k semantic evidence mismatch")


def _validate_mtp_qat_evidence(qat: object) -> None:
    if not isinstance(qat, dict):
        raise ValueError("Gemma4 MTP QAT evidence must be an object")
    _require_exact_keys(
        qat,
        {
            "cases",
            "donorSequence",
            "eagerEquivalence",
            "selectionContract",
            "tokenOrdering",
        },
        "Gemma4 MTP QAT evidence",
    )
    if qat["selectionContract"] != _MTP_QAT_SELECTION_CONTRACT:
        raise ValueError("Gemma4 MTP QAT selection contract mismatch")
    if qat["eagerEquivalence"] != {
        "allClose": True,
        "atol": 1e-4,
        "rtol": 1e-3,
    }:
        raise ValueError("Gemma4 MTP eager-equivalence evidence mismatch")
    donor_sequence = qat["donorSequence"]
    cases = qat["cases"]
    if donor_sequence != _MTP_QAT_DONOR_SEQUENCE or not isinstance(cases, list):
        raise ValueError("Gemma4 MTP QAT donor sequence mismatch")
    if len(cases) != len(donor_sequence):
        raise ValueError("Gemma4 MTP QAT case count mismatch")
    for index, (case, donor_length) in enumerate(zip(cases, donor_sequence)):
        _validate_mtp_qat_case(case, index, donor_length)
    if (
        cases[0]["inputSha256"] != cases[-1]["inputSha256"]
        or cases[0]["outputs"] != cases[-1]["outputs"]
        or cases[0]["topk"] != cases[-1]["topk"]
    ):
        raise ValueError("Gemma4 MTP QAT replay evidence mismatch")

    token_ordering = qat["tokenOrdering"]
    if not isinstance(token_ordering, dict):
        raise ValueError("Gemma4 MTP token-ordering evidence must be an object")
    base_token_keys = {
        "max",
        "min",
        "numel",
        "permutationExact",
        "rawShape",
        "sha256",
        "shape",
        "uniqueCount",
    }
    _require_exact_keys(
        token_ordering,
        base_token_keys | {"loaded", "raw", "rawLoadedByteExact", "rawSha256"},
        "Gemma4 MTP token-ordering evidence",
    )
    effective = {key: token_ordering[key] for key in base_token_keys}
    _validate_mtp_token_record(effective, "effective")
    _validate_mtp_token_record(token_ordering["raw"], "raw")
    _validate_mtp_token_record(token_ordering["loaded"], "loaded")
    if (
        token_ordering["rawLoadedByteExact"] is not True
        or not _is_hex_digest(token_ordering["rawSha256"], 64)
        or token_ordering["raw"] != token_ordering["loaded"]
        or effective != token_ordering["loaded"]
        or token_ordering["rawSha256"] != token_ordering["sha256"]
    ):
        raise ValueError("Gemma4 MTP raw/loaded token-ordering identity mismatch")


def _validate_mtp_evidence(evidence: object) -> None:  # noqa: C901
    if not isinstance(evidence, dict):
        raise ValueError("Gemma4 MTP evidence must be an object")
    _require_exact_keys(
        evidence,
        {
            "assistant_checkpoint",
            "k2_abi",
            "lowering",
            "qat_selection",
            "target_checkpoint",
        },
        "Gemma4 MTP evidence",
    )
    if evidence["assistant_checkpoint"] != ASSISTANT_CHECKPOINT_ACQUISITION:
        raise ValueError("Gemma4 MTP assistant evidence mismatch")
    if evidence["target_checkpoint"] != CHECKPOINT_ACQUISITION:
        raise ValueError("Gemma4 MTP target evidence mismatch")
    _validate_mtp_k2_evidence(evidence["k2_abi"])
    _validate_mtp_qat_evidence(evidence["qat_selection"])
    lowering = evidence["lowering"]
    if not isinstance(lowering, dict):
        raise ValueError("Gemma4 MTP lowering evidence must be an object")
    if lowering != {
        "delegate_count": 1,
        "edge": MTP_EDGE_CENSUS,
        "portable_operator_count": 0,
    }:
        raise ValueError("Gemma4 MTP lowering census mismatch")


def validate_mtp_manifest(root: Path, manifest: Mapping[str, object]) -> None:
    provenance = manifest.get("provenance")
    requires_source_receipt = False
    if provenance == MTP_ACCEPTED_PROVENANCE:
        expected_top_level = {
            "acquisition",
            "artifacts",
            "export",
            "model",
            "provenance",
            "ptd_order",
            "schema_version",
        }
    elif provenance in (
        MTP_PENDING_SOURCE_PROVENANCE,
        MTP_SOURCE_VERIFIED_PROVENANCE,
    ):
        expected_top_level = {
            "acquisition",
            "artifacts",
            "evidence",
            "export",
            "model",
            "provenance",
            "ptd_order",
            "schema_version",
        }
        _validate_mtp_evidence(manifest.get("evidence"))
        requires_source_receipt = provenance == MTP_SOURCE_VERIFIED_PROVENANCE
    else:
        raise ValueError("Gemma4 MTP provenance mismatch")
    _require_exact_keys(manifest, expected_top_level, "Gemma4 MTP manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Gemma4 MTP manifest artifacts must be a list")
    _require_unique_mtp_artifact_paths(artifacts)
    validate_manifest(root, manifest)
    if manifest.get("acquisition") != {
        "assistant": ASSISTANT_CHECKPOINT_ACQUISITION,
        "target": CHECKPOINT_ACQUISITION,
    }:
        raise ValueError("Gemma4 MTP checkpoint acquisition identity mismatch")
    if manifest.get("model") != {
        "architecture": ARCHITECTURE_FINGERPRINT,
        "assistant": ASSISTANT_MODEL_CONTRACT,
        "source_config": MTP_SOURCE_CONFIG,
    }:
        raise ValueError("Gemma4 MTP model/source-config identity mismatch")
    if manifest.get("export") != MTP_EXPORT_CONTRACT:
        raise ValueError("Gemma4 MTP export contract mismatch")
    ptd_order = manifest.get("ptd_order")
    assert isinstance(ptd_order, list)
    if len(ptd_order) != 3:
        raise ValueError("Gemma4 MTP manifest requires exactly three ordered PTDs")
    roles = [
        artifact.get("role") for artifact in artifacts if isinstance(artifact, dict)
    ]
    expected_source_count = 1 if requires_source_receipt else 0
    if (
        roles.count("pte") != 1
        or roles.count("ptd") != 3
        or roles.count("source") != expected_source_count
        or len(roles) != 4 + expected_source_count
    ):
        raise ValueError("Gemma4 MTP manifest has unexpected artifact roles")
    if requires_source_receipt:
        _validate_source_receipt(root, artifacts)
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        _require_exact_keys(
            artifact,
            {"bytes", "path", "role", "sha256"},
            "Gemma4 MTP artifact",
        )
        if (
            artifact.get("role") == "ptd"
            and int(artifact["bytes"]) >= WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES
        ):
            raise ValueError("Gemma4 MTP PTD exceeds the WebGPU constant limit")
    expected_paths = {
        str(artifact["path"]) for artifact in artifacts if isinstance(artifact, dict)
    }
    if len(expected_paths) != len(artifacts):
        raise ValueError("Gemma4 MTP artifact paths must be distinct")
    if any(len(Path(path).parts) != 1 for path in expected_paths):
        raise ValueError("Gemma4 MTP artifact staging directory must be flat")
    actual_paths = {entry.name for entry in root.iterdir()}
    if actual_paths != expected_paths:
        raise ValueError(
            "Gemma4 MTP artifact staging contains missing or extra entries"
        )


def _reject_pending_provenance(provenance: object, label: str) -> None:
    if not isinstance(provenance, dict):
        raise ValueError(f"Gemma4 {label} provenance must be an object")
    closure = provenance.get("source_closure")
    if not isinstance(closure, str):
        raise ValueError(f"Gemma4 {label} provenance has no source closure")
    if closure in MTP_PENDING_SOURCE_CLOSURES or closure.startswith("pending"):
        raise ValueError(
            f"Gemma4 {label} source closure is still pending ({closure}); a "
            "pending manifest is never source complete"
        )


def _validate_source_complete_mtp_manifest(
    root: Path, manifest: Mapping[str, object]
) -> None:
    validate_mtp_manifest(root, manifest)
    _reject_pending_provenance(manifest.get("provenance"), "MTP manifest")
    if manifest.get("provenance") != MTP_SOURCE_VERIFIED_PROVENANCE:
        raise ValueError(
            "Gemma4 combined runtime requires a source-verified MTP manifest"
        )
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, list)
    if not any(
        isinstance(artifact, dict) and artifact.get("role") == "source"
        for artifact in artifacts
    ):
        raise ValueError(
            "Gemma4 source-verified MTP manifest requires a hashed source receipt"
        )


def _contained_regular_file(root: Path, path: Path) -> tuple[Path, str]:
    resolved_root = root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise ValueError(f"runtime staging rejects symlink: {path}")
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"runtime artifact escapes staging root: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"runtime artifact is not a regular file: {path}")
    return resolved, relative.as_posix()


def _file_identity(root: Path, path: Path) -> dict[str, object]:
    resolved, relative = _contained_regular_file(root, path)
    return {
        "bytes": resolved.stat().st_size,
        "path": relative,
        "sha256": _sha256(resolved),
    }


def _validate_file_identity(root: Path, identity: object, label: str) -> None:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} identity must be an object")
    _require_exact_keys(identity, {"bytes", "path", "sha256"}, label)
    path = identity["path"]
    if not isinstance(path, str) or Path(path).is_absolute():
        raise ValueError(f"{label} path must be relative")
    if not isinstance(identity["bytes"], int) or identity["bytes"] <= 0:
        raise ValueError(f"{label} must not be empty")
    if not _is_hex_digest(identity["sha256"], 64):
        raise ValueError(f"{label} has an invalid SHA-256")
    observed = _file_identity(root, Path(path))
    if observed != identity:
        raise ValueError(f"{label} byte or SHA-256 identity mismatch")


def _runtime_artifact_identity(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"runtime artifact is not a regular file: {path}")
    byte_count = path.stat().st_size
    if byte_count <= 0:
        raise ValueError(f"runtime artifact must not be empty: {path}")
    return {"bytes": byte_count, "sha256": _sha256(path)}


def _validate_target_prefill_binding(
    target_prefill_receipt_path: Path,
    runtime_source_receipt_path: Path,
) -> None:
    runtime_source_receipt = _load_json(runtime_source_receipt_path)
    fbsource_commit = runtime_source_receipt.get("fbsource_commit")
    if not _is_hex_digest(fbsource_commit, 40):
        raise ValueError("Gemma4 runtime source receipt has an invalid fbsource commit")
    producer_path = reviewed_producer_source_path()
    producer_identity = target_prefill_file_identity(producer_path)
    validate_target_prefill_receipt(
        _load_json(target_prefill_receipt_path),
        expected_checkpoint_acquisition=CHECKPOINT_ACQUISITION,
        expected_producer_path=producer_path,
        expected_producer_sha256=str(producer_identity["sha256"]),
        expected_runtime_source_identity=_runtime_artifact_identity(
            runtime_source_receipt_path
        ),
        expected_fbsource_commit=str(fbsource_commit),
    )


def _closure_identity(path: Path, relative_path: str) -> dict[str, object]:
    return {"path": relative_path, **_runtime_artifact_identity(path)}


def _identity_bytes_and_hash(identity: Mapping[str, object]) -> tuple[object, object]:
    return identity.get("bytes"), identity.get("sha256")


def _runtime_role_identity(
    build: Mapping[str, object], model: str, role: str
) -> Mapping[str, object]:
    identity = build.get(role)
    if not isinstance(identity, dict):
        raise ValueError(f"Gemma4 {model} {role} identity must be an object")
    return identity


def _reject_aliased_runtime_roles(runtime: Mapping[str, object]) -> None:
    for model in ("mtp", "plain"):
        builds = runtime.get(model)
        if not isinstance(builds, dict):
            raise ValueError(f"Gemma4 {model} runtime build receipt must be an object")
        wall = builds.get("wall")
        profile = builds.get("profile")
        if not isinstance(wall, dict) or not isinstance(profile, dict):
            raise ValueError(
                f"Gemma4 {model} wall/profile build receipts are incomplete"
            )
        wall_javascript = _runtime_role_identity(wall, model, "javascript")
        profile_javascript = _runtime_role_identity(profile, model, "javascript")
        wall_wasm = _runtime_role_identity(wall, model, "wasm")
        profile_wasm = _runtime_role_identity(profile, model, "wasm")
        wall_pair = (
            _identity_bytes_and_hash(wall_javascript),
            _identity_bytes_and_hash(wall_wasm),
        )
        profile_pair = (
            _identity_bytes_and_hash(profile_javascript),
            _identity_bytes_and_hash(profile_wasm),
        )
        if wall_pair == profile_pair:
            raise ValueError(f"Gemma4 {model} wall/profile pair identities must differ")
        if _identity_bytes_and_hash(wall_wasm) == _identity_bytes_and_hash(
            profile_wasm
        ):
            raise ValueError(f"Gemma4 {model} wall/profile wasm identities must differ")
        for role in ("recipe",):
            wall_identity = _runtime_role_identity(wall, model, role)
            profile_identity = _runtime_role_identity(profile, model, role)
            if _identity_bytes_and_hash(wall_identity) == _identity_bytes_and_hash(
                profile_identity
            ):
                raise ValueError(
                    f"Gemma4 {model} wall/profile {role} identities must differ"
                )
        if model == "mtp" and _identity_bytes_and_hash(
            wall_javascript
        ) == _identity_bytes_and_hash(profile_javascript):
            raise ValueError(
                "Gemma4 MTP wall/profile javascript identities must differ"
            )


def _validate_runtime_product_basenames(
    runtime_paths: Mapping[str, Mapping[str, Mapping[str, Path]]],
) -> None:
    for model, builds in runtime_paths.items():
        for flavor, paths in builds.items():
            output_stem = RUNTIME_OUTPUT_STEMS[model][flavor]
            for kind, suffix in (("javascript", "js"), ("wasm", "wasm")):
                if paths[kind].name != f"{output_stem}.{suffix}":
                    display_kind = "JavaScript" if kind == "javascript" else "WASM"
                    raise ValueError(
                        f"Gemma4 {model} {flavor} {display_kind} basename mismatch"
                    )


def _validate_runtime_source_receipt(
    receipt: Mapping[str, object],
    runtime_paths: Mapping[str, Mapping[str, Mapping[str, Path]]],
    manifest_paths: Mapping[str, Path],
    model_roots: Mapping[str, Path],
    source_input_paths: Mapping[str, Path],
    build_recipe_paths: Mapping[str, Mapping[str, Path]],
) -> None:
    _require_exact_keys(
        receipt,
        {
            "fbsource_commit",
            "model_manifests",
            "oss_commit",
            "runtime",
            "schema_version",
            "source_inputs",
            "source_current",
            "verification",
        },
        "Gemma4 runtime source receipt",
    )
    if receipt.get("schema_version") != RUNTIME_SOURCE_SCHEMA_VERSION:
        raise ValueError("Gemma4 runtime source receipt requires schema version 4")
    if receipt.get("source_current") is not True:
        raise ValueError("Gemma4 runtime source receipt is not source-current")
    if receipt.get("verification") != {
        "build_execution": "not_attested",
        "recipe": "validated",
        "source_checkout": "verified",
        "wgsl_codegen": "verified",
    }:
        raise ValueError("Gemma4 runtime source verification claims are invalid")
    source_inputs = receipt["source_inputs"]
    if not isinstance(source_inputs, dict):
        raise ValueError("Gemma4 runtime source inputs must be an object")
    _require_exact_keys(
        source_inputs, set(CLOSURE_SOURCE_PATHS), "Gemma4 runtime source inputs"
    )
    if set(source_input_paths) != set(CLOSURE_SOURCE_PATHS):
        raise ValueError("Gemma4 runtime source-input paths are incomplete")
    for label, relative_path in CLOSURE_SOURCE_PATHS.items():
        expected = source_inputs[label]
        if not isinstance(expected, dict):
            raise ValueError(f"Gemma4 {label} identity must be an object")
        _require_exact_keys(
            expected, {"bytes", "path", "sha256"}, f"Gemma4 {label} identity"
        )
        if expected != _closure_identity(source_input_paths[label], relative_path):
            raise ValueError(f"Gemma4 {label} is not bound to its source receipt")
    source_manifest = _load_json(source_input_paths["source_manifest"])
    wgsl_manifest = _load_json(source_input_paths["wgsl_manifest"])
    try:
        validate_source_manifest(source_manifest)
    except ValueError as error:
        raise ValueError(f"Gemma4 source manifest is invalid: {error}") from error
    try:
        validate_wgsl_manifest(wgsl_manifest)
    except ValueError as error:
        raise ValueError(f"Gemma4 WGSL manifest is invalid: {error}") from error
    checkouts = source_manifest["checkouts"]
    assert isinstance(checkouts, dict)
    fbsource_checkout = checkouts["fbsource"]
    assert isinstance(fbsource_checkout, dict)
    if wgsl_manifest.get("fbsource_commit") != fbsource_checkout.get("head"):
        raise ValueError("Gemma4 source and WGSL manifests have different heads")
    for label, key in (("fbsource", "fbsource_commit"), ("oss", "oss_commit")):
        checkout = checkouts[label]
        assert isinstance(checkout, dict)
        if receipt.get(key) != checkout.get("head"):
            raise ValueError(f"Gemma4 runtime source receipt has invalid {key}")
    model_manifests = receipt["model_manifests"]
    if not isinstance(model_manifests, dict):
        raise ValueError("Gemma4 model-manifest bindings must be an object")
    _require_exact_keys(
        model_manifests, {"mtp", "plain"}, "Gemma4 model-manifest bindings"
    )
    if set(manifest_paths) != {"mtp", "plain"}:
        raise ValueError("Gemma4 model-manifest paths are incomplete")
    if set(model_roots) != {"mtp", "plain"}:
        raise ValueError("Gemma4 model artifact roots are incomplete")
    for label in ("plain", "mtp"):
        expected = model_manifests[label]
        if not isinstance(expected, dict):
            raise ValueError(f"Gemma4 {label} manifest identity must be an object")
        _require_exact_keys(
            expected, {"bytes", "sha256"}, f"Gemma4 {label} manifest identity"
        )
        if expected != _runtime_artifact_identity(manifest_paths[label]):
            raise ValueError(
                f"Gemma4 {label} manifest is not bound to its source receipt"
            )
        manifest = _load_json(manifest_paths[label])
        if label == "plain":
            validate_plain_manifest(model_roots[label], manifest)
        else:
            _validate_source_complete_mtp_manifest(model_roots[label], manifest)
        model_source = _model_source_receipt(model_roots[label], manifest, label)
        if model_source.get("fbsource_commit") != receipt.get(
            "fbsource_commit"
        ) or model_source.get("oss_commit") != receipt.get("oss_commit"):
            raise ValueError(f"Gemma4 {label} source receipt checkout head mismatch")
        if (
            model_source.get("source_manifest") != source_manifest
            or model_source.get("wgsl_manifest") != wgsl_manifest
        ):
            raise ValueError(f"Gemma4 {label} source receipt closure mismatch")
    runtime = receipt["runtime"]
    if not isinstance(runtime, dict):
        raise ValueError("Gemma4 runtime source receipt runtime must be an object")
    _require_exact_keys(runtime, set(RUNTIME_BUILD_TARGETS), "Gemma4 runtime builds")
    if set(runtime_paths) != set(RUNTIME_BUILD_TARGETS):
        raise ValueError("Gemma4 runtime paths are incomplete")
    if set(build_recipe_paths) != set(RUNTIME_BUILD_TARGETS):
        raise ValueError("Gemma4 build-recipe paths are incomplete")
    for model, target in RUNTIME_BUILD_TARGETS.items():
        builds = runtime[model]
        if not isinstance(builds, dict):
            raise ValueError(f"Gemma4 {model} build receipt must be an object")
        flavors = RUNTIME_PROFILING_MODES[model]
        _require_exact_keys(
            builds,
            {"target", *flavors},
            f"Gemma4 {model} build receipt",
        )
        if builds.get("target") != target:
            raise ValueError(f"Gemma4 {model} runtime build target mismatch")
        model_paths = runtime_paths[model]
        model_recipe_paths = build_recipe_paths[model]
        if set(model_paths) != set(flavors):
            raise ValueError(f"Gemma4 {model} runtime paths are incomplete")
        if set(model_recipe_paths) != set(flavors):
            raise ValueError(f"Gemma4 {model} build-recipe paths are incomplete")
        for flavor, profiling_enabled in flavors.items():
            build = builds[flavor]
            if not isinstance(build, dict):
                raise ValueError(
                    f"Gemma4 {model} {flavor} build receipt must be an object"
                )
            _require_exact_keys(
                build,
                {
                    "factory",
                    "javascript",
                    "output_stem",
                    "profiling_enabled",
                    "recipe",
                    "wasm",
                },
                f"Gemma4 {model} {flavor} build receipt",
            )
            if build["profiling_enabled"] is not profiling_enabled:
                raise ValueError(f"Gemma4 {model} {flavor} profiling mode mismatch")
            recipe = build["recipe"]
            if not isinstance(recipe, dict):
                raise ValueError(
                    f"Gemma4 {model} {flavor} recipe identity must be an object"
                )
            _require_exact_keys(
                recipe,
                {"bytes", "path", "sha256"},
                f"Gemma4 {model} {flavor} recipe identity",
            )
            if recipe != _closure_identity(
                model_recipe_paths[flavor], CLOSURE_RECIPE_PATHS[model][flavor]
            ):
                raise ValueError(
                    f"Gemma4 {model} {flavor} recipe is not bound to its build receipt"
                )
            recipe_document = _validate_build_recipe(
                model_recipe_paths[flavor], model, flavor
            )
            if build["factory"] != recipe_document["factory"]:
                raise ValueError(f"Gemma4 {model} {flavor} factory mismatch")
            if build["output_stem"] != recipe_document["output_stem"]:
                raise ValueError(f"Gemma4 {model} {flavor} output_stem mismatch")
            paths = model_paths[flavor]
            if set(paths) != {"javascript", "wasm"}:
                raise ValueError(
                    f"Gemma4 {model} {flavor} runtime paths are incomplete"
                )
            for kind in ("javascript", "wasm"):
                expected = build[kind]
                if not isinstance(expected, dict):
                    raise ValueError(
                        f"Gemma4 {model} {flavor} {kind} identity must be an object"
                    )
                _require_exact_keys(
                    expected,
                    {"bytes", "sha256"},
                    f"Gemma4 {model} {flavor} {kind} identity",
                )
                if expected != _runtime_artifact_identity(paths[kind]):
                    raise ValueError(
                        f"Gemma4 {model} {flavor} {kind} is not bound to its build receipt"
                    )
    _reject_aliased_runtime_roles(runtime)


def _runtime_paths(root: Path) -> dict[str, dict[str, dict[str, Path]]]:
    return {
        model: {
            flavor: {
                "javascript": root / f"runtime/{model}/{flavor}.js",
                "wasm": root / f"runtime/{model}/{flavor}.wasm",
            }
            for flavor in flavors
        }
        for model, flavors in {
            "mtp": ("profile", "wall"),
            "plain": ("profile", "wall"),
        }.items()
    }


def _closure_source_paths(root: Path) -> dict[str, Path]:
    return {label: root / path for label, path in CLOSURE_SOURCE_PATHS.items()}


def _closure_recipe_paths(root: Path) -> dict[str, dict[str, Path]]:
    return {
        model: {flavor: root / path for flavor, path in recipes.items()}
        for model, recipes in CLOSURE_RECIPE_PATHS.items()
    }


def create_runtime_source_receipt(
    *,
    fbsource_root: Path,
    oss_root: Path,
    backend_root: Path,
    source_manifest_path: Path,
    wgsl_manifest_path: Path,
    manifest_paths: Mapping[str, Path],
    model_roots: Mapping[str, Path],
    runtime_paths: Mapping[str, Mapping[str, Mapping[str, Path]]],
    build_command_paths: Mapping[str, Mapping[str, Path]],
) -> dict[str, object]:
    source_manifest = _load_json(source_manifest_path)
    wgsl_manifest = _load_json(wgsl_manifest_path)
    try:
        validate_source_manifest(source_manifest)
    except ValueError as error:
        raise ValueError(f"Gemma4 source manifest is invalid: {error}") from error
    try:
        validate_wgsl_manifest(wgsl_manifest)
    except ValueError as error:
        raise ValueError(f"Gemma4 WGSL manifest is invalid: {error}") from error
    live_source_manifest = create_source_manifest(fbsource_root, oss_root)
    if source_manifest != live_source_manifest:
        raise ValueError(
            "Gemma4 source manifest does not match the live clean checkouts"
        )
    live_wgsl_manifest = create_wgsl_manifest(backend_root)
    if wgsl_manifest != live_wgsl_manifest:
        raise ValueError(
            "Gemma4 WGSL manifest does not match the live generator closure"
        )
    checkouts = source_manifest["checkouts"]
    assert isinstance(checkouts, dict)
    fbsource_checkout = checkouts["fbsource"]
    oss_checkout = checkouts["oss"]
    assert isinstance(fbsource_checkout, dict) and isinstance(oss_checkout, dict)
    fbsource_commit = str(fbsource_checkout["head"])
    oss_commit = str(oss_checkout["head"])
    _require_exact_keys(manifest_paths, {"mtp", "plain"}, "Gemma4 model-manifest paths")
    _require_exact_keys(model_roots, {"mtp", "plain"}, "Gemma4 model artifact roots")
    _require_exact_keys(
        runtime_paths, set(RUNTIME_BUILD_TARGETS), "Gemma4 runtime paths"
    )
    _require_exact_keys(
        build_command_paths,
        set(RUNTIME_BUILD_TARGETS),
        "Gemma4 build-command paths",
    )
    for model, recipes in build_command_paths.items():
        if _identity_bytes_and_hash(
            _runtime_artifact_identity(recipes["wall"])
        ) == _identity_bytes_and_hash(_runtime_artifact_identity(recipes["profile"])):
            raise ValueError(
                f"Gemma4 {model} wall/profile recipe identities must differ"
            )

    runtime: dict[str, object] = {}
    for model, target in RUNTIME_BUILD_TARGETS.items():
        expected_flavors = RUNTIME_PROFILING_MODES[model]
        model_runtime_paths = runtime_paths[model]
        model_command_paths = build_command_paths[model]
        _require_exact_keys(
            model_runtime_paths,
            set(expected_flavors),
            f"Gemma4 {model} runtime paths",
        )
        _require_exact_keys(
            model_command_paths,
            set(expected_flavors),
            f"Gemma4 {model} build-command paths",
        )
        builds: dict[str, object] = {"target": target}
        for flavor, profiling_enabled in expected_flavors.items():
            _validate_build_recipe(model_command_paths[flavor], model, flavor)
            artifacts = model_runtime_paths[flavor]
            _require_exact_keys(
                artifacts,
                {"javascript", "wasm"},
                f"Gemma4 {model} {flavor} runtime paths",
            )
            builds[flavor] = {
                "factory": RUNTIME_FACTORY_NAMES[model][flavor],
                "javascript": _runtime_artifact_identity(artifacts["javascript"]),
                "output_stem": RUNTIME_OUTPUT_STEMS[model][flavor],
                "profiling_enabled": profiling_enabled,
                "recipe": _closure_identity(
                    model_command_paths[flavor],
                    CLOSURE_RECIPE_PATHS[model][flavor],
                ),
                "wasm": _runtime_artifact_identity(artifacts["wasm"]),
            }
        runtime[model] = builds

    source_input_paths = {
        "source_manifest": source_manifest_path,
        "wgsl_manifest": wgsl_manifest_path,
    }
    receipt: dict[str, object] = {
        "fbsource_commit": fbsource_commit,
        "model_manifests": {
            label: _runtime_artifact_identity(path)
            for label, path in manifest_paths.items()
        },
        "oss_commit": oss_commit,
        "runtime": runtime,
        "schema_version": RUNTIME_SOURCE_SCHEMA_VERSION,
        "source_inputs": {
            label: _closure_identity(source_input_paths[label], relative_path)
            for label, relative_path in CLOSURE_SOURCE_PATHS.items()
        },
        "source_current": True,
        "verification": {
            "build_execution": "not_attested",
            "recipe": "validated",
            "source_checkout": "verified",
            "wgsl_codegen": "verified",
        },
    }
    _validate_runtime_source_receipt(
        receipt,
        runtime_paths,
        manifest_paths,
        model_roots,
        source_input_paths,
        build_command_paths,
    )
    _validate_runtime_product_basenames(runtime_paths)
    return receipt


def _staged_regular_files(root: Path) -> set[str]:
    files: set[str] = set()
    for directory, directories, filenames in os.walk(root):
        current = Path(directory)
        for name in directories:
            path = current / name
            if path.is_symlink():
                raise ValueError(f"runtime staging rejects symlink: {path}")
        for name in filenames:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"runtime staging rejects non-regular file: {path}")
            files.add(path.relative_to(root).as_posix())
    return files


def _source_receipt_identity(
    root: Path, manifest: Mapping[str, object], label: str
) -> dict[str, object]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"Gemma4 {label} manifest has no artifacts")
    entries = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("role") == "source"
    ]
    if len(entries) != 1:
        raise ValueError(f"Gemma4 {label} manifest requires one source receipt")
    entry = entries[0]
    return {key: entry[key] for key in ("bytes", "path", "sha256")}


def _source_verification(
    root: Path,
    plain_manifest: Mapping[str, object],
    mtp_manifest: Mapping[str, object],
) -> dict[str, object]:
    verification: dict[str, object] = {}
    for label, manifest in (("mtp", mtp_manifest), ("plain", plain_manifest)):
        provenance = manifest.get("provenance")
        if label == "mtp":
            _reject_pending_provenance(provenance, "MTP manifest")
        verification[label] = {
            "provenance": copy.deepcopy(provenance) if provenance else None,
            "source_receipt": _source_receipt_identity(root / label, manifest, label),
        }
    return verification


def create_combined_runtime_envelope(root: Path) -> dict[str, object]:
    plain_receipt_path = Path("receipts/plain.json")
    mtp_receipt_path = Path("receipts/mtp.json")
    runtime_source_path = Path("receipts/runtime_source.json")
    target_prefill_path = Path("receipts/target_prefill.json")

    plain_receipt = _load_json(root / plain_receipt_path)
    mtp_receipt = _load_json(root / mtp_receipt_path)
    runtime_source_receipt = _load_json(root / runtime_source_path)
    validate_plain_manifest(root / "plain", plain_receipt)
    _validate_source_complete_mtp_manifest(root / "mtp", mtp_receipt)
    _validate_runtime_source_receipt(
        runtime_source_receipt,
        _runtime_paths(root),
        {"mtp": root / mtp_receipt_path, "plain": root / plain_receipt_path},
        {"mtp": root / "mtp", "plain": root / "plain"},
        _closure_source_paths(root),
        _closure_recipe_paths(root),
    )
    _validate_target_prefill_binding(
        root / target_prefill_path, root / runtime_source_path
    )
    runtime = {
        model: {
            flavor: {
                kind: _file_identity(root, path) for kind, path in artifacts.items()
            }
            for flavor, artifacts in builds.items()
        }
        for model, builds in _runtime_paths(root).items()
    }

    envelope: dict[str, object] = {
        "contract": COMBINED_RUNTIME_CONTRACT,
        "receipts": {
            "mtp": {
                **_file_identity(root, mtp_receipt_path),
                "root": "mtp",
            },
            "plain": {
                **_file_identity(root, plain_receipt_path),
                "root": "plain",
            },
            "target_prefill": _file_identity(root, target_prefill_path),
        },
        "runtime": {**runtime, "source": _file_identity(root, runtime_source_path)},
        "schema_version": _COMBINED_RUNTIME_SCHEMA_VERSION,
        "source_verification": _source_verification(root, plain_receipt, mtp_receipt),
        "views": COMBINED_RUNTIME_VIEWS,
    }
    return envelope


def validate_combined_runtime_envelope(
    root: Path, envelope: Mapping[str, object]
) -> None:
    _require_exact_keys(
        envelope,
        {
            "contract",
            "receipts",
            "runtime",
            "schema_version",
            "source_verification",
            "views",
        },
        "Gemma4 combined runtime envelope",
    )
    if envelope.get("schema_version") != _COMBINED_RUNTIME_SCHEMA_VERSION:
        raise ValueError("Gemma4 combined runtime schema version mismatch")
    if envelope.get("contract") != COMBINED_RUNTIME_CONTRACT:
        raise ValueError("Gemma4 combined runtime contract mismatch")
    if envelope.get("views") != COMBINED_RUNTIME_VIEWS:
        raise ValueError("Gemma4 combined runtime views mismatch")
    source_verification = envelope.get("source_verification")
    if not isinstance(source_verification, dict):
        raise ValueError("Gemma4 combined source verification must be an object")
    _require_exact_keys(
        source_verification, {"mtp", "plain"}, "Gemma4 combined source verification"
    )
    for label in ("mtp", "plain"):
        entry = source_verification[label]
        if not isinstance(entry, dict):
            raise ValueError(f"Gemma4 {label} source verification must be an object")
        _require_exact_keys(
            entry,
            {"provenance", "source_receipt"},
            f"Gemma4 {label} source verification",
        )
        receipt = entry["source_receipt"]
        if not isinstance(receipt, dict):
            raise ValueError(f"Gemma4 {label} source receipt must be an object")
        _require_exact_keys(
            receipt, {"bytes", "path", "sha256"}, f"Gemma4 {label} source receipt"
        )
        _validate_file_identity(root / label, receipt, f"Gemma4 {label} source receipt")
    _reject_pending_provenance(
        source_verification["mtp"].get("provenance"), "MTP manifest"
    )
    if source_verification["mtp"].get("provenance") != MTP_SOURCE_VERIFIED_PROVENANCE:
        raise ValueError("Gemma4 combined runtime requires source-verified MTP")

    receipts = envelope.get("receipts")
    if not isinstance(receipts, dict):
        raise ValueError("Gemma4 combined receipts must be an object")
    _require_exact_keys(
        receipts,
        {"mtp", "plain", "target_prefill"},
        "Gemma4 combined receipts",
    )
    expected_paths = {
        "receipts/mtp.json",
        "receipts/plain.json",
        "receipts/runtime_source.json",
        "receipts/target_prefill.json",
    }
    expected_paths.update(
        path.relative_to(root).as_posix()
        for builds in _runtime_paths(root).values()
        for artifacts in builds.values()
        for path in artifacts.values()
    )
    expected_paths.update(CLOSURE_SOURCE_PATHS.values())
    expected_paths.update(
        path for recipes in CLOSURE_RECIPE_PATHS.values() for path in recipes.values()
    )
    for label in ("plain", "mtp"):
        receipt = receipts[label]
        if not isinstance(receipt, dict):
            raise ValueError(f"Gemma4 {label} receipt must be an object")
        _require_exact_keys(
            receipt, {"bytes", "path", "root", "sha256"}, f"Gemma4 {label} receipt"
        )
        receipt_identity = {key: receipt[key] for key in ("bytes", "path", "sha256")}
        _validate_file_identity(root, receipt_identity, f"Gemma4 {label} receipt")
        if receipt.get("root") != label:
            raise ValueError(f"Gemma4 {label} receipt root mismatch")
        manifest = _load_json(root / str(receipt["path"]))
        if label == "plain":
            validate_plain_manifest(root / label, manifest)
        else:
            _validate_source_complete_mtp_manifest(root / label, manifest)
        artifacts = manifest.get("artifacts")
        assert isinstance(artifacts, list)
        expected_paths.update(
            f"{label}/{artifact['path']}"
            for artifact in artifacts
            if isinstance(artifact, dict)
        )

    target_prefill = receipts["target_prefill"]
    if not isinstance(target_prefill, dict):
        raise ValueError("Gemma4 target-prefill receipt must be an object")
    _require_exact_keys(
        target_prefill,
        {"bytes", "path", "sha256"},
        "Gemma4 target-prefill receipt",
    )
    if target_prefill.get("path") != "receipts/target_prefill.json":
        raise ValueError("Gemma4 target-prefill receipt path mismatch")
    _validate_file_identity(root, target_prefill, "Gemma4 target-prefill receipt")

    runtime = envelope.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("Gemma4 combined runtime must be an object")
    _require_exact_keys(runtime, {"mtp", "plain", "source"}, "Gemma4 combined runtime")
    _validate_file_identity(root, runtime["source"], "Gemma4 runtime source")
    source_identity = runtime["source"]
    assert isinstance(source_identity, dict)
    _validate_runtime_source_receipt(
        _load_json(root / str(source_identity["path"])),
        _runtime_paths(root),
        {
            "mtp": root / "receipts/mtp.json",
            "plain": root / "receipts/plain.json",
        },
        {"mtp": root / "mtp", "plain": root / "plain"},
        _closure_source_paths(root),
        _closure_recipe_paths(root),
    )
    _validate_target_prefill_binding(
        root / "receipts/target_prefill.json",
        root / str(source_identity["path"]),
    )
    for model, expected_builds in _runtime_paths(root).items():
        model_runtime = runtime[model]
        if not isinstance(model_runtime, dict):
            raise ValueError(f"Gemma4 {model} runtime must be an object")
        _require_exact_keys(
            model_runtime, set(expected_builds), f"Gemma4 {model} runtime"
        )
        for flavor in expected_builds:
            artifacts = model_runtime[flavor]
            if not isinstance(artifacts, dict):
                raise ValueError(f"Gemma4 {model} {flavor} runtime must be an object")
            _require_exact_keys(
                artifacts,
                {"javascript", "wasm"},
                f"Gemma4 {model} {flavor} runtime",
            )
            _validate_file_identity(
                root,
                artifacts["javascript"],
                f"Gemma4 {model} {flavor} JavaScript",
            )
            _validate_file_identity(
                root, artifacts["wasm"], f"Gemma4 {model} {flavor} WASM"
            )
    extra_paths = (
        _staged_regular_files(root)
        - expected_paths
        - {"gemma4_webgpu_combined_runtime.json"}
    )
    if extra_paths:
        raise ValueError(
            f"Gemma4 combined runtime contains extra files: {sorted(extra_paths)}"
        )
    if envelope != create_combined_runtime_envelope(root):
        raise ValueError("Gemma4 combined runtime has non-canonical role bindings")


def _copy_manifest_artifacts(
    source_root: Path,
    manifest: Mapping[str, object],
    destination_root: Path,
) -> None:
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, list)
    destination_root.mkdir()
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        path = Path(str(artifact["path"]))
        source, relative = _contained_regular_file(source_root, path)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def stage_combined_runtime(
    destination_root: Path,
    plain_root: Path,
    plain_receipt_path: Path,
    mtp_root: Path,
    mtp_receipt_path: Path,
    runtime_source_receipt_path: Path,
    mtp_wall_javascript_path: Path,
    mtp_wall_wasm_path: Path,
    mtp_profile_javascript_path: Path,
    mtp_profile_wasm_path: Path,
    *,
    plain_profile_javascript_path: Path,
    plain_profile_wasm_path: Path,
    plain_wall_javascript_path: Path,
    plain_wall_wasm_path: Path,
    source_manifest_path: Path,
    wgsl_manifest_path: Path,
    build_recipe_paths: Mapping[str, Mapping[str, Path]],
    target_prefill_receipt_path: Path,
) -> None:
    if destination_root.exists() or destination_root.is_symlink():
        raise ValueError(
            f"runtime staging destination already exists: {destination_root}"
        )
    destination_root.parent.mkdir(parents=True, exist_ok=True)

    for receipt_path in (
        plain_receipt_path,
        mtp_receipt_path,
        runtime_source_receipt_path,
        target_prefill_receipt_path,
    ):
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError(
                f"runtime staging receipt is not a regular file: {receipt_path}"
            )
    plain_receipt = _load_json(plain_receipt_path)
    mtp_receipt = _load_json(mtp_receipt_path)
    validate_plain_manifest(plain_root, plain_receipt)
    _validate_source_complete_mtp_manifest(mtp_root, mtp_receipt)
    runtime_inputs = {
        "mtp": {
            "profile": {
                "javascript": mtp_profile_javascript_path,
                "wasm": mtp_profile_wasm_path,
            },
            "wall": {
                "javascript": mtp_wall_javascript_path,
                "wasm": mtp_wall_wasm_path,
            },
        },
        "plain": {
            "profile": {
                "javascript": plain_profile_javascript_path,
                "wasm": plain_profile_wasm_path,
            },
            "wall": {
                "javascript": plain_wall_javascript_path,
                "wasm": plain_wall_wasm_path,
            },
        },
    }
    source_inputs = {
        "source_manifest": source_manifest_path,
        "wgsl_manifest": wgsl_manifest_path,
    }
    for builds in runtime_inputs.values():
        for artifacts in builds.values():
            for path in artifacts.values():
                if path.is_symlink() or not path.is_file():
                    raise ValueError(
                        f"runtime staging input is not a regular file: {path}"
                    )
    closure_inputs = list(source_inputs.values())
    for builds in build_recipe_paths.values():
        closure_inputs.extend(builds.values())
    for path in closure_inputs:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"runtime closure input is not a regular file: {path}")
    if (
        runtime_source_receipt_path.is_symlink()
        or not runtime_source_receipt_path.is_file()
    ):
        raise ValueError(
            "runtime staging source receipt is not a regular file: "
            f"{runtime_source_receipt_path}"
        )
    _validate_runtime_source_receipt(
        _load_json(runtime_source_receipt_path),
        runtime_inputs,
        {"mtp": mtp_receipt_path, "plain": plain_receipt_path},
        {"mtp": mtp_root, "plain": plain_root},
        source_inputs,
        build_recipe_paths,
    )
    _validate_target_prefill_binding(
        target_prefill_receipt_path, runtime_source_receipt_path
    )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.", dir=destination_root.parent
        )
    )
    try:
        _copy_manifest_artifacts(plain_root, plain_receipt, temporary_root / "plain")
        _copy_manifest_artifacts(mtp_root, mtp_receipt, temporary_root / "mtp")
        receipts_root = temporary_root / "receipts"
        receipts_root.mkdir()
        shutil.copyfile(plain_receipt_path, receipts_root / "plain.json")
        shutil.copyfile(mtp_receipt_path, receipts_root / "mtp.json")
        shutil.copyfile(
            runtime_source_receipt_path, receipts_root / "runtime_source.json"
        )
        shutil.copyfile(
            target_prefill_receipt_path, receipts_root / "target_prefill.json"
        )
        for label, destination in CLOSURE_SOURCE_PATHS.items():
            staged_path = temporary_root / destination
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_inputs[label], staged_path)
        for model, recipes in CLOSURE_RECIPE_PATHS.items():
            for flavor, destination in recipes.items():
                staged_path = temporary_root / destination
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(build_recipe_paths[model][flavor], staged_path)
        for model, builds in runtime_inputs.items():
            runtime_root = temporary_root / "runtime" / model
            runtime_root.mkdir(parents=True)
            for flavor, artifacts in builds.items():
                for kind, source in artifacts.items():
                    suffix = "js" if kind == "javascript" else "wasm"
                    shutil.copyfile(source, runtime_root / f"{flavor}.{suffix}")

        envelope = create_combined_runtime_envelope(temporary_root)
        validate_combined_runtime_envelope(temporary_root, envelope)
        (temporary_root / "gemma4_webgpu_combined_runtime.json").write_text(
            json.dumps(envelope, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_root, destination_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


@dataclasses.dataclass(slots=True)
class _PublishedArtifact:
    destination: Path
    expected_device: int
    expected_inode: int
    owned: bool = False


def _path_identity(path: Path) -> tuple[int, int] | None:
    try:
        observed = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    return observed.st_dev, observed.st_ino


def _quarantine_owned_path(
    path: Path,
    expected_identity: tuple[int, int],
    quarantine_parent: Path,
) -> bool:
    quarantine_root = Path(
        tempfile.mkdtemp(prefix=".mtp-publication-quarantine.", dir=quarantine_parent)
    )
    quarantined = quarantine_root / path.name
    try:
        try:
            os.rename(path, quarantined)
        except FileNotFoundError:
            return False

        observed_identity = _path_identity(quarantined)
        if observed_identity == expected_identity:
            quarantined.unlink()
            return True
        if observed_identity is None:
            return False

        try:
            os.link(quarantined, path, follow_symlinks=False)
        except OSError as error:
            raise RuntimeError(
                f"foreign publication entry retained for recovery at {quarantined}"
            ) from error
        if _path_identity(path) != observed_identity:
            raise RuntimeError(
                f"foreign publication entry retained for recovery at {quarantined}"
            )
        quarantined.unlink()
        return False
    finally:
        try:
            quarantine_root.rmdir()
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise


def _link_and_claim(
    staged: Path,
    destination: Path,
    publication: _PublishedArtifact,
) -> None:
    """Defer SIGINT through a normal-return transition to owned.

    SIGKILL or another exception with uncertain syscall completion can leave a
    pre-receipt partial file. The final receipt is the commit witness, and a
    later no-clobber publication attempt fails closed on that partial.
    """
    if threading.current_thread() is not threading.main_thread():
        os.link(staged, destination, follow_symlinks=False)
        publication.owned = True
        return

    received_sigint = False

    def defer_sigint(_signum: int, _frame: FrameType | None) -> None:
        nonlocal received_sigint
        received_sigint = True

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, defer_sigint)
    try:
        os.link(staged, destination, follow_symlinks=False)
        # Keep this as the only operation between a normal link return and OWNED.
        publication.owned = True
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        if received_sigint:
            signal.raise_signal(signal.SIGINT)


def _publish_no_clobber(
    staged: Path,
    destination: Path,
    published: list[_PublishedArtifact],
) -> None:
    source_stat = staged.stat(follow_symlinks=False)
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError(f"staged publication is not regular: {staged}")
    if destination.parent.stat(follow_symlinks=False).st_dev != source_stat.st_dev:
        raise ValueError("staged and final artifacts must use the same filesystem")

    expected_identity = (source_stat.st_dev, source_stat.st_ino)
    publication = _PublishedArtifact(destination, *expected_identity)
    published.append(publication)
    try:
        _link_and_claim(staged, destination, publication)
    except FileExistsError as error:
        raise ValueError(
            f"refusing to overwrite existing artifact: {destination}"
        ) from error
    except OSError as error:
        if error.errno == errno.EXDEV:
            raise ValueError(
                "staged and final artifacts must use the same filesystem"
            ) from error
        raise
    if _path_identity(destination) != expected_identity:
        publication.owned = False
        raise ValueError(f"published artifact ownership changed: {destination}")
    if not _quarantine_owned_path(staged, expected_identity, destination.parent):
        raise ValueError(f"staged artifact ownership changed: {staged}")
    if _path_identity(destination) != expected_identity:
        publication.owned = False
        raise ValueError(f"published artifact ownership changed: {destination}")


def _rollback_publications(
    published: Sequence[_PublishedArtifact],
) -> list[tuple[Path, BaseException]]:
    failures: list[tuple[Path, BaseException]] = []
    for publication in reversed(published):
        if not publication.owned:
            continue
        try:
            _quarantine_owned_path(
                publication.destination,
                (publication.expected_device, publication.expected_inode),
                publication.destination.parent,
            )
        except BaseException as error:
            failures.append((publication.destination, error))
    return failures


def _annotate_rollback_failures(
    original_error: BaseException,
    failures: Sequence[tuple[Path, BaseException]],
) -> None:
    for destination, cleanup_error in failures:
        original_error.add_note(
            f"rollback cleanup failed for {destination}: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )


def finalize_mtp_export(
    staging_root: Path,
    output_path: Path,
    receipt_path: Path,
    staged_pte: Path,
    staged_ptds: Sequence[Path],
    source_receipt_path: Path | None,
    evidence: Mapping[str, object],
) -> Path:
    if staged_pte.name != output_path.name:
        raise ValueError("Gemma4 MTP staged and final PTE names must match")
    if len(staged_ptds) != 3:
        raise ValueError("Gemma4 MTP export requires exactly three staged PTDs")
    staged_artifacts = [staged_pte, *staged_ptds]
    artifact_names = [path.name for path in staged_artifacts]
    _require_unique_mtp_artifact_paths([{"path": name} for name in artifact_names])
    for path in staged_artifacts:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Gemma4 MTP staged artifact is not regular: {path}")

    role_paths: dict[str, Path] = {"pte": staged_pte}
    staged_source: Path | None = None
    if source_receipt_path is not None:
        if source_receipt_path.is_symlink() or not source_receipt_path.is_file():
            raise ValueError(
                "Gemma4 MTP source receipt must be a regular non-symlink file"
            )
        _require_unique_mtp_artifact_paths(
            [
                *({"path": name} for name in artifact_names),
                {"path": source_receipt_path.name},
            ]
        )
        staged_source = staging_root / source_receipt_path.name
        if staged_source.exists() or staged_source.is_symlink():
            raise ValueError(
                "Gemma4 MTP duplicate normalized artifact path: "
                f"{source_receipt_path.name}"
            )
        shutil.copyfile(source_receipt_path, staged_source)
        role_paths["source"] = staged_source

    receipt = create_mtp_manifest(staging_root, role_paths, staged_ptds)
    receipt["evidence"] = copy.deepcopy(evidence)
    validate_mtp_manifest(staging_root, receipt)

    artifact_root = output_path.parent
    artifact_root.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    publications = [(path, artifact_root / path.name) for path in staged_ptds]
    if staged_source is not None:
        publications.append((staged_source, artifact_root / staged_source.name))
    publications.append((staged_pte, output_path))

    with tempfile.TemporaryDirectory(
        prefix=f".{receipt_path.stem}.", dir=receipt_path.parent
    ) as receipt_staging_directory:
        staged_receipt = Path(receipt_staging_directory) / receipt_path.name
        staged_receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        published: list[_PublishedArtifact] = []
        try:
            for staged, destination in publications:
                _publish_no_clobber(staged, destination, published)
            validate_mtp_manifest(artifact_root, receipt)
            _publish_no_clobber(staged_receipt, receipt_path, published)
            validate_mtp_manifest(artifact_root, _load_json(receipt_path))
        except BaseException as error:
            _annotate_rollback_failures(error, _rollback_publications(published))
            raise
    return receipt_path


def _role_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or not role or not path or role in result:
            raise ValueError(f"invalid or duplicate ROLE=PATH: {value}")
        result[role] = Path(path)
    return result


def _write_json(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _handle_closure_creation(args: argparse.Namespace) -> bool:
    if args.command == "create-source-manifest":
        _write_json(
            args.output,
            create_source_manifest(args.fbsource_root, args.oss_root),
        )
        return True
    if args.command == "create-wgsl-manifest":
        _write_json(args.output, create_wgsl_manifest(args.backend_root))
        return True
    if args.command == "create-build-recipe":
        _write_json(args.output, canonical_build_recipe(args.model, args.flavor))
        return True
    if args.command == "create-source-receipt":
        _write_json(
            args.output,
            create_source_closure_receipt(
                args.fbsource_root, args.oss_root, args.backend_root
            ),
        )
        return True
    if args.command != "create-runtime-source":
        return False
    receipt = create_runtime_source_receipt(
        fbsource_root=args.fbsource_root,
        oss_root=args.oss_root,
        backend_root=args.backend_root,
        source_manifest_path=args.source_manifest,
        wgsl_manifest_path=args.wgsl_manifest,
        manifest_paths={"mtp": args.mtp_manifest, "plain": args.plain_manifest},
        model_roots={"mtp": args.mtp_root, "plain": args.plain_root},
        runtime_paths={
            "mtp": {
                "profile": {
                    "javascript": args.mtp_profile_javascript,
                    "wasm": args.mtp_profile_wasm,
                },
                "wall": {
                    "javascript": args.mtp_wall_javascript,
                    "wasm": args.mtp_wall_wasm,
                },
            },
            "plain": {
                "profile": {
                    "javascript": args.plain_profile_javascript,
                    "wasm": args.plain_profile_wasm,
                },
                "wall": {
                    "javascript": args.plain_wall_javascript,
                    "wasm": args.plain_wall_wasm,
                },
            },
        },
        build_command_paths={
            "mtp": {
                "profile": args.mtp_profile_recipe,
                "wall": args.mtp_wall_recipe,
            },
            "plain": {
                "profile": args.plain_profile_recipe,
                "wall": args.plain_wall_recipe,
            },
        },
    )
    _write_json(args.output, receipt)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquisition = subparsers.add_parser("validate-acquisition")
    acquisition.add_argument("--checkpoint-root", type=Path, required=True)
    assistant_acquisition = subparsers.add_parser("validate-assistant-acquisition")
    assistant_acquisition.add_argument("--checkpoint-root", type=Path, required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--role", action="append", default=[])
    create.add_argument("--ptd", action="append", type=Path, default=[])
    create_mtp = subparsers.add_parser("create-mtp")
    create_mtp.add_argument("--root", type=Path, required=True)
    create_mtp.add_argument("--output", type=Path, required=True)
    create_mtp.add_argument("--role", action="append", default=[])
    create_mtp.add_argument("--ptd", action="append", type=Path, default=[])
    create_mtp.add_argument("--evidence", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    create_source = subparsers.add_parser("create-source-manifest")
    create_source.add_argument("--fbsource-root", type=Path, required=True)
    create_source.add_argument("--oss-root", type=Path, required=True)
    create_source.add_argument("--output", type=Path, required=True)
    create_wgsl = subparsers.add_parser("create-wgsl-manifest")
    create_wgsl.add_argument("--backend-root", type=Path, required=True)
    create_wgsl.add_argument("--output", type=Path, required=True)
    create_source_receipt = subparsers.add_parser("create-source-receipt")
    create_source_receipt.add_argument("--fbsource-root", type=Path, required=True)
    create_source_receipt.add_argument("--oss-root", type=Path, required=True)
    create_source_receipt.add_argument("--backend-root", type=Path, required=True)
    create_source_receipt.add_argument("--output", type=Path, required=True)
    validate_mtp = subparsers.add_parser("validate-mtp")
    validate_mtp.add_argument("--root", type=Path, required=True)
    validate_mtp.add_argument("--manifest", type=Path, required=True)
    stage_runtime = subparsers.add_parser("stage-runtime")
    stage_runtime.add_argument("--destination-root", type=Path, required=True)
    stage_runtime.add_argument("--plain-root", type=Path, required=True)
    stage_runtime.add_argument("--plain-receipt", type=Path, required=True)
    stage_runtime.add_argument("--mtp-root", type=Path, required=True)
    stage_runtime.add_argument("--mtp-receipt", type=Path, required=True)
    stage_runtime.add_argument("--runtime-source-receipt", type=Path, required=True)
    stage_runtime.add_argument("--target-prefill-receipt", type=Path, required=True)
    stage_runtime.add_argument("--plain-profile-javascript", type=Path, required=True)
    stage_runtime.add_argument("--plain-profile-wasm", type=Path, required=True)
    stage_runtime.add_argument("--plain-wall-javascript", type=Path, required=True)
    stage_runtime.add_argument("--plain-wall-wasm", type=Path, required=True)
    stage_runtime.add_argument("--mtp-wall-javascript", type=Path, required=True)
    stage_runtime.add_argument("--mtp-wall-wasm", type=Path, required=True)
    stage_runtime.add_argument("--mtp-profile-javascript", type=Path, required=True)
    stage_runtime.add_argument("--mtp-profile-wasm", type=Path, required=True)
    stage_runtime.add_argument("--source-manifest", type=Path, required=True)
    stage_runtime.add_argument("--wgsl-manifest", type=Path, required=True)
    stage_runtime.add_argument("--plain-profile-recipe", type=Path, required=True)
    stage_runtime.add_argument("--plain-wall-recipe", type=Path, required=True)
    stage_runtime.add_argument("--mtp-wall-recipe", type=Path, required=True)
    stage_runtime.add_argument("--mtp-profile-recipe", type=Path, required=True)
    create_runtime = subparsers.add_parser("create-runtime-source")
    create_runtime.add_argument("--output", type=Path, required=True)
    create_runtime.add_argument("--fbsource-root", type=Path, required=True)
    create_runtime.add_argument("--oss-root", type=Path, required=True)
    create_runtime.add_argument("--backend-root", type=Path, required=True)
    create_runtime.add_argument("--source-manifest", type=Path, required=True)
    create_runtime.add_argument("--wgsl-manifest", type=Path, required=True)
    create_runtime.add_argument("--plain-manifest", type=Path, required=True)
    create_runtime.add_argument("--mtp-manifest", type=Path, required=True)
    create_runtime.add_argument("--plain-root", type=Path, required=True)
    create_runtime.add_argument("--mtp-root", type=Path, required=True)
    create_runtime.add_argument("--plain-profile-javascript", type=Path, required=True)
    create_runtime.add_argument("--plain-profile-wasm", type=Path, required=True)
    create_runtime.add_argument("--plain-profile-recipe", type=Path, required=True)
    create_runtime.add_argument("--plain-wall-javascript", type=Path, required=True)
    create_runtime.add_argument("--plain-wall-wasm", type=Path, required=True)
    create_runtime.add_argument("--plain-wall-recipe", type=Path, required=True)
    create_runtime.add_argument("--mtp-wall-javascript", type=Path, required=True)
    create_runtime.add_argument("--mtp-wall-wasm", type=Path, required=True)
    create_runtime.add_argument("--mtp-wall-recipe", type=Path, required=True)
    create_runtime.add_argument("--mtp-profile-javascript", type=Path, required=True)
    create_runtime.add_argument("--mtp-profile-wasm", type=Path, required=True)
    create_runtime.add_argument("--mtp-profile-recipe", type=Path, required=True)
    create_recipe = subparsers.add_parser("create-build-recipe")
    create_recipe.add_argument(
        "--model", choices=sorted(RUNTIME_BUILD_TARGETS), required=True
    )
    create_recipe.add_argument("--flavor", choices=("profile", "wall"), required=True)
    create_recipe.add_argument("--output", type=Path, required=True)
    validate_runtime = subparsers.add_parser("validate-runtime")
    validate_runtime.add_argument("--root", type=Path, required=True)
    validate_runtime.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "validate-acquisition":
        validate_export_identity(args.checkpoint_root)
        return 0
    if _handle_closure_creation(args):
        return 0
    if args.command == "validate-assistant-acquisition":
        validate_assistant_export_identity(args.checkpoint_root)
        return 0
    if args.command == "stage-runtime":
        stage_combined_runtime(
            args.destination_root,
            args.plain_root,
            args.plain_receipt,
            args.mtp_root,
            args.mtp_receipt,
            args.runtime_source_receipt,
            args.mtp_wall_javascript,
            args.mtp_wall_wasm,
            args.mtp_profile_javascript,
            args.mtp_profile_wasm,
            plain_profile_javascript_path=args.plain_profile_javascript,
            plain_profile_wasm_path=args.plain_profile_wasm,
            plain_wall_javascript_path=args.plain_wall_javascript,
            plain_wall_wasm_path=args.plain_wall_wasm,
            source_manifest_path=args.source_manifest,
            wgsl_manifest_path=args.wgsl_manifest,
            build_recipe_paths={
                "mtp": {
                    "profile": args.mtp_profile_recipe,
                    "wall": args.mtp_wall_recipe,
                },
                "plain": {
                    "profile": args.plain_profile_recipe,
                    "wall": args.plain_wall_recipe,
                },
            },
            target_prefill_receipt_path=args.target_prefill_receipt,
        )
        return 0
    if args.command in {"create", "create-mtp"}:
        create_fn = (
            create_plain_manifest if args.command == "create" else create_mtp_manifest
        )
        manifest = create_fn(args.root, _role_paths(args.role), args.ptd)
        if args.command == "create-mtp":
            manifest["evidence"] = _load_json(args.evidence)
            validate_mtp_manifest(args.root, manifest)
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0

    manifest = _load_json(args.manifest)
    if args.command == "validate":
        validate_plain_manifest(args.root, manifest)
    elif args.command == "validate-mtp":
        validate_mtp_manifest(args.root, manifest)
    else:
        validate_combined_runtime_envelope(args.root, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
