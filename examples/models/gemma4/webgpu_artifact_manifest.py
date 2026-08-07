# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Gemma 4 E2B acquisition and WebGPU artifact manifest contract."""

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess

from pathlib import Path
from typing import Any, Mapping, Sequence

from executorch.backends.webgpu.scripts.webgpu_artifact_manifest import (
    create_manifest,
    validate_manifest,
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
)


def _source_config_path() -> Path:
    return Path(__file__).parent / "config" / "e2b_config.json"


def _single_file_manifest(
    path: str, byte_count: int, sha256: str
) -> dict[str, object]:
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


def validate_plain_manifest(
    root: Path,
    manifest: Mapping[str, object],
    require_source_receipt: bool = True,
) -> None:
    validate_manifest(root, manifest)
    if manifest.get("acquisition") != CHECKPOINT_ACQUISITION:
        raise ValueError("Gemma4 checkpoint acquisition identity mismatch")
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("architecture") != ARCHITECTURE_FINGERPRINT:
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
        artifact.get("role")
        for artifact in artifacts
        if isinstance(artifact, dict)
    }
    if "pte" not in roles or (require_source_receipt and "source" not in roles):
        raise ValueError("Gemma4 plain manifest is missing PTE/source receipt roles")
    if require_source_receipt:
        _validate_source_receipt(root, artifacts)

    expected_paths = {
        str(artifact["path"])
        for artifact in artifacts
        if isinstance(artifact, dict)
    }
    if any(len(Path(path).parts) != 1 for path in expected_paths):
        raise ValueError("Gemma4 artifact staging directory must be flat")
    actual_paths = {entry.name for entry in root.iterdir()}
    if actual_paths != expected_paths:
        raise ValueError("Gemma4 artifact staging contains missing or extra entries")


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
    if args.command == "create-source-receipt":
        _write_json(
            args.output,
            create_source_closure_receipt(
                args.fbsource_root, args.oss_root, args.backend_root
            ),
        )
        return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquisition = subparsers.add_parser("validate-acquisition")
    acquisition.add_argument("--checkpoint-root", type=Path, required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--role", action="append", default=[])
    create.add_argument("--ptd", action="append", type=Path, default=[])
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
    args = parser.parse_args(argv)

    if args.command == "validate-acquisition":
        validate_export_identity(args.checkpoint_root)
        return 0
    if _handle_closure_creation(args):
        return 0
    if args.command == "create":
        manifest = create_plain_manifest(
            args.root, _role_paths(args.role), args.ptd
        )
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0

    manifest = _load_json(args.manifest)
    validate_plain_manifest(args.root, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
