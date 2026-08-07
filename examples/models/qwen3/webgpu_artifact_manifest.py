# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-0.6B WebGPU acquisition/artifact manifest over the generic validator.

The generic byte-role layer (hashing, containment, symlink rejection, ordered
PTD bookkeeping) lives in
``executorch.backends.webgpu.scripts.webgpu_artifact_manifest`` and is reused
here rather than reimplemented. This module adds only what is Qwen3-specific:
pinned checkpoint/tokenizer acquisition, the required artifact roles and method
set, the export capacities, and a fail-closed scan for undeclared files.

Runtime selection is declared explicitly. ``LlmConfig`` exposes no WebGPU
backend field, so ``backend.vulkan`` remains the *serialization* mechanism that
produces the program; it is not by itself a WebGPU selection. The export
contract in ``manifests/qwen3_0_6b_webgpu.json`` therefore records
``target_runtime: webgpu`` alongside ``serialization_backend: vulkan``, and
both are validated here.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from executorch.backends.webgpu.scripts.webgpu_artifact_manifest import (
    create_manifest,
    validate_manifest,
)

MODEL_ID = "qwen3_0_6b"
TARGET_RUNTIME = "webgpu"
# LlmConfig has no WebGPU backend field; Vulkan serialization is the mechanism
# that produces a WebGPU-consumable program, not the runtime claim itself.
SERIALIZATION_BACKEND = "vulkan"
# The WebGPU delegate consumes the Vulkan-partitioner program and registers
# under the Vulkan backend id; a graph carrying any other id is not WebGPU.
WEBGPU_BACKEND_ID = "VulkanBackend"
REQUIRED_METHODS = ("forward",)
REQUIRED_ROLES = ("javascript", "pte", "wasm")
ROLE_SUFFIXES = {"javascript": ".js", "pte": ".pte", "wasm": ".wasm"}
ACQUISITION_KINDS = ("checkpoint", "tokenizer")
MAX_INPUT_LEN = 512
MAX_CONTEXT_LEN = 8960

CONTRACT_PATH = Path(__file__).parent / "manifests" / "qwen3_0_6b_webgpu.json"

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")


class ManifestError(ValueError):
    """Raised when a Qwen3 WebGPU manifest or its artifacts fail validation."""


def _wrap(action: str, fn, *args):
    try:
        return fn(*args)
    except ManifestError:
        raise
    except (OSError, ValueError) as error:
        raise ManifestError(f"{action}: {error}") from error


def load_contract(path: Path | None = None) -> dict[str, Any]:
    contract = json.loads((path or CONTRACT_PATH).read_text(encoding="utf-8"))
    _check_acquisition(contract["acquisition"])
    check_export_config(contract["export"])
    check_methods(contract["methods"])
    return contract


def expected_export_config() -> dict[str, Any]:
    return {
        "target_runtime": TARGET_RUNTIME,
        "serialization_backend": SERIALIZATION_BACKEND,
        "max_input_len": MAX_INPUT_LEN,
        "max_context_len": MAX_CONTEXT_LEN,
    }


def check_export_config(config: Mapping[str, Any]) -> None:
    if config.get("target_runtime") != TARGET_RUNTIME:
        raise ManifestError(
            f"target_runtime must be {TARGET_RUNTIME}: {config.get('target_runtime')}"
        )
    if config.get("serialization_backend") != SERIALIZATION_BACKEND:
        raise ManifestError(
            "serialization_backend must be "
            f"{SERIALIZATION_BACKEND}: {config.get('serialization_backend')}"
        )
    for field, expected in (
        ("max_input_len", MAX_INPUT_LEN),
        ("max_context_len", MAX_CONTEXT_LEN),
    ):
        if config.get(field) != expected:
            raise ManifestError(f"{field} must be {expected}: {config.get(field)}")


def check_methods(observed: Sequence[str]) -> None:
    if sorted(observed) != sorted(REQUIRED_METHODS):
        raise ManifestError(
            f"method set must be {sorted(REQUIRED_METHODS)}: {sorted(observed)}"
        )


def check_delegation(
    backend_ids: Sequence[str],
    operator_names: Sequence[str],
) -> None:
    if operator_names:
        raise ManifestError(
            f"graph retains portable operators: {sorted(set(operator_names))}"
        )
    unique = sorted(set(backend_ids))
    if unique != [WEBGPU_BACKEND_ID]:
        raise ManifestError(
            f"graph backend ids must be [{WEBGPU_BACKEND_ID}]: {unique}"
        )


def check_role_suffixes(artifacts: Sequence[Mapping[str, Any]]) -> None:
    for artifact in artifacts:
        role = artifact.get("role")
        path = str(artifact.get("path", ""))
        expected = ROLE_SUFFIXES.get(str(role))
        if expected is not None and not path.endswith(expected):
            raise ManifestError(
                f"artifact role confusion: role {role} expects {expected}: {path}"
            )


def check_no_extra_files(root: Path, declared: Iterable[str]) -> None:
    allowed = set(declared)
    for path in sorted(root.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in allowed:
            raise ManifestError(f"undeclared artifact under manifest root: {relative}")


def _check_acquisition(acquisition: Mapping[str, Any]) -> None:
    for kind in ACQUISITION_KINDS:
        entry = acquisition.get(kind)
        if not isinstance(entry, Mapping):
            raise ManifestError(f"missing {kind} acquisition pin")
        for field in ("repo", "filename"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ManifestError(f"{kind} acquisition {field} must be a string")
        revision = entry.get("revision")
        if not isinstance(revision, str) or not _GIT_SHA.match(revision):
            raise ManifestError(
                f"{kind} acquisition revision must be a 40-hex commit: {revision}"
            )
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not _SHA256.match(digest):
            raise ManifestError(
                f"{kind} acquisition sha256 must be 64 lowercase hex digits"
            )
        size = entry.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ManifestError(f"{kind} acquisition bytes must be a positive integer")
    unknown = sorted(set(acquisition) - set(ACQUISITION_KINDS))
    if unknown:
        raise ManifestError(f"unsupported acquisition kinds: {unknown}")


def check_contract_agreement(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    if manifest.get("model") != contract.get("model"):
        raise ManifestError("manifest model disagrees with the export contract")
    for kind in ACQUISITION_KINDS:
        recorded = manifest["acquisition"][kind]
        pinned = contract["acquisition"][kind]
        for field in ("repo", "revision", "filename", "sha256", "bytes"):
            if recorded.get(field) != pinned.get(field):
                raise ManifestError(
                    f"acquisition {field} mismatch for {kind}: "
                    f"{recorded.get(field)} != {pinned.get(field)}"
                )
    for field in ("target_runtime", "serialization_backend"):
        if manifest["export"].get(field) != contract["export"].get(field):
            raise ManifestError(f"export {field} disagrees with the export contract")


def create_qwen3_manifest(
    root: Path,
    role_paths: Mapping[str, Path],
    ptd_paths: Sequence[Path],
    acquisition: Mapping[str, Any],
) -> dict[str, Any]:
    _check_acquisition(acquisition)
    missing = sorted(set(REQUIRED_ROLES) - set(role_paths))
    if missing:
        raise ManifestError(f"missing required artifact roles: {missing}")
    manifest: dict[str, Any] = _wrap(
        "artifact", create_manifest, root, role_paths, ptd_paths
    )
    check_role_suffixes(manifest["artifacts"])
    manifest["model"] = MODEL_ID
    manifest["export"] = expected_export_config()
    manifest["methods"] = list(REQUIRED_METHODS)
    manifest["acquisition"] = {
        kind: dict(acquisition[kind]) for kind in ACQUISITION_KINDS
    }
    return manifest


def validate_qwen3_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    expected_acquisition: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None,
) -> None:
    if manifest.get("model") != MODEL_ID:
        raise ManifestError(
            f"manifest model must be {MODEL_ID}: {manifest.get('model')}"
        )
    export = manifest.get("export")
    if not isinstance(export, Mapping):
        raise ManifestError("manifest export config must be an object")
    check_export_config(export)
    methods = manifest.get("methods")
    if not isinstance(methods, list):
        raise ManifestError("manifest method list must be a list")
    check_methods(methods)

    acquisition = manifest.get("acquisition")
    if not isinstance(acquisition, Mapping):
        raise ManifestError("manifest acquisition block must be an object")
    _check_acquisition(acquisition)
    if expected_acquisition is not None:
        for kind in ACQUISITION_KINDS:
            for field in ("sha256", "bytes", "revision"):
                recorded = acquisition[kind].get(field)
                pinned = expected_acquisition[kind].get(field)
                if recorded != pinned:
                    raise ManifestError(
                        f"acquisition pin mismatch for {kind}.{field}: "
                        f"{recorded} != {pinned}"
                    )
    if contract is not None:
        check_contract_agreement(manifest, contract)

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ManifestError("manifest artifacts must be a list")
    observed_roles = {a.get("role") for a in artifacts if isinstance(a, Mapping)}
    missing = sorted(set(REQUIRED_ROLES) - observed_roles)
    if missing:
        raise ManifestError(f"missing required artifact roles: {missing}")
    check_role_suffixes(artifacts)

    _wrap("artifact", validate_manifest, root, manifest)
    declared = {a["path"] for a in artifacts if isinstance(a, Mapping) and "path" in a}
    check_no_extra_files(root, declared)


def _parse_roles(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or not role or not path or role in result:
            raise ManifestError(f"invalid or duplicate ROLE=PATH: {value}")
        result[role] = Path(path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--contract", type=Path, default=None)
    create.add_argument("--role", action="append", default=[])
    create.add_argument("--ptd", action="append", type=Path, default=[])
    validate = commands.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--contract", type=Path, default=None)
    args = parser.parse_args(argv)

    contract = load_contract(args.contract)
    if args.command == "create":
        manifest = create_qwen3_manifest(
            args.root, _parse_roles(args.role), args.ptd, contract["acquisition"]
        )
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_qwen3_manifest(args.root, manifest, contract["acquisition"], contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
