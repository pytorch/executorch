# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Create and validate reproducible WebGPU artifact manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
ALLOWED_SINGLE_ROLES = frozenset(
    {"javascript", "wasm", "pte", "source", "object", "link_map"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained_file(root: Path, path: Path) -> tuple[Path, str]:
    resolved_root = root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise ValueError(f"artifact symlink escapes are not allowed: {path}")
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"artifact escapes manifest root: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"artifact is not a regular file: {path}")
    return resolved, relative.as_posix()


def create_manifest(
    root: Path,
    role_paths: Mapping[str, Path],
    ptd_paths: Sequence[Path] = (),
) -> dict[str, object]:
    unknown_roles = set(role_paths) - ALLOWED_SINGLE_ROLES
    if unknown_roles:
        raise ValueError(f"unsupported artifact roles: {sorted(unknown_roles)}")

    artifacts: list[dict[str, object]] = []
    for role in sorted(role_paths):
        resolved, relative = _contained_file(root, role_paths[role])
        artifacts.append(
            {
                "bytes": resolved.stat().st_size,
                "path": relative,
                "role": role,
                "sha256": _sha256(resolved),
            }
        )

    ptd_order: list[str] = []
    for ptd in ptd_paths:
        resolved, relative = _contained_file(root, ptd)
        if relative in ptd_order:
            raise ValueError(f"duplicate PTD path: {relative}")
        ptd_order.append(relative)
        artifacts.append(
            {
                "bytes": resolved.stat().st_size,
                "path": relative,
                "role": "ptd",
                "sha256": _sha256(resolved),
            }
        )

    return {
        "artifacts": artifacts,
        "ptd_order": ptd_order,
        "schema_version": SCHEMA_VERSION,
    }


def _validate_artifact_entry(
    root: Path,
    artifact: object,
    seen_single_roles: set[str],
) -> tuple[str, str]:
    if not isinstance(artifact, dict):
        raise ValueError("artifact entry must be an object")
    role = artifact.get("role")
    path_value = artifact.get("path")
    if not isinstance(role, str) or not isinstance(path_value, str):
        raise ValueError("artifact role/path must be strings")
    if Path(path_value).is_absolute():
        raise ValueError("manifest stores an absolute artifact path")
    if role != "ptd":
        if role not in ALLOWED_SINGLE_ROLES:
            raise ValueError(f"unsupported artifact role: {role}")
        if role in seen_single_roles:
            raise ValueError(f"duplicate singleton artifact role: {role}")
        seen_single_roles.add(role)

    resolved, normalized = _contained_file(root, Path(path_value))
    if normalized != path_value:
        raise ValueError(f"non-canonical artifact path: {path_value}")
    if artifact.get("bytes") != resolved.stat().st_size:
        raise ValueError(f"artifact byte count mismatch: {path_value}")
    if artifact.get("sha256") != _sha256(resolved):
        raise ValueError(f"artifact SHA-256 mismatch: {path_value}")
    return role, path_value


def validate_manifest(root: Path, manifest: Mapping[str, object]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema version")
    artifacts = manifest.get("artifacts")
    ptd_order = manifest.get("ptd_order")
    if not isinstance(artifacts, list) or not isinstance(ptd_order, list):
        raise ValueError("manifest artifacts/PTD order must be lists")

    seen_single_roles: set[str] = set()
    observed_ptds: list[str] = []
    for artifact in artifacts:
        role, path_value = _validate_artifact_entry(root, artifact, seen_single_roles)
        if role == "ptd":
            observed_ptds.append(path_value)

    if observed_ptds != ptd_order:
        raise ValueError("manifest PTD order does not match artifact order")


def _parse_roles(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or not role or not path or role in result:
            raise ValueError(f"invalid or duplicate ROLE=PATH: {value}")
        result[role] = Path(path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--role", action="append", default=[])
    create.add_argument("--ptd", action="append", type=Path, default=[])
    validate = commands.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "create":
        manifest = create_manifest(args.root, _parse_roles(args.role), args.ptd)
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(args.root, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
