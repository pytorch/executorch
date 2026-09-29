# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metadata for finalized Core AI bundles, independent of their delivery mode."""

import hashlib
import stat
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

_DIGEST_DOMAIN = b"ExecuTorch.CoreAI.raw-assets.sha256.v1\0"
_HASH_CHUNK_SIZE = 1024 * 1024


def _validate_component(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or any(c in name for c in ("/", "\\", "\0"))
    ):
        raise ValueError(f"unsafe Core AI asset path component: {name!r}")
    name.encode("utf-8")


def _declared_bundles(manifest: Mapping[str, Any]) -> List[str]:
    model_hash = manifest["hash"]
    _validate_component(model_hash)
    packaging = manifest["packaging"]
    if packaging in ("inline", "sidecar"):
        bundles = {"model.aimodel": manifest["path"]}
    elif packaging in ("aot_compiled_inline", "aot_compiled_sidecar"):
        archs = manifest["archs"]
        if not isinstance(archs, dict) or not archs:
            raise ValueError("Core AI assets require declared architectures")
        bundles = {}
        for arch, path in archs.items():
            _validate_component(arch)
            bundles[f"model.{arch}.aimodelc"] = path
    else:
        raise ValueError(f"unknown Core AI asset packaging: {packaging!r}")
    for bundle, path in bundles.items():
        if path != f"{model_hash}/{bundle}":
            raise ValueError(f"invalid declared Core AI bundle path: {path!r}")
    return sorted(bundles, key=lambda name: name.encode("utf-8"))


def _collect_bundle_files(
    root_dir: Path, bundles: List[str]
) -> Dict[str, List[Tuple[str, Path, int]]]:
    if not stat.S_ISDIR(root_dir.lstat().st_mode):
        raise ValueError(
            f"Core AI asset root must be a non-symlink directory: {root_dir}"
        )
    if {path.name for path in root_dir.iterdir()} != set(bundles):
        raise ValueError("Core AI asset root must contain exactly the declared bundles")

    bundle_files: Dict[str, List[Tuple[str, Path, int]]] = {}

    def visit(path: Path, files: List[Tuple[str, Path, int]]) -> None:
        _validate_component(path.name)
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            for child in path.iterdir():
                visit(child, files)
        elif stat.S_ISREG(info.st_mode):
            files.append((path.relative_to(root_dir).as_posix(), path, info.st_size))
        else:
            raise ValueError(
                f"Core AI assets reject symlinks and special files: {path}"
            )

    for bundle in bundles:
        path = root_dir / bundle
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise ValueError(f"Core AI bundle must be a non-symlink directory: {path}")
        files: List[Tuple[str, Path, int]] = []
        visit(path, files)
        if not files:
            raise ValueError(f"Core AI bundle contains no files: {path}")
        bundle_files[bundle] = sorted(files, key=lambda item: item[0].encode("utf-8"))
    return bundle_files


def collect_asset_metadata(
    root_dir: Path,
    manifest: Mapping[str, Any],
    *,
    register_payload: Optional[Callable[[str, bytes], None]] = None,
) -> Dict[str, Any]:
    """Validate the final tree and hash its files, optionally registering buffers.

    Inline delivery shares each read buffer with the registration callback.
    Sidecar delivery streams files without retaining their payloads.
    """
    bundle_files = _collect_bundle_files(root_dir, _declared_bundles(manifest))
    sizes = {}
    digests = {}
    for bundle, files in bundle_files.items():
        digest = hashlib.sha256(_DIGEST_DOMAIN)
        digest.update(len(files).to_bytes(8, "big"))
        for relative_path, path, size in files:
            name = relative_path.encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(size.to_bytes(8, "big"))
            if register_payload is not None:
                payload = path.read_bytes()
                if len(payload) != size:
                    raise ValueError(
                        f"Core AI asset size changed while reading: {path}"
                    )
                digest.update(payload)
                register_payload(relative_path, payload)
            else:
                actual_size = 0
                with path.open("rb") as source:
                    while chunk := source.read(_HASH_CHUNK_SIZE):
                        digest.update(chunk)
                        actual_size += len(chunk)
                if actual_size != size:
                    raise ValueError(
                        f"Core AI asset size changed while reading: {path}"
                    )
            sizes[relative_path] = size
        digests[bundle] = digest.hexdigest()
    return {"version": 2, "files": sizes, "bundle_digests": digests}
