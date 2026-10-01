#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Finalize release dependencies after their stable artifacts are published."""

import argparse
import json
import re
import runpy
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Match

from prepare_release import (  # type: ignore[import-not-found]
    _release_version_from_file,
    _test_index_versions,
    _TORCH_VERSION_PATTERN,
    _write_if_changed,
    pytorch_commit_for_release,
    sync_pytorch_source,
)
from release_versions import (  # type: ignore[import-not-found]
    is_release_version,
    release_base_version,
)

_TORCHVISION_VERSION_PATTERN = re.compile(
    r'^TORCHVISION_VERSION\s*=\s*"(?P<version>[^"]+)"$', re.MULTILINE
)
_TORCHAUDIO_VERSION_PATTERN = re.compile(
    r'^TORCHAUDIO_VERSION\s*=\s*"(?P<version>[^"]+)"$', re.MULTILINE
)
_TORCHCODEC_PATTERN = re.compile(r"torchcodec==\d+\.\d+\.\d+")
_TOKENIZERS_REQUIREMENT_PATTERN = re.compile(
    r'(?m)^(?P<indent>\s*)"pytorch-tokenizers(?:>=\d+\.\d+\.\d+)?",$'
)
_TORCHAO_ASSIGNMENT_PATTERN = re.compile(
    r'^TORCHAO_NIGHTLY_VERSION\s*=\s*"(?P<version>[^"]+)"$', re.MULTILINE
)
_ROCM_TORCHAO_ASSIGNMENT_PATTERN = re.compile(
    r'^ROCM_TORCHAO_NIGHTLY_VERSION\s*=\s*"(?P<version>[^"]+)"$', re.MULTILINE
)
_ROCM_VERSION_PATTERN = re.compile(
    r'^ROCM_VERSION="\$\{ROCM_VERSION:-(?P<version>\d+\.\d+)\}"$', re.MULTILINE
)
_TORCH_URL_BASE_PATTERN = re.compile(
    r'^TORCH_URL_BASE\s*=\s*"https://download\.pytorch\.org/whl(?:/test)?"$',
    re.MULTILINE,
)
_FINALIZED_PATTERN = re.compile(
    r"^RELEASE_DEPENDENCIES_FINALIZED\s*=\s*(?:True|False)$", re.MULTILINE
)
_TORCHCODEC_PATHS = (
    ".ci/scripts/test-rocm-voxtral.sh",
    ".ci/scripts/test_model_e2e.sh",
    "examples/models/moshi/mimi/install_requirements.sh",
)
_ROCM_PATHS = (
    ".ci/scripts/test-rocm-aoti.sh",
    ".ci/scripts/test-rocm-voxtral.sh",
)
_QNN_TEST_INDEX = (
    '"$PIPBIN" install torch=="${TORCH_VERSION}" '
    '--extra-index-url "https://download.pytorch.org/whl/test"'
)
_QNN_STABLE_INDEX = (
    '"$PIPBIN" install --no-cache-dir torch=="${TORCH_VERSION}" '
    '--index-url "https://download.pytorch.org/whl/cpu"'
)
_SUBMODULE_RELEASES = (
    ("extension/llm/tokenizers", "https://github.com/meta-pytorch/tokenizers.git"),
    ("third-party/ao", "https://github.com/pytorch/ao.git"),
)


def stable_base_version(version: str) -> str:
    """Convert a development or prerelease version to its final base version."""
    return release_base_version(version)


def current_torchao_version(install_requirements_path: Path) -> str:
    """Read the generic TorchAO pin without importing the installer module."""
    match = _TORCHAO_ASSIGNMENT_PATTERN.search(install_requirements_path.read_text())
    if match is None:
        raise RuntimeError(
            f"could not find TORCHAO_NIGHTLY_VERSION in {install_requirements_path}"
        )
    return match.group("version")


def latest_pypi_version(package: str) -> str:
    """Return the latest stable version published for a package."""
    with urllib.request.urlopen(
        f"https://pypi.org/pypi/{package}/json", timeout=30
    ) as response:
        version = json.load(response)["info"]["version"]
    if not is_release_version(version, allow_prerelease=False):
        raise RuntimeError(f"latest {package} version is not stable: {version!r}")
    return version


def require_pypi_release(package: str, version: str) -> None:
    """Fail before mutation if an expected stable package is unavailable."""
    if not is_release_version(version, allow_prerelease=False):
        raise RuntimeError(f"invalid {package} version {version!r}")
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{package}/{version}/json", timeout=30
        ):
            pass
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise RuntimeError(
                f"{package}=={version} is not published on PyPI"
            ) from error
        raise


def require_stable_cuda_releases(versions: Iterable[tuple[str, str]]) -> None:
    """Fail unless every final cu134 package is on the stable index."""
    missing = [
        f"{package}=={version}+cu134"
        for package, version in versions
        if version not in _test_index_versions(package, "cu134", channel="")
    ]
    if missing:
        raise RuntimeError(
            "stable PyTorch cu134 releases are unavailable: " + ", ".join(missing)
        )


def require_remote_tag(repository: str, version: str) -> None:
    """Fail before mutation if a matching submodule release tag is unavailable."""
    result = subprocess.run(
        [
            "git",
            "ls-remote",
            "--exit-code",
            "--tags",
            repository,
            f"refs/tags/v{version}",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{repository} does not have tag v{version}")


def require_rocm_torchao_release(repo_root: Path, version: str) -> None:
    """Verify the stable ROCm TorchAO artifact used by release CI exists."""
    rocm_script = repo_root / _ROCM_PATHS[0]
    match = _ROCM_VERSION_PATTERN.search(rocm_script.read_text())
    if match is None:
        raise RuntimeError(f"could not determine ROCm version from {rocm_script}")
    rocm_version = match.group("version")
    url = (
        f"https://download.pytorch.org/whl/rocm{rocm_version}/"
        f"torchao-{version}%2Brocm{rocm_version}-"
        "cp310-abi3-manylinux_2_28_x86_64.whl"
    )
    try:
        with urllib.request.urlopen(url, timeout=30):
            pass
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise RuntimeError(
                f"TorchAO {version} is not published for ROCm {rocm_version}"
            ) from error
        raise


def planned_torch_release(torch_pin_path: Path, torch_version: str) -> str:
    """Plan the RC-to-final pin update without changing the checkout."""
    if not is_release_version(torch_version, allow_prerelease=False):
        raise RuntimeError(
            f"final PyTorch version must be X.Y.Z, got {torch_version!r}"
        )
    text = torch_pin_path.read_text()
    updated, version_count = _TORCH_VERSION_PATTERN.subn(
        f'TORCH_VERSION = "{torch_version}"', text
    )
    if version_count != 1:
        raise RuntimeError(f"expected one TORCH_VERSION assignment in {torch_pin_path}")
    for pattern, package in (
        (_TORCHVISION_VERSION_PATTERN, "torchvision"),
        (_TORCHAUDIO_VERSION_PATTERN, "torchaudio"),
    ):
        current = pattern.search(updated)
        if current is None:
            raise RuntimeError(f"expected {package} assignment in {torch_pin_path}")
        expected = stable_base_version(current.group("version"))

        def replace_version(match: Match[str], replacement: str = expected) -> str:
            return match.group(0).replace(match.group("version"), replacement)

        updated = pattern.sub(replace_version, updated)
    updated, count = _FINALIZED_PATTERN.subn(
        "RELEASE_DEPENDENCIES_FINALIZED = True", updated
    )
    if count != 1:
        raise RuntimeError(
            f"expected one RELEASE_DEPENDENCIES_FINALIZED assignment in {torch_pin_path}"
        )
    return updated


def finalize_torch_release(torch_pin_path: Path, torch_version: str) -> int:
    """Replace the RC pin with the final release and mark finalization complete."""
    updated = planned_torch_release(torch_pin_path, torch_version)
    return int(_write_if_changed(torch_pin_path, updated))


def plan_dependency_text(
    repo_root: Path,
    torchao_version: str,
    tokenizers_version: str,
    torchcodec_version: str,
) -> dict[Path, str]:
    """Validate and calculate every text edit without changing the checkout."""
    updates: dict[Path, str] = {}

    def read(path: Path) -> str:
        if not path.exists():
            raise RuntimeError(f"expected release dependency file {path}")
        return updates.get(path, path.read_text())

    path = repo_root / "install_requirements.py"
    updated, count = _TORCHAO_ASSIGNMENT_PATTERN.subn(
        f'TORCHAO_NIGHTLY_VERSION = "{torchao_version}"', read(path)
    )
    if count != 1:
        raise RuntimeError(f"expected one TORCHAO_NIGHTLY_VERSION in {path}")
    updated, count = _ROCM_TORCHAO_ASSIGNMENT_PATTERN.subn(
        f'ROCM_TORCHAO_NIGHTLY_VERSION = "{torchao_version}"', updated
    )
    if count != 1:
        raise RuntimeError(f"expected one ROCM_TORCHAO_NIGHTLY_VERSION in {path}")
    updated, count = _TORCH_URL_BASE_PATTERN.subn(
        'TORCH_URL_BASE = "https://download.pytorch.org/whl"', updated
    )
    if count != 1:
        raise RuntimeError(f"expected one TORCH_URL_BASE in {path}")
    updates[path] = updated

    path = repo_root / "setup.py"
    updated, count = _TOKENIZERS_REQUIREMENT_PATTERN.subn(
        rf'\g<indent>"pytorch-tokenizers>={tokenizers_version}",', read(path)
    )
    if count != 1:
        raise RuntimeError(f"expected one pytorch-tokenizers requirement in {path}")
    updates[path] = updated

    for relative in _TORCHCODEC_PATHS:
        path = repo_root / relative
        updated, count = _TORCHCODEC_PATTERN.subn(
            f"torchcodec=={torchcodec_version}", read(path)
        )
        if count == 0:
            raise RuntimeError(f"expected a torchcodec pin in {path}")
        updates[path] = updated.replace(
            "--extra-index-url https://download.pytorch.org/whl/test/cpu",
            "--index-url https://download.pytorch.org/whl/cpu",
        )

    for relative in _ROCM_PATHS:
        path = repo_root / relative
        updates[path] = (
            read(path)
            .replace(
                "https://download.pytorch.org/whl/test/rocm${ROCM_VERSION}",
                "https://download.pytorch.org/whl/rocm${ROCM_VERSION}",
            )
            .replace(
                "https://download.pytorch.org/whl/nightly/rocm${ROCM_VERSION}",
                "https://download.pytorch.org/whl/rocm${ROCM_VERSION}",
            )
        )

    qnn_path = repo_root / ".ci/scripts/test_wheel_package_qnn.sh"
    updated = read(qnn_path).replace(_QNN_TEST_INDEX, _QNN_STABLE_INDEX)
    if _QNN_STABLE_INDEX not in updated:
        raise RuntimeError(
            f"could not find the QNN Torch install command in {qnn_path}"
        )
    updates[qnn_path] = updated
    return {path: text for path, text in updates.items() if text != path.read_text()}


def finalize_dependency_text(
    repo_root: Path,
    torchao_version: str,
    tokenizers_version: str,
    torchcodec_version: str,
) -> int:
    """Update all recurring stable dependency declarations and index URLs."""
    updates = plan_dependency_text(
        repo_root, torchao_version, tokenizers_version, torchcodec_version
    )
    for path, updated in updates.items():
        _write_if_changed(path, updated)
    return len(updates)


def prepare_submodule_tags(
    repo_root: Path, releases: Iterable[tuple[str, str, str]]
) -> list[tuple[Path, str, str]]:
    """Fetch and resolve every submodule tag before changing any gitlink."""
    prepared = []
    for relative, _repository, version in releases:
        path = repo_root / relative
        if (path / ".git").exists() and subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout:
            raise RuntimeError(f"submodule {path} has local changes")
        subprocess.run(
            ["git", "submodule", "update", "--init", "--", relative],
            cwd=repo_root,
            check=True,
        )
        if subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout:
            raise RuntimeError(f"submodule {path} has local changes")
        subprocess.run(
            ["git", "fetch", "--depth=1", "origin", "tag", f"v{version}"],
            cwd=path,
            check=True,
        )
        current = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        target = subprocess.run(
            ["git", "rev-parse", f"v{version}^{{commit}}"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        prepared.append((path, current, target))
    return prepared


def update_submodule_tags(prepared: Iterable[tuple[Path, str, str]]) -> None:
    """Apply prepared submodule updates, restoring prior heads on failure."""
    changed = []
    try:
        for path, current, target in prepared:
            subprocess.run(
                ["git", "checkout", "--detach", target], cwd=path, check=True
            )
            changed.append((path, current))
    except Exception:
        for path, current in reversed(changed):
            subprocess.run(
                ["git", "checkout", "--detach", current], cwd=path, check=True
            )
        raise


def main() -> None:
    """Preflight and apply stable dependency finalization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-version", help="release version in MAJOR.MINOR form")
    parser.add_argument("--torch-version", help="final stable PyTorch version")
    parser.add_argument("--torchao-version", help="stable TorchAO version")
    parser.add_argument(
        "--tokenizers-version", help="stable pytorch-tokenizers version"
    )
    parser.add_argument("--torchcodec-version", help="stable TorchCodec version")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify packages and tags without changing the checkout",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    release_version = args.release_version or _release_version_from_file(repo_root)
    release_config = runpy.run_path(str(repo_root / "torch_pin.py"))
    if release_config.get("RELEASE_WHEEL") is not True:
        raise RuntimeError(
            "release preparation has not run; use cut-release-branch.sh first"
        )
    torch_version = args.torch_version or stable_base_version(
        release_config["TORCH_VERSION"]
    )
    torchvision_version = stable_base_version(release_config["TORCHVISION_VERSION"])
    torchaudio_version = stable_base_version(release_config["TORCHAUDIO_VERSION"])
    torchao_version = args.torchao_version or stable_base_version(
        current_torchao_version(repo_root / "install_requirements.py")
    )
    tokenizers_version = args.tokenizers_version or f"{release_version}.0"
    torchcodec_version = args.torchcodec_version or latest_pypi_version("torchcodec")
    pytorch_commit = pytorch_commit_for_release(torch_version)

    releases = (
        ("extension/llm/tokenizers", _SUBMODULE_RELEASES[0][1], tokenizers_version),
        ("third-party/ao", _SUBMODULE_RELEASES[1][1], torchao_version),
    )
    for package, version in (
        ("torch", torch_version),
        ("torchvision", torchvision_version),
        ("torchaudio", torchaudio_version),
        ("torchao", torchao_version),
        ("pytorch-tokenizers", tokenizers_version),
        ("torchcodec", torchcodec_version),
    ):
        require_pypi_release(package, version)
    require_stable_cuda_releases(
        (
            ("torch", torch_version),
            ("torchvision", torchvision_version),
            ("torchaudio", torchaudio_version),
        )
    )
    for _path, repository, version in releases:
        require_remote_tag(repository, version)
    require_rocm_torchao_release(repo_root, torchao_version)

    print(
        "Release dependency artifacts are available: "
        f"torch=={torch_version}, torchvision=={torchvision_version}, "
        f"torchaudio=={torchaudio_version}, torchao=={torchao_version}, "
        f"pytorch-tokenizers=={tokenizers_version}, torchcodec=={torchcodec_version}"
    )
    if args.preflight_only:
        return

    dependency_updates = plan_dependency_text(
        repo_root, torchao_version, tokenizers_version, torchcodec_version
    )
    torch_pin_path = repo_root / "torch_pin.py"
    torch_pin_update = planned_torch_release(torch_pin_path, torch_version)
    prepared_submodules = prepare_submodule_tags(repo_root, releases)
    text_updates = {**dependency_updates, torch_pin_path: torch_pin_update}
    original_text = {path: path.read_text() for path in text_updates}
    source_paths = [repo_root / ".ci/docker/ci_commit_pins/pytorch.txt"]
    for relative in (
        "runtime/core/portable_type/c10/c10",
        "runtime/core/portable_type/c10/torch/headeronly",
    ):
        source_paths.extend(
            path for path in (repo_root / relative).rglob("*") if path.is_file()
        )
    original_source = {path: path.read_bytes() for path in source_paths}

    try:
        sync_pytorch_source(repo_root, pytorch_commit)
        update_submodule_tags(prepared_submodules)
        for path, updated in text_updates.items():
            _write_if_changed(path, updated)
    except Exception:
        for path, original in original_text.items():
            path.write_text(original)
        for path, original in original_source.items():
            path.write_bytes(original)
        for path, original, _target in reversed(prepared_submodules):
            subprocess.run(
                ["git", "checkout", "--detach", original], cwd=path, check=False
            )
        raise

    changed = sum(
        updated != original_text[path] for path, updated in text_updates.items()
    )
    changed_source = sum(
        path.read_bytes() != original for path, original in original_source.items()
    )
    print(
        f"Finalized release dependencies; changed {changed} text files, "
        f"{changed_source} PyTorch source files, and 2 submodules"
    )
    print(
        "Review and stage the resulting release-only changes with git add "
        "install_requirements.py setup.py torch_pin.py .ci/scripts/ "
        ".ci/docker/ci_commit_pins/pytorch.txt runtime/core/portable_type/c10/ "
        "examples/models/moshi/mimi/install_requirements.sh "
        "extension/llm/tokenizers third-party/ao"
    )


if __name__ == "__main__":
    main()
