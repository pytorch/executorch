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
from typing import Iterable

from prepare_release import _release_version_from_file, set_torch_version

_VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+")
_TORCHCODEC_PATTERN = re.compile(r"torchcodec==\d+\.\d+\.\d+")
_TOKENIZERS_REQUIREMENT_PATTERN = re.compile(
    r'(?m)^(?P<indent>\s*)"pytorch-tokenizers(?:>=\d+\.\d+\.\d+)?",$'
)
_TORCHAO_ASSIGNMENT_PATTERN = re.compile(
    r'^TORCHAO_NIGHTLY_VERSION\s*=\s*"(?P<version>[^"]+)"$', re.MULTILINE
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
    match = re.match(r"(\d+\.\d+\.\d+)", version)
    if match is None:
        raise RuntimeError(f"could not derive a stable version from {version!r}")
    return match.group(1)


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
    if _VERSION_PATTERN.fullmatch(version) is None:
        raise RuntimeError(f"latest {package} version is not stable: {version!r}")
    return version


def require_pypi_release(package: str, version: str) -> None:
    """Fail before mutation if an expected stable package is unavailable."""
    if _VERSION_PATTERN.fullmatch(version) is None:
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


def _write_if_changed(path: Path, updated: str) -> bool:
    original = path.read_text()
    if updated == original:
        return False
    path.write_text(updated)
    return True


def mark_release_dependencies_finalized(torch_pin_path: Path) -> bool:
    """Record that delayed dependency finalization completed successfully."""
    text = torch_pin_path.read_text()
    updated, count = _FINALIZED_PATTERN.subn(
        "RELEASE_DEPENDENCIES_FINALIZED = True", text
    )
    if count != 1:
        raise RuntimeError(
            f"expected one RELEASE_DEPENDENCIES_FINALIZED assignment in {torch_pin_path}"
        )
    return _write_if_changed(torch_pin_path, updated)


def finalize_torch_release(torch_pin_path: Path, torch_version: str) -> int:
    """Replace the RC pin with the final release and mark finalization complete."""
    changed = int(set_torch_version(torch_pin_path, torch_version))
    changed += int(mark_release_dependencies_finalized(torch_pin_path))
    return changed


def _finalize_torchao(repo_root: Path, version: str) -> set[Path]:
    path = repo_root / "install_requirements.py"
    text = path.read_text()
    updated, count = _TORCHAO_ASSIGNMENT_PATTERN.subn(
        f'TORCHAO_NIGHTLY_VERSION = "{version}"', text
    )
    if count != 1:
        raise RuntimeError(f"expected one TORCHAO_NIGHTLY_VERSION in {path}")
    return {path} if _write_if_changed(path, updated) else set()


def _finalize_tokenizers(repo_root: Path, version: str) -> set[Path]:
    path = repo_root / "setup.py"
    text = path.read_text()
    updated, count = _TOKENIZERS_REQUIREMENT_PATTERN.subn(
        rf'\g<indent>"pytorch-tokenizers>={version}",', text
    )
    if count != 1:
        raise RuntimeError(f"expected one pytorch-tokenizers requirement in {path}")
    return {path} if _write_if_changed(path, updated) else set()


def _finalize_torchcodec(repo_root: Path, version: str) -> set[Path]:
    changed_paths = set()
    for relative in _TORCHCODEC_PATHS:
        path = repo_root / relative
        if not path.exists():
            raise RuntimeError(f"expected release dependency file {path}")
        text = path.read_text()
        updated, count = _TORCHCODEC_PATTERN.subn(f"torchcodec=={version}", text)
        if count == 0:
            raise RuntimeError(f"expected a torchcodec pin in {path}")
        updated = updated.replace(
            "--extra-index-url https://download.pytorch.org/whl/test/cpu",
            "--index-url https://download.pytorch.org/whl/cpu",
        )
        if _write_if_changed(path, updated):
            changed_paths.add(path)
    return changed_paths


def _finalize_stable_indexes(repo_root: Path) -> set[Path]:
    changed_paths = set()
    for relative in _ROCM_PATHS:
        path = repo_root / relative
        if not path.exists():
            raise RuntimeError(f"expected release dependency file {path}")
        text = path.read_text()
        updated = text.replace(
            "https://download.pytorch.org/whl/test/rocm${ROCM_VERSION}",
            "https://download.pytorch.org/whl/rocm${ROCM_VERSION}",
        ).replace(
            "https://download.pytorch.org/whl/nightly/rocm${ROCM_VERSION}",
            "https://download.pytorch.org/whl/rocm${ROCM_VERSION}",
        )
        if _write_if_changed(path, updated):
            changed_paths.add(path)

    qnn_path = repo_root / ".ci/scripts/test_wheel_package_qnn.sh"
    if not qnn_path.exists():
        raise RuntimeError(f"expected release dependency file {qnn_path}")
    text = qnn_path.read_text()
    updated = text.replace(_QNN_TEST_INDEX, _QNN_STABLE_INDEX)
    if _QNN_STABLE_INDEX not in updated:
        raise RuntimeError(
            f"could not find the QNN Torch install command in {qnn_path}"
        )
    if _write_if_changed(qnn_path, updated):
        changed_paths.add(qnn_path)
    return changed_paths


def finalize_dependency_text(
    repo_root: Path,
    torchao_version: str,
    tokenizers_version: str,
    torchcodec_version: str,
) -> int:
    """Update all recurring stable dependency declarations and index URLs."""
    changed_paths = _finalize_torchao(repo_root, torchao_version)
    changed_paths |= _finalize_tokenizers(repo_root, tokenizers_version)
    changed_paths |= _finalize_torchcodec(repo_root, torchcodec_version)
    changed_paths |= _finalize_stable_indexes(repo_root)
    return len(changed_paths)


def update_submodule_tags(
    repo_root: Path, releases: Iterable[tuple[str, str, str]]
) -> None:
    """Check out verified release tags in the dependency submodules."""
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
        subprocess.run(
            ["git", "checkout", "--detach", f"v{version}"], cwd=path, check=True
        )


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
    torchao_version = args.torchao_version or stable_base_version(
        current_torchao_version(repo_root / "install_requirements.py")
    )
    tokenizers_version = args.tokenizers_version or f"{release_version}.0"
    torchcodec_version = args.torchcodec_version or latest_pypi_version("torchcodec")

    releases = (
        ("extension/llm/tokenizers", _SUBMODULE_RELEASES[0][1], tokenizers_version),
        ("third-party/ao", _SUBMODULE_RELEASES[1][1], torchao_version),
    )
    for package, version in (
        ("torch", torch_version),
        ("torchao", torchao_version),
        ("pytorch-tokenizers", tokenizers_version),
        ("torchcodec", torchcodec_version),
    ):
        require_pypi_release(package, version)
    for _path, repository, version in releases:
        require_remote_tag(repository, version)

    print(
        "Release dependency artifacts are available: "
        f"torch=={torch_version}, torchao=={torchao_version}, "
        f"pytorch-tokenizers=={tokenizers_version}, torchcodec=={torchcodec_version}"
    )
    if args.preflight_only:
        return

    update_submodule_tags(repo_root, releases)
    changed = finalize_dependency_text(
        repo_root, torchao_version, tokenizers_version, torchcodec_version
    )
    changed += finalize_torch_release(repo_root / "torch_pin.py", torch_version)
    print(
        f"Finalized release dependencies; changed {changed} text files and 2 submodules"
    )
    print(
        "Review and stage the resulting release-only changes with git add "
        "install_requirements.py setup.py torch_pin.py .ci/scripts/ "
        "examples/models/moshi/mimi/install_requirements.sh "
        "extension/llm/tokenizers third-party/ao"
    )


if __name__ == "__main__":
    main()
