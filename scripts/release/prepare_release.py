#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply and validate the deterministic changes required on release branches."""

import argparse
import re
import runpy
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List
from urllib.parse import unquote

from release_versions import (  # type: ignore[import-not-found]
    release_key,
    torch_release_tag,
    torch_requirement,
    torchvision_version,
)

_RELEASE_VERSION_PATTERN = re.compile(r"\d+\.\d+")
_RELEASE_WHEEL_PATTERN = re.compile(
    r"^RELEASE_WHEEL\s*=\s*(?:True|False)$", re.MULTILINE
)
_TORCH_VERSION_PATTERN = re.compile(r'^TORCH_VERSION\s*=\s*"([^"]+)"$', re.MULTILINE)
_TORCHVISION_VERSION_PATTERN = re.compile(
    r'^TORCHVISION_VERSION\s*=\s*"([^"]+)"$', re.MULTILINE
)
_TORCHAUDIO_VERSION_PATTERN = re.compile(
    r'^TORCHAUDIO_VERSION\s*=\s*"([^"]+)"$', re.MULTILINE
)
_TEST_INFRA_MAIN_PATTERN = re.compile(r"(pytorch/test-infra/[^\s\"']+)@main\b")
_TEST_INFRA_REF_MAIN_PATTERN = re.compile(r"(test-infra-ref:\s*)main\b")
_TEST_INFRA_BRANCH_PATTERN = re.compile(r"pytorch/test-infra/[^\s\"'@]+@([^\s\"']+)")
_TEST_INFRA_INPUT_PATTERN = re.compile(r"test-infra-ref:\s*([^\s#]+)")
_CLONE_BRANCH_PATTERN = re.compile(r"(?P<prefix>-b\s+)viable/strict\b")
_STABLE_SWIFTPM_PATTERN = re.compile(r"swiftpm-\d+\.\d+\.\d+(?![.\d-])")
_DOCUMENTATION_PATHS = (
    "CONTRIBUTING.md",
    "docs",
    "extension/benchmark/apple/Benchmark",
)


def _validate_release_version(release_version: str) -> None:
    if _RELEASE_VERSION_PATTERN.fullmatch(release_version) is None:
        raise RuntimeError(
            f"release version must be MAJOR.MINOR, received {release_version!r}"
        )


def _write_if_changed(path: Path, updated: str) -> bool:
    original = path.read_text()
    if updated == original:
        return False
    path.write_text(updated)
    return True


def enable_release_wheel(torch_pin_path: Path) -> bool:
    """Enable release-only wheel metadata in the PyTorch pin file."""
    text = torch_pin_path.read_text()
    updated, count = _RELEASE_WHEEL_PATTERN.subn("RELEASE_WHEEL = True", text)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one RELEASE_WHEEL assignment in {torch_pin_path}"
        )
    return _write_if_changed(torch_pin_path, updated)


def set_torch_version(torch_pin_path: Path, torch_version: str) -> bool:
    """Set the Torch release pin used by installs and release wheel metadata."""
    if re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", torch_version) is None:
        raise RuntimeError(f"invalid Torch release version {torch_version!r}")
    text = torch_pin_path.read_text()
    updated, count = _TORCH_VERSION_PATTERN.subn(
        f'TORCH_VERSION = "{torch_version}"', text
    )
    if count != 1:
        raise RuntimeError(
            f"expected exactly one TORCH_VERSION assignment in {torch_pin_path}"
        )
    return _write_if_changed(torch_pin_path, updated)


def set_companion_versions(torch_pin_path: Path, torch_version: str) -> bool:
    """Move torchvision to the selected PyTorch release train."""
    text = torch_pin_path.read_text()
    updated, vision_count = _TORCHVISION_VERSION_PATTERN.subn(
        f'TORCHVISION_VERSION = "{torchvision_version(torch_version)}"', text
    )
    if vision_count != 1 or _TORCHAUDIO_VERSION_PATTERN.search(updated) is None:
        raise RuntimeError(
            f"expected exactly one torchvision and torchaudio assignment in {torch_pin_path}"
        )
    return _write_if_changed(torch_pin_path, updated)


def newest_torch_test_release(
    available_versions: Iterable[str], newer_than: str = ""
) -> str:
    """Return the newest RC/final on a newer PyTorch release train."""
    candidates = []
    for version in available_versions:
        match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:(rc)(\d+))?", version)
        if match is None:
            continue
        key = release_key(version)
        if newer_than and key[:2] <= release_key(newer_than)[:2]:
            continue
        candidates.append((key, version))
    if not candidates:
        qualifier = f" newer than {newer_than}" if newer_than else ""
        raise RuntimeError(
            f"PyTorch test index has no release candidate or final release{qualifier}"
        )
    return max(candidates)[1]


def _test_index_versions(package: str) -> set[str]:
    with urllib.request.urlopen(
        f"https://download.pytorch.org/whl/test/cpu/{package}/", timeout=30
    ) as response:
        index = unquote(response.read().decode())
    return set(
        re.findall(
            rf"{re.escape(package)}-(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)(?:\+[^-]+)?-",
            index,
        )
    )


def latest_torch_test_release(newer_than: str) -> str:
    """Look up the newest PyTorch release candidate on its CPU test index."""
    return newest_torch_test_release(
        _test_index_versions("torch"), newer_than=newer_than
    )


def require_companion_releases(torch_pin_path: Path, torch_version: str) -> None:
    """Fail before mutation unless matching domain-library releases exist."""
    config = runpy.run_path(str(torch_pin_path))
    expected = {
        "torchvision": torchvision_version(torch_version),
        "torchaudio": config["TORCHAUDIO_VERSION"],
    }
    missing = [
        f"{package}=={version}"
        for package, version in expected.items()
        if version not in _test_index_versions(package)
    ]
    if missing:
        raise RuntimeError(
            "PyTorch companion releases are not available on the test index: "
            + ", ".join(missing)
        )


def torch_version_for_release(torch_pin_path: Path, override: str = "") -> str:
    """Choose a pin once, then preserve it on subsequent preparation runs."""
    config = runpy.run_path(str(torch_pin_path))
    current = config["TORCH_VERSION"]
    if override:
        current_key = release_key(current)
        override_key = release_key(override)
        already_prepared = config.get("RELEASE_WHEEL") is True
        invalid = (
            override_key < current_key
            if already_prepared
            else override_key[:2] <= current_key[:2]
        )
        if invalid:
            raise RuntimeError(
                f"Torch release {override} cannot replace current version {current}"
            )
        return override
    if config.get("RELEASE_WHEEL") is True:
        return current
    return latest_torch_test_release(current)


def test_infra_branch_for_torch(torch_version: str) -> str:
    """Return the test-infra release branch matching a PyTorch release."""
    match = re.match(r"(\d+)\.(\d+)\.", torch_version)
    if match is None:
        raise RuntimeError(
            f"could not derive a test-infra branch from Torch {torch_version!r}"
        )
    return f"release/{match.group(1)}.{match.group(2)}"


def require_test_infra_branch(test_infra_branch: str) -> None:
    """Fail before editing if the selected test-infra branch does not exist."""
    result = subprocess.run(
        [
            "git",
            "ls-remote",
            "--exit-code",
            "--heads",
            "https://github.com/pytorch/test-infra.git",
            test_infra_branch,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pytorch/test-infra does not have branch {test_infra_branch!r}"
        )


def pytorch_commit_for_release(torch_version: str) -> str:
    """Resolve the selected PyTorch release tag before changing the checkout."""
    tag = torch_release_tag(torch_version)
    result = subprocess.run(
        [
            "git",
            "ls-remote",
            "https://github.com/pytorch/pytorch.git",
            f"refs/tags/{tag}",
            f"refs/tags/{tag}^{{}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    references = {
        reference: commit
        for commit, reference in (
            line.split("\t", 1) for line in result.stdout.splitlines() if "\t" in line
        )
    }
    commit = references.get(f"refs/tags/{tag}^{{}}") or references.get(
        f"refs/tags/{tag}"
    )
    if commit is None:
        raise RuntimeError(f"PyTorch release tag {tag} does not exist")
    return commit


def sync_pytorch_source(repo_root: Path, commit: str) -> None:
    """Pin PyTorch source and synchronize the vendored c10 headers."""
    subprocess.run(
        [
            sys.executable,
            str(repo_root / ".github/scripts/update_pytorch_pin.py"),
            "--commit",
            commit,
        ],
        cwd=repo_root,
        check=True,
    )


def validate_release_build(torch_pin_path: Path) -> str:
    """Return the release PyTorch requirement or raise for invalid config."""
    config = runpy.run_path(str(torch_pin_path))
    if config.get("RELEASE_WHEEL") is not True:
        raise RuntimeError(f"{torch_pin_path} does not enable release wheel metadata")

    version = config.get("TORCH_VERSION")
    if (
        not isinstance(version, str)
        or re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", version) is None
    ):
        raise RuntimeError(f"{torch_pin_path} has invalid TORCH_VERSION {version!r}")
    expected_vision = torchvision_version(version)
    if config.get("TORCHVISION_VERSION") != expected_vision:
        raise RuntimeError(
            f"{torch_pin_path} has TORCHVISION_VERSION="
            f"{config.get('TORCHVISION_VERSION')!r}, expected {expected_vision!r}"
        )
    audio_version = config.get("TORCHAUDIO_VERSION")
    if (
        not isinstance(audio_version, str)
        or re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", audio_version) is None
    ):
        raise RuntimeError(
            f"{torch_pin_path} has invalid TORCHAUDIO_VERSION {audio_version!r}"
        )
    return torch_requirement(version, "cpu")


def configure_release_version(version_path: Path, release_version: str) -> str:
    """Finalize a development version without resetting an existing patch release."""
    current = version_path.read_text().strip()
    if re.fullmatch(rf"{re.escape(release_version)}\.\d+", current):
        return current
    release_full_version = f"{release_version}.0"
    _write_if_changed(version_path, f"{release_full_version}\n")
    return release_full_version


def configure_workflows(workflow_paths: Iterable[Path], test_infra_branch: str) -> int:
    """Pin test-infra actions and reusable workflows to its release branch."""
    changed = 0
    for path in workflow_paths:
        text = path.read_text()
        updated = _TEST_INFRA_MAIN_PATTERN.sub(rf"\1@{test_infra_branch}", text)
        updated = _TEST_INFRA_REF_MAIN_PATTERN.sub(rf"\1{test_infra_branch}", updated)
        changed += _write_if_changed(path, updated)
    return changed


def documentation_paths(repo_root: Path) -> List[Path]:
    """Return the maintained Markdown files that contain branch instructions."""
    paths: List[Path] = []
    for relative in _DOCUMENTATION_PATHS:
        path = repo_root / relative
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(path.rglob("*.md"))
    return sorted(paths)


def configure_documentation(
    paths: Iterable[Path], release_branch: str, release_full_version: str
) -> int:
    """Point checkout and stable SwiftPM examples at the new release."""
    changed = 0
    for path in paths:
        text = path.read_text()
        updated = _CLONE_BRANCH_PATTERN.sub(rf"\g<prefix>{release_branch}", text)
        updated = _STABLE_SWIFTPM_PATTERN.sub(
            f"swiftpm-{release_full_version}", updated
        )
        changed += _write_if_changed(path, updated)
    return changed


def validate_release_references(
    workflow_paths: Iterable[Path],
    documentation: Iterable[Path],
    test_infra_branch: str,
    release_version: str,
    release_full_version: str,
) -> None:
    """Reject release branches that retain moving branch references."""
    errors = []
    found_test_infra_reference = False
    for path in workflow_paths:
        text = path.read_text()
        references = _TEST_INFRA_BRANCH_PATTERN.findall(text)
        references.extend(_TEST_INFRA_INPUT_PATTERN.findall(text))
        found_test_infra_reference |= bool(references)
        for reference in references:
            if reference != test_infra_branch:
                errors.append(
                    f"{path}: test-infra references {reference}, expected "
                    f"{test_infra_branch}"
                )
    if not found_test_infra_reference:
        errors.append("no test-infra workflow references were found")
    expected_release_branch = f"release/{release_version}"
    expected_swiftpm_branch = f"swiftpm-{release_full_version}"
    for path in documentation:
        text = path.read_text()
        if _CLONE_BRANCH_PATTERN.search(text):
            errors.append(
                f"{path}: clone command still references viable/strict, expected "
                f"{expected_release_branch}"
            )
        for swiftpm_branch in _STABLE_SWIFTPM_PATTERN.findall(text):
            if swiftpm_branch != expected_swiftpm_branch:
                errors.append(
                    f"{path}: stable SwiftPM example references {swiftpm_branch}, "
                    f"expected {expected_swiftpm_branch}"
                )
    if errors:
        raise RuntimeError("release preparation is incomplete:\n" + "\n".join(errors))


def validate_release_files(repo_root: Path, release_version: str) -> str:
    """Validate the release version."""
    actual_version = (repo_root / "version.txt").read_text().strip()
    if re.fullmatch(rf"{re.escape(release_version)}\.\d+", actual_version) is None:
        raise RuntimeError(
            f"version.txt contains {actual_version!r}, expected a final "
            f"{release_version}.x version"
        )

    manifest_dir = repo_root / "backends/arm/public_api_manifests"
    if (manifest_dir / "api_manifest_running.toml").exists():
        expected_manifest = (
            manifest_dir / f"api_manifest_{release_version.replace('.', '_')}.toml"
        )
        if not expected_manifest.exists():
            raise RuntimeError(
                f"missing {expected_manifest}; merge the Arm API snapshot into main "
                "before cutting the release branch"
            )

    return actual_version


def configured_test_infra_branch(workflow_paths: Iterable[Path]) -> str:
    """Read the one test-infra branch already recorded in release workflows."""
    references = set()
    for path in workflow_paths:
        text = path.read_text()
        references.update(_TEST_INFRA_BRANCH_PATTERN.findall(text))
        references.update(_TEST_INFRA_INPUT_PATTERN.findall(text))
    if len(references) != 1:
        raise RuntimeError(
            "expected exactly one configured test-infra branch, found "
            + repr(sorted(references))
        )
    return references.pop()


def prepare_release(
    repo_root: Path,
    release_version: str,
    test_infra_branch: str,
    torch_version: str,
) -> str:
    """Apply every deterministic branch-cut edit and return the Torch requirement."""
    _validate_release_version(release_version)
    if not test_infra_branch:
        raise RuntimeError("test-infra branch must not be empty")

    workflow_paths = sorted((repo_root / ".github/workflows").glob("*.yml"))
    documentation = documentation_paths(repo_root)
    set_torch_version(repo_root / "torch_pin.py", torch_version)
    set_companion_versions(repo_root / "torch_pin.py", torch_version)
    enable_release_wheel(repo_root / "torch_pin.py")
    release_full_version = configure_release_version(
        repo_root / "version.txt", release_version
    )
    workflow_count = configure_workflows(workflow_paths, test_infra_branch)
    documentation_count = configure_documentation(
        documentation, f"release/{release_version}", release_full_version
    )
    validate_release_references(
        workflow_paths,
        documentation,
        test_infra_branch=test_infra_branch,
        release_version=release_version,
        release_full_version=release_full_version,
    )
    validate_release_files(repo_root, release_version)
    requirement = validate_release_build(repo_root / "torch_pin.py")
    print(
        f"Prepared release/{release_version}: {requirement}; changed "
        f"{workflow_count} workflow files and {documentation_count} documentation files"
    )
    return requirement


def _release_version_from_file(repo_root: Path) -> str:
    version = (repo_root / "version.txt").read_text().strip()
    match = re.match(r"(\d+\.\d+)", version)
    if match is None:
        raise RuntimeError(f"could not derive release version from {version!r}")
    return match.group(1)


def main() -> None:
    """Update or validate the repository's release configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the release configuration without changing it",
    )
    parser.add_argument("--release-version", help="release version in MAJOR.MINOR form")
    parser.add_argument(
        "--test-infra-branch",
        help="test-infra branch (defaults to the selected PyTorch release line)",
    )
    parser.add_argument(
        "--torch-version",
        help="Torch RC version (defaults to the newest release on the test index)",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    release_version = args.release_version or _release_version_from_file(repo_root)
    workflow_paths = sorted((repo_root / ".github/workflows").glob("*.yml"))
    documentation = documentation_paths(repo_root)

    if args.check:
        _validate_release_version(release_version)
        requirement = validate_release_build(repo_root / "torch_pin.py")
        torch_version = runpy.run_path(str(repo_root / "torch_pin.py"))["TORCH_VERSION"]
        test_infra_branch = args.test_infra_branch or configured_test_infra_branch(
            workflow_paths
        )
        release_full_version = validate_release_files(repo_root, release_version)
        validate_release_references(
            workflow_paths,
            documentation,
            test_infra_branch=test_infra_branch,
            release_version=release_version,
            release_full_version=release_full_version,
        )
        print(f"Release configuration is valid: {requirement}")
    else:
        torch_version = torch_version_for_release(
            repo_root / "torch_pin.py", args.torch_version or ""
        )
        require_companion_releases(repo_root / "torch_pin.py", torch_version)
        test_infra_branch = args.test_infra_branch or test_infra_branch_for_torch(
            torch_version
        )
        require_test_infra_branch(test_infra_branch)
        pytorch_commit = pytorch_commit_for_release(torch_version)
        prepare_release(repo_root, release_version, test_infra_branch, torch_version)
        sync_pytorch_source(repo_root, pytorch_commit)


if __name__ == "__main__":
    main()
