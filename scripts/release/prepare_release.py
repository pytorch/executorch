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
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable, List
from urllib.parse import unquote

_RELEASE_VERSION_PATTERN = re.compile(r"\d+\.\d+")
_RELEASE_WHEEL_PATTERN = re.compile(
    r"^RELEASE_WHEEL\s*=\s*(?:True|False)$", re.MULTILINE
)
_TORCH_VERSION_PATTERN = re.compile(r'^TORCH_VERSION\s*=\s*"([^"]+)"$', re.MULTILINE)
_TEST_INFRA_MAIN_PATTERN = re.compile(r"(pytorch/test-infra/[^\s\"']+)@main\b")
_TEST_INFRA_REF_MAIN_PATTERN = re.compile(r"(test-infra-ref:\s*)main\b")
_TEST_INFRA_BRANCH_PATTERN = re.compile(r"pytorch/test-infra/[^\s\"'@]+@([^\s\"']+)")
_TEST_INFRA_INPUT_PATTERN = re.compile(r"test-infra-ref:\s*([^\s#]+)")
_CLONE_BRANCH_PATTERN = re.compile(
    r"(?P<prefix>-b\s+)(?P<branch>viable/strict|release/\d+\.\d+)\b"
)
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


def _write_if_changed(path: Path, original: str, updated: str) -> bool:
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
    return _write_if_changed(torch_pin_path, text, updated)


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
    return _write_if_changed(torch_pin_path, text, updated)


def newest_torch_test_release(available_versions: Iterable[str]) -> str:
    """Return the newest RC or final-form version on the PyTorch test index."""
    candidates = []
    stage_rank = {"a": 0, "b": 1, "rc": 2, None: 3}
    for version in available_versions:
        match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?", version)
        if match is None:
            continue
        major, minor, patch = (int(part) for part in match.groups()[:3])
        stage = match.group(4)
        stage_number = int(match.group(5) or 0)
        candidates.append(
            ((major, minor, patch, stage_rank[stage], stage_number), version)
        )
    if not candidates:
        raise RuntimeError("PyTorch test index has no release candidate wheels")
    return max(candidates)[1]


def latest_torch_test_release() -> str:
    """Look up the newest PyTorch release candidate on its CPU test index."""
    with urllib.request.urlopen(
        "https://download.pytorch.org/whl/test/cpu/torch/", timeout=30
    ) as response:
        index = unquote(response.read().decode())
    versions = re.findall(
        r"torch-(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)(?:\+[^-]+)?-", index
    )
    return newest_torch_test_release(versions)


def torch_version_for_release(torch_pin_path: Path, override: str = "") -> str:
    """Choose a pin once, then preserve it on subsequent preparation runs."""
    if override:
        return override
    config = runpy.run_path(str(torch_pin_path))
    if config.get("RELEASE_WHEEL") is True:
        return config["TORCH_VERSION"]
    return latest_torch_test_release()


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
    return f"torch>={version}"


def configure_release_version(version_path: Path, release_version: str) -> bool:
    """Set the package version to the first final release in this series."""
    return _write_if_changed(
        version_path, version_path.read_text(), f"{release_version}.0\n"
    )


def configure_workflows(workflow_paths: Iterable[Path], test_infra_branch: str) -> int:
    """Pin test-infra actions and reusable workflows to its release branch."""
    changed = 0
    for path in workflow_paths:
        text = path.read_text()
        updated = _TEST_INFRA_MAIN_PATTERN.sub(rf"\1@{test_infra_branch}", text)
        updated = _TEST_INFRA_REF_MAIN_PATTERN.sub(rf"\1{test_infra_branch}", updated)
        changed += _write_if_changed(path, text, updated)
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
    paths: Iterable[Path], release_branch: str, release_version: str
) -> int:
    """Point checkout and stable SwiftPM examples at the new release."""
    changed = 0
    for path in paths:
        text = path.read_text()
        updated = _CLONE_BRANCH_PATTERN.sub(rf"\g<prefix>{release_branch}", text)
        updated = _STABLE_SWIFTPM_PATTERN.sub(f"swiftpm-{release_version}.0", updated)
        changed += _write_if_changed(path, text, updated)
    return changed


def freeze_arm_public_api(repo_root: Path, release_version: str) -> bool:
    """Snapshot the Arm API and retain the two newest release manifests."""
    manifest_dir = repo_root / "backends/arm/public_api_manifests"
    running = manifest_dir / "api_manifest_running.toml"
    if not running.exists():
        return False

    target = manifest_dir / f"api_manifest_{release_version.replace('.', '_')}.toml"
    created = False
    if not target.exists():
        shutil.copyfile(running, target)
        created = True

    static_manifests = sorted(
        manifest_dir.glob("api_manifest_[0-9]*_[0-9]*.toml"),
        key=lambda path: tuple(
            int(part) for part in path.stem.removeprefix("api_manifest_").split("_")
        ),
    )
    for obsolete in static_manifests[:-2]:
        obsolete.unlink()
    return created


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
        for match in _CLONE_BRANCH_PATTERN.finditer(text):
            if match.group("branch") != expected_release_branch:
                errors.append(
                    f"{path}: clone command references {match.group('branch')}, "
                    f"expected {expected_release_branch}"
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
    """Validate release version, API snapshot, and stable QNN installation."""
    actual_version = (repo_root / "version.txt").read_text().strip()
    if re.fullmatch(rf"{re.escape(release_version)}\.\d+", actual_version) is None:
        raise RuntimeError(
            f"version.txt contains {actual_version!r}, expected a final "
            f"{release_version}.x version"
        )

    manifest_dir = repo_root / "backends/arm/public_api_manifests"
    if manifest_dir.exists():
        expected_manifest = (
            manifest_dir / f"api_manifest_{release_version.replace('.', '_')}.toml"
        )
        if not expected_manifest.exists():
            raise RuntimeError(f"missing release API snapshot {expected_manifest}")
        static_manifests = list(manifest_dir.glob("api_manifest_[0-9]*_[0-9]*.toml"))
        if len(static_manifests) > 2:
            raise RuntimeError("more than two static Arm API manifests remain")

    return actual_version


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
    enable_release_wheel(repo_root / "torch_pin.py")
    configure_release_version(repo_root / "version.txt", release_version)
    workflow_count = configure_workflows(workflow_paths, test_infra_branch)
    documentation_count = configure_documentation(
        documentation, f"release/{release_version}", release_version
    )
    manifest_created = freeze_arm_public_api(repo_root, release_version)
    validate_release_references(
        workflow_paths,
        documentation,
        test_infra_branch=test_infra_branch,
        release_version=release_version,
        release_full_version=f"{release_version}.0",
    )
    validate_release_files(repo_root, release_version)
    requirement = validate_release_build(repo_root / "torch_pin.py")
    print(
        f"Prepared release/{release_version}: {requirement}; changed "
        f"{workflow_count} workflow files and {documentation_count} documentation files; "
        f"Arm API snapshot {'created' if manifest_created else 'already present'}"
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
        help="test-infra release branch (defaults to release/MAJOR.MINOR)",
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
    test_infra_branch = args.test_infra_branch or f"release/{release_version}"
    workflow_paths = sorted((repo_root / ".github/workflows").glob("*.yml"))
    documentation = documentation_paths(repo_root)

    if args.check:
        _validate_release_version(release_version)
        requirement = validate_release_build(repo_root / "torch_pin.py")
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
        prepare_release(repo_root, release_version, test_infra_branch, torch_version)


if __name__ == "__main__":
    main()
