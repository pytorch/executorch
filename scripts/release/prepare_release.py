#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply and validate the deterministic changes required on release branches."""

import argparse
import html
import io
import re
import runpy
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List
from urllib.error import HTTPError
from urllib.parse import unquote, urljoin

from release_versions import (  # type: ignore[import-not-found]
    is_release_version,
    release_key,
    torch_requirement,
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
_CU134_TORCH_PACKAGES_PATTERN = re.compile(
    r"^CU134_TORCH_PACKAGES = \[\n.*?^\]\n", re.MULTILINE | re.DOTALL
)
_TORCHAO_URL_PATTERN = re.compile(r'^TORCHAO_URL_BASE\s*=\s*"[^"]+"$', re.MULTILINE)
_TORCHAO_VERSION_PATTERN = re.compile(
    r'^TORCHAO_NIGHTLY_VERSION\s*=\s*"[^"]+"$', re.MULTILINE
)
_CU134_TORCHAO_VERSION_PATTERN = re.compile(
    r'^CU134_TORCHAO_NIGHTLY_VERSION\s*=\s*"[^"]+"$', re.MULTILINE
)
_SUPPORTED_CUDA_PATTERN = re.compile(
    r"^SUPPORTED_CUDA_VERSIONS: List\[str\] = \[[^\n]*\]$", re.MULTILINE
)
_RELEASE_CUDA_CANDIDATES_PATTERN = re.compile(
    r"^RELEASE_CUDA_CANDIDATES: List\[str\] = \[[^\n]*\]$", re.MULTILINE
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


@dataclass(frozen=True)
class WheelLink:
    """One package-index wheel and its optional PEP 658 metadata."""

    version: str
    installed_version: str
    url: str
    metadata_url: str | None


class _WheelIndexParser(HTMLParser):
    def __init__(self, index_url: str, package: str) -> None:
        super().__init__()
        self.index_url = index_url
        self.package = package
        self.wheels: dict[str, list[WheelLink]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        href = attributes.get("href")
        if not href:
            return
        wheel_url = urljoin(self.index_url, html.unescape(href).split("#", 1)[0])
        filename = unquote(wheel_url).rsplit("/", 1)[-1]
        match = re.match(rf"{re.escape(self.package)}-([^-]+)-.*\.whl$", filename)
        if match is None:
            return
        installed_version = match.group(1)
        version = installed_version.partition("+")[0]
        if not is_release_version(version):
            return
        has_metadata = any(
            name in attributes
            for name in ("data-core-metadata", "data-dist-info-metadata")
        )
        self.wheels.setdefault(version, []).append(
            WheelLink(
                version=version,
                installed_version=installed_version,
                url=wheel_url,
                metadata_url=f"{wheel_url}.metadata" if has_metadata else None,
            )
        )


class _RemoteWheel(io.RawIOBase):
    """Seekable range reader used to inspect a wheel without downloading it."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.position = 0
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            content_range = response.headers.get("Content-Range", "")
        if "/" not in content_range:
            raise RuntimeError(f"wheel server does not support byte ranges for {url}")
        self.length = int(content_range.rsplit("/", 1)[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.length + offset
        else:
            raise ValueError(f"invalid seek mode {whence}")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size == 0 or self.position >= self.length:
            return b""
        end = (
            self.length - 1
            if size < 0
            else min(self.length - 1, self.position + size - 1)
        )
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
        self.position += len(data)
        return data


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
    if not is_release_version(torch_version, allow_prerelease=False):
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


def set_companion_versions(
    torch_pin_path: Path, torchvision: str, torchaudio: str
) -> bool:
    """Set domain-library releases that declare the selected Torch version."""
    text = torch_pin_path.read_text()
    updated, vision_count = _TORCHVISION_VERSION_PATTERN.subn(
        f'TORCHVISION_VERSION = "{torchvision}"', text
    )
    updated, audio_count = _TORCHAUDIO_VERSION_PATTERN.subn(
        f'TORCHAUDIO_VERSION = "{torchaudio}"', updated
    )
    if vision_count != 1 or audio_count != 1:
        raise RuntimeError(
            f"expected exactly one torchvision and torchaudio assignment in {torch_pin_path}"
        )
    return _write_if_changed(torch_pin_path, updated)


def set_cu134_versions(
    install_requirements_path: Path,
    torch_version: str,
    torchvision: str,
    torchaudio: str,
) -> bool:
    """Move the exceptional cu134 package list to the selected release train."""
    text = install_requirements_path.read_text()
    packages = (
        "CU134_TORCH_PACKAGES = [\n"
        f'    "torch=={torch_version}+cu134",\n'
        f'    "torchvision=={torchvision}+cu134",\n'
        f'    "torchaudio=={torchaudio}+cu134",\n'
        "]\n"
    )
    updated, count = _CU134_TORCH_PACKAGES_PATTERN.subn(packages, text)
    if count != 1:
        raise RuntimeError(
            f"expected one CU134_TORCH_PACKAGES list in {install_requirements_path}"
        )
    return _write_if_changed(install_requirements_path, updated)


def set_release_torchao(install_requirements_path: Path, version: str) -> bool:
    """Pin TorchAO to a retained test-index release instead of a nightly."""
    if not is_release_version(version, allow_prerelease=False):
        raise RuntimeError(f"invalid TorchAO release version {version!r}")
    text = install_requirements_path.read_text()
    updated, url_count = _TORCHAO_URL_PATTERN.subn(
        'TORCHAO_URL_BASE = "https://download.pytorch.org/whl/test"', text
    )
    updated, version_count = _TORCHAO_VERSION_PATTERN.subn(
        f'TORCHAO_NIGHTLY_VERSION = "{version}"', updated
    )
    updated, cu_count = _CU134_TORCHAO_VERSION_PATTERN.subn(
        f'CU134_TORCHAO_NIGHTLY_VERSION = "{version}"', updated
    )
    if (url_count, version_count, cu_count) != (1, 1, 1):
        raise RuntimeError(
            f"expected one TorchAO URL and two version pins in {install_requirements_path}"
        )
    return _write_if_changed(install_requirements_path, updated)


def configured_cuda_variants(filter_path: Path) -> list[str]:
    """Read the CUDA wheel trains selected by the release matrix filter."""
    match = _SUPPORTED_CUDA_PATTERN.search(filter_path.read_text())
    if match is None:
        raise RuntimeError(f"could not read supported CUDA versions from {filter_path}")
    return re.findall(r'"(cu\d+)"', match.group(0))


def release_cuda_candidates(filter_path: Path) -> list[str]:
    """Read every CUDA train that release preparation may select."""
    match = _RELEASE_CUDA_CANDIDATES_PATTERN.search(filter_path.read_text())
    if match is None:
        raise RuntimeError(f"could not read release CUDA candidates from {filter_path}")
    return re.findall(r'"(cu\d+)"', match.group(0))


def set_cuda_variants(filter_path: Path, variants: Iterable[str]) -> bool:
    """Drop CUDA trains for which the selected upstream release has no wheels."""
    selected = list(variants)
    if not selected:
        raise RuntimeError("the selected PyTorch release has no supported CUDA trains")
    if any(re.fullmatch(r"cu\d+", variant) is None for variant in selected):
        raise RuntimeError(f"invalid CUDA release variants {selected!r}")
    replacement = "SUPPORTED_CUDA_VERSIONS: List[str] = " + repr(selected).replace(
        "'", '"'
    )
    text = filter_path.read_text()
    updated, count = _SUPPORTED_CUDA_PATTERN.subn(replacement, text)
    if count != 1:
        raise RuntimeError(f"expected one supported CUDA list in {filter_path}")
    return _write_if_changed(filter_path, updated)


def newest_torch_test_release(
    available_versions: Iterable[str], newer_than: str = ""
) -> str:
    """Return the newest final-form wheel on a newer PyTorch release train."""
    candidates = []
    for version in available_versions:
        if not is_release_version(version, allow_prerelease=False):
            continue
        key = release_key(version)
        if newer_than and key[:2] <= release_key(newer_than)[:2]:
            continue
        candidates.append((key, version))
    if not candidates:
        qualifier = f" newer than {newer_than}" if newer_than else ""
        raise RuntimeError(f"PyTorch test index has no release wheel{qualifier}")
    return max(candidates)[1]


@lru_cache(maxsize=None)
def _test_index_wheels(
    package: str, variant: str = "cpu", channel: str = "test"
) -> dict[str, list[WheelLink]]:
    channel_path = f"{channel}/" if channel else ""
    index_url = f"https://download.pytorch.org/whl/{channel_path}{variant}/{package}/"  # @lint-ignore
    try:
        with urllib.request.urlopen(index_url, timeout=30) as response:
            index = response.read().decode()
    except HTTPError as error:
        if error.code in (403, 404) and variant != "cpu":
            return {}
        raise
    parser = _WheelIndexParser(index_url, package)
    parser.feed(index)
    return parser.wheels


def _test_index_versions(
    package: str, variant: str = "cpu", channel: str = "test"
) -> set[str]:
    return set(_test_index_wheels(package, variant, channel))


@lru_cache(maxsize=None)
def _wheel_member(wheel_url: str, suffix: str) -> str:
    with zipfile.ZipFile(_RemoteWheel(wheel_url)) as wheel:
        matches = [name for name in wheel.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one {suffix} in {wheel_url}, found {len(matches)}"
            )
        return wheel.read(matches[0]).decode()


@lru_cache(maxsize=None)
def _wheel_metadata(wheel: WheelLink) -> str:
    if wheel.metadata_url:
        with urllib.request.urlopen(wheel.metadata_url, timeout=30) as response:
            return response.read().decode()
    return _wheel_member(wheel.url, ".dist-info/METADATA")


def _preferred_wheel(wheels: list[WheelLink]) -> WheelLink:
    """Choose a common Linux wheel, with a deterministic fallback."""
    return min(
        wheels,
        key=lambda wheel: (
            "manylinux" not in wheel.url or "x86_64" not in wheel.url,
            "cp310" not in wheel.url,
            wheel.url,
        ),
    )


def companion_release_for_torch(
    package: str, torch_version: str, torch_installed_version: str = ""
) -> str:
    """Find the newest domain-library wheel compatible with this Torch build."""
    wheels = _test_index_wheels(package)
    installed = torch_installed_version or torch_version
    for version in sorted(wheels, key=release_key, reverse=True):
        if companion_release_supports_torch(package, version, installed):
            return version
    raise RuntimeError(
        f"PyTorch test index has no {package} release compatible with torch "
        f"{installed}"
    )


def companion_release_supports_torch(
    package: str,
    package_version: str,
    torch_installed_version: str,
    channel: str = "test",
) -> bool:
    """Whether any wheel for a companion release accepts the selected Torch build."""
    for wheel in _test_index_wheels(package, channel=channel).get(package_version, []):
        metadata = Parser().parsestr(_wheel_metadata(wheel))
        for raw_requirement in metadata.get_all("Requires-Dist", []):
            match = re.match(
                r"\s*torch\s*(?:\(\s*([^)]*)\s*\)|([^;]*))?",
                raw_requirement,
                re.IGNORECASE,
            )
            if match and _specifier_allows_version(
                (match.group(1) or match.group(2) or "").strip(),
                torch_installed_version,
            ):
                return True
    return False


def _specifier_allows_version(specifier: str, installed_version: str) -> bool:
    """Evaluate the release specifiers used by PyTorch companion wheels."""
    installed_public = installed_version.partition("+")[0]
    installed_key = release_key(installed_public)
    for clause in filter(None, (part.strip() for part in specifier.split(","))):
        match = re.fullmatch(r"(===|==|!=|<=|>=|<|>|~=)\s*([^\s]+)", clause)
        if match is None:
            raise RuntimeError(f"unsupported Torch requirement {specifier!r}")
        operator, wanted = match.groups()
        wanted_public = wanted.partition("+")[0]
        if re.fullmatch(r"\d+\.\d+", wanted_public):
            wanted_public += ".0"
        wanted_key = release_key(wanted_public)
        equal = (
            installed_version == wanted
            if "+" in wanted
            else installed_public == wanted_public
        )
        accepted = {
            "==": equal,
            "===": installed_version == wanted,
            "!=": not equal,
            "<=": installed_key <= wanted_key,
            ">=": installed_key >= wanted_key,
            "<": installed_key < wanted_key,
            ">": installed_key > wanted_key,
        }.get(operator)
        if operator == "~=":
            release = wanted_public.split(".")
            upper = (
                (int(release[0]) + 1, 0)
                if len(release) == 2
                else (
                    int(release[0]),
                    int(release[1]) + 1,
                )
            )
            accepted = installed_key >= wanted_key and installed_key[:2] < upper
        if not accepted:
            return False
    return True


def companion_releases_for_torch(torch_version: str) -> dict[str, str]:
    """Resolve compatible CPU companions for the selected Torch wheel."""
    torch_wheels = _test_index_wheels("torch").get(torch_version, [])
    if not torch_wheels:
        raise RuntimeError(f"PyTorch test index does not contain torch {torch_version}")
    torch_installed_version = _preferred_wheel(torch_wheels).installed_version
    expected = {
        package: companion_release_for_torch(
            package, torch_version, torch_installed_version
        )
        for package in ("torchvision", "torchaudio")
    }
    expected["torch"] = torch_version
    return expected


def available_cuda_variants(
    candidates: Iterable[str], releases: dict[str, str]
) -> list[str]:
    """Keep CUDA trains for which every selected PyTorch package exists."""
    return [
        variant
        for variant in candidates
        if all(
            version in _test_index_versions(package, variant)
            for package, version in releases.items()
        )
    ]


def torch_version_for_release(torch_pin_path: Path, override: str = "") -> str:
    """Choose a pin once, then preserve it on subsequent preparation runs."""
    config = runpy.run_path(str(torch_pin_path))
    current = config["TORCH_VERSION"]
    if override:
        if not is_release_version(override, allow_prerelease=False):
            raise RuntimeError(
                "Torch wheel versions on the test index must use final X.Y.Z form, "
                f"received {override!r}"
            )
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
    return newest_torch_test_release(_test_index_versions("torch"), newer_than=current)


def test_infra_branch_for_torch(torch_version: str) -> str:
    """Return the test-infra release branch matching a PyTorch release."""
    major, minor, _patch, _stage, _stage_number = release_key(torch_version)
    return f"release/{major}.{minor}"


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


def pytorch_commit_for_wheel(torch_version: str, channel: str = "test") -> str:
    """Read the exact source commit embedded in the selected binary wheel."""
    wheels = _test_index_wheels("torch", channel=channel).get(torch_version, [])
    if not wheels:
        raise RuntimeError(f"PyTorch test index has no torch {torch_version} wheel")
    version_module = _wheel_member(_preferred_wheel(wheels).url, "torch/version.py")
    version_match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]+)", version_module, re.MULTILINE
    )
    commit_match = re.search(
        r"^git_version\s*=\s*['\"]([0-9a-f]{40})", version_module, re.MULTILINE
    )
    if version_match is None or commit_match is None:
        raise RuntimeError("selected PyTorch wheel does not record its build commit")
    if version_match.group(1).partition("+")[0] != torch_version:
        raise RuntimeError(
            f"selected wheel reports {version_match.group(1)}, expected {torch_version}"
        )
    return commit_match.group(1)


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


def _release_torchao_version(
    install_requirements_path: Path, finalized: bool
) -> tuple[str, str]:
    requirements_text = install_requirements_path.read_text()
    url_match = _TORCHAO_URL_PATTERN.search(requirements_text)
    version_match = _TORCHAO_VERSION_PATTERN.search(requirements_text)
    expected_index = "/whl" if finalized else "/whl/test"
    if url_match is None or not url_match.group(0).endswith(f'{expected_index}"'):
        raise RuntimeError(f"release TorchAO must come from the {expected_index} index")
    if version_match is None:
        raise RuntimeError("release TorchAO pin is missing")
    torchao_version = version_match.group(0).split('"')[1]
    if not is_release_version(torchao_version, allow_prerelease=False):
        raise RuntimeError("release TorchAO must use a non-nightly version")
    return torchao_version, requirements_text


def _validate_cuda_releases(
    releases: dict[str, str],
    requirements_text: str,
    cuda_filter_path: Path,
    verify_index: bool,
    channel: str,
) -> None:
    variants = configured_cuda_variants(cuda_filter_path)
    if verify_index:
        missing = [
            f"{package}=={package_version}+{variant}"
            for variant in variants
            for package, package_version in releases.items()
            if package_version
            not in _test_index_versions(package, variant, channel=channel)
        ]
        if missing:
            raise RuntimeError(
                "configured CUDA trains are missing release packages: "
                + ", ".join(missing)
            )
    if "cu134" not in variants:
        return
    expected = {
        f"{package}=={package_version}+cu134"
        for package, package_version in releases.items()
        if package != "torchao"
    }
    actual = set(
        re.findall(
            r'"(torch(?:vision|audio)?==[^"]+\+cu134)"',
            requirements_text,
        )
    )
    if actual != expected:
        raise RuntimeError(
            f"cu134 pins {sorted(actual)!r} do not match {sorted(expected)!r}"
        )


def validate_release_build(
    torch_pin_path: Path,
    install_requirements_path: Path | None = None,
    cuda_filter_path: Path | None = None,
    verify_index: bool = True,
) -> str:
    """Return the release PyTorch requirement or raise for invalid config."""
    config = runpy.run_path(str(torch_pin_path))
    if config.get("RELEASE_WHEEL") is not True:
        raise RuntimeError(f"{torch_pin_path} does not enable release wheel metadata")

    version = config.get("TORCH_VERSION")
    if not isinstance(version, str) or not is_release_version(
        version, allow_prerelease=False
    ):
        raise RuntimeError(f"{torch_pin_path} has invalid TORCH_VERSION {version!r}")
    for name in ("TORCHVISION_VERSION", "TORCHAUDIO_VERSION"):
        if not is_release_version(config.get(name), allow_prerelease=False):
            raise RuntimeError(
                f"{torch_pin_path} has invalid {name} {config.get(name)!r}"
            )
    if install_requirements_path is None or cuda_filter_path is None:
        return torch_requirement(version, "cpu")

    releases = {"torch": version}
    for package, name in (
        ("torchvision", "TORCHVISION_VERSION"),
        ("torchaudio", "TORCHAUDIO_VERSION"),
    ):
        package_version = config[name]
        releases[package] = package_version

    finalized = config.get("RELEASE_DEPENDENCIES_FINALIZED") is True
    channel = "" if finalized else "test"
    if verify_index:
        torch_wheels = _test_index_wheels("torch", channel=channel).get(version, [])
        if not torch_wheels:
            raise RuntimeError(f"test index no longer contains torch {version}")
        installed = _preferred_wheel(torch_wheels).installed_version
        for package, name in (
            ("torchvision", "TORCHVISION_VERSION"),
            ("torchaudio", "TORCHAUDIO_VERSION"),
        ):
            if not companion_release_supports_torch(
                package, config[name], installed, channel=channel
            ):
                raise RuntimeError(
                    f"{package} {config[name]} is not compatible with torch {installed}"
                )

    torchao_version, requirements_text = _release_torchao_version(
        install_requirements_path, finalized
    )
    releases["torchao"] = torchao_version
    _validate_cuda_releases(
        releases, requirements_text, cuda_filter_path, verify_index, channel
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
    companions: dict[str, str],
    torchao_version: str,
    cuda_variants: list[str],
    verify_index: bool = True,
) -> str:
    """Apply every deterministic branch-cut edit and return the Torch requirement."""
    _validate_release_version(release_version)
    if not test_infra_branch:
        raise RuntimeError("test-infra branch must not be empty")

    workflow_paths = sorted((repo_root / ".github/workflows").glob("*.yml"))
    documentation = documentation_paths(repo_root)
    set_torch_version(repo_root / "torch_pin.py", torch_version)
    set_companion_versions(
        repo_root / "torch_pin.py",
        companions["torchvision"],
        companions["torchaudio"],
    )
    install_requirements_path = repo_root / "install_requirements.py"
    if "cu134" in cuda_variants:
        set_cu134_versions(
            install_requirements_path,
            torch_version,
            companions["torchvision"],
            companions["torchaudio"],
        )
    set_release_torchao(install_requirements_path, torchao_version)
    cuda_filter_path = repo_root / ".github/scripts/filter_cuda_matrix.py"
    set_cuda_variants(cuda_filter_path, cuda_variants)
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
    requirement = validate_release_build(
        repo_root / "torch_pin.py",
        install_requirements_path,
        cuda_filter_path,
        verify_index=verify_index,
    )
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
        help="Torch wheel version (defaults to the newest release on the test index)",
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
        requirement = validate_release_build(
            repo_root / "torch_pin.py",
            repo_root / "install_requirements.py",
            repo_root / ".github/scripts/filter_cuda_matrix.py",
        )
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
        current_config = runpy.run_path(str(repo_root / "torch_pin.py"))
        reuse_selected_dependencies = (
            current_config.get("RELEASE_WHEEL") is True and not args.torch_version
        )
        torch_version = torch_version_for_release(
            repo_root / "torch_pin.py", args.torch_version or ""
        )
        cuda_filter_path = repo_root / ".github/scripts/filter_cuda_matrix.py"
        if reuse_selected_dependencies:
            companions = {
                "torch": torch_version,
                "torchvision": current_config["TORCHVISION_VERSION"],
                "torchaudio": current_config["TORCHAUDIO_VERSION"],
            }
            requirements_text = (repo_root / "install_requirements.py").read_text()
            torchao_match = _TORCHAO_VERSION_PATTERN.search(requirements_text)
            if torchao_match is None:
                raise RuntimeError("release TorchAO pin is missing")
            torchao_version = torchao_match.group(0).split('"')[1]
            releases = dict(companions)
            releases["torchao"] = torchao_version
            cuda_variants = available_cuda_variants(
                release_cuda_candidates(cuda_filter_path), releases
            )
        else:
            companions = companion_releases_for_torch(torch_version)
            torchao_version = newest_torch_test_release(_test_index_versions("torchao"))
            releases = dict(companions)
            releases["torchao"] = torchao_version
            cuda_variants = available_cuda_variants(
                release_cuda_candidates(cuda_filter_path), releases
            )
        pytorch_commit = pytorch_commit_for_wheel(torch_version)
        test_infra_branch = args.test_infra_branch or test_infra_branch_for_torch(
            torch_version
        )
        require_test_infra_branch(test_infra_branch)
        sync_pytorch_source(repo_root, pytorch_commit)
        prepare_release(
            repo_root,
            release_version,
            test_infra_branch,
            torch_version,
            companions,
            torchao_version,
            cuda_variants,
            verify_index=True,
        )


if __name__ == "__main__":
    main()
