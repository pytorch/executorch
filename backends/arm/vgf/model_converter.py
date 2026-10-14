# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import re
import subprocess  # nosec B404 - invoked only for trusted local converter tools
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from packaging.version import InvalidVersion, Version

MODEL_CONVERTER_BINARY = "model-converter"
_MODEL_CONVERTER_FALLBACK_BINARY = "model_converter"
MIN_MODEL_CONVERTER_VERSION = Version("0.11.0")
# Keep the old name as an alias while tests/callers migrate.
MIN_MODEL_CONVERTER_VERSION_FOR_VGF_TESTS = MIN_MODEL_CONVERTER_VERSION
_MODEL_CONVERTER_VERSION_PATTERN = re.compile(r"\b\d+\.\d+\.\d+(?:[A-Za-z0-9_.+-]*)?\b")
_MODEL_CONVERTER_BUILD_VERSION_ALIASES = {
    "d8c1b8e": Version("0.9.0"),
    "19d1d0f": Version("0.10.0"),
}

STATUS_OK = "PASS"
STATUS_FAIL = "FAIL"


@dataclass(frozen=True)
class ModelConverterEnvironmentCheck:
    """One model-converter environment preflight result.

    This lives in the same module that resolves and launches the converter so
    the standalone VGF preflight CLI cannot drift from the actual compiler path.

    """

    name: str
    status: str
    detail: str
    action: str | None = None

    @property
    def ok(self) -> bool:
        """Return True when the check did not fail."""
        return self.status != STATUS_FAIL

    def to_dict(self) -> dict[str, str | None]:
        """Return the check as a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "action": self.action,
        }


def _bundled_model_converter_binary() -> Optional[str]:
    # Buck's live test packages expose this resource through a build-owned symlink.
    resource = Path(__file__).with_name("_model_converter_binary")
    try:
        executable = resource.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not _safe_is_file(executable) or not os.access(executable, os.X_OK):
        return None
    return str(executable)


def _model_converter_scripts_dir() -> Path:
    """Return the trusted scripts directory."""
    scripts_dir = sysconfig.get_path("scripts")
    if not scripts_dir:
        raise RuntimeError(
            "Unable to determine the scripts directory for the active "
            "Python environment."
        )
    return Path(scripts_dir).resolve()


def _model_converter_candidate_names() -> tuple[str, ...]:
    """Return supported model-converter executable names."""
    return (
        MODEL_CONVERTER_BINARY,
        _MODEL_CONVERTER_FALLBACK_BINARY,
    )


def find_model_converter_binary() -> Optional[str]:
    """Return a packaged converter or one from Python's scripts directory.

    Buck may supply the converter as a resource with a fixed name. Otherwise,
    use the active Python environment. PATH and MODEL_CONVERTER_PATH do not
    participate in executable selection.

    """
    binary = _bundled_model_converter_binary()
    if binary is not None:
        return binary

    try:
        scripts_dir = _model_converter_scripts_dir()
    except (OSError, RuntimeError):
        return None

    for name in _model_converter_candidate_names():
        candidate = scripts_dir / name

        # Do not follow a symlink out of the trusted scripts directory.
        try:
            if candidate.is_symlink():
                continue
        except OSError:
            continue

        if not _safe_is_file(candidate) or not os.access(candidate, os.X_OK):
            continue

        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue

        # Defense in depth: the resolved executable must still belong directly
        # to the trusted scripts directory.
        if resolved.parent != scripts_dir:
            continue

        return str(resolved)

    return None


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def model_converter_env() -> dict[str, str]:
    """Return an env dict suitable for running model-converter as a subprocess.

    If MODEL_CONVERTER_LIB_DIR is set, it is prepended to LD_LIBRARY_PATH so the
    binary can find a compatible libstdc++ (or other shared libs) without
    polluting the parent process environment.

    """
    env = dict(os.environ)
    lib_dir = env.pop("MODEL_CONVERTER_LIB_DIR", None)
    if lib_dir:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    return env


def require_model_converter_binary() -> str:
    """Return the trusted model-converter executable or raise an error."""
    binary = find_model_converter_binary()
    if binary is None:
        try:
            scripts_dir = _model_converter_scripts_dir()
            location = str(scripts_dir)
        except (OSError, RuntimeError):
            location = "<unknown>"

        raise RuntimeError(
            "Unable to locate model-converter in the active Python "
            f"environment's scripts directory ({location}). "
            "Install the VGF dependencies into this Python environment. "
            "PATH and MODEL_CONVERTER_PATH are intentionally not used for "
            "executable discovery."
        )

    return binary


def resolve_model_converter_executable(binary: str) -> Path | None:
    """Validate an already-resolved model-converter executable.

    Executable names are deliberately not resolved through PATH here. The caller
    must provide an absolute path obtained from the trusted Python environment
    resolver.

    """
    path = Path(binary)

    if not path.is_absolute():
        return None

    try:
        if path.is_symlink():
            return None
    except OSError:
        return None

    if not _safe_is_file(path) or not os.access(path, os.X_OK):
        return None

    try:
        return path.resolve(strict=True)
    except OSError:
        return None


def require_model_converter_executable() -> Path:
    """Return the trusted model-converter executable path."""
    binary = require_model_converter_binary()
    executable = resolve_model_converter_executable(binary)
    if executable is None:
        raise RuntimeError(
            f"Resolved trusted converter candidate {binary!r}, but it is not "
            "an executable regular file."
        )
    return executable


def _command_output(result: subprocess.CompletedProcess[str]) -> str:
    text = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    lines = text.splitlines()
    if not lines:
        return "<no output>"
    return "\n".join(lines[:4])


def get_model_converter_version_text() -> str | None:
    """Return the raw ``model-converter --version`` output, if available."""
    binary = find_model_converter_binary()
    if binary is None:
        return None

    executable = resolve_model_converter_executable(binary)
    if executable is None:
        return None

    try:
        result = subprocess.run(  # nosec B603 - trusted local converter tool
            [str(executable), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
            env=model_converter_env(),
        )
    except Exception:
        return None

    version_text = (result.stdout or result.stderr).strip()
    return version_text or None


def parse_model_converter_version(version_text: str) -> Version | None:
    """Parse a comparable model-converter version from ``--version`` output."""
    match = _MODEL_CONVERTER_VERSION_PATTERN.search(version_text)
    if match is not None:
        try:
            return Version(match.group(0))
        except InvalidVersion:
            pass

    for revision, version in _MODEL_CONVERTER_BUILD_VERSION_ALIASES.items():
        if revision in version_text:
            return version
    return None


def get_model_converter_minimum_version_failure_reason(
    version_text: str,
    minimum_version: Version,
    *,
    requirement_name: str,
) -> str | None:
    """Return a reason when the installed converter is unsupported.

    The converter is unsupported when it is below ``minimum_version``.

    """
    version = parse_model_converter_version(version_text)
    if version is None or version >= minimum_version:
        return None
    return (
        f"{version_text} is below the minimum supported version "
        f"{minimum_version} required for {requirement_name}"
    )


def check_model_converter_environment() -> ModelConverterEnvironmentCheck:
    """Check the model-converter dependency used by VGF compilation."""
    binary = find_model_converter_binary()
    if binary is None:
        try:
            scripts_dir = _model_converter_scripts_dir()
            location = str(scripts_dir)
        except (OSError, RuntimeError):
            location = "<unknown>"

        return ModelConverterEnvironmentCheck(
            "MLSDK model converter",
            STATUS_FAIL,
            "Could not find model-converter in the active Python "
            f"environment's scripts directory ({location}).",
            "Install VGF AoT dependencies into the active Python environment "
            "with python -m pip install 'executorch[vgf]' or, in a source "
            "checkout, python -m pip install -r "
            "backends/arm/requirements-arm-vgf.txt. PATH and "
            "MODEL_CONVERTER_PATH are intentionally not used for executable "
            "discovery.",
        )

    executable = resolve_model_converter_executable(binary)
    if executable is None:
        return ModelConverterEnvironmentCheck(
            "MLSDK model converter",
            STATUS_FAIL,
            f"Resolved trusted converter candidate {binary!r}, but it is not "
            "an executable regular file.",
            "Reinstall the model-converter package in the active Python "
            "environment.",
        )

    try:
        result = subprocess.run(  # nosec B603 - local converter executable
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env=model_converter_env(),
        )
    except Exception as exc:
        return ModelConverterEnvironmentCheck(
            "MLSDK model converter",
            STATUS_FAIL,
            f"Found {executable}, but running '--version' failed: {exc}",
            "Check MODEL_CONVERTER_LIB_DIR and the process loader paths. "
            "Reinstall model-converter into the active Python environment "
            "if necessary.",
        )

    if result.returncode != 0:
        return ModelConverterEnvironmentCheck(
            "MLSDK model converter",
            STATUS_FAIL,
            f"{executable} --version exited with {result.returncode}:\n"
            f"{_command_output(result)}",
            "Check that the model-converter binary and its shared libraries are "
            "from the same MLSDK install.",
        )

    version_text = _command_output(result)
    version = parse_model_converter_version(version_text)
    if version is None:
        return ModelConverterEnvironmentCheck(
            "MLSDK model converter",
            STATUS_FAIL,
            f"{executable} --version succeeded, but its version could not be "
            f"parsed:\n{version_text}",
            f"ExecuTorch VGF requires model-converter "
            f">={MIN_MODEL_CONVERTER_VERSION}, but the installed converter "
            "version could not be verified. Install the VGF dependencies from "
            "this ExecuTorch checkout. If this is a known compatible custom "
            "build, add its revision to _MODEL_CONVERTER_BUILD_VERSION_ALIASES.",
        )

    if version < MIN_MODEL_CONVERTER_VERSION:
        return ModelConverterEnvironmentCheck(
            "MLSDK model converter",
            STATUS_FAIL,
            f"Found model-converter {version}, but ExecuTorch VGF requires "
            f">={MIN_MODEL_CONVERTER_VERSION} for the pinned VGF toolchain.",
            "Install the VGF dependencies from this ExecuTorch checkout with "
            "python -m pip install -r backends/arm/requirements-arm-vgf.txt, "
            "or reinstall the matching executorch[vgf] extra.",
        )

    return ModelConverterEnvironmentCheck(
        "MLSDK model converter",
        STATUS_OK,
        f"{executable} --version succeeded (version={version}):\n{version_text}",
    )


def check_model_converter_lib_dir_environment() -> ModelConverterEnvironmentCheck:
    """Check MODEL_CONVERTER_LIB_DIR used by model_converter_env()."""
    lib_dir = os.environ.get("MODEL_CONVERTER_LIB_DIR")
    if not lib_dir:
        return ModelConverterEnvironmentCheck(
            "MODEL_CONVERTER_LIB_DIR",
            STATUS_OK,
            "MODEL_CONVERTER_LIB_DIR is not set; relying on the process loader "
            "paths. This is OK when model-converter --version succeeds.",
        )

    path = Path(lib_dir).expanduser()
    if _safe_is_dir(path):
        return ModelConverterEnvironmentCheck(
            "MODEL_CONVERTER_LIB_DIR",
            STATUS_OK,
            f"MODEL_CONVERTER_LIB_DIR points to existing directory: {path}",
        )

    return ModelConverterEnvironmentCheck(
        "MODEL_CONVERTER_LIB_DIR",
        STATUS_FAIL,
        f"MODEL_CONVERTER_LIB_DIR={lib_dir!r} does not exist or is not a directory.",
        "Unset MODEL_CONVERTER_LIB_DIR or set it to the converter library directory.",
    )
