# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import subprocess  # nosec B404 - invoked only for trusted local converter tools
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Optional

MODEL_CONVERTER_BINARY = "model-converter"
_MODEL_CONVERTER_FALLBACK_BINARY = "model_converter"

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
        return self.status != STATUS_FAIL

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "action": self.action,
        }


def find_model_converter_binary() -> Optional[str]:
    """Return the path/name of the first model converter executable found."""
    env_path = os.environ.get("MODEL_CONVERTER_PATH")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    for candidate in (MODEL_CONVERTER_BINARY, _MODEL_CONVERTER_FALLBACK_BINARY):
        if which(candidate):
            return candidate
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
    """Return a usable model converter executable or raise a helpful error."""

    binary = find_model_converter_binary()
    if binary is None:
        tried = ", ".join((MODEL_CONVERTER_BINARY, _MODEL_CONVERTER_FALLBACK_BINARY))
        raise RuntimeError(
            "Unable to locate a model converter executable. "
            f"Tried: {tried}. Ensure the Model Converter is installed and on PATH."
        )
    return binary


def resolve_model_converter_executable(binary: str) -> Path | None:
    """Resolve a converter candidate to an executable path, if possible.

    This is shared by the VGF compiler path and the preflight checker so both
    agree on what a usable converter executable means.

    """

    path = Path(binary)
    if path.is_absolute() or path.parent != Path("."):
        if _safe_is_file(path) and os.access(path, os.X_OK):
            return path
        return None

    resolved = which(binary)
    if resolved:
        return Path(resolved)
    return None


def require_model_converter_executable() -> Path:
    """Return a usable converter executable path or raise a helpful error."""

    binary = require_model_converter_binary()
    executable = resolve_model_converter_executable(binary)
    if executable is None:
        raise RuntimeError(
            f"Resolved converter candidate {binary!r}, but it is not executable. "
            "Fix MODEL_CONVERTER_PATH or place model-converter on PATH."
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


def check_model_converter_environment() -> ModelConverterEnvironmentCheck:
    """Check the model-converter dependency used by VGF compilation."""

    binary = find_model_converter_binary()
    if binary is None:
        return ModelConverterEnvironmentCheck(
            "MLSDK model converter",
            STATUS_FAIL,
            "Could not find model-converter on PATH and MODEL_CONVERTER_PATH "
            "does not point to an executable file.",
            "Install VGF AoT dependencies with "
            "python -m pip install 'executorch[vgf]' or, in a source checkout, "
            "python -m pip install -r backends/arm/requirements-arm-vgf.txt. "
            "Alternatively set MODEL_CONVERTER_PATH to the converter executable.",
        )

    executable = resolve_model_converter_executable(binary)
    if executable is None:
        return ModelConverterEnvironmentCheck(
            "MLSDK model converter",
            STATUS_FAIL,
            f"Resolved converter candidate {binary!r}, but it is not executable.",
            "Fix MODEL_CONVERTER_PATH or place model-converter on PATH.",
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
            "For source setup, source examples/arm/arm-scratch/setup_path.sh.",
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

    return ModelConverterEnvironmentCheck(
        "MLSDK model converter",
        STATUS_OK,
        f"{executable} --version succeeded:\n{_command_output(result)}",
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
