# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from shutil import which
from typing import Optional

MODEL_CONVERTER_BINARY = "model-converter"
_MODEL_CONVERTER_FALLBACK_BINARY = "model_converter"


def find_model_converter_binary() -> Optional[str]:
    """Return the path/name of the first model converter executable found."""
    env_path = os.environ.get("MODEL_CONVERTER_PATH")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    for candidate in (MODEL_CONVERTER_BINARY, _MODEL_CONVERTER_FALLBACK_BINARY):
        if which(candidate):
            return candidate
    return None


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
