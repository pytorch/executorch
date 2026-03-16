# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from shutil import which
from typing import Optional

MODEL_CONVERTER_BINARY = "model-converter"
_MODEL_CONVERTER_FALLBACK_BINARY = "model_converter"


def find_model_converter_binary() -> Optional[str]:
    """Return the name of the first model converter executable on PATH."""

    for candidate in (MODEL_CONVERTER_BINARY, _MODEL_CONVERTER_FALLBACK_BINARY):
        if which(candidate):
            return candidate
    return None


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
