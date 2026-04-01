# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Public entry points for the Arm backend.

Public API is defined by explicit module exports (e.g., ``.vgf``, ``.ethosu``,
``.quantizer``). Selected symbols are re-exported here for convenience.

"""

from __future__ import annotations

import importlib
from typing import Any

# Public for tooling (manifest generation and API validation).
LAZY_IMPORTS = {
    "EthosUBackend": ("executorch.backends.arm.ethosu", "EthosUBackend"),
    "EthosUCompileSpec": ("executorch.backends.arm.ethosu", "EthosUCompileSpec"),
    "EthosUPartitioner": ("executorch.backends.arm.ethosu", "EthosUPartitioner"),
    "VgfBackend": ("executorch.backends.arm.vgf", "VgfBackend"),
    "VgfCompileSpec": ("executorch.backends.arm.vgf", "VgfCompileSpec"),
    "VgfPartitioner": ("executorch.backends.arm.vgf", "VgfPartitioner"),
    "EthosUQuantizer": ("executorch.backends.arm.quantizer", "EthosUQuantizer"),
    "VgfQuantizer": ("executorch.backends.arm.quantizer", "VgfQuantizer"),
    ("get_symmetric_quantization_config"): (
        "executorch.backends.arm.quantizer",
        "get_symmetric_quantization_config",
    ),
    ("get_symmetric_a16w8_quantization_config"): (
        "executorch.backends.arm.quantizer",
        "get_symmetric_a16w8_quantization_config",
    ),
}


def __getattr__(name: str) -> Any:
    if name in LAZY_IMPORTS:
        module_name, attr = LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(LAZY_IMPORTS))
