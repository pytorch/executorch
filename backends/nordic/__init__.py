# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Public entry points for the Nordic backend.

Public API is defined by explicit module exports (``.axon``).
Selected symbols are re-exported here for convenience.
"""

from __future__ import annotations

import importlib
from typing import Any

# Public for tooling (manifest generation and API validation).
LAZY_IMPORTS = {
    "AxonBackend": ("executorch.backends.nordic.axon", "AxonBackend"),
    "AxonCompileSpec": ("executorch.backends.nordic.axon", "AxonCompileSpec"),
    "AxonPartitioner": ("executorch.backends.nordic.axon", "AxonPartitioner"),
    "AxonQuantizer": ("executorch.backends.nordic.axon", "AxonQuantizer"),
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
