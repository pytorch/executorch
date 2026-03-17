# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deprecated compatibility wrapper for the Arm AOT compiler CLI.

This script has been moved to
``executorch/backends/arm/scripts/aot_arm_compiler.py``.
"""

import sys
import warnings

from pathlib import Path

# Add Executorch root to path so this script can be run from anywhere
_EXECUTORCH_DIR = Path(__file__).resolve().parents[2]
_EXECUTORCH_DIR_STR = str(_EXECUTORCH_DIR)
if _EXECUTORCH_DIR_STR not in sys.path:
    sys.path.insert(0, _EXECUTORCH_DIR_STR)

from backends.arm.scripts.aot_arm_compiler import main as _main


def main() -> None:
    warnings.warn(
        "examples/arm/aot_arm_compiler.py is deprecated and has moved to "
        "backends/arm/scripts/aot_arm_compiler.py.",
        DeprecationWarning,
        stacklevel=2,
    )
    _main()


if __name__ == "__main__":
    main()
