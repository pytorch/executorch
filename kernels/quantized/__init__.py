# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

# The platform-specific naming prefix for the AOT library. On Linux and Mac, the
# name is prefix with lib. On Windows, it is not.
LIB_PREFIX = "" if sys.platform == "win32" else "lib"

try:
    from pathlib import Path

    libs = list(Path(__file__).parent.resolve().glob(f"**/{LIB_PREFIX}quantized_ops_aot_lib.*"))
    del Path
    assert len(libs) == 1, f"Expected 1 library but got {len(libs)}"
    import torch as _torch

    _torch.ops.load_library(libs[0])
    del _torch
except:
    import logging

    logging.info("{LIB_PREFIX}quantized_ops_aot_lib is not loaded")
    del logging
