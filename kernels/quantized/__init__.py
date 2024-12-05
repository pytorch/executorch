# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

try:
    from pathlib import Path

    libs = list(Path(__file__).parent.resolve().glob("**/libquantized_ops_aot_lib.*"))
    del Path
    assert len(libs) == 1, f"Expected 1 library but got {len(libs)}"
    import torch as _torch

    _torch.ops.load_library(libs[0])
    del _torch
except:
    import logging

    logging.info("libquantized_ops_aot_lib is not loaded")
    del logging
