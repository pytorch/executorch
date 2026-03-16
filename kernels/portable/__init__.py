# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

try:
    from pathlib import Path

    libs = list(Path(__file__).parent.resolve().glob("*portable_custom_ops_aot_lib.*"))
    del Path
    if not libs:
        raise RuntimeError("No portable_custom_ops_aot_lib library found.")
    if len(libs) > 1:
        raise RuntimeError(
            f"Expected 1 portable_custom_ops_aot_lib library but found {len(libs)}."
        )
    import torch as _torch

    _torch.ops.load_library(str(libs[0]))
    del _torch
except Exception:  # noqa: E722
    import logging

    logging.info("portable_custom_ops_aot_lib is not loaded")
    del logging
