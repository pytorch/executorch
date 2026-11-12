# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

try:
    import os
    import sys
    from pathlib import Path

    # The library depends on the shipped runtime and quantized kernels DLLs in
    # executorch/lib, and Windows records no search path in a DLL. Not resolved:
    # in an editable install this directory is a symlink, and executorch/lib
    # sits beside the link.
    _lib_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "lib"
        )
    )
    if sys.platform == "win32" and os.path.isdir(_lib_dir):
        os.add_dll_directory(_lib_dir)
    del os, sys, _lib_dir

    libs = list(Path(__file__).parent.resolve().glob("**/*quantized_ops_aot_lib.*"))
    del Path
    assert len(libs) == 1, f"Expected 1 library but got {len(libs)}"
    import torch as _torch

    _torch.ops.load_library(libs[0])
    del _torch
except:
    import logging

    logging.info("libquantized_ops_aot_lib is not loaded")
    del logging
