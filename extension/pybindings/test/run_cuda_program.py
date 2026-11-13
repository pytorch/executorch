# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runs model.pte with its .ptd through cuda_lib, without LibTorch.

Usage: run_cuda_program DIR. Reads DIR/model.pte, the one .ptd in DIR, and
DIR/input.npy, and writes the first output to DIR/output.npy.
"""

import sys
from pathlib import Path

import numpy as np
from executorch.extension.pybindings import cuda_lib


def main() -> None:
    root = Path(sys.argv[1])
    (ptd,) = root.glob("*.ptd")
    module = cuda_lib._load_for_executorch(str(root / "model.pte"), data_path=str(ptd))
    (output,) = module.forward([np.load(root / "input.npy")])
    np.save(root / "output.npy", np.asarray(output))
