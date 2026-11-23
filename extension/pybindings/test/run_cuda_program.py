# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runs model.pte with its .ptd through cuda_lib, without LibTorch.

Usage: run_cuda_program DIR. Reads DIR/model.pte, the one .ptd in DIR, and
DIR/input.npy, and writes the first output to DIR/output.npy plus any
further outputs to DIR/output_1.npy, DIR/output_2.npy, and so on. When
DIR/input2.npy also exists, the same loaded module runs once more on it
and writes DIR/output2.npy (again with _1, _2, ... siblings).
"""

import sys
from pathlib import Path

import numpy as np
from executorch.extension.pybindings import cuda_lib


def _save_outputs(root: Path, prefix: str, outputs) -> None:
    for index, output in enumerate(outputs):
        suffix = "" if index == 0 else f"_{index}"
        np.save(root / f"{prefix}{suffix}.npy", np.asarray(output))


def main() -> None:
    root = Path(sys.argv[1])
    (ptd,) = root.glob("*.ptd")
    module = cuda_lib._load_for_executorch(str(root / "model.pte"), data_path=str(ptd))
    _save_outputs(root, "output", module.forward([np.load(root / "input.npy")]))
    second_input = root / "input2.npy"
    if second_input.exists():
        _save_outputs(root, "output2", module.forward([np.load(second_input)]))
