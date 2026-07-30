# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 - 2026 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import shutil

import numpy as np

from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    torch_type_to_numpy_type,
)
from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec
from executorch.devtools.visualization.visualization_utils import (
    visualize_with_clusters,
)
from executorch.exir import ExecutorchProgramManager
from torch._subclasses import FakeTensor


def save_pte_program(
    prog: ExecutorchProgramManager, model_name: str, output_dir: str = ""
) -> str:
    if model_name.endswith(".pte"):
        filename = model_name
        visualize_file_name = f"{model_name}.json"
    else:
        filename = os.path.join(output_dir, f"{model_name}.pte")
        visualize_file_name = os.path.join(output_dir, f"{model_name}.json")
    try:
        with open(filename, "wb") as file:
            prog.write_to_file(file)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")

    visualize_with_clusters(prog.exported_program(), visualize_file_name, False)
    return filename


def change_filepath_extension(path: str, extension: str) -> str:
    base, _ = os.path.splitext(path)
    return base + "." + extension


def store_txt_input_tensor(
    input_tensor_path: str,
    tensor_spec: ModelInputSpec | FakeTensor,
    quant_dataset: bool = False,
):
    dtype = np.int8 if quant_dataset else torch_type_to_numpy_type(tensor_spec.dtype)
    input_tensor = np.fromfile(input_tensor_path, dtype=dtype)
    int__max = np.iinfo(np.int32).max

    with open(change_filepath_extension(input_tensor_path, "txt"), "w") as f:
        f.write("Flattened tensor shape:" + str(input_tensor.shape))
        f.write("\nOriginal tensor shape:" + str(list(tensor_spec.shape)) + "\n")
        f.write(np.array2string(input_tensor, threshold=int__max))


def archive_test_dir(test_dir: str):
    shutil.make_archive(test_dir, "zip", test_dir)
