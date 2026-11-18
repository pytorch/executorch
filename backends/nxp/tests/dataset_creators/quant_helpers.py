# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import shutil
from dataclasses import dataclass

import numpy as np
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    torch_type_to_numpy_type,
)
from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec
from executorch.exir.scalar_type import ScalarType


@dataclass
class InputQuantizationSpec:
    name: str
    scale: float
    zp: int
    dtype: ScalarType


def _replace_input_binary_tensor_with_quantized_variant(
    input_bin_tensor_path: str,
    input_spec: ModelInputSpec,
    q_params: InputQuantizationSpec,
) -> None:
    tensor = np.fromfile(
        input_bin_tensor_path, dtype=torch_type_to_numpy_type(input_spec.dtype)
    )
    if q_params.dtype == ScalarType.CHAR:
        tensor = np.add(np.round(np.divide(tensor, [q_params.scale])), [q_params.zp])
        tensor = np.clip(tensor, -128, 127).astype(np.int8)
    else:
        raise ValueError(f"Unknown quantization type: '{q_params.dtype}'.")
    tensor.tofile(input_bin_tensor_path)


def create_quantized_variant_of_dataset(
    dataset_dir: str,
    dataset_dir_quant: str,
    input_quant_spec: list[InputQuantizationSpec],
    input_spec: list[ModelInputSpec],
) -> None:
    """
    Create quantized dataset from provided quantization spec. Dataset is cloned from directory 'dataset_dir'.

    :param dataset_dir: Original (float) dataset directory.
    :param dataset_dir_quant: Quantized dataset directory.
    :param input_quant_spec: Quantization parameters used for dataset quantization.
    :param input_spec: Model inputs specification.
    """
    assert len(input_quant_spec) > 0

    shutil.copytree(dataset_dir, dataset_dir_quant, dirs_exist_ok=True)

    if len(input_quant_spec) == 1:
        # Single input dataset - quantize only files in dataset's root dir with first input_quant_spec
        input_spec = input_spec[0]
        input_quant_spec = input_quant_spec[0]

        for file in os.listdir(dataset_dir_quant):
            input_bin_tensor_path = os.path.join(dataset_dir_quant, file)
            _replace_input_binary_tensor_with_quantized_variant(
                input_bin_tensor_path, input_spec, input_quant_spec
            )
    else:
        # Iterate over samples (subfolders)
        for dir_ in os.listdir(dataset_dir_quant):
            # Iterate over each input in sample
            sample_dir = os.path.join(dataset_dir_quant, dir_)

            for idx, input_ in enumerate(sorted(os.listdir(sample_dir))):
                _replace_input_binary_tensor_with_quantized_variant(
                    os.path.join(sample_dir, input_),
                    input_spec[idx],
                    input_quant_spec[idx],
                )
