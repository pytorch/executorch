# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import os.path
import shutil
from os import mkdir
from pathlib import Path
from random import seed

import numpy as np
import torch
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    torch_type_to_numpy_type,
)
from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec


_RNG_SEED = 42


class BaseDatasetCreator(abc.ABC):
    """Abstract base class for all dataset creators."""

    def __init__(self):
        seed(_RNG_SEED)
        np.random.seed(_RNG_SEED)
        torch.manual_seed(_RNG_SEED)

    @staticmethod
    def _create_calibration_and_test_dataset_dirs(
        dataset_dir_path: str | Path,
    ) -> tuple[str, str]:
        """Create 'calibration' and 'test' subdirectories inside dataset_dir_path and return their paths."""
        mkdir(calibration_dir_path := os.path.join(dataset_dir_path, "calibration"))
        mkdir(test_dir_path := os.path.join(dataset_dir_path, "test"))
        return calibration_dir_path, test_dir_path

    @staticmethod
    def _permute_nchw_to_nhwc(
        tensor: np.ndarray, expected_shape: list[int] | tuple[int, ...]
    ) -> np.ndarray:
        """Permute a contiguous NCHW tensor (or a batch of NCHW tensors) to NHWC layout."""
        expected = tuple(expected_shape)
        shape = tensor.shape
        if shape == expected:
            # Single 4D tensor.
            return np.moveaxis(tensor, 1, -1)
        elif shape[1:] == expected:
            # A batch of 4D tensors.
            return np.moveaxis(tensor, 2, -1)
        else:
            raise ValueError(
                f"Cannot permute a tensor of shape {shape} to channels last."
            )

    @staticmethod
    def _derive_test_from_calibration(
        calibration_dir_path: str,
        test_dir_path: str,
        input_spec: list[ModelInputSpec],
    ):
        """Populate test_dir_path by copying files from calibration_dir_path, permuting to NHWC where channels_last is requested."""

        def _copy(src, dst, spec):
            if spec.dim_order == torch.channels_last:
                tensor = np.fromfile(
                    src, dtype=torch_type_to_numpy_type(spec.dtype)
                ).reshape(spec.shape)
                BaseDatasetCreator._permute_nchw_to_nhwc(tensor, spec.shape).tofile(dst)
            else:
                shutil.copy(src, dst)

        if len(input_spec) == 1:
            spec = input_spec[0]
            for file_name in os.listdir(calibration_dir_path):
                src = os.path.join(calibration_dir_path, file_name)
                dst = os.path.join(test_dir_path, file_name)

                _copy(src, dst, spec)
        else:
            for example_dir_name in sorted(os.listdir(calibration_dir_path)):
                src_example_dir_path = os.path.join(
                    calibration_dir_path, example_dir_name
                )
                dst_example_dir_path = os.path.join(test_dir_path, example_dir_name)
                mkdir(dst_example_dir_path)

                for spec_idx, spec in enumerate(input_spec):
                    file_name = f"{str(spec_idx).zfill(2)}.bin"
                    src = os.path.join(src_example_dir_path, file_name)
                    dst = os.path.join(dst_example_dir_path, file_name)

                    _copy(src, dst, spec)

    def generate_samples(
        self, output_dir_path: str | Path, input_spec: list[ModelInputSpec]
    ) -> tuple[str, str]:
        """Generate calibration and test datasets under output_dir_path.

        Creates two subdirectories ('calibration' and 'test') inside output_dir_path.
        The calibration directory is populated by create_calibration_dataset and always
        contains contiguous (NCHW) tensors. The test directory is derived from it by
        permuting tensors to the dim_order requested in input_spec where applicable.

        :param output_dir_path: Root directory under which 'calibration' and 'test' subdirectories are created.
        :param input_spec: List of ModelInputSpec describing shape, dtype, and dim_order of each model input.

        :returns: Tuple (calibration_dir_path, test_dir_path).
        """
        calibration_dir_path, test_dir_path = (
            self._create_calibration_and_test_dataset_dirs(output_dir_path)
        )

        self.create_calibration_dataset(calibration_dir_path, input_spec)

        # Copy calibration data to test dir, converting to the requested dim_order where needed.
        self._derive_test_from_calibration(
            calibration_dir_path, test_dir_path, input_spec
        )

        return calibration_dir_path, test_dir_path

    @abc.abstractmethod
    def create_calibration_dataset(self, calibration_dir_path, input_spec):
        """Populate calibration_dir_path with binary sample files for the given input_spec.

        Implementations must write one .bin file per sample input in contiguous (NCHW)
        memory order. For multi-input models each sample must be placed in a zero-padded
        numbered subdirectory (e.g. '0000/'), with one file per input named '<index>.bin'.

        :param calibration_dir_path: Directory to write the calibration sample files into.
        :param input_spec: List of ModelInputSpec describing shape, dtype, and dim_order of each model input.
        """
        pass
