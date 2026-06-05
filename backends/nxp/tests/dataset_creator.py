# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import os.path
import shutil
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from os import mkdir
from random import sample, seed

import numpy as np
import torch
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    torch_type_to_numpy_type,
)
from executorch.backends.nxp.tests.calibration_dataset import CalibrationDataset
from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec
from executorch.exir.scalar_type import ScalarType
from torch import Tensor


def _get_calibration_and_testing_dataset_directory_names(
    dataset_dir_name: str,
) -> tuple[str, str]:
    """Return the names of the directories which contain calibration data for quantization, and testing data for the
    inference. The difference is that the testing data may contain channels last samples.
    """
    mkdir(calibration_path := os.path.join(dataset_dir_name, "calibration"))
    mkdir(test_path := os.path.join(dataset_dir_name, "test"))
    return calibration_path, test_path


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
):
    tensor = np.fromfile(
        input_bin_tensor_path, dtype=torch_type_to_numpy_type(input_spec.dtype)
    )
    if q_params.dtype == ScalarType.CHAR:
        tensor = np.add(np.round(np.divide(tensor, [q_params.scale])), [q_params.zp])
        tensor = np.clip(tensor, -128, 127).astype(np.int8)
    else:
        raise ValueError(f"Unknown quantization type: '{q_params.dtype}.")
    tensor.tofile(input_bin_tensor_path)


def create_quantized_variant_of_dataset(
    dataset_dir: str,
    dataset_dir_quant: str,
    input_quant_spec: list[InputQuantizationSpec],
    input_spec: list[ModelInputSpec],
):
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


class DatasetCreator(abc.ABC):

    @abc.abstractmethod
    def generate_samples(
        self, dataset_dir, input_spec: list[ModelInputSpec]
    ) -> tuple[str, str]:
        pass


class RandomDatasetCreator(DatasetCreator):
    """Dataset creator that generates random input samples."""

    def __init__(self, num_samples=2, low=0.0, high=1.0):
        self._num_samples = num_samples
        self.low = low
        self.high = high

    def generate_samples(
        self, dataset_dir: str, input_spec: list[ModelInputSpec]
    ) -> tuple[str, str]:
        assert isinstance(input_spec, list) and all(
            isinstance(spec, ModelInputSpec) for spec in input_spec
        ), "Input_spec must be a list of ModelInputSpec."

        calibration_dir, test_dir = (
            _get_calibration_and_testing_dataset_directory_names(dataset_dir)
        )

        rng_seed = 42
        if any(spec.dim_order == torch.channels_last for spec in input_spec):
            # We will need to generate a separate testing dataset, containing the same data as is in the calibration
            #  dataset, just permuted to channels last where necessary.
            self._gen_samples(test_dir, input_spec, rng_seed)

        else:
            # Use the calibration dataset for testing as well.
            test_dir = calibration_dir

        # Make sure the calibration dataset contains contiguous tensors.
        contiguous_input_spec = deepcopy(input_spec)
        for spec in contiguous_input_spec:
            spec.dim_order = torch.contiguous_format

        # Generate the calibration dataset. Use the same rng seed as for the testing dataset, to make sure they contain
        #  the same data (except for the permutation).
        self._gen_samples(calibration_dir, contiguous_input_spec, rng_seed)

        return calibration_dir, test_dir

    def _gen_samples(
        self, dataset_dir: str, input_spec: list[ModelInputSpec], rng_seed: int
    ):
        rng = np.random.default_rng(rng_seed)
        for idx in range(self._num_samples):
            sample_dir = dataset_dir

            # Multi-input, use a subdirectory containing the inputs for each sample
            if len(input_spec) > 1:
                sample_dir = os.path.join(dataset_dir, f"{str(idx).zfill(4)}")
                mkdir(sample_dir)

            for spec_idx, spec in enumerate(input_spec):
                match spec.dim_order:
                    case torch.contiguous_format:
                        shape = spec.shape
                    case torch.channels_last:
                        shape = tuple(
                            translator.dims_to_channels_last(list(spec.shape))
                        )
                    case _:
                        raise ValueError(f"Unsupported dim_order: {spec.dim_order}")

                sample_vector = (
                    rng.uniform(self.low, self.high, size=np.prod(shape))
                    .astype(torch_type_to_numpy_type(spec.dtype))
                    .reshape(shape)
                )
                file_name = (
                    f"{str(spec_idx).zfill(2)}.bin"
                    if len(input_spec) > 1
                    else f"{str(idx).zfill(4)}.bin"
                )
                sample_vector.tofile(os.path.join(sample_dir, file_name))


class CopyDatasetCreator(DatasetCreator):
    """Creator that just copies data from other directory."""

    def __init__(self, source_dir: str):
        self._source_dir = source_dir

    def generate_samples(self, dataset_dir, input_spec) -> tuple[str, str]:
        assert (
            len(input_spec) == 1
        ), "Only one input is supported for `CopyDatasetCreator` right now."

        calibration_dataset_dir, testing_dataset_dir = (
            _get_calibration_and_testing_dataset_directory_names(dataset_dir)
        )
        if input_spec[0].dim_order != torch.channels_last:
            # Use the calibration dataset for testing as well.
            testing_dataset_dir = calibration_dataset_dir

        for sample_name in os.listdir(self._source_dir):
            sample_path = os.path.join(self._source_dir, sample_name)

            # Store the sample in the calibration dataset.
            shutil.copy(sample_path, calibration_dataset_dir)

            if input_spec[0].dim_order == torch.channels_last:
                # Permute the sample to channels last and store it in the testing dataset.
                tensor = np.fromfile(
                    sample_path, dtype=torch_type_to_numpy_type(input_spec[0].dtype)
                ).reshape(input_spec[0].shape)

                if (
                    list(tensor.shape) == list(input_spec[0].shape)
                    and len(tensor.shape) == 4
                ):
                    # 4D tensor.
                    tensor = np.moveaxis(tensor, 1, -1)
                else:
                    raise ValueError(
                        f"Cannot permute a tensor of shape {tensor.shape} to channels last."
                    )

                tensor.tofile(os.path.join(testing_dataset_dir, sample_name))

        return calibration_dataset_dir, testing_dataset_dir


class FromCalibrationDataDatasetCreator(DatasetCreator):
    """Creator that uses CalibrationDataset archive file."""

    def __init__(
        self,
        dataset: CalibrationDataset,
        num_examples: int,
        idx_to_label: dict[int, str],
    ):
        self._dataset = dataset
        self._num_examples = num_examples
        self._idx_to_label = idx_to_label
        seed(42)

    @staticmethod
    def _get_example_np_data(example):
        if isinstance(example, tuple):
            if len(example) == 2:
                data, _ = example
            elif len(example) == 1:
                data = example[0]
            else:
                raise ValueError(f"Unexpected number of elements in {example}.")
        else:
            raise NotImplementedError("Examples other than tuple are not supported.")

        if isinstance(data, Tensor):
            return [data.unsqueeze(0).numpy()]
        elif isinstance(data, list) and all(isinstance(dt, Tensor) for dt in data):
            return [dt.unsqueeze(0).numpy() for dt in data]
        else:
            raise TypeError("Data must be a single Tensor or a list of Tensors.")

    def generate_samples(self, dataset_dir, input_spec) -> tuple[str, str]:
        os.makedirs(dataset_dir, exist_ok=True)
        assert (
            type(self._dataset[0]) is tuple and len(self._dataset[0]) == 2
        ), "Provide calibration data with examples and labels"

        # We need to use ordered collection for deterministic selection of samples
        classes = OrderedDict([(cl, None) for _, cl in self._dataset])
        examples_per_class = self._num_examples // len(classes)
        idx_list = []
        for cl in classes.keys():
            cl_idx_list = [
                idx for idx in range(len(self._dataset)) if self._dataset[idx][1] == cl
            ]
            class_indices = list(
                zip(sample(cl_idx_list, examples_per_class), [cl] * examples_per_class)
            )
            idx_list.extend(class_indices)

        assert (
            isinstance(input_spec, list) and len(input_spec) == 1
        )  # Other cases are not implemented yet.

        calibration_dir, test_dir = (
            _get_calibration_and_testing_dataset_directory_names(dataset_dir)
        )

        if any(spec.dim_order == torch.channels_last for spec in input_spec):
            # We will need to generate a separate testing dataset, containing the same data as is in the calibration
            #  dataset, just permuted to channels last where necessary.
            self._gen_samples(test_dir, input_spec, idx_list)

        else:
            # Use the calibration dataset for testing as well.
            test_dir = calibration_dir

        # Generate the calibration dataset. Make sure it contains only `contiguous` tensors.
        contiguous_input_spec = deepcopy(input_spec)
        for spec in contiguous_input_spec:
            spec.dim_order = torch.contiguous_format
        self._gen_samples(calibration_dir, contiguous_input_spec, idx_list)

        return calibration_dir, test_dir

    def _gen_samples(
        self, dataset_dir: str, input_spec: list[ModelInputSpec], idx_list
    ):
        for i, (idx, cl) in enumerate(idx_list):
            label = self._idx_to_label[cl]
            example = self._dataset[idx]
            data = self._get_example_np_data(example)
            for inp_idx, dt in enumerate(data):
                if input_spec[0].dim_order == torch.channels_last:
                    if (
                        list(dt.shape) == list(input_spec[0].shape)
                        and len(dt.shape) == 4
                    ):
                        # 4D tensor.
                        dt = np.moveaxis(dt, 1, -1)
                    elif (
                        list(dt.shape)[1:] == list(input_spec[0].shape)
                        and len(dt.shape) == 5
                    ):
                        # Multiple 4D tensors.
                        dt = np.asarray([np.moveaxis(d, 1, -1) for d in dt])
                    else:
                        raise ValueError(
                            f"Cannot permute a tensor of shape {dt.shape} to channels last."
                        )

                bin_file_name = f"{dataset_dir}/example_{label}_{cl}_{i}_i{str(inp_idx).zfill(2)}.bin"
                dt.tofile(bin_file_name)


class LinearRampDatasetCreator(DatasetCreator):
    """Dataset creator that generates deterministic linear ramp input samples.

    The generated data forms a monotonic sequence where values are evenly
    distributed between a specified range (low to high) and span the full
    interval. The first element is equal to `low` and the last element is
    equal to `high`, with increments depending on the total number of elements.
    """

    def __init__(self, num_samples=2, low=0.0, high=1.0):
        self._num_samples = num_samples
        self.low = low
        self.high = high

    def generate_samples(
        self, dataset_dir: str, input_spec: list[ModelInputSpec]
    ) -> tuple[str, str]:
        assert isinstance(input_spec, list) and all(
            isinstance(spec, ModelInputSpec) for spec in input_spec
        ), "Input_spec must be a list of ModelInputSpec."

        calibration_dir, test_dir = (
            _get_calibration_and_testing_dataset_directory_names(dataset_dir)
        )

        if any(spec.dim_order == torch.channels_last for spec in input_spec):
            # We will need to generate a separate testing dataset, containing the same data as is in the calibration
            #  dataset, just permuted to channels last where necessary.
            self._gen_samples(test_dir, input_spec)

        else:
            # Use the calibration dataset for testing as well.
            test_dir = calibration_dir

        # Make sure the calibration dataset contains contiguous tensors.
        contiguous_input_spec = deepcopy(input_spec)
        for spec in contiguous_input_spec:
            spec.dim_order = torch.contiguous_format

        # Generate the calibration dataset. Calibration amd testing dataset s will contain
        #  the same data (except for the permutation).
        self._gen_samples(calibration_dir, contiguous_input_spec)

        return calibration_dir, test_dir

    def _gen_samples(self, dataset_dir: str, input_spec: list[ModelInputSpec]):
        for idx in range(self._num_samples):
            sample_dir = dataset_dir

            # Multi-input, use a subdirectory containing the inputs for each sample
            if len(input_spec) > 1:
                sample_dir = os.path.join(dataset_dir, f"{str(idx).zfill(4)}")
                mkdir(sample_dir)

            for spec_idx, spec in enumerate(input_spec):
                match spec.dim_order:
                    case torch.contiguous_format:
                        shape = spec.shape
                    case torch.channels_last:
                        shape = tuple(
                            translator.dims_to_channels_last(list(spec.shape))
                        )
                    case _:
                        raise ValueError(f"Unsupported dim_order: {spec.dim_order}")

                sample_vector = (
                    np.linspace(self.low, self.high, num=np.prod(shape))
                    .astype(torch_type_to_numpy_type(spec.dtype))
                    .reshape(shape)
                )
                file_name = (
                    f"{str(spec_idx).zfill(2)}.bin"
                    if len(input_spec) > 1
                    else f"{str(idx).zfill(4)}.bin"
                )
                sample_vector.tofile(os.path.join(sample_dir, file_name))
