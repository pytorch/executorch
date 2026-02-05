# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import os.path
import shutil
from collections import OrderedDict
from os import mkdir
from random import sample, seed

import numpy as np
import torch
from torch import Tensor

from executorch.backends.nxp.tests_models.model_input_spec import ModelInputSpec
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.examples.nxp.models.calibration_dataset import CalibrationDataset


class DatasetCreator(abc.ABC):

    @abc.abstractmethod
    def generate_samples(self, dataset_dir, input_spec: list[ModelInputSpec]):
        pass


class RandomDatasetCreator(DatasetCreator):
    """ Dataset creator that generates random input samples. """

    def __init__(self, num_samples=2):
        self._num_samples = num_samples

    def generate_samples(self, dataset_dir, input_spec):
        assert isinstance(input_spec, list) and all([isinstance(spec, ModelInputSpec) for spec in input_spec]), \
            "Input_spec must be a list of ModelInputSpec."
        rng = np.random.default_rng(42)

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
                        shape = tuple(translator.dims_to_channels_last(list(spec.shape)))
                    case _:
                        raise ValueError(f"Unsupported dim_order: {spec.dim_order}")

                sample_vector = rng.random(np.prod(shape), spec.type).reshape(shape)
                sample_vector.tofile(os.path.join(sample_dir, f"{str(spec_idx).zfill(2)}.bin"))


class CopyDatasetCreator(DatasetCreator):
    """ Creator that just copies data from other directory. """

    def __init__(self, source_dir: str):
        self._source_dir = source_dir

    def generate_samples(self, dataset_dir, input_spec):
        for sample_name in os.listdir(self._source_dir):
            sample_path = os.path.join(self._source_dir, sample_name)
            shutil.copy(sample_path, dataset_dir)


class FromCalibrationDataDatasetCreator(DatasetCreator):
    """ Creator that uses CalibrationDataset archive file."""

    def __init__(self, dataset: CalibrationDataset, num_examples: int, idx_to_label: dict[int, str]):
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

    def generate_samples(self, dataset_dir, input_spec):
        os.makedirs(dataset_dir, exist_ok=True)
        assert type(self._dataset[0]) is tuple and len(self._dataset[0]) == 2, \
            "Provide calibration data with examples and labels"

        # We need to use ordered collection for deterministic selection of samples
        classes = OrderedDict([(cl, None) for _, cl in self._dataset])
        examples_per_class = self._num_examples // len(classes)
        idx_list = []
        for cl in classes.keys():
            cl_idx_list = [idx for idx in range(len(self._dataset)) if self._dataset[idx][1] == cl]
            class_indices = list(zip(sample(cl_idx_list, examples_per_class), [cl] * examples_per_class))
            idx_list.extend(class_indices)

        for i, (idx, cl) in enumerate(idx_list):
            label = self._idx_to_label[cl]
            example = self._dataset[idx]
            data = self._get_example_np_data(example)
            for inp_idx, dt in enumerate(data):
                bin_file_name = f"{dataset_dir}/example_{label}_{cl}_{i}_i{str(inp_idx).zfill(2)}.bin"
                dt.tofile(bin_file_name)
