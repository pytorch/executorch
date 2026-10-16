# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import shutil
from random import sample

import numpy as np
import torch

from executorch.backends.nxp.tests.calibration_dataset import CalibrationDataset
from executorch.backends.nxp.tests.dataset_creators.base_dataset_creator import (
    BaseDatasetCreator,
)


class CopyDatasetCreator(BaseDatasetCreator):
    """Creator that just copies data from other directory."""

    def __init__(self, source_dir: str):
        super().__init__()
        self._source_dir = source_dir

    def create_calibration_dataset(self, calibration_dir_path, input_spec):
        # Copy all source files into the calibration directory (always contiguous).
        for example_name in os.listdir(self._source_dir):
            shutil.copy(
                os.path.join(self._source_dir, example_name), calibration_dir_path
            )


class FromCalibrationDataDatasetCreator(BaseDatasetCreator):
    """Creator that uses CalibrationDataset archive file."""

    def __init__(
        self,
        dataset: CalibrationDataset,
        num_examples: int,
        idx_to_label: dict[int, str],
    ):
        super().__init__()
        self._dataset = dataset
        assert (
            type(self._dataset[0]) is tuple and len(self._dataset[0]) == 2
        ), "Provide calibration data with examples and labels"

        self._num_examples = num_examples
        self._idx_to_label = idx_to_label

    def create_calibration_dataset(self, calibration_dir_path, input_spec):
        assert (
            isinstance(input_spec, list) and len(input_spec) == 1
        ), "Only one input is supported right now."

        examples_to_classes = self._sample_examples()
        self._write_examples(calibration_dir_path, examples_to_classes)

    def _sample_examples(self) -> list[tuple[int, int]]:
        """Sample up to num_examples (dataset_index, class_id) pairs, balanced across all classes."""
        # Use ordered collection for deterministic selection of samples
        unique_classes = dict.fromkeys(cl for _, cl in self._dataset)

        # If num_examples is less than number of classes, sample only from first num_examples classes
        examples_per_class = (
            self._num_examples // len(unique_classes)
            if self._num_examples >= len(unique_classes)
            else 1
        )
        examples_to_classes: list[tuple[int, int]] = []
        for cl in list(unique_classes.keys())[: self._num_examples]:
            cl_idx_list = [
                idx for idx in range(len(self._dataset)) if self._dataset[idx][1] == cl
            ]
            examples_to_classes.extend(
                (idx, cl) for idx in sample(cl_idx_list, examples_per_class)
            )
        return examples_to_classes

    def _write_examples(
        self,
        dataset_dir_path: str,
        examples_to_classes: list[tuple[int, int]],
    ):
        """Write binary .bin sample files for each (dataset_index, class_id) pair to dataset_dir_path."""
        for i, (idx, cl) in enumerate(examples_to_classes):
            label = self._idx_to_label[cl]
            raw, _ = self._dataset[idx]

            # normalize to list[np.ndarray] regardless of dataset type
            if isinstance(raw, torch.Tensor):
                np_data = [raw.numpy()]
            elif isinstance(raw, np.ndarray):
                np_data = [raw]
            else:
                np_data = raw

            for spec_idx, dt in enumerate(np_data):
                bin_file_name = f"{dataset_dir_path}/example_{label}_{cl}_{i}_i{str(spec_idx).zfill(2)}.bin"
                dt.tofile(bin_file_name)
