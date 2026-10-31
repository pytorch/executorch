# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os.path
from abc import abstractmethod
from os import mkdir
from pathlib import Path

import numpy as np
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    torch_type_to_numpy_type,
)
from executorch.backends.nxp.tests.dataset_creators.base_dataset_creator import (
    BaseDatasetCreator,
)
from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec


class SyntheticDatasetCreator(BaseDatasetCreator):
    """Base class for dataset creators that synthesize example data from scratch."""

    def __init__(self, num_examples: int = 2) -> None:
        super().__init__()
        self._num_examples = num_examples

    @abstractmethod
    def generate_examples(self, shape: tuple) -> np.ndarray:
        """Generate an array of artificial values with the requested shape
        to be used as the dataset examples.

        :param shape: Shape of the output array.

        :returns: numpy array of generated values with the given shape.
        """
        pass

    def create_calibration_dataset(
        self, calibration_dir_path: str | Path, input_spec: list[ModelInputSpec]
    ) -> None:
        for idx in range(self._num_examples):
            example_dir_path = calibration_dir_path

            # Multi-input: use a subdirectory containing the inputs for each example
            if len(input_spec) > 1:
                example_dir_path = os.path.join(
                    calibration_dir_path, f"{str(idx).zfill(4)}"
                )
                mkdir(example_dir_path)

            for spec_idx, spec in enumerate(input_spec):
                # calibration data is always stored in contiguous (NCHW) format
                shape = tuple(spec.shape)
                example_vector = self.generate_examples(shape).astype(
                    torch_type_to_numpy_type(spec.dtype)
                )
                file_name = (
                    f"{str(spec_idx).zfill(2)}.bin"
                    if len(input_spec) > 1
                    else f"{str(idx).zfill(4)}.bin"
                )
                example_vector.tofile(os.path.join(example_dir_path, file_name))


class RandomDatasetCreator(SyntheticDatasetCreator):
    """Dataset creator that generates data from uniform distribution."""

    def __init__(
        self, num_examples: int = 2, low: float = 0.0, high: float = 1.0
    ) -> None:
        super().__init__(num_examples)
        self.low = low
        self.high = high
        self._rng = np.random.default_rng(self._seed)

    def generate_examples(self, shape: tuple) -> np.ndarray:
        return self._rng.uniform(self.low, self.high, size=shape)


class LinearRampDatasetCreator(SyntheticDatasetCreator):
    """Dataset creator that generates deterministic linear ramp input examples.

    The generated data forms a monotonic sequence where values are evenly
    distributed between a specified range (low to high) and span the full
    interval.
    """

    def __init__(
        self, num_examples: int = 2, low: float = -1.0, high: float = 1.0
    ) -> None:
        super().__init__(num_examples)
        self.low = low
        self.high = high

    def generate_examples(self, shape: tuple) -> np.ndarray:
        return np.linspace(self.low, self.high, num=int(np.prod(shape))).reshape(shape)
