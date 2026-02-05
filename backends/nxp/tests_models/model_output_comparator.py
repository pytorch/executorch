# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import os
from abc import abstractmethod
from pathlib import Path

import numpy as np
import polars as pl

from executorch.backends.nxp.backend.ir.converter.conversion.translator import torch_type_to_numpy_type


class BaseOutputComparator(abc.ABC):

    def compare_results(self, cpu_results_dir, npu_results_dir, output_tensor_spec):
        """
        Check if tensors in result dirs corresponds. Directory with CPU results is taken as
        a reference for compared binary files. Result dir should have the following hierarchy:

        result_dir
        |-- sample_0
        |---- 0000.bin
        |-- some_other_sample
        |---- first_output.bin
        |---- second_output.bin

        :param cpu_results_dir: Path to directory with CPU results.
        :param npu_results_dir: Path to directory with NPU (delegated) results.
        :param output_tensor_spec: List of output tensor specifications.
        """
        sample_dirs = [os.path.join(cpu_results_dir, file) for file in os.listdir(cpu_results_dir)]
        sample_dirs = [file for file in sample_dirs if os.path.isdir(file)]

        assert len(sample_dirs), "No samples to compare."

        for sample_dir in sample_dirs:
            npu_output_tensors = []
            cpu_output_tensors = []

            for idx, output_tensor_name in enumerate(os.listdir(sample_dir)):
                sample_dir = os.path.basename(sample_dir)
                tensor_path = os.path.join(sample_dir, output_tensor_name)

                cpu_tensor_path = os.path.join(cpu_results_dir, tensor_path)
                npu_tensor_path = os.path.join(npu_results_dir, tensor_path)

                tensor_spec = output_tensor_spec[idx]

                cpu_tensor = np.fromfile(cpu_tensor_path, dtype=torch_type_to_numpy_type(tensor_spec.dtype))
                np.reshape(cpu_tensor, tensor_spec.shape)
                cpu_output_tensors.append((output_tensor_name, cpu_tensor))

                npu_tensor = np.fromfile(npu_tensor_path, dtype=torch_type_to_numpy_type(tensor_spec.dtype))
                np.reshape(npu_tensor, tensor_spec.shape)
                npu_output_tensors.append((output_tensor_name, npu_tensor))

            self.compare_sample(sample_dir, cpu_output_tensors, npu_output_tensors)

    @abstractmethod
    def compare_sample(self, sample_dir,
                       cpu_output_tensors: list[tuple[str, np.ndarray]],
                       npu_output_tensors: list[tuple[str, np.ndarray]]):
        raise NotImplementedError


class AllCloseOutputComparator(BaseOutputComparator):

    def __init__(self, atol=1e-7):
        self.atol = atol

    def compare_sample(self, sample_dir, cpu_output_tensors, npu_output_tensors):
        for idx in range(len(cpu_output_tensors)):
            (cpu_output_name, cpu_tensor) = cpu_output_tensors[idx]
            (npu_output_name, npu_tensor) = npu_output_tensors[idx]

            assert cpu_output_name == npu_output_name
            assert np.any(cpu_tensor), "Output tensor contains only zeros. This is suspicious."
            assert np.allclose(cpu_tensor, npu_tensor, atol=self.atol)


class NumericalStatsOutputComparator(BaseOutputComparator):

    def __init__(self, max_mse_error=3.5e-4, fail_if_not_close=True,
                 output_filename: None | str = "numerical_stats.csv",
                 is_classification_task=False):
        self._max_mse_error = max_mse_error
        self._fail_if_not_close = fail_if_not_close
        self._output_filename = output_filename
        self._stats_data = None
        self._is_classification_task = is_classification_task

    def compare_results(self, cpu_results_dir, npu_results_dir, output_tensor_spec):
        self._stats_data = []
        super().compare_results(cpu_results_dir, npu_results_dir, output_tensor_spec)

        stats = pl.from_dicts(self._stats_data)
        print(stats.sort("name"))
        name_contains_class = stats.select(pl.col("name").str.extract(r"example_(\w+)_", group_index=1)).item(0, 0)
        if name_contains_class is not None:
            print("Stats per label class:\n",
                  stats.group_by(pl.col("name").str.extract(r"example_(\w+)_", group_index=1).alias("label"))
                  .agg(
                      pl.col("mse").mean().alias("mean_mse"),
                      pl.col("max_nominal_error").mean().alias("mean_max_nominal_error")
                  )
                  .sort("label"))

        if self._output_filename:
            test_results_dir = Path(cpu_results_dir).resolve().parent
            stats.write_csv(os.path.join(test_results_dir, self._output_filename))

        if self._fail_if_not_close:
            error_samples = stats.filter(pl.col("mse") > self._max_mse_error)

            if len(error_samples) > 0:
                error_msg = f"Some samples didn't match max MSE error threshold.\n{error_samples}"
                raise AssertionError(error_msg)

    def compare_sample(self, sample_dir, cpu_output_tensors, npu_output_tensors):
        for idx in range(len(cpu_output_tensors)):
            (cpu_output_name, cpu_tensor) = cpu_output_tensors[idx]
            (npu_output_name, npu_tensor) = npu_output_tensors[idx]

            assert cpu_output_name == npu_output_name
            assert np.any(cpu_tensor), "Output tensor contains only zeros. This is suspicious."

            mse = np.square(np.subtract(cpu_tensor, npu_tensor)).mean()
            max_error = np.max(np.abs(cpu_tensor - npu_tensor))

            stats = {
                "name": f"{os.path.basename(sample_dir)}/{cpu_output_name}",
                "shape": str(cpu_tensor.shape),
                "mse": mse,
                "max_nominal_error": max_error,
            }

            if self._is_classification_task:
                stats["argmax_cpu"] = np.argmax(cpu_tensor, axis=-1)
                stats["argmax_npu"] = np.argmax(npu_tensor, axis=-1)

            self._stats_data.append(stats)
