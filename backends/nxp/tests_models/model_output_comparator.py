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

from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    torch_type_to_numpy_type,
)


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
        sample_dirs = [
            os.path.join(cpu_results_dir, file) for file in os.listdir(cpu_results_dir)
        ]
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

                cpu_tensor = np.fromfile(
                    cpu_tensor_path, dtype=torch_type_to_numpy_type(tensor_spec.dtype)
                )
                np.reshape(cpu_tensor, tensor_spec.shape)
                cpu_output_tensors.append((output_tensor_name, cpu_tensor))

                npu_tensor = np.fromfile(
                    npu_tensor_path, dtype=torch_type_to_numpy_type(tensor_spec.dtype)
                )
                np.reshape(npu_tensor, tensor_spec.shape)
                npu_output_tensors.append((output_tensor_name, npu_tensor))

            self.compare_sample(sample_dir, cpu_output_tensors, npu_output_tensors)

    @abstractmethod
    def compare_sample(
        self,
        sample_dir,
        cpu_output_tensors: list[tuple[str, np.ndarray]],
        npu_output_tensors: list[tuple[str, np.ndarray]],
    ):
        raise NotImplementedError


class AllCloseOutputComparator(BaseOutputComparator):

    def __init__(self, atol=1e-7):
        self.atol = atol

    def compare_sample(self, sample_dir, cpu_output_tensors, npu_output_tensors):
        for idx in range(len(cpu_output_tensors)):
            (cpu_output_name, cpu_tensor) = cpu_output_tensors[idx]
            (npu_output_name, npu_tensor) = npu_output_tensors[idx]

            assert cpu_output_name == npu_output_name
            assert np.any(
                cpu_tensor
            ), "Output tensor contains only zeros. This is suspicious."
            assert np.allclose(cpu_tensor, npu_tensor, atol=self.atol)


class ClassificationAccuracyOutputComparator(BaseOutputComparator):

    def __init__(self, class_dict: dict[int, str], tolerance=0.0):
        """
        Comparator for comparing model prediction accuracies based on a ground-truth annotations.
        The comparator passes if finetuned model results have higher accuracy than baseline (accounting for a tolerance).

        :param class_dict: Dictionary mapping class names to class indices.
        :param tolerance: Tolerance threshold for accuracy comparison.
                            Used for checking `baseline_acc + tolerance < finetuned_acc`.
        """
        self.tolerance = tolerance
        self.inv_class_dict = {v: k for k, v in class_dict.items()}

    def compare_results(
        self, baseline_results_dir, finetuned_results_dir, output_tensor_spec
    ):
        """
        Based on the result in the results dirs, estimate prediction accuracy and compare with tolerance.
        Finetuned model is expected to have higher prediction accuracy than baseline one, therefore if the accuracy is lower, the comparator fails.
        Result dir should have the following hierarchy:

        result_dir
        |-- example_classname_0
        |---- 0000.bin
        |-- example_classname_1
        |---- first_output.bin
        |---- second_output.bin

        :param finetuned_results_dir: Path to directory results generated with finetuned model.
        :param baseline_results_dir: Path to directory results generated with baseline model.
        :param output_tensor_spec: List of output tensor specifications.
        """
        sample_dirs = [
            os.path.join(baseline_results_dir, file)
            for file in os.listdir(baseline_results_dir)
        ]
        sample_dirs = [file for file in sample_dirs if os.path.isdir(file)]

        assert len(sample_dirs), "No samples to compare."

        finetuned_total_correct = 0
        baseline_total_correct = 0
        total_samples = 0

        for sample_dir in sample_dirs:
            finetuned_output_tensors = []
            baseline_output_tensors = []

            for idx, output_tensor_name in enumerate(os.listdir(sample_dir)):
                sample_dir = os.path.basename(sample_dir)
                tensor_path = os.path.join(sample_dir, output_tensor_name)

                baseline_tensor_path = os.path.join(baseline_results_dir, tensor_path)
                finetuned_tensor_path = os.path.join(finetuned_results_dir, tensor_path)

                tensor_spec = output_tensor_spec[idx]

                baseline_tensor = np.fromfile(
                    baseline_tensor_path,
                    dtype=torch_type_to_numpy_type(tensor_spec.dtype),
                )
                np.reshape(baseline_tensor, tensor_spec.shape)
                baseline_output_tensors.append((output_tensor_name, baseline_tensor))

                finetuned_tensor = np.fromfile(
                    finetuned_tensor_path,
                    dtype=torch_type_to_numpy_type(tensor_spec.dtype),
                )
                np.reshape(finetuned_tensor, tensor_spec.shape)
                finetuned_output_tensors.append((output_tensor_name, finetuned_tensor))

            finetuned_correct, baseline_correct, total = self.compare_sample(
                sample_dir, baseline_output_tensors, finetuned_output_tensors
            )

            finetuned_total_correct += finetuned_correct
            baseline_total_correct += baseline_correct
            total_samples += total

        baseline_accuracy = baseline_total_correct / total_samples
        finetuned_accuracy = finetuned_total_correct / total_samples

        if (baseline_accuracy + self.tolerance) > finetuned_accuracy:
            raise AssertionError(
                f"Finetuned model accuracy ({finetuned_accuracy} < baseline accuracy + tolerance "
                + f"({baseline_accuracy} + {self.tolerance}). "
                + "This might be a sign that something is not working properly. "
                + "Hint: Try adjusting training hyperparameters or select negative tolerance."
            )

    def compare_sample(
        self, sample_dir, baseline_output_tensors, finetuned_output_tensors
    ) -> tuple[int, int, int]:
        baseline_correct = 0
        finetuned_correct = 0

        if not isinstance(sample_dir, str) or len(sample_dir.split("_")) < 2:
            raise ValueError(
                f"Sample dir format invalid. Expected format: 'example_classname_0', got {sample_dir}"
            )

        class_name = sample_dir.split("_")[1]
        class_id = self.inv_class_dict[class_name]

        for idx in range(len(baseline_output_tensors)):
            (baseline_output_name, baseline_tensor) = baseline_output_tensors[idx]
            (finetuned_output_name, finetuned_tensor) = finetuned_output_tensors[idx]

            assert baseline_output_name == finetuned_output_name
            assert np.any(
                baseline_tensor
            ), "Output tensor contains only zeros. This is suspicious."

            finetuned_class = np.argmax(finetuned_tensor, axis=-1)
            baseline_class = np.argmax(baseline_tensor, axis=-1)

            baseline_correct += baseline_class == class_id
            finetuned_correct += finetuned_class == class_id

        return finetuned_correct, baseline_correct, len(baseline_output_tensors)


class NumericalStatsOutputComparator(BaseOutputComparator):

    def __init__(
        self,
        max_mse_error=3.5e-4,
        fail_if_not_close=True,
        output_filename: None | str = "numerical_stats.csv",
        use_softmax=False,
        is_classification_task=False,
    ):
        self._max_mse_error = max_mse_error
        self._fail_if_not_close = fail_if_not_close
        self._output_filename = output_filename
        self._stats_data = None
        self.use_softmax = use_softmax
        self._is_classification_task = is_classification_task

    def compare_results(self, cpu_results_dir, npu_results_dir, output_tensor_spec):
        self._stats_data = []
        super().compare_results(cpu_results_dir, npu_results_dir, output_tensor_spec)

        stats = pl.from_dicts(self._stats_data)
        print(stats.sort("name"))
        name_contains_class = stats.select(
            pl.col("name").str.extract(r"example_(\w+)_", group_index=1)
        ).item(0, 0)
        if name_contains_class is not None:
            print(
                "Stats per label class:\n",
                stats.group_by(
                    pl.col("name")
                    .str.extract(r"example_(\w+)_", group_index=1)
                    .alias("label")
                )
                .agg(
                    pl.col("mse").mean().alias("mean_mse"),
                    pl.col("max_nominal_error").mean().alias("mean_max_nominal_error"),
                )
                .sort("label"),
            )

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
            assert np.any(
                cpu_tensor
            ), "Output tensor contains only zeros. This is suspicious."

            if self.use_softmax:
                cpu_tensor = np.exp(cpu_tensor) / sum(np.exp(cpu_tensor))
                npu_tensor = np.exp(npu_tensor) / sum(np.exp(npu_tensor))

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
