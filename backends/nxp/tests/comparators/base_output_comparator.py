# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    torch_type_to_numpy_type,
)
from executorch.backends.nxp.tests.utils import archive_test_dir, store_txt_input_tensor


@dataclass
class TensorResult:
    """A single output tensor produced for one sample.

    :param name: Name of the output tensor (the result file name).
    :param path: Absolute path to the binary file the tensor was read from.
    :param data: The tensor data as a numpy array (reshaped to its spec shape).
    """

    name: str
    path: str
    data: np.ndarray


@dataclass
class SamplePair:
    """Reference and candidate results for a single sample.
    The reference results are treated as the ground truth that the
    candidate results are compared against.

    :param sample_name: Basename of the sample directory.
    :param reference: Output tensors produced by the reference model.
    :param candidate: Output tensors produced by the candidate model.
    """

    sample_name: str
    reference: list[TensorResult]
    candidate: list[TensorResult]


@dataclass
class ComparisonResult:
    """Result of the final comparison of per-sample values.
    If comparison is not satisfied, then `error_message` is filled in.
    `stats` field contains additional statistics about the per-sample values.
    """

    verdict: bool
    error_message: str
    stats: pl.DataFrame


class SampleResultsReader:
    """Reads matching samples from a reference and a candidate results directory.

    Both directories are expected to have the same hierarchy:

        result_dir
        |-- sample_0
        |---- 0000.bin
        |-- some_other_sample
        |---- first_output.bin
        |---- second_output.bin

    The reference directory is used as the source of truth for which samples and
    output tensors exist. Tensors are read using the dtype derived from the
    matching entry in `output_tensor_spec` and reshaped to its declared shape.
    """

    def __init__(self, reference_dir: str, candidate_dir: str, output_tensor_spec):
        self._reference_dir = reference_dir
        self._candidate_dir = candidate_dir
        self._output_tensor_spec = output_tensor_spec

    def _sample_dirs(self) -> list[str]:
        entries = [
            os.path.join(self._reference_dir, name)
            for name in os.listdir(self._reference_dir)
        ]
        return [entry for entry in entries if os.path.isdir(entry)]

    def _load_tensor(self, results_dir: str, rel_path: str, tensor_spec) -> np.ndarray:
        tensor_path = os.path.join(results_dir, rel_path)
        tensor = np.fromfile(
            tensor_path, dtype=torch_type_to_numpy_type(tensor_spec.dtype)
        )
        return np.reshape(tensor, tensor_spec.shape)

    def read(self) -> list[SamplePair]:
        sample_dirs = self._sample_dirs()
        assert len(sample_dirs), "No samples to compare."

        samples: list[SamplePair] = []

        for sample_dir in sample_dirs:
            sample_name = os.path.basename(sample_dir)

            reference_tensors: list[TensorResult] = []
            candidate_tensors: list[TensorResult] = []

            for idx, output_tensor_name in enumerate(sorted(os.listdir(sample_dir))):
                rel_path = os.path.join(sample_name, output_tensor_name)
                tensor_spec = self._output_tensor_spec[idx]

                reference_tensors.append(
                    TensorResult(
                        name=output_tensor_name,
                        path=os.path.join(self._reference_dir, rel_path),
                        data=self._load_tensor(
                            self._reference_dir, rel_path, tensor_spec
                        ),
                    )
                )
                candidate_tensors.append(
                    TensorResult(
                        name=output_tensor_name,
                        path=os.path.join(self._candidate_dir, rel_path),
                        data=self._load_tensor(
                            self._candidate_dir, rel_path, tensor_spec
                        ),
                    )
                )

            samples.append(
                SamplePair(
                    sample_name=sample_name,
                    reference=reference_tensors,
                    candidate=candidate_tensors,
                )
            )

        return samples


class BaseOutputComparator(abc.ABC):
    """Template for comparing a reference results directory with a candidate one.

    Subclasses implement two steps:

    * `evaluate_sample` collects whatever per-sample data is needed for the
      comparison (differences, predictions, statistics, ...).
    * `assert_comparison` receives the collected per-sample evaluations and is
      the single place where the pass/fail decision is made by returning bool value
      and additional information (error messages, assertion statistics, ...).
    """

    def __init__(self, stats_out_filename: str = "stats.csv") -> None:
        self._stats_out_filename = stats_out_filename

    @staticmethod
    def _save_test_dir_debug(reference_dir: str) -> None:
        """Archive the enclosing test directory for post-mortem inspection.

        No-op unless DEBUG logging is enabled.
        """
        if not logging.root.isEnabledFor(logging.DEBUG):
            return
        archive_test_dir(os.path.dirname(reference_dir))

    @staticmethod
    def _save_samples_diff_debug(
        sample: SamplePair, reference_dir: str, output_tensor_spec
    ) -> None:
        """Store per-tensor diff binaries and txt dumps for one sample.

        No-op unless DEBUG logging is enabled.
        """
        if not logging.root.isEnabledFor(logging.DEBUG):
            return

        diff_dir = os.path.join(os.path.dirname(reference_dir), "diff_cpu_npu_results")

        for idx, (ref, cand) in enumerate(zip(sample.reference, sample.candidate)):
            tensor_spec = output_tensor_spec[idx]

            diff_tensor = np.abs(ref.data - cand.data)
            diff_sample_dir = os.path.join(diff_dir, sample.sample_name)
            os.makedirs(diff_sample_dir, exist_ok=True)
            diff_tensor_path = os.path.join(diff_sample_dir, ref.name)
            diff_tensor.tofile(diff_tensor_path)

            store_txt_input_tensor(ref.path, tensor_spec)
            store_txt_input_tensor(cand.path, tensor_spec)
            store_txt_input_tensor(diff_tensor_path, tensor_spec)

    @staticmethod
    def _save_stats_debug(stats: pl.DataFrame, file_path: str | Path) -> None:
        """Report/persist the overall comparison result.

        No-op unless DEBUG logging is enabled. Called once, after
        every sample has been evaluated, and is always run, even if assertion fails.
        """
        if not logging.root.isEnabledFor(logging.DEBUG):
            return

        stats.write_csv(file_path)

    def compare_results(self, reference_dir, candidate_dir, output_tensor_spec):
        """Compare candidate results against reference results.

        The reference directory is taken as the source of truth for the set of
        samples and output tensors. Both directories must share the hierarchy:

            result_dir
            |-- sample_0
            |---- 0000.bin
            |-- some_other_sample
            |---- first_output.bin
            |---- second_output.bin

        :param reference_dir: Path to the directory with reference results.
        :param candidate_dir: Path to the directory with candidate results.
        :param output_tensor_spec: List of output tensor specifications.
        """
        reader = SampleResultsReader(reference_dir, candidate_dir, output_tensor_spec)
        test_results_dir = Path(reference_dir).resolve().parent

        samples = reader.read()
        evaluations = []
        comparison_stats: pl.DataFrame | None = None

        try:
            for sample in samples:
                self._save_samples_diff_debug(sample, reference_dir, output_tensor_spec)
                evaluations.append(self.evaluate_sample(sample))

            result: ComparisonResult = self.assert_comparison(evaluations)
            comparison_stats = result.stats

            # Final assert if comparison of candidate to reference is satisfactory.
            assert result.verdict, result.error_message

        finally:
            stats_destination = os.path.join(test_results_dir, self._stats_out_filename)

            if comparison_stats is not None:
                self._save_stats_debug(comparison_stats, stats_destination)
            self._save_test_dir_debug(reference_dir)

    @abstractmethod
    def evaluate_sample(self, sample: SamplePair):
        """Evaluate a single sample.

        Implementations collect whatever per-sample data the comparison needs
        (differences, predictions, statistics, ...) and return it. The returned
        value is collected across all samples and passed to `assert_comparison`.
        This method must not decide pass/fail on its own.

        :param sample: SamplePair of candidate + reference tensors
        """
        raise NotImplementedError

    @abstractmethod
    def assert_comparison(self, evaluations: list) -> ComparisonResult:
        """Make the pass/fail decision for the whole comparison.

        Receives the list of per-sample values returned by `evaluate_sample`
        (one entry per sample) and must return a `ComparisonResult` whose
        `stats` field is a `polars.DataFrame` combining the per-sample
        evaluations. This is the single place where a comparator decides
        whether the comparison passed.
        """
        raise NotImplementedError
