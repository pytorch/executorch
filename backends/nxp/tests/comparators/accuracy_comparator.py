# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import numpy as np
import polars as pl

from executorch.backends.nxp.tests.comparators.base_output_comparator import (
    BaseOutputComparator,
    ComparisonResult,
    SamplePair,
)


def _default_postprocess_fn(outputs: np.ndarray, _: str):
    return np.argmax(outputs, axis=-1)


def _parse_class_id_from_sample_dir(
    sample_dir: str, inv_class_dict: dict[str, int]
) -> int:
    if not isinstance(sample_dir, str) or len(sample_dir.split("_")) < 3:
        raise ValueError(
            f"Sample dir format invalid. Expected format: 'example_classname_0', got {sample_dir}"
        )

    dir_parts = sample_dir.split("_")
    first_numerical_index = next(
        (i for i, s in enumerate(dir_parts) if s.isdigit()), -1
    )

    if first_numerical_index < 2:
        raise ValueError(
            f"Sample dir format invalid. Expected format: 'example_classname_0', got {sample_dir}"
        )

    class_name = "_".join(dir_parts[1:first_numerical_index])
    return inv_class_dict[class_name]


class AccuracyOutputComparator(BaseOutputComparator):

    def __init__(
        self,
        class_dict: dict[int, str],
        max_accuracy_drop: float = 0.05,
        postprocess_fn: Callable[
            [np.ndarray, str], np.ndarray
        ] = _default_postprocess_fn,
        min_reference_accuracy: float | None = None,
        parse_class_id_fn: Callable[
            [str, dict[str, int]], int
        ] = _parse_class_id_from_sample_dir,
        stats_out_filename: str = "accuracy_stats.csv",
    ):
        """Compare classification accuracy of a reference vs a candidate model.

        Both models are evaluated against the ground-truth annotations encoded
        in the sample directory names (e.g. 'example_classname_0', as produced
        by `FromCalibrationDataDatasetCreator`). The comparison fails if the
        accuracy drop (reference accuracy - candidate accuracy) exceeds a
        configured threshold.

        :param class_dict: Dictionary mapping class indices to class names.
        :param max_accuracy_drop: Maximum allowed accuracy drop
                                   (reference_accuracy - candidate_accuracy),
                                   given as a fraction in the range [0, 1]. The
                                   test fails if the candidate model is less
                                   accurate than the reference model by more
                                   than this value.
        :param postprocess_fn: Optional callback mapping a model output into a
                               classification prediction.
        :param min_reference_accuracy: Optional lower bound on the reference
                                       model accuracy. If provided and the
                                       reference accuracy is below this value the
                                       test fails - this guards against a
                                       meaningless comparison where both models
                                       are equally bad. Given as a fraction in
                                       the range [0, 1].
        :param parse_class_id_fn: Callback used to derive the ground-truth class
                                  index for a sample. It receives the sample
                                  directory basename and the inverse class
                                  dictionary (class name to class index) and
                                  returns the class index. Defaults to
                                  `_parse_class_id_from_sample_dir`.
        :param stats_out_filename: Name of the CSV file the per-sample accuracy
                                 stats are written to (relative to the test
                                 results directory), when DEBUG logging is
                                 enabled.
        """
        super().__init__(stats_out_filename=stats_out_filename)
        self.postprocess_fn = postprocess_fn
        self.max_accuracy_drop = max_accuracy_drop
        self.min_reference_accuracy = min_reference_accuracy
        self.parse_class_id_fn = parse_class_id_fn
        self.inv_class_dict = {v: k for k, v in class_dict.items()}

    def evaluate_sample(self, sample: SamplePair) -> dict:
        reference_correct_total = 0
        candidate_correct_total = 0
        total = 0

        class_id = self.parse_class_id_fn(sample.sample_name, self.inv_class_dict)

        for ref, cand in zip(sample.reference, sample.candidate):
            reference_class = self.postprocess_fn(ref.data, ref.path)
            candidate_class = self.postprocess_fn(cand.data, cand.path)

            reference_correct = reference_class == class_id
            candidate_correct = candidate_class == class_id

            reference_correct_total += (
                reference_correct
                if np.isscalar(reference_correct)
                else sum(reference_correct)
            )
            candidate_correct_total += (
                candidate_correct
                if np.isscalar(candidate_correct)
                else sum(candidate_correct)
            )
            total += 1 if np.isscalar(reference_correct) else len(reference_correct)

        return {
            "name": sample.sample_name,
            "class_id": class_id,
            "reference_correct": reference_correct_total,
            "candidate_correct": candidate_correct_total,
            "total": total,
        }

    def assert_comparison(self, evaluations: list[dict]) -> ComparisonResult:
        stats = pl.from_dicts(evaluations)

        reference_num_correct = stats["reference_correct"].sum()
        candidate_num_correct = stats["candidate_correct"].sum()
        total_samples = stats["total"].sum()

        reference_accuracy = reference_num_correct / total_samples
        candidate_accuracy = candidate_num_correct / total_samples
        accuracy_drop = reference_accuracy - candidate_accuracy

        if (
            self.min_reference_accuracy is not None
            and reference_accuracy < self.min_reference_accuracy
        ):
            error_msg = (
                f"Reference model accuracy ({reference_accuracy:.4f}) is below the "
                f"minimum required accuracy ({self.min_reference_accuracy:.4f}). "
                "The accuracy comparison is not meaningful when the reference model is inaccurate.\n"
            )
            return ComparisonResult(verdict=False, error_message=error_msg, stats=stats)

        if accuracy_drop > self.max_accuracy_drop:
            error_msg = (
                f"Candidate model accuracy ({candidate_accuracy:.4f}) dropped by "
                f"{accuracy_drop:.4f} relative to the reference model accuracy "
                f"({reference_accuracy:.4f}), which exceeds the maximum allowed drop "
                f"({self.max_accuracy_drop:.4f}).\n"
            )
            return ComparisonResult(verdict=False, error_message=error_msg, stats=stats)

        return ComparisonResult(verdict=True, error_message="", stats=stats)
