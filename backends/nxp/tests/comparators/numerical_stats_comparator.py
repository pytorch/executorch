# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import polars as pl

from executorch.backends.nxp.tests.comparators.base_output_comparator import (
    BaseOutputComparator,
    ComparisonResult,
    SamplePair,
)


class NumericalStatsOutputComparator(BaseOutputComparator):

    def __init__(
        self,
        max_mse_error=3.5e-4,
        fail_if_not_close=True,
        stats_out_filename: str = "numerical_stats.csv",
        use_softmax=False,
        is_classification_task=False,
    ):
        super().__init__(stats_out_filename=stats_out_filename)
        self._max_mse_error = max_mse_error
        self._fail_if_not_close = fail_if_not_close
        self.use_softmax = use_softmax
        self._is_classification_task = is_classification_task

    def evaluate_sample(self, sample: SamplePair) -> list[dict]:
        sample_stats = []

        for ref, cand in zip(sample.reference, sample.candidate):
            reference_data = ref.data
            candidate_data = cand.data

            if self.use_softmax:
                reference_data = np.exp(reference_data) / sum(np.exp(reference_data))
                candidate_data = np.exp(candidate_data) / sum(np.exp(candidate_data))

            mse = np.square(np.subtract(reference_data, candidate_data)).mean()
            max_error = np.max(np.abs(reference_data - candidate_data))

            stats = {
                "name": f"{sample.sample_name}/{ref.name}",
                "shape": str(reference_data.shape),
                "mse": mse,
                "max_nominal_error": max_error,
            }

            if self._is_classification_task:
                stats["argmax_cpu"] = np.argmax(reference_data, axis=-1).item()
                stats["argmax_npu"] = np.argmax(candidate_data, axis=-1).item()

            sample_stats.append(stats)

        return sample_stats

    def _create_stats_table(self, evaluations: list[list[dict]]) -> pl.DataFrame:
        stats = pl.from_dicts(
            [row for sample_stats in evaluations for row in sample_stats]
        )
        stats = stats.sort("name")

        name_contains_class = stats.select(
            pl.col("name").str.extract(r"example_(\w+)_", group_index=1)
        ).item(0, 0)

        # if label is available, retrieve more info about it
        if name_contains_class is not None:
            stats = stats.with_columns(
                pl.col("name")
                .str.extract(r"example_(\w+)_", group_index=1)
                .alias("label")
            )
            per_label_stats = stats.group_by("label").agg(
                pl.col("mse").mean().alias("mean_mse"),
                pl.col("max_nominal_error").mean().alias("mean_max_nominal_error"),
            )
            stats = stats.join(per_label_stats, on="label", how="left").sort("name")

        return stats

    def assert_comparison(self, evaluations: list[list[dict]]) -> ComparisonResult:
        stats = self._create_stats_table(evaluations)

        if not self._fail_if_not_close:
            return ComparisonResult(verdict=True, error_message="", stats=stats)

        error_samples = stats.filter(pl.col("mse") > self._max_mse_error)

        if len(error_samples) > 0:
            error_msg = (
                f"Some samples didn't match max MSE error threshold.\n"
                f"{error_samples}"
            )
            return ComparisonResult(verdict=False, error_message=error_msg, stats=stats)

        return ComparisonResult(verdict=True, error_message="", stats=stats)
