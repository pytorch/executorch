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


class AllCloseOutputComparator(BaseOutputComparator):

    def __init__(self, atol=1e-7, stats_out_filename: str = "all_close_stats.csv"):
        super().__init__(stats_out_filename=stats_out_filename)
        self.atol = atol

    def evaluate_sample(self, sample: SamplePair) -> list[dict]:
        sample_stats = []
        for ref, cand in zip(sample.reference, sample.candidate):
            max_diff = np.abs(np.float32(ref.data) - np.float32(cand.data)).max()
            sample_stats.append(
                {
                    "name": f"{sample.sample_name}/{ref.name}",
                    "shape": str(ref.data.shape),
                    "max_nominal_error": max_diff,
                }
            )
        return sample_stats

    def assert_comparison(self, evaluations: list[list[dict]]) -> ComparisonResult:
        stats = pl.from_dicts(
            [row for sample_stats in evaluations for row in sample_stats]
        )
        error_samples = stats.filter(pl.col("max_nominal_error") > self.atol)

        if len(error_samples) > 0:
            error_msg = (
                "NPU output doesn't match reference. "
                f"Some samples exceeded the absolute tolerance ({self.atol}).\n{error_samples}"
            )
            return ComparisonResult(verdict=False, error_message=error_msg, stats=stats)

        return ComparisonResult(verdict=True, error_message="", stats=stats)
