# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

from executorch.devtools.intermediate_output_tap._reducers import (
    ABS_MAX_ONLY,
    DEFAULT_STATS,
    FULL_TENSOR,
    get_reducer,
    MIN_MAX_MEAN,
    StatReducer,
)


class ReducersTest(unittest.TestCase):
    def test_get_reducer_by_name(self):
        self.assertIs(get_reducer("DEFAULT_STATS"), DEFAULT_STATS)
        self.assertIs(get_reducer("FULL_TENSOR"), FULL_TENSOR)
        self.assertIs(get_reducer("MIN_MAX_MEAN"), MIN_MAX_MEAN)
        self.assertIs(get_reducer("ABS_MAX_ONLY"), ABS_MAX_ONLY)

    def test_get_reducer_passthrough(self):
        custom = StatReducer(name="X", fields=("a",), emit=lambda g, n: n)
        self.assertIs(get_reducer(custom), custom)

    def test_get_reducer_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_reducer("DOES_NOT_EXIST")

    def test_reducer_field_counts(self):
        self.assertEqual(FULL_TENSOR.fields, ())
        self.assertEqual(ABS_MAX_ONLY.fields, ("abs_max",))
        self.assertEqual(MIN_MAX_MEAN.fields, ("min", "max", "mean"))
        self.assertEqual(
            DEFAULT_STATS.fields,
            ("min", "max", "mean", "abs_max"),
        )

    def test_reducer_names_unique(self):
        names = {r.name for r in (FULL_TENSOR, ABS_MAX_ONLY, MIN_MAX_MEAN, DEFAULT_STATS)}
        self.assertEqual(len(names), 4)

    def test_default_stats_eager_correctness(self):
        """Confirm DEFAULT_STATS spec has 4 fields (std/nan_count/inf_count excluded)."""
        self.assertEqual(len(DEFAULT_STATS.fields), 4)
