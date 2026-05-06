# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.devtools.intermediate_output_tap._reducers import (
    FULL_TENSOR,
    get_reducer,
    StatReducer,
    STATS,
)


class ReducersTest(unittest.TestCase):
    def test_get_reducer_by_name(self):
        self.assertIs(get_reducer("FULL_TENSOR"), FULL_TENSOR)
        self.assertIs(get_reducer("STATS"), STATS)

    def test_get_reducer_passthrough(self):
        custom = StatReducer(
            name="X",
            fields=("a",),
            emit=lambda g, n: n,
            eager=lambda t: t,
        )
        self.assertIs(get_reducer(custom), custom)

    def test_get_reducer_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_reducer("DOES_NOT_EXIST")

    def test_reducer_field_counts(self):
        self.assertEqual(FULL_TENSOR.fields, ())
        self.assertEqual(
            STATS.fields,
            (
                "min",
                "max",
                "mean",
                "abs_max",
                "abs_mean",
                "std",
                "rms",
                "l1_norm",
                "l2_norm",
                "nan_count",
                "inf_count",
                "zero_count",
                "p99_abs",
            ),
        )

    def test_reducer_names_unique(self):
        names = {r.name for r in (FULL_TENSOR, STATS)}
        self.assertEqual(len(names), 2)

    def test_full_tensor_eager_is_identity(self):
        t = torch.randn(2, 3, 4)
        out = FULL_TENSOR.eager(t)
        self.assertEqual(out.shape, t.shape)
        torch.testing.assert_close(out, t.detach())

    def test_stats_eager_correctness(self):
        torch.manual_seed(0)
        t = torch.randn(64)
        out = STATS.eager(t)
        self.assertEqual(out.shape, (len(STATS.fields),))

        f = t.to(torch.float32)
        expected = {
            "min": float(f.amin()),
            "max": float(f.amax()),
            "mean": float(f.mean()),
            "abs_max": float(f.abs().amax()),
            "abs_mean": float(f.abs().mean()),
            "rms": float(f.pow(2).mean().sqrt()),
            "l1_norm": float(f.abs().sum()),
            "l2_norm": float(f.pow(2).sum().sqrt()),
            "nan_count": 0.0,
            "inf_count": 0.0,
            "zero_count": float((f == 0).to(torch.float32).sum()),
        }
        for i, field in enumerate(STATS.fields):
            if field in expected:
                torch.testing.assert_close(
                    float(out[i]), expected[field], rtol=1e-4, atol=1e-5
                )
        # std uses E[x^2] - E[x]^2 (population variance); compare to that.
        pop_var = float((f.pow(2).mean() - f.mean().pow(2)).abs())
        torch.testing.assert_close(
            float(out[STATS.fields.index("std")]) ** 2,
            pop_var,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_stats_p99_abs_matches_topk(self):
        torch.manual_seed(0)
        t = torch.randn(1000)
        out = STATS.eager(t)
        numel = t.numel()
        k = max(1, (numel + 99) // 100)
        expected = float(
            torch.topk(t.abs().reshape(-1), k=k, largest=True, sorted=True).values[
                k - 1
            ]
        )
        torch.testing.assert_close(
            float(out[STATS.fields.index("p99_abs")]),
            expected,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_stats_counts_nan_and_inf(self):
        t = torch.tensor(
            [1.0, float("nan"), 2.0, float("inf"), 0.0, -float("inf"), 0.0]
        )
        out = STATS.eager(t)
        i_nan = STATS.fields.index("nan_count")
        i_inf = STATS.fields.index("inf_count")
        i_zero = STATS.fields.index("zero_count")
        self.assertEqual(float(out[i_nan]), 1.0)
        self.assertEqual(float(out[i_inf]), 2.0)
        self.assertEqual(float(out[i_zero]), 2.0)
