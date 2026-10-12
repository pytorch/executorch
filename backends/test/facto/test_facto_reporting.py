# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import unittest

import torch

from .test_facto import _summarize_tensor, _summarize_value


class TestFactoJsonSerialization(unittest.TestCase):
    def test_non_finite_values_are_json_safe(self) -> None:
        record = {
            "scalar_nan": _summarize_value(float("nan")),
            "scalar_pos_inf": _summarize_value(float("inf")),
            "scalar_neg_inf": _summarize_value(float("-inf")),
            "tensor": _summarize_tensor(
                torch.tensor([0.0, float("nan"), float("inf"), float("-inf")])
            ),
        }

        encoded = json.dumps(record, sort_keys=True, allow_nan=False)

        self.assertIn('"scalar_nan": "NaN"', encoded)
        self.assertIn('"scalar_pos_inf": "Infinity"', encoded)
        self.assertIn('"scalar_neg_inf": "-Infinity"', encoded)
