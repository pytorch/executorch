# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the dtype-override module transform."""

import unittest

import torch

from executorch.backends.qualcomm.genai_pipeline.source_transform.dtype_override import (
    apply_dtype_override,
)
from torch import nn


class TestApplyDtypeOverride(unittest.TestCase):
    """``--dtype-override`` casts the whole module before export."""

    def test_casts_to_requested_dtype(self):
        module = nn.Linear(2, 2)

        result = apply_dtype_override(module, dtype_override="fp16")

        self.assertEqual(result.weight.dtype, torch.float16)

    def test_no_op_when_unset(self):
        module = nn.Linear(2, 2)

        result = apply_dtype_override(module, dtype_override=None)

        self.assertEqual(result.weight.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
