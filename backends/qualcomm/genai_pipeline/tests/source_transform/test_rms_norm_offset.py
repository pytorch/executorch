# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the RMSNorm state-dict transform."""

import unittest

import torch

from executorch.backends.qualcomm.genai_pipeline.source_transform.rms_norm_offset import (
    gemma_rmsnorm_offset,
)


class TestGemmaRmsnormOffset(unittest.TestCase):
    """Gemma computes ``(x * w).to(fp16)``, Llama ``x.to(fp16) * w``.

    The static decoder implements the Llama form, so Gemma's norm weights carry
    a ``+1`` offset folded in at load time.
    """

    def test_adds_one_to_norm_weights(self):
        result = gemma_rmsnorm_offset(
            {"layers.0.attention_norm.weight": torch.zeros(3)}
        )
        torch.testing.assert_close(
            result["layers.0.attention_norm.weight"], torch.ones(3)
        )

    def test_leaves_non_norm_weights_untouched(self):
        result = gemma_rmsnorm_offset({"layers.0.attention.wq.weight": torch.zeros(3)})
        torch.testing.assert_close(
            result["layers.0.attention.wq.weight"], torch.zeros(3)
        )

    def test_promotes_norm_weights_to_float32(self):
        result = gemma_rmsnorm_offset(
            {"norm.weight": torch.zeros(3, dtype=torch.bfloat16)}
        )
        self.assertEqual(result["norm.weight"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
