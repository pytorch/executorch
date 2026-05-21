# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import ModuleType
from unittest.mock import patch

import torch

with patch.dict(
    "sys.modules",
    {"executorch.extension.llm.custom_ops.custom_ops": ModuleType("custom_ops")},
):
    from executorch.examples.models.voxtral_realtime.model import StandardRingKVCache


class StandardRingKVCacheTest(unittest.TestCase):
    def test_additive_mask_uses_finite_negative_values(self):
        cache = StandardRingKVCache(window_size=4, n_heads=1, head_dim=2)

        mask = cache.create_causal_mask(
            torch.tensor(0), seq_len=1, dtype=torch.bfloat16
        )

        self.assertEqual(mask.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(mask).all())
        self.assertEqual(mask[0, 0].item(), 0)
        self.assertLess(mask[0, 1].float().item(), -1e8)

    def test_bool_mask_keeps_bool_dtype(self):
        cache = StandardRingKVCache(window_size=4, n_heads=1, head_dim=2)

        mask = cache.create_causal_mask(torch.tensor(3), seq_len=2, bool_mask=True)

        self.assertEqual(mask.dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()
