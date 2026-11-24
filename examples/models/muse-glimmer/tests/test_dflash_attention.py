# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Position-based oracle for draft sliding-window attention."""

from unittest import TestCase

import torch
from executorch.examples.models.muse_glimmer.model.dflash_model import (
    build_dflash_swa_mask,
)


class DFlashAttentionTest(TestCase):
    def test_window_mask_uses_absolute_query_positions(self):
        for window in (1, 3, 8):
            for context in (0, 1, window - 1, window, window + 1, 3 * window):
                for block in (1, 4, 16):
                    with self.subTest(window=window, context=context, block=block):
                        slots = [-1] * window
                        for position in range(context):
                            slots[position % window] = position
                        keys = slots + list(range(context, context + block))
                        expected = torch.tensor(
                            [
                                [key >= 0 and abs(query - key) < window for key in keys]
                                for query in range(context, context + block)
                            ]
                        )[None, None]
                        torch.testing.assert_close(
                            build_dflash_swa_mask(context, block, window, window),
                            expected,
                        )
