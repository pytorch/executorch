# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401
import torch


class RopeHfSingleTest(unittest.TestCase):
    def test_reference_matches_rotate_half(self) -> None:
        x = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]]]]
        )
        freqs_cos = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [0.5, 0.25, 0.5, 0.25]]
        )
        freqs_sin = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [0.5, 0.75, 0.5, 0.75]]
        )
        expected = torch.tensor(
            [[[[
                -1.0,
                -2.5,
                2.0,
                2.5,
            ], [
                1.0,
                2.5,
                -2.0,
                -2.5,
            ]]]]
        )

        actual = torch.ops.et_vk.apply_rotary_emb_hf_single.default(
            x, freqs_cos, freqs_sin, 1
        )

        torch.testing.assert_close(expected, actual)


