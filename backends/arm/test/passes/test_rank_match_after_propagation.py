# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Regression test: view/permute propagation can leave a binary op with
mismatched operand ranks (rank-4 activation vs rank-1 per-channel constant),
which TOSA rejects. MatchArgRanksPass must run after propagation to keep the
graph TOSA-legal, so this guards that a per-channel-affine module lowers for U55.
"""

from typing import Tuple

import torch

from executorch.backends.arm.test.tester.test_pipeline import EthosU55PipelineINT

input_t = Tuple[torch.Tensor]


class PerChannelAffineRank4(torch.nn.Module):
    """Per-channel scale-mul + bias-add with a transpose to engage
    propagation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.rand(240) * 0.1 + 0.9)
        self.bias = torch.nn.Parameter(torch.randn(240) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.reshape(1, 1, 1, x.shape[-1])
        y = y * self.scale + self.bias
        y = y.permute(0, 3, 1, 2)
        y = y.reshape(1, x.shape[-1])
        y = y * self.scale + self.bias
        return y

    data = (torch.randn(1, 240),)


def test_per_channel_affine_rank4_u55_INT() -> None:
    pipeline = EthosU55PipelineINT[input_t](
        PerChannelAffineRank4(),
        PerChannelAffineRank4.data,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=False,
        a16w8_quantization=True,
    )
    pipeline.run()
