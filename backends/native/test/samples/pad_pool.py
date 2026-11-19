# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Zero pad and 2-D max pooling cases: pad widths and ranks, pool kernels,
strides and padding."""

import torch
import torch.nn.functional as F
from executorch.backends.native.test.samples.cases import Case, draws


class _Pad(torch.nn.Module):
    def __init__(self, pad: tuple[int, ...]) -> None:
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self.pad)


def pad_cases() -> list[Case]:
    return [
        Case("last_dim", _Pad((1, 2)), draws(4, (4, 64))),
        Case("two_dims", _Pad((2, 0, 1, 3)), draws(2, (2, 16, 32))),
        Case("rank4", _Pad((1, 1, 2, 2)), draws(2, (1, 8, 16, 16))),
        Case("wide", _Pad((0, 125)), draws(2, (2, 3, 131))),
    ]


def max_pool2d_cases() -> list[Case]:
    return [
        Case("kernel2", torch.nn.MaxPool2d(2), draws(2, (1, 8, 32, 32))),
        Case("kernel3_stride1", torch.nn.MaxPool2d(3, 1), draws(2, (1, 8, 16, 16))),
        Case("kernel3_stride2", torch.nn.MaxPool2d(3, 2), draws(2, (2, 4, 17, 17))),
        Case("padding", torch.nn.MaxPool2d(3, 2, padding=1), draws(2, (1, 16, 20, 20))),
        Case(
            "rectangular", torch.nn.MaxPool2d((2, 3), (2, 1)), draws(2, (1, 4, 12, 18))
        ),
    ]
