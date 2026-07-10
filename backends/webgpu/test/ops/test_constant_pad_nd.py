# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.constant_pad_nd.default` module (DaViT window padding) for op-tests."""

import torch


class PadModule(torch.nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = tuple(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(x, self.pad, value=0.0)


def _randn(shape) -> torch.Tensor:
    g = torch.Generator().manual_seed(sum(int(x) for x in shape))
    return torch.randn(*shape, generator=g) * 0.1
