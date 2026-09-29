# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.split_with_sizes_copy.default` module for the WebGPU op-test framework.

split_with_sizes is on YOLO's Detect head (splitting the concatenated
box/objectness/class predictions). `torch.split` by a size list lowers to
`split_with_sizes_copy` in the edge dialect.
"""

import torch


class SplitWithSizesModule(torch.nn.Module):
    def __init__(self, sizes, dim: int, out_order=None) -> None:
        super().__init__()
        self.sizes = sizes
        self.dim = dim
        # Reorder the returned chunks so a non-first chunk can be output 0 (the
        # op-test framework compares out_index 0) -- exercises the running offset.
        self.out_order = out_order

    def forward(self, x: torch.Tensor):
        chunks = torch.split(x, self.sizes, self.dim)
        if self.out_order is not None:
            return tuple(chunks[i] for i in self.out_order)
        return chunks


def _det_input(shape) -> torch.Tensor:
    # ((i % 23) - 11) / 16: exact in fp32, spans negatives through positives.
    n = 1
    for s in shape:
        n *= s
    idx = torch.arange(n, dtype=torch.int64)
    return (((idx % 23) - 11).to(torch.float32) / 16.0).reshape(shape)
