# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten._native_batch_norm_legit_no_training.default` module for the WebGPU op-test framework.

Inference batch norm is on MODNet's decoder and many CNN backbones; `nn.BatchNorm2d`
in eval mode lowers to `_native_batch_norm_legit_no_training`, applying the per-channel
running-stat affine. Deterministic running stats + affine make the golden non-trivial.
"""

import torch


class BatchNorm2dModule(torch.nn.Module):
    def __init__(
        self, num_features: int, affine: bool = True, eps: float = 1e-5
    ) -> None:
        super().__init__()
        bn = torch.nn.BatchNorm2d(num_features, eps=eps, affine=affine)
        with torch.no_grad():
            bn.running_mean.copy_(torch.linspace(-1.0, 1.0, num_features))
            bn.running_var.copy_(torch.linspace(0.5, 2.0, num_features))
            if affine:
                bn.weight.copy_(torch.linspace(0.5, 1.5, num_features))
                bn.bias.copy_(torch.linspace(-0.5, 0.5, num_features))
        bn.eval()
        self.bn = bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


def _det_input(shape) -> torch.Tensor:
    # ((i % 23) - 11) / 16: exact in fp32, spans negatives through positives.
    n = 1
    for s in shape:
        n *= s
    idx = torch.arange(n, dtype=torch.int64)
    return (((idx % 23) - 11).to(torch.float32) / 16.0).reshape(shape)
