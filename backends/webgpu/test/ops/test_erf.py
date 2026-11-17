# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.erf.default` modules + inputs for the WebGPU tests.

Imported by `cases.py`, `test_webgpu_partitioner.py` and the dynamic-shape
export. `ErfGeluModule` is the hand-written exact GELU of Hugging Face ESM-2
and BERT-style models, x * 0.5 * (1 + erf(x / sqrt(2))).
"""

import math

import torch


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def wide_ramp(shape) -> torch.Tensor:
    """Deterministic linspace over [-10, 10]: both tails saturate to +-1."""
    return torch.linspace(-10.0, 10.0, _numel(shape), dtype=torch.float32).reshape(
        shape
    )


def signed_small(shape) -> torch.Tensor:
    """+-geomspace(1e-30, 1), alternating sign, where erf(x) ~ 2x / sqrt(pi)
    and only a relative-accurate kernel matches."""
    n = _numel(shape)
    mag = torch.logspace(-30.0, 0.0, n, dtype=torch.float64)
    sign = torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0).double()
    return (sign * mag).float().reshape(shape)


class ErfModule(torch.nn.Module):
    """aten.erf; with `param_shape`, on a parameter added to the input."""

    def __init__(self, param_shape=None) -> None:
        super().__init__()
        self.param = (
            torch.nn.Parameter(wide_ramp(param_shape), requires_grad=False)
            if param_shape is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.param is not None:
            return torch.erf(self.param) + x
        return torch.erf(x)


class ErfGeluModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
