# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""No-param unary activation modules + input gens for the WebGPU op-test framework.

`UNARY_G1` (op name -> (torch fn, input gen)) is imported by `cases.py` to drive the
declarative suites; each op mirrors the Vulkan `add_unary_op_node` activations. Inputs
are deterministic and range-bounded per op (positive for sqrt/rsqrt; spanning the ±3
knees for hardswish; reaching the ±15 clamp for tanh) so the fp64 golden is well-defined.
"""

import torch
import torch.nn.functional as F


class UnaryModule(torch.nn.Module):
    """Applies a fixed unary op; `fn` is traced by torch.export (not a parameter)."""

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class ClampModule(torch.nn.Module):
    """aten.clamp.default with baked bounds; `lo`/`hi` may be None (-> ±inf)."""

    def __init__(self, lo, hi) -> None:
        super().__init__()
        self.lo = lo
        self.hi = hi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.lo, self.hi)


class HardtanhModule(torch.nn.Module):
    """aten.hardtanh.default with baked min/max bounds."""

    def __init__(self, lo, hi) -> None:
        super().__init__()
        self.lo = lo
        self.hi = hi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardtanh(x, self.lo, self.hi)


# name -> (lo, hi) construct kwargs. `min_none` exercises the None -> -inf path.
CLAMP_CONFIGS = {
    "both": (-2.0, 3.0),
    "min_none": (None, 3.0),
}
HARDTANH_CONFIGS = {
    "default": (-1.0, 1.0),
    "wide": (-2.0, 2.0),
}


def _lin(lo: float, hi: float):
    """Deterministic linspace input of the requested shape over [lo, hi]."""

    def gen(shape: tuple[int, ...]) -> torch.Tensor:
        n = 1
        for d in shape:
            n *= d
        return torch.linspace(lo, hi, n, dtype=torch.float32).reshape(shape)

    return gen


# op name -> (torch reference fn, input generator). Ranges keep each op numerically
# well-defined and away from asymptotes/NaN.
UNARY_G1 = {
    "abs": (torch.abs, _lin(-6.0, 6.0)),
    "exp": (torch.exp, _lin(-5.0, 5.0)),
    "sqrt": (torch.sqrt, _lin(0.05, 12.0)),
    "rsqrt": (torch.rsqrt, _lin(0.5, 12.0)),
    "sin": (torch.sin, _lin(-6.0, 6.0)),
    "cos": (torch.cos, _lin(-6.0, 6.0)),
    "tanh": (torch.tanh, _lin(-20.0, 20.0)),
    "round": (torch.round, _lin(-6.0, 6.0)),
    "neg": (torch.neg, _lin(-6.0, 6.0)),
    "hardswish": (F.hardswish, _lin(-6.0, 6.0)),
}
# tan + hardsigmoid deferred: absent from the Vulkan partitioner (op_registry.py),
# so they can't be delegated yet; porting needs a partitioner extension (own diff).
