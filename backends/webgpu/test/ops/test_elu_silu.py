# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.elu.default` / `aten.silu.default` modules + inputs for the WebGPU tests.

Imported by `cases.py` and `test_webgpu_partitioner.py`. Inputs reach deep
negatives, where ELU saturates at -scale * alpha and SiLU tends to -0.
"""

import torch


def extreme_values(shape) -> torch.Tensor:
    """+-inf, NaN, +-100, +-88.8 (past exp's fp32 overflow) and +-0.0, tiled."""
    values = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 100.0, -100.0, 88.8, -88.8]
        + [-0.0, 0.0, 1.0]
    )
    n = 1
    for d in shape:
        n *= d
    return values.repeat((n + values.numel() - 1) // values.numel())[:n].reshape(shape)


def signed_ramp(shape) -> torch.Tensor:
    """Deterministic linspace over [-12, 6] reshaped to `shape`."""
    n = 1
    for d in shape:
        n *= d
    return torch.linspace(-12.0, 6.0, n, dtype=torch.float32).reshape(shape)


class EluModule(torch.nn.Module):
    """aten.elu with baked alpha / scale / input_scale.

    With `param_shape`, ELU runs on a parameter (a constant input to the
    delegate) and the result is added to the runtime input.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        scale: float = 1.0,
        input_scale: float = 1.0,
        param_shape=None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.scale = scale
        self.input_scale = input_scale
        self.param = (
            torch.nn.Parameter(signed_ramp(param_shape), requires_grad=False)
            if param_shape is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = self.param if self.param is not None else x
        y = torch.ops.aten.elu.default(src, self.alpha, self.scale, self.input_scale)
        return y + x if self.param is not None else y


class SiluModule(torch.nn.Module):
    """aten.silu; with `param_shape`, on a parameter added to the input."""

    def __init__(self, param_shape=None) -> None:
        super().__init__()
        self.param = (
            torch.nn.Parameter(signed_ramp(param_shape), requires_grad=False)
            if param_shape is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.param is not None:
            return torch.nn.functional.silu(self.param) + x
        return torch.nn.functional.silu(x)


class SiluMulModule(torch.nn.Module):
    """silu(gate) * up, the SwiGLU pattern."""

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(gate) * up


class SiluCondModule(torch.nn.Module):
    """silu inside a torch.cond branch (a control-flow submodule)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cond(
            x.sum() > 0,
            lambda t: torch.nn.functional.silu(t),
            lambda t: t + 1.0,
            (x,),
        )


# name -> (alpha, scale, input_scale).
ELU_CONFIGS = {
    "default": (1.0, 1.0, 1.0),
    "alpha_half": (0.5, 1.0, 1.0),
    "alpha_scaled": (1.6732632, 1.0507010, 1.0),
    "input_scale": (1.3, 0.8, 0.5),
}
