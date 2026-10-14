# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""No-param unary activation modules + input gens for the WebGPU op-test framework.

`UNARY_G1` (op name -> (torch fn, input gen)) is imported by `cases.py` to drive the
declarative suites; each op mirrors the Vulkan `add_unary_op_node` activations. Inputs
are deterministic and range-bounded per op (positive for sqrt/rsqrt; spanning the ±3
knees for hardsigmoid and hardswish; reaching the ±15 clamp for tanh) so the fp64 golden is well-defined.
"""

import os

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


class PowScalarModule(torch.nn.Module):
    """aten.pow.Tensor_Scalar with a baked exponent (exponent → the min slot)."""

    def __init__(self, exponent) -> None:
        super().__init__()
        self.exponent = exponent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(x, self.exponent)


# name -> exponent construct kwarg; the suite uses a positive base to avoid NaN.
POW_SCALAR_CONFIGS = {
    "square": 2.0,
    "sqrt": 0.5,
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
    "hardsigmoid": (F.hardsigmoid, _lin(-6.0, 6.0)),
    "hardswish": (F.hardswish, _lin(-6.0, 6.0)),
}
# tan deferred: absent from the Vulkan partitioner (op_registry.py), so it can't be
# delegated yet; porting needs a partitioner extension (own diff).


# Unary ops whose fp16 programs the WebGPU backend must refuse at load: the unary
# shaders read array<f32>, so an fp16 operand would be misread, not computed.
UNARY_FP16_NEGATIVE = ("hardsigmoid", "hardswish", "abs")


def export_unary_fp16_negative(out_dir: str) -> None:
    """Export fp16 unary programs as unary_fp16_<op>.pte, for the native test.

    An even element count passes the 4-byte size check, so only the dtype check
    refuses them. Asserts each still delegates to VulkanBackend, so the native
    test exercises the runtime check rather than a CPU fallback.
    """
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    os.makedirs(out_dir, exist_ok=True)
    for name in UNARY_FP16_NEGATIVE:
        torch_fn, _ = UNARY_G1[name]
        x = torch.linspace(-6.0, 6.0, 32, dtype=torch.float16).reshape(4, 8)
        ep = torch.export.export(UnaryModule(torch_fn), (x,))
        et_program = to_edge_transform_and_lower(
            ep, partitioner=[VulkanPartitioner()]
        ).to_executorch()
        delegated = any(
            d.id == "VulkanBackend"
            for plan in et_program.executorch_program.execution_plan
            for d in plan.delegates
        )
        if not delegated:
            raise RuntimeError(f"{name}: expected VulkanBackend delegation")
        with open(os.path.join(out_dir, f"unary_fp16_{name}.pte"), "wb") as f:
            f.write(et_program.buffer)
        print(f"Exported unary_fp16_{name}.pte")
