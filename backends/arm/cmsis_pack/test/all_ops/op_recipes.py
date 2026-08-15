#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Per-operator model recipes for the all-ops execution test.

A recipe knows how to build a tiny torch model + example inputs that exercise
one operator component (one op_*.cpp / one #ifdef guard).
generate_test_models.py turns each recipe into a .pte plus a reference output.

Coverage is driven by op_guards.discover_components: every discovered component
must either have a recipe or an explicit skip reason, otherwise it is reported
as an uncovered gap (no silent gaps). Recipes are registered in bulk for the
large elementwise classes and individually for the structured ops.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class Recipe:
    """How to exercise one operator component."""

    category: str  # "Portable" | "Quantized" | "Cortex-M"
    make: Callable[[], tuple]  # () -> (nn.Module, example_inputs_tuple)
    atol: float = 1e-6
    rtol: float = 1e-6
    # Non-empty => intentionally not executed; value explains why. The op still
    # gets build/link coverage via the all-ops cproject; only runtime execution
    # is skipped (e.g. nondeterministic or data-dependent-shape ops).
    skip: str = ""


# Registry keyed by (category, op-component-name).
RECIPES: dict[tuple[str, str], Recipe] = {}
# Explicit skips keyed the same way: ops we deliberately do not execute.
SKIPS: dict[tuple[str, str], str] = {}
# Recipes that export for the CPU targets but not through the Ethos-U flow,
# with the reason. These are backend limitations, not missing recipes, so they
# only apply to --ethos-u; the same op still runs on the CPU variants.
ETHOS_U_SKIPS: dict[tuple[str, str], str] = {
    ("Portable", "isnan"): (
        "NaN test inputs break the quantizer's histogram observer "
        "(torch.histc: range of [nan, nan] is not finite)"
    ),
    ("Portable", "max"): "ConvertMinMaxPass does not handle the (values, indices) form",
    ("Portable", "min"): "ConvertMinMaxPass does not handle the (values, indices) form",
    ("Portable", "select_scatter"): (
        "backend rejects the _to_dim_order_copy introduced by the decomposition"
    ),
}
# Extra shapes for EXISTING components, keyed by (category, variant-name) and
# carrying the base component name so kernel-presence/arity checks validate
# against the real kernel. Exported in addition to the per-component sweep;
# variants do not participate in component<->recipe reconciliation.
VARIANTS: dict[tuple[str, str], tuple[str, "Recipe"]] = {}


# torch.nn layers initialize their weights randomly. The wrappers below pin
# every parameter to 1.0 so exports (and the reference outputs embedded in the
# pte) are reproducible run to run.
class FixedParametersConv2D(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, **kwargs, bias=False)
        self.conv.weight.data.fill_(1.0)

    def forward(self, x):
        return self.conv(x)


class FixedParametersEmbedding(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embedding = torch.nn.Embedding(*args, **kwargs)
        self.embedding.weight.data.fill_(1.0)

    def forward(self, x):
        return self.embedding(x)


class FixedParametersLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(*args, **kwargs)
        self.linear.weight.data.fill_(1.0)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(1.0)

    def forward(self, x):
        return self.linear(x)


class FixedParametersConvTranspose2d(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(*args, **kwargs)
        self.conv_transpose.weight.data.fill_(1.0)
        if self.conv_transpose.bias is not None:
            self.conv_transpose.bias.data.fill_(1.0)

    def forward(self, x):
        return self.conv_transpose(x)


class _Fn(torch.nn.Module):
    """Wrap a plain callable as a module so torch.export can trace it."""

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _ramp(lo: float, hi: float, shape: tuple) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= s
    return torch.linspace(lo, hi, n).reshape(shape)


def _reg(category: str, name: str, make: Callable, **kw) -> None:
    RECIPES[(category, name)] = Recipe(category=category, make=make, **kw)


def _portable(name: str, make: Callable, **kw) -> None:
    _reg("Portable", name, make, **kw)


# --------------------------------------------------------------------------
# Bulk classes
# --------------------------------------------------------------------------

# Unary float elementwise: op(x), default domain. (lo, hi) chosen per op so the
# input stays in the op's valid range.
_UNARY_FLOAT = {
    "abs": (torch.abs, -2.0, 2.0),
    "acos": (torch.acos, -0.9, 0.9),
    "acosh": (torch.acosh, 1.1, 3.0),
    "asin": (torch.asin, -0.9, 0.9),
    "asinh": (torch.asinh, -2.0, 2.0),
    "atan": (torch.atan, -3.0, 3.0),
    "atanh": (torch.atanh, -0.9, 0.9),
    "ceil": (torch.ceil, -2.5, 2.5),
    "conj_physical": (torch.conj_physical, -2.0, 2.0),
    "cos": (torch.cos, -3.0, 3.0),
    "cosh": (torch.cosh, -2.0, 2.0),
    "erf": (torch.erf, -2.0, 2.0),
    "exp": (torch.exp, -2.0, 2.0),
    "expm1": (torch.expm1, -2.0, 2.0),
    "floor": (torch.floor, -2.5, 2.5),
    "log": (torch.log, 0.1, 4.0),
    "log10": (torch.log10, 0.1, 4.0),
    "log1p": (torch.log1p, -0.5, 4.0),
    "log2": (torch.log2, 0.1, 4.0),
    "neg": (torch.neg, -2.0, 2.0),
    "reciprocal": (torch.reciprocal, 0.2, 3.0),
    "relu": (torch.relu, -2.0, 2.0),
    "round": (torch.round, -2.5, 2.5),
    "rsqrt": (torch.rsqrt, 0.2, 4.0),
    "sigmoid": (torch.sigmoid, -4.0, 4.0),
    "sign": (torch.sign, -2.0, 2.0),
    "sin": (torch.sin, -3.0, 3.0),
    "sinh": (torch.sinh, -2.0, 2.0),
    "sqrt": (torch.sqrt, 0.0, 4.0),
    "tan": (torch.tan, -1.0, 1.0),
    "tanh": (torch.tanh, -3.0, 3.0),
    "trunc": (torch.trunc, -2.5, 2.5),
    "logit": (lambda x: torch.logit(x, eps=1e-6), 0.1, 0.9),
    "gelu": (torch.nn.functional.gelu, -3.0, 3.0),
}
for _n, (_fn, _lo, _hi) in _UNARY_FLOAT.items():
    _portable(_n, (lambda fn=_fn, lo=_lo, hi=_hi: (_Fn(fn), (_ramp(lo, hi, (2, 3)),))))

# Unary that returns bool / needs special handling.
_portable("isinf", lambda: (_Fn(torch.isinf), (torch.tensor([[1.0, float("inf")]]),)))
_portable("isnan", lambda: (_Fn(torch.isnan), (torch.tensor([[1.0, float("nan")]]),)))
_portable(
    "logical_not", lambda: (_Fn(torch.logical_not), (torch.tensor([[True, False]]),))
)
_portable(
    "bitwise_not",
    lambda: (_Fn(torch.bitwise_not), (torch.tensor([[1, 2, 3]], dtype=torch.int32),)),
)

# Binary float elementwise: op(x, y).
_BINARY_FLOAT = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
    "atan2": torch.atan2,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
    "fmod": torch.fmod,
    "remainder": torch.remainder,
    "pow": torch.pow,
    "rsub": lambda a, b: torch.rsub(a, b),
}
for _n, _fn in _BINARY_FLOAT.items():
    lo = 0.5 if _n == "pow" else -2.0
    _portable(
        _n,
        (
            lambda fn=_fn, lo=lo: (
                _Fn(fn),
                (_ramp(lo, 2.0, (2, 3)), _ramp(1.0, 2.0, (2, 3))),
            )
        ),
    )

# Comparison ops (return bool).
for _n, _fn in {
    "eq": torch.eq,
    "ne": torch.ne,
    "lt": torch.lt,
    "le": torch.le,
    "gt": torch.gt,
    "ge": torch.ge,
}.items():
    _portable(
        _n,
        (
            lambda fn=_fn: (
                _Fn(fn),
                (_ramp(-2.0, 2.0, (2, 3)), _ramp(-1.0, 1.0, (2, 3))),
            )
        ),
    )

# Logical ops (bool inputs).
for _n, _fn in {
    "logical_and": torch.logical_and,
    "logical_or": torch.logical_or,
    "logical_xor": torch.logical_xor,
}.items():
    _portable(
        _n,
        (
            lambda fn=_fn: (
                _Fn(fn),
                (
                    torch.tensor([[True, False, True]]),
                    torch.tensor([[True, True, False]]),
                ),
            )
        ),
    )

# Integer bitwise / shift ops.
for _n, _fn in {
    "bitwise_and": torch.bitwise_and,
    "bitwise_or": torch.bitwise_or,
    "bitwise_xor": torch.bitwise_xor,
    "bitwise_left_shift": torch.bitwise_left_shift,
    "bitwise_right_shift": torch.bitwise_right_shift,
    "floor_divide": torch.floor_divide,
}.items():
    _portable(
        _n,
        (
            lambda fn=_fn: (
                _Fn(fn),
                (
                    torch.tensor([[1, 2, 6]], dtype=torch.int32),
                    torch.tensor([[1, 1, 2]], dtype=torch.int32),
                ),
            )
        ),
    )


# --------------------------------------------------------------------------
# Activations / clamps with attributes
# --------------------------------------------------------------------------
_portable(
    "clamp", lambda: (_Fn(lambda x: torch.clamp(x, -0.5, 0.5)), (_ramp(-2, 2, (2, 3)),))
)
_portable("elu", lambda: (torch.nn.ELU(), (_ramp(-3, 3, (2, 3)),)))
_portable("hardtanh", lambda: (torch.nn.Hardtanh(), (_ramp(-3, 3, (2, 3)),)))
_portable("leaky_relu", lambda: (torch.nn.LeakyReLU(0.1), (_ramp(-3, 3, (2, 3)),)))
_portable("glu", lambda: (torch.nn.GLU(dim=-1), (_ramp(-2, 2, (2, 4)),)))
_portable(
    "where",
    lambda: (
        _Fn(lambda c, a, b: torch.where(c, a, b)),
        (torch.tensor([[True, False, True]]), _ramp(0, 1, (1, 3)), _ramp(2, 3, (1, 3))),
    ),
)
_portable(
    "masked_fill",
    lambda: (
        _Fn(lambda x, m: x.masked_fill(m, 0.0)),
        (
            _ramp(-2, 2, (2, 3)),
            torch.tensor([[True, False, True], [False, True, False]]),
        ),
    ),
)


# --------------------------------------------------------------------------
# Reductions (need dim / keepdim)
# --------------------------------------------------------------------------
def _red(name: str, fn: Callable, **kw):
    _portable(name, lambda fn=fn: (_Fn(fn), (_ramp(-2, 2, (3, 4)),)), **kw)


_red("sum", lambda x: torch.sum(x, dim=1))
_red("mean", lambda x: torch.mean(x, dim=1))
_red("amax", lambda x: torch.amax(x, dim=1))
_red("amin", lambda x: torch.amin(x, dim=1))
_red("prod", lambda x: torch.prod(x, dim=1))
_red("argmax", lambda x: torch.argmax(x, dim=1))
_red("argmin", lambda x: torch.argmin(x, dim=1))
_red("any", lambda x: torch.any(x > 0, dim=1))
_red("var", lambda x: torch.var(x, dim=1))
_red("var_mean", lambda x: torch.var_mean(x, dim=1))
_red("max", lambda x: tuple(torch.max(x, dim=1)))
_red("min", lambda x: tuple(torch.min(x, dim=1)))
_red("cumsum", lambda x: torch.cumsum(x, dim=1))
_red("softmax", lambda x: torch.softmax(x, dim=1))
_red("log_softmax", lambda x: torch.log_softmax(x, dim=1))


# --------------------------------------------------------------------------
# Matmul / linear
# --------------------------------------------------------------------------
_portable("mm", lambda: (_Fn(torch.mm), (_ramp(-1, 1, (3, 4)), _ramp(-1, 1, (4, 2)))))
_portable(
    "bmm", lambda: (_Fn(torch.bmm), (_ramp(-1, 1, (2, 3, 4)), _ramp(-1, 1, (2, 4, 2))))
)
_portable(
    "addmm",
    lambda: (
        _Fn(lambda b, x, w: torch.addmm(b, x, w)),
        (_ramp(0, 1, (2,)), _ramp(-1, 1, (3, 4)), _ramp(-1, 1, (4, 2))),
    ),
)


# --------------------------------------------------------------------------
# Conv / pool / norm
# --------------------------------------------------------------------------
_portable(
    "convolution",
    lambda: (FixedParametersConv2D(2, 3, 3), (_ramp(-1, 1, (1, 2, 8, 8)),)),
)
_portable("avg_pool2d", lambda: (torch.nn.AvgPool2d(2), (_ramp(-1, 1, (1, 2, 8, 8)),)))
_portable(
    "adaptive_avg_pool2d",
    lambda: (torch.nn.AdaptiveAvgPool2d((2, 2)), (_ramp(-1, 1, (1, 2, 8, 8)),)),
)
_portable(
    "max_pool2d_with_indices",
    lambda: (torch.nn.MaxPool2d(2, return_indices=True), (_ramp(-1, 1, (1, 2, 8, 8)),)),
)
_portable("native_layer_norm", lambda: (torch.nn.LayerNorm(4), (_ramp(-2, 2, (2, 4)),)))
_portable(
    "native_group_norm", lambda: (torch.nn.GroupNorm(2, 4), (_ramp(-2, 2, (1, 4, 4)),))
)
_portable(
    "native_batch_norm",
    lambda: (torch.nn.BatchNorm2d(2).eval(), (_ramp(-2, 2, (1, 2, 4, 4)),)),
)
_portable(
    "constant_pad_nd",
    lambda: (
        _Fn(lambda x: torch.nn.functional.pad(x, (1, 1), value=0.0)),
        (_ramp(-1, 1, (2, 3)),),
    ),
)
for _n, _pad in {
    "reflection_pad1d": (torch.nn.ReflectionPad1d(1), (1, 3, 6)),
    "reflection_pad2d": (torch.nn.ReflectionPad2d(1), (1, 3, 6, 6)),
    "reflection_pad3d": (torch.nn.ReflectionPad3d(1), (1, 3, 4, 6, 6)),
    "replication_pad1d": (torch.nn.ReplicationPad1d(1), (1, 3, 6)),
    "replication_pad2d": (torch.nn.ReplicationPad2d(1), (1, 3, 6, 6)),
    "replication_pad3d": (torch.nn.ReplicationPad3d(1), (1, 3, 4, 6, 6)),
}.items():
    _portable(_n, lambda mod=_pad[0], sh=_pad[1]: (mod, (_ramp(-1, 1, sh),)))
_portable(
    "upsample_nearest2d",
    lambda: (
        _Fn(
            lambda x: torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        ),
        (_ramp(-1, 1, (1, 2, 4, 4)),),
    ),
)
_portable(
    "upsample_bilinear2d",
    lambda: (
        _Fn(
            lambda x: torch.nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear"
            )
        ),
        (_ramp(-1, 1, (1, 2, 4, 4)),),
    ),
)
_portable(
    "pixel_shuffle", lambda: (torch.nn.PixelShuffle(2), (_ramp(-1, 1, (1, 8, 4, 4)),))
)
_portable(
    "pixel_unshuffle",
    lambda: (torch.nn.PixelUnshuffle(2), (_ramp(-1, 1, (1, 2, 8, 8)),)),
)


# --------------------------------------------------------------------------
# Shape / view ops
# --------------------------------------------------------------------------
_VIEWS = {
    "view_copy": (lambda x: x.reshape(6), (2, 3)),
    "permute_copy": (lambda x: x.permute(1, 0), (2, 3)),
    "transpose_copy": (lambda x: x.transpose(0, 1), (2, 3)),
    "squeeze_copy": (lambda x: x.squeeze(0), (1, 3)),
    "unsqueeze_copy": (lambda x: x.unsqueeze(0), (2, 3)),
    "expand_copy": (lambda x: x.expand(2, 3), (1, 3)),
    "repeat": (lambda x: x.repeat(2, 1), (2, 3)),
    "flip": (lambda x: torch.flip(x, [1]), (2, 3)),
    "roll": (lambda x: torch.roll(x, 1, 1), (2, 3)),
    "t_copy": (lambda x: x.t(), (2, 3)),
    "tril": (lambda x: torch.tril(x), (3, 3)),
    "diagonal_copy": (lambda x: torch.diagonal(x), (3, 3)),
    "narrow_copy": (lambda x: torch.narrow(x, 1, 0, 2), (2, 3)),
    "select_copy": (lambda x: torch.select(x, 1, 0), (2, 3)),
    "slice_copy": (lambda x: x[:, 0:2], (2, 3)),
    "unfold_copy": (lambda x: x.unfold(1, 2, 1), (2, 4)),
    "detach_copy": (lambda x: x.detach(), (2, 3)),
    "alias_copy": (lambda x: torch.ops.aten.alias_copy(x), (2, 3)),
    "clone": (lambda x: x.clone(), (2, 3)),
    "lift_fresh_copy": (lambda x: x.clone(), (2, 3)),
    "as_strided_copy": (
        lambda x: torch.as_strided(x, (2, 2), (3, 1)),
        (2, 3),
    ),
    "repeat_interleave": (lambda x: x.repeat_interleave(2, dim=1), (2, 3)),
}
for _n, (_fn, _sh) in _VIEWS.items():
    _portable(_n, lambda fn=_fn, sh=_sh: (_Fn(fn), (_ramp(-1, 1, sh),)))

_portable(
    "cat",
    lambda: (
        _Fn(lambda a, b: torch.cat([a, b], 1)),
        (_ramp(0, 1, (2, 2)), _ramp(2, 3, (2, 2))),
    ),
)
_portable(
    "stack",
    lambda: (
        _Fn(lambda a, b: torch.stack([a, b], 0)),
        (_ramp(0, 1, (2, 2)), _ramp(2, 3, (2, 2))),
    ),
)


# --------------------------------------------------------------------------
# Indexing / gather / scatter
# --------------------------------------------------------------------------
_portable(
    "index_select",
    lambda: (
        _Fn(lambda x: torch.index_select(x, 1, torch.tensor([0, 2]))),
        (_ramp(-1, 1, (2, 3)),),
    ),
)
_portable(
    "gather",
    lambda: (
        _Fn(lambda x: torch.gather(x, 1, torch.tensor([[0, 1], [2, 0]]))),
        (_ramp(-1, 1, (2, 3)),),
    ),
)
_portable(
    "embedding",
    lambda: (FixedParametersEmbedding(5, 3), (torch.tensor([[0, 2], [1, 4]]),)),
)
_portable(
    "scatter_add",
    lambda: (
        _Fn(
            lambda x: torch.zeros(2, 3).scatter_add(
                1, torch.tensor([[0, 1, 2], [2, 1, 0]]), x
            )
        ),
        (_ramp(-1, 1, (2, 3)),),
    ),
)


# --------------------------------------------------------------------------
# Creation ops (input is a seed tensor to keep one input signature)
# --------------------------------------------------------------------------
_portable(
    "full_like",
    lambda: (_Fn(lambda x: torch.full_like(x, 3.0)), (_ramp(0, 1, (2, 3)),)),
)
_portable("fill", lambda: (_Fn(lambda x: torch.fill(x, 2.0)), (_ramp(0, 1, (2, 3)),)))


# Multi-output / scatter / indexing ops.
_portable(
    "topk",
    lambda: (_Fn(lambda x: tuple(torch.topk(x, 2, dim=1))), (_ramp(-2, 2, (2, 4)),)),
)
_portable(
    "split_copy",
    lambda: (_Fn(lambda x: torch.split(x, 1, dim=1)), (_ramp(-1, 1, (2, 3)),)),
)
_portable(
    "split_with_sizes_copy",
    lambda: (_Fn(lambda x: torch.split(x, [1, 2], dim=1)), (_ramp(-1, 1, (2, 3)),)),
)
_portable(
    "unbind_copy",
    lambda: (_Fn(lambda x: torch.unbind(x, dim=0)), (_ramp(-1, 1, (2, 3)),)),
)
_portable(
    "index",
    lambda: (
        _Fn(lambda x: x[torch.tensor([0, 2])]),
        (_ramp(-1, 1, (3, 4)),),
    ),
)
_portable(
    "index_put",
    lambda: (
        _Fn(lambda x: x.index_put((torch.tensor([0]),), torch.zeros(3))),
        (_ramp(-1, 1, (2, 3)),),
    ),
)
_portable(
    "scatter",
    lambda: (
        _Fn(
            lambda x: torch.zeros(2, 3).scatter(
                1, torch.tensor([[0, 1, 2], [2, 1, 0]]), x
            )
        ),
        (_ramp(-1, 1, (2, 3)),),
    ),
)
_portable(
    "select_scatter",
    lambda: (
        _Fn(lambda x: torch.select_scatter(x, torch.zeros(3), 0, 0)),
        (_ramp(-1, 1, (2, 3)),),
    ),
)
_portable(
    "slice_scatter",
    lambda: (
        _Fn(lambda x: torch.slice_scatter(x, torch.zeros(2, 2), 1, 0, 2)),
        (_ramp(-1, 1, (2, 3)),),
    ),
)
_portable(
    "masked_scatter",
    lambda: (
        _Fn(
            lambda x: x.masked_scatter(
                torch.tensor([[True, False, True], [False, True, False]]), torch.ones(6)
            )
        ),
        (_ramp(-1, 1, (2, 3)),),
    ),
)
_portable(
    "to_copy", lambda: (_Fn(lambda x: x.to(torch.int32)), (_ramp(-2, 2, (2, 3)),))
)
_portable(
    "copy",
    lambda: (
        _Fn(lambda x, y: torch.ops.aten.copy.default(x, y)),
        (_ramp(-1, 1, (2, 3)), _ramp(2, 3, (2, 3))),
    ),
)


# --------------------------------------------------------------------------
# Cortex-M ops (quantized; exercised via the CortexM export flow). Inputs are
# float; the CortexMQuantizer + passes lower them to cortex_m::* kernels.
# --------------------------------------------------------------------------
def _cm(name: str, make: Callable, **kw):
    _reg("Cortex-M", name, make, atol=2e-2, rtol=2e-2, **kw)


_cm(
    "quantized_add",
    lambda: (_Fn(lambda a, b: a + b), (_ramp(-2, 2, (1, 8)), _ramp(-1, 1, (1, 8)))),
)
_cm(
    "quantized_mul",
    lambda: (_Fn(lambda a, b: a * b), (_ramp(-2, 2, (1, 8)), _ramp(-1, 1, (1, 8)))),
)
_cm(
    "maximum",
    lambda: (_Fn(torch.maximum), (_ramp(-2, 2, (1, 8)), _ramp(-1, 1, (1, 8)))),
)
_cm(
    "minimum",
    lambda: (_Fn(torch.minimum), (_ramp(-2, 2, (1, 8)), _ramp(-1, 1, (1, 8)))),
)
_cm(
    "quantized_linear",
    lambda: (FixedParametersLinear(8, 4, bias=False), (_ramp(-2, 2, (1, 8)),)),
)
_cm(
    "softmax", lambda: (_Fn(lambda x: torch.softmax(x, dim=1)), (_ramp(-2, 2, (1, 8)),))
)
_cm("transpose", lambda: (_Fn(lambda x: x.transpose(1, 2)), (_ramp(-2, 2, (1, 4, 6)),)))
_cm(
    "pad",
    lambda: (
        _Fn(lambda x: torch.nn.functional.pad(x, (1, 1))),
        (_ramp(-2, 2, (1, 4, 6)),),
    ),
)
_cm(
    "quantized_conv2d",
    lambda: (
        FixedParametersConv2D(2, 4, 3),
        (_ramp(1, 5, (1, 2, 5, 5)).to(memory_format=torch.channels_last),),
    ),
)
_cm(
    "quantized_depthwise_conv2d",
    lambda: (
        FixedParametersConv2D(4, 4, 3, groups=4),
        (_ramp(1, 5, (1, 4, 8, 8)).to(memory_format=torch.channels_last),),
    ),
)


# Scratch-buffer danger shapes. The AoT scratch is sized per target core's
# CMSIS-NN path (generate_test_models.py --target-core: m55->MVE, m7/m4->DSP,
# m0plus->SCALAR). These shapes cross the thresholds where the DSP requirement
# EXCEEDS the MVE size, so exporting for a DSP-class core (--target-core m7)
# yields a visibly larger scratch buffer than the M55/MVE default -- they verify
# the per-core AoT sizing is correct (a model sized for the wrong path would
# under-allocate on the DSP kernel, whose only guard is a NULL check):
#   * depthwise, ch > 248, non-3x3: DSP needs 2*ch*kw*kh B, but the MVE
#     size (4*CH_IN_BLOCK_MVE*kw*kh) is channel-independent.
#   * conv with input h==1 and padding: DSP dispatches to the 1_x_n kernel
#     whose requirement is not covered by the MVE 1_x_n formula for all
#     padded shapes.
def _cm_variant(base: str, variant: str, make: Callable, **kw):
    VARIANTS[("Cortex-M", variant)] = (
        base,
        Recipe(category="Cortex-M", make=make, atol=2e-2, rtol=2e-2, **kw),
    )


_cm_variant(
    "quantized_depthwise_conv2d",
    "quantized_depthwise_conv2d_deep",
    lambda: (
        FixedParametersConv2D(384, 384, 5, groups=384, padding=2),
        (_ramp(1, 5, (1, 384, 6, 6)).to(memory_format=torch.channels_last),),
    ),
)
_cm_variant(
    "quantized_conv2d",
    "quantized_conv2d_1xn_pad",
    lambda: (
        FixedParametersConv2D(4, 8, (1, 5), padding=(0, 2)),
        (_ramp(1, 5, (1, 4, 1, 16)).to(memory_format=torch.channels_last),),
    ),
)
_cm(
    "quantized_transpose_conv2d",
    lambda: (
        FixedParametersConvTranspose2d(2, 4, 3),
        (_ramp(1, 5, (1, 2, 5, 5)).to(memory_format=torch.channels_last),),
    ),
)
_cm(
    "quantized_avg_pool2d",
    lambda: (
        torch.nn.AvgPool2d(kernel_size=2, stride=2),
        (_ramp(0, 15, (1, 1, 4, 4)),),
    ),
)
_cm(
    "quantized_max_pool2d",
    lambda: (
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        (_ramp(-50, 50, (1, 1, 6, 6)),),
    ),
)
_cm(
    "quantized_batch_matmul",
    lambda: (_Fn(torch.bmm), (_ramp(-1, 1, (1, 4, 6)), _ramp(-1, 1, (1, 6, 4)))),
)
# quantize/dequantize_per_tensor are exercised as a side effect of any quantized
# model above; a dedicated round-trip keeps them attributable.
_cm(
    "quantize_per_tensor",
    lambda: (_Fn(lambda x: x + 0.0), (_ramp(-2, 2, (1, 8)),)),
)
_cm(
    "dequantize_per_tensor",
    lambda: (_Fn(lambda x: x + 0.0), (_ramp(-2, 2, (1, 8)),)),
)


# --------------------------------------------------------------------------
# Explicit skips: ops we deliberately do not execute at runtime. They still get
# build/link coverage from the all-ops cproject; only the .pte/reference step is
# skipped. Each must give a reason so the gap is documented, not silent.
# --------------------------------------------------------------------------
def _skip(category: str, name: str, reason: str) -> None:
    SKIPS[(category, name)] = reason


for _n, _r in {
    # Nondeterministic outputs -> no stable reference.
    "rand": "nondeterministic (RNG)",
    "randn": "nondeterministic (RNG)",
    "native_dropout": "nondeterministic (RNG)",
    "empty": "uninitialized contents",
    "empty_dim_order": "uninitialized contents",
    # Data-dependent output shape -> not expressible as a static .pte/reference.
    "nonzero": "data-dependent output shape",
    "masked_select": "data-dependent output shape",
    # Backward/training-only ops: no forward inference recipe.
    "convolution_backward": "backward op (training only)",
    "max_pool2d_with_indices_backward": "backward op (training only)",
    # Creation ops are constant-folded at export, so no runtime kernel call to
    # observe; build/link is still covered by the all-ops cproject.
    "arange": "creation op constant-folded at export",
    "full": "creation op constant-folded at export",
    "ones": "creation op constant-folded at export",
    "zeros": "creation op constant-folded at export",
    "scalar_tensor": "creation op constant-folded at export",
    # Dim-order conversions are a no-op for the default layout here.
    "clone_dim_order": "dim-order no-op for default layout; recipe TODO",
    "to_dim_order_copy": "dim-order no-op for default layout; recipe TODO",
    # Exotic / low-value to model here; covered by build/link, recipe TODO.
    "grid_sampler_2d": "recipe TODO (sampler grid setup)",
    "cdist_forward": "recipe TODO",
    "pdist_forward": "recipe TODO",
    "upsample_bilinear2d_aa": "antialiased upsample; recipe TODO",
    "view_as_real_copy": "complex dtype unsupported on target",
    "allclose": "returns scalar bool; ET out-variant shape TODO",
    # Host<->device transfer op: needs a registered DeviceAllocator and a
    # non-CPU device; bare-metal Corstone has neither.
    "device_copy": "requires a DeviceAllocator-backed device (none on target)",
}.items():
    _skip("Portable", _n, _r)

# LUT-based fused activation (sigmoid/tanh/silu): the CortexMQuantizer lowering
# that emits it does not exist in the current export environment.
_skip(
    "Cortex-M",
    "quantized_activation",
    "needs 1.4 AOT lowering (CortexMQuantizer activation LUT); recipe TODO",
)

# Quantized (quantized_decomposed::*) ops need the PT2E quantization flow to be
# emitted into the graph; a portable float export does not exercise them. They
# get build/link coverage now; execution recipes are a follow-up.
for _n in (
    "add",
    "choose_qparams",
    "dequantize",
    "embedding",
    "embedding2b",
    "embedding4b",
    "mixed_linear",
    "mixed_mm",
    "quantize",
):
    _skip(
        "Quantized", _n, "quantized_decomposed op; needs PT2E quant flow (recipe TODO)"
    )


def all_keys() -> set:
    """Every (category, name) that has a recipe or an explicit skip."""
    return set(RECIPES) | set(SKIPS)
