# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Folds the clamping activations into the fused op that feeds them.

An activation reading ``P``'s output on grid ``A`` and writing grid ``B`` lets
``P`` quantize straight to ``B``, with the activation's clamp as ``P``'s output
range: ``Q_B`` is monotone, so ``Q_B(clamp(y, lo, hi)) = clamp(Q_B(y), Q_B(lo),
Q_B(hi))``. The CMSIS-NN kernels clamp to that range, their
``activation_min`` and ``activation_max``, and the activation is gone. That
drops the rounding to ``A``, which the current flow, folding the same way, never
has.

``P`` must own its output grid and clamp to its output range, or the range must
be the dtype's, which saturating already is. A max pool writes on its input's
grid, so it folds only an activation on that grid; the quantizer gives one to
an activation reading a max pool. A hardsigmoid, ``clamp(y / 6 + 1 / 2, 0, 1)``,
folds into an op that requantizes its result: the output grid's scale grows by
6 and its zero point moves by ``1 / (2 * scale)``, rounded, which can put a
value one step off.

Only an op the activation is the sole reader of folds, through data movement,
and only if the fold leaves it as lowerable as it was: an add's fixed-point
parameters, for one, depend on its output scale.
"""

import operator
from typing import cast, Optional

import torch
from executorch.backends.cortex_m.fused_quant.elementwise import LUT_TARGETS
from executorch.backends.cortex_m.fused_quant.utils import Context, QParams, Rules
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.backends.transforms.fused_quant.graph_utils import get_qparams_from_node
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch import fx
from torch.export import ExportedProgram

_aten = exir_ops.edge.aten
_fused_quant = exir_ops.edge.fused_quant

_CLAMPS = {
    _fused_quant.relu.default: lambda node: (0.0, None),
    _fused_quant.hard_tanh.default: lambda node: (
        get_arg(node, "min_val"),
        get_arg(node, "max_val"),
    ),
    _fused_quant.clamp.default: lambda node: (
        get_arg(node, "min"),
        get_arg(node, "max"),
    ),
}
_HARDSIGMOID = _fused_quant.hardsigmoid.default

# Requantize a fresh result to any grid and clamp it to the output range.
_REQUANTIZING = frozenset(
    {
        _fused_quant.conv1d.default,
        _fused_quant.conv2d.default,
        _fused_quant.convolution.default,
        _fused_quant.convolution_channels_last.default,
        _fused_quant.linear.default,
        _fused_quant.add.default,
        _fused_quant.sub.default,
    }
)
# A table holds any function of its input, clamped or not.
_CLAMPING = _REQUANTIZING | frozenset(LUT_TARGETS)
# Write any grid, but saturate only to the dtype's range.
_SATURATING = frozenset(
    {
        _fused_quant.mul.default,
        _fused_quant.div.default,
        _fused_quant.bmm.default,
        _fused_quant.mean.dim,
    }
)
# Write their input's grid and clamp to the output range.
_SAME_GRID = frozenset(
    {
        _fused_quant.max_pool2d_with_indices.default,
        _fused_quant.max_pool2d_with_indices_channels_last.default,
    }
)
_PRODUCERS = _CLAMPING | _SATURATING | _SAME_GRID
_DTYPES = (torch.int8, torch.int16)
_OUT_NAMES = ("out_scale", "out_zero_point", "out_quant_min", "out_quant_max")


def _passes_through(node: fx.Node) -> bool:
    """Moves the bytes it reads without reading their grid."""
    return node.target in (_aten.view_copy.default, _aten.permute_copy.default) or (
        node.target is operator.getitem and bool(node.args[1] == 0)
    )


def _producer(activation: fx.Node) -> Optional[fx.Node]:
    """The fused op whose only reader, through data movement, is activation."""
    cursor = get_arg(activation, "inp", fx.Node)
    while _passes_through(cursor) and len(cursor.users) == 1:
        cursor = cast(fx.Node, cursor.args[0])
    if cursor.target not in _PRODUCERS or len(cursor.users) != 1:
        return None
    return cursor


def _bound(qparams: QParams, value: object, default: int) -> int:
    return default if value is None else qparams.quantize(float(cast(float, value)))


def _folded_output(
    ctx: Context, activation: fx.Node, producer: fx.Node
) -> Optional[tuple[object, ...]]:
    """The producer's output qparams with activation folded in, if it folds."""
    produced = ctx.qparams(producer, "out", _DTYPES)
    read = ctx.qparams(activation, "inp", _DTYPES)
    written = ctx.qparams(activation, "out", _DTYPES)
    if (
        produced is None
        or written is None
        # The activation must read the bytes on the grid they were written.
        or produced != read
        or produced.dtype != written.dtype
    ):
        return None
    qparams = get_qparams_from_node(activation, "out")
    assert qparams is not None

    if activation.target == _HARDSIGMOID:
        if producer.target not in _REQUANTIZING:
            return None
        zero_point = written.zero_point + round(1 / (2 * written.scale))
        info = torch.iinfo(written.dtype)
        if not info.min <= zero_point <= info.max:
            return None
        scale_dtype = cast(torch.Tensor, ctx.constant(qparams.scale)).dtype
        zero_point_dtype = cast(torch.Tensor, ctx.constant(qparams.zero_point)).dtype
        return (
            ctx.add_constant(
                producer,
                "hardsigmoid_scale",
                torch.tensor(6 * written.scale, dtype=scale_dtype),
            ),
            ctx.add_constant(
                producer,
                "hardsigmoid_zero_point",
                torch.tensor(zero_point, dtype=zero_point_dtype),
            ),
            written.quantize(0.0),
            written.quantize(1.0),
        )

    low, high = _CLAMPS[activation.target](activation)
    clamped = QParams(
        written.scale,
        written.zero_point,
        _bound(written, low, written.quant_min),
        _bound(written, high, written.quant_max),
        written.dtype,
    )
    if producer.target in _SAME_GRID and clamped.grid != produced.grid:
        return None
    if producer.target in _SATURATING and not clamped.is_full_range:
        return None
    return qparams.scale, qparams.zero_point, clamped.quant_min, clamped.quant_max


class FoldActivations(ExportedProgramPassBase):
    """Folds each relu, hardtanh, clamp and hardsigmoid into the fused op that
    feeds it, where that op takes it; see the module docstring.

    ``rules`` are the lowering's, so that a fold never costs a kernel.
    """

    def __init__(self, rules: Rules, target_config: CortexMTargetConfig) -> None:
        super().__init__()
        self.rules = rules
        self.target_config = target_config

    def _lowerable(self, ctx: Context, node: fx.Node) -> bool:
        rule = self.rules.get(cast(EdgeOpOverload, node.target))
        return rule is not None and rule.can_lower(ctx, node)

    def _fold(self, ctx: Context, activation: fx.Node) -> bool:
        producer = _producer(activation)
        if producer is None:
            return False
        output = _folded_output(ctx, activation, producer)
        if output is None:
            return False
        lowerable = self._lowerable(ctx, producer)
        before: list[fx.node.Argument] = [
            get_arg(producer, name) for name in _OUT_NAMES
        ]
        for name, value in zip(_OUT_NAMES, output):
            set_arg(producer, name, cast(fx.node.Argument, value))
        if lowerable and not self._lowerable(ctx, producer):
            for name, value in zip(_OUT_NAMES, before):
                set_arg(producer, name, value)
            return False
        activation.replace_all_uses_with(get_arg(activation, "inp", fx.Node))
        activation.graph.erase_node(activation)
        return True

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        ctx = Context(exported_program, self.target_config)
        modified = False
        for node in list(exported_program.graph.nodes):
            if node.op == "call_function" and (
                node.target in _CLAMPS or node.target == _HARDSIGMOID
            ):
                modified |= self._fold(ctx, node)
        if modified:
            exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, modified)
