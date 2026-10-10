# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import executorch.backends.fused_quant.ops  # noqa: F401
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from executorch.backends.fused_quant.graph_utils import (
    get_qparams_from_node,
    get_scale,
    get_zero_point,
)
from executorch.backends.transforms.utils import compute_meta_val
from torch import fx
from torch.export import ExportedProgram


def _fold_into_dequant_quant(ep: ExportedProgram, node: fx.Node, scalar: float) -> bool:
    """Fold scalar into dequant scale when both inp and out qparams are present.

    ``Q_out(c * DQ_inp(x))`` becomes ``DQ_{inp*c}(x) -> Q_out``.
    """
    inp_qparams = get_qparams_from_node(node, "inp")
    out_qparams = get_qparams_from_node(node, "out")
    assert inp_qparams is not None and out_qparams is not None

    if not inp_qparams.is_per_tensor() or not out_qparams.is_per_tensor():
        return False

    dequant_scale = get_scale(ep, inp_qparams) * scalar
    if dequant_scale <= 0.0 or not math.isfinite(dequant_scale):
        return False

    inp = get_arg(node, "inp", fx.Node)
    with node.graph.inserting_before(node):
        dequant = node.graph.call_function(
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(
                inp,
                dequant_scale,
                get_zero_point(ep, inp_qparams),
                inp_qparams.quant_min,
                inp_qparams.quant_max,
                inp.meta["val"].dtype,
            ),
            kwargs={"out_dtype": inp_qparams.dtype},
        )
        dequant.meta = node.meta.copy()
        dequant.meta["val"] = compute_meta_val(dequant)

        quant = node.graph.call_function(
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(
                dequant,
                get_scale(ep, out_qparams),
                get_zero_point(ep, out_qparams),
                out_qparams.quant_min,
                out_qparams.quant_max,
                out_qparams.dtype,
            ),
        )
        quant.meta = node.meta.copy()
        quant.meta["val"] = compute_meta_val(quant)

    node.replace_all_uses_with(quant)
    node.graph.erase_node(node)
    return True


def _fold_into_quant(ep: ExportedProgram, node: fx.Node, scalar: float) -> bool:
    """Fold scalar into quant scale when only out qparams are present.

    ``Q_out(float_x * c)`` becomes ``Q_{out/c}(float_x)`` by dividing the
    quantize scale by the scalar: ``round(x * c / s) = round(x / (s / c))``.
    """
    out_qparams = get_qparams_from_node(node, "out")
    assert out_qparams is not None

    if not out_qparams.is_per_tensor():
        return False

    new_scale = get_scale(ep, out_qparams) / scalar
    if new_scale <= 0.0 or not math.isfinite(new_scale):
        return False

    inp = get_arg(node, "inp", fx.Node)
    with node.graph.inserting_before(node):
        quant = node.graph.call_function(
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(
                inp,
                new_scale,
                get_zero_point(ep, out_qparams),
                out_qparams.quant_min,
                out_qparams.quant_max,
                out_qparams.dtype,
            ),
        )
        quant.meta = node.meta.copy()
        quant.meta["val"] = compute_meta_val(quant)

    node.replace_all_uses_with(quant)
    node.graph.erase_node(node)
    return True


def _fold_scalar_mul(ep: ExportedProgram, node: fx.Node) -> bool:
    other = get_arg(node, "other")
    if not isinstance(other, (int, float)):
        return False
    scalar = float(other)
    if scalar <= 0.0 or not math.isfinite(scalar):
        return False

    inp_qparams = get_qparams_from_node(node, "inp")
    out_qparams = get_qparams_from_node(node, "out")

    if inp_qparams is not None and out_qparams is not None:
        return _fold_into_dequant_quant(ep, node, scalar)

    if inp_qparams is None and out_qparams is not None:
        return _fold_into_quant(ep, node, scalar)

    return False


class FoldScalarMul(ExportedProgramPassBase):
    """Fold a positive scalar multiply into neighboring quantization ops.

    Handles two cases of ``fused_quant.mul.Scalar(x, c)``:

    1. Both input and output qparams present (fully quantized):
       ``Q_B(c * DQ_A(x))`` -> ``DQ_{A*c}(x) -> Q_B``, folding ``c`` into the
       dequant scale so later quant absorption can consume it.

    2. Only output qparams present (float input, quantized output):
       ``Q_B(float_x * c)`` -> ``Q_{B/c}(float_x)``, folding ``c`` into the
       quant scale by dividing it, eliminating the multiply entirely.

    Negative, zero, and non-finite scalars are left unchanged (they cannot be
    represented by a positive quantization scale). Per-channel qparams are
    also left unchanged.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        modified = False
        for node in graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.mul.Scalar
        ):
            modified |= _fold_scalar_mul(exported_program, node)

        if modified:
            exported_program = constant_prop_pass(exported_program)
        return ExportedProgramPassResult(exported_program, modified)
