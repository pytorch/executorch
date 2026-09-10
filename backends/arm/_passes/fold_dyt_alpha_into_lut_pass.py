# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict
"""Fold quantized tanh(alpha * x) into one exact INT8 TOSA table."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast, Optional, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node, get_param_tensor
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.operators.op_tosa_rescale import (
    _compute_multiplier_and_shift,
)
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind
from torch.fx import GraphModule, Node


@dataclass(frozen=True)
class _RescaleParams:
    scale: float
    input_zp: int
    output_zp: int
    output_dtype: torch.dtype

    @classmethod
    def from_node(cls, node: Node) -> Optional["_RescaleParams"]:
        if node.target != exir_ops.backend.tosa.RESCALE.default:
            return None
        scales = cast(list[float], node.args[2])
        if len(scales) != 1:
            return None
        if node.kwargs.get("input_unsigned", False) or node.kwargs.get(
            "output_unsigned", False
        ):
            return None
        return cls(
            scale=float(scales[0]),
            input_zp=cast(int, node.args[3]),
            output_zp=cast(int, node.args[4]),
            output_dtype=cast(torch.dtype, node.args[1]),
        )


def _apply_tosa_rescale(
    values: torch.Tensor,
    params: _RescaleParams,
) -> torch.Tensor:
    """Apply TOSA-1.0 RESCALE SINGLE_ROUND exactly."""
    multipliers, shifts = _compute_multiplier_and_shift([params.scale])
    multiplier = multipliers[0]
    shift = shifts[0]
    # _compute_multiplier_and_shift asserts shift is in [2, 62], so shift - 1 is
    # never negative here. Do not "guard" this by zeroing rounding on small
    # shifts: that would silently change SINGLE_ROUND's round-half behaviour and
    # break the bit-exactness this whole pass depends on.
    rounding = 1 << (shift - 1)
    centered = values.to(torch.int64) - params.input_zp
    scaled = (centered * multiplier + rounding) >> shift
    shifted = scaled + params.output_zp
    dtype_range = torch.iinfo(params.output_dtype)
    return shifted.clamp(dtype_range.min, dtype_range.max).to(params.output_dtype)


def _generate_dyt_lut(
    *,
    activation_qargs: QuantArgs,
    alpha_code: torch.Tensor,
    activation_rescale: _RescaleParams,
    alpha_rescale: _RescaleParams,
    mul_output_rescale: _RescaleParams,
    tanh_input_qargs: QuantArgs,
    tanh_output_qargs: QuantArgs,
) -> torch.Tensor:
    """Compose the integer Mul/RESCALE path with the existing tanh mapping."""
    if alpha_code.numel() != 1:
        raise ValueError(f"Expected scalar DyT alpha, got shape {alpha_code.shape}")

    domain = InsertTableOpsPass._get_8bit_table_domain().clamp(
        activation_qargs.qmin, activation_qargs.qmax
    )
    activation_i32 = _apply_tosa_rescale(domain, activation_rescale)
    alpha_i32 = _apply_tosa_rescale(alpha_code, alpha_rescale)
    product = (activation_i32.to(torch.int64) * alpha_i32.to(torch.int64)).to(
        torch.int32
    )
    mul_codes = _apply_tosa_rescale(product, mul_output_rescale)
    mul_codes = mul_codes.clamp(tanh_input_qargs.qmin, tanh_input_qargs.qmax)
    tanh_values = torch.tanh(tanh_input_qargs.dequantize_value(mul_codes))
    return tanh_output_qargs.quantize_value(tanh_values).to(torch.int8).reshape(-1)


@dataclass(frozen=True)
class _DyTMatch:
    tanh: Node
    activation: Node
    activation_qargs: QuantArgs
    activation_rescale: _RescaleParams
    alpha_code: torch.Tensor
    alpha_rescale: _RescaleParams
    mul_output_rescale: _RescaleParams
    tanh_input_qargs: QuantArgs
    tanh_output_qargs: QuantArgs


@dataclass(frozen=True)
class _MulOperand:
    index: int
    source: Node
    rescale: _RescaleParams
    scalar_constant: Optional[torch.Tensor]
    had_view: bool


class FoldDyTAlphaIntoLUTPass(ArmPass):
    """Replace an INT8 DyT alpha Mul, requantize, and tanh with one TABLE."""

    _passes_required_after: Set[Type[ExportPass]] = {InsertTableOpsPass}

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program

    @staticmethod
    def _unwrap_view(node: Node) -> Node:
        while (
            node.target == exir_ops.edge.aten.view_copy.default
            and len(node.args) > 0
            and isinstance(node.args[0], Node)
        ):
            node = node.args[0]
        return node

    def _get_scalar_constant(self, rescale_node: Node) -> Optional[torch.Tensor]:
        source = rescale_node.args[0]
        if not isinstance(source, Node):
            return None
        try:
            value = get_param_tensor(self.exported_program, source)
        except RuntimeError:
            return None
        if value is None or value.numel() != 1:
            return None
        return value

    @staticmethod
    def _single_qargs(node: Node, key: str) -> Optional[QuantArgs]:
        qparams = cast(dict[int, QuantArgs], node.meta.get(key, {}))
        if len(qparams) != 1:
            return None
        qargs = next(iter(qparams.values()))
        if qargs.per_channel:
            return None
        return qargs

    @staticmethod
    def _source_qargs(
        source: Node,
        rescale: _RescaleParams,
        int32_qargs: QuantArgs,
    ) -> Optional[QuantArgs]:
        candidates = cast(
            dict[int, QuantArgs],
            source.meta.get("output_qparams", {}),
        )
        for qargs in candidates.values():
            if qargs.per_channel or qargs.dtype != torch.int8:
                continue
            if qargs.get_zp_per_tensor() != rescale.input_zp:
                continue
            expected_scale = (
                qargs.get_scale_per_tensor() / int32_qargs.get_scale_per_tensor()
            )
            if math.isclose(expected_scale, rescale.scale, rel_tol=1e-6, abs_tol=0.0):
                return qargs
        return None

    def _match_mul_operands(
        self, mul: Node
    ) -> Optional[tuple[_MulOperand, _MulOperand]]:
        if len(mul.args) < 2:
            return None

        operands = []
        for index, arg in enumerate(mul.args[:2]):
            if not isinstance(arg, Node):
                return None
            rescale_node = self._unwrap_view(arg)
            rescale = _RescaleParams.from_node(rescale_node)
            if rescale is None or rescale.output_dtype != torch.int32:
                return None
            source = rescale_node.args[0]
            if not isinstance(source, Node):
                return None
            operands.append(
                _MulOperand(
                    index=index,
                    source=source,
                    rescale=rescale,
                    scalar_constant=self._get_scalar_constant(rescale_node),
                    had_view=rescale_node is not arg,
                )
            )
        return operands[0], operands[1]

    @staticmethod
    def _match_output_rescale_and_mul(
        tanh: Node,
    ) -> Optional[tuple[Node, _RescaleParams]]:
        if tanh.target != exir_ops.edge.aten.tanh.default:
            return None
        if len(tanh.args) != 1 or not isinstance(tanh.args[0], Node):
            return None

        output_rescale_node = tanh.args[0]
        output_rescale = _RescaleParams.from_node(output_rescale_node)
        if (
            output_rescale is None
            or output_rescale.output_dtype != torch.int8
            or len(output_rescale_node.users) != 1
            or not isinstance(output_rescale_node.args[0], Node)
        ):
            return None

        mul = output_rescale_node.args[0]
        if mul.target != exir_ops.edge.aten.mul.Tensor or len(mul.users) != 1:
            return None
        return mul, output_rescale

    def _match_mul_and_operands(
        self, mul: Node
    ) -> Optional[tuple[_MulOperand, _MulOperand, QuantArgs]]:
        mul_qparams = cast(dict[int, QuantArgs], mul.meta.get("input_qparams", {}))
        if len(mul_qparams) != 2:
            return None

        operands = self._match_mul_operands(mul)
        if operands is None:
            return None
        scalar_operands = [
            operand for operand in operands if operand.scalar_constant is not None
        ]
        if len(scalar_operands) != 1:
            return None
        alpha_operand = scalar_operands[0]
        activation_operand = next(
            operand for operand in operands if operand is not alpha_operand
        )
        # A view of the proven-scalar alpha only changes its broadcast shape.
        # An activation-side view can change layout/indexing and is unsupported.
        if activation_operand.had_view:
            return None

        activation_int32_qargs = mul_qparams.get(activation_operand.index)
        if (
            activation_int32_qargs is None
            or activation_int32_qargs.dtype != torch.int32
        ):
            return None
        return alpha_operand, activation_operand, activation_int32_qargs

    def _match_qargs(
        self,
        tanh: Node,
        activation_operand: _MulOperand,
        activation_int32_qargs: QuantArgs,
        output_rescale: _RescaleParams,
    ) -> Optional[tuple[QuantArgs, QuantArgs, QuantArgs]]:
        activation_qargs = self._source_qargs(
            activation_operand.source,
            activation_operand.rescale,
            activation_int32_qargs,
        )
        tanh_input_qargs = self._single_qargs(tanh, "input_qparams")
        tanh_output_qargs = self._single_qargs(tanh, "output_qparams")
        if (
            activation_qargs is None
            or tanh_input_qargs is None
            or tanh_output_qargs is None
            or tanh_input_qargs.dtype != torch.int8
            or tanh_output_qargs.dtype != torch.int8
            or output_rescale.output_zp != tanh_input_qargs.get_zp_per_tensor()
        ):
            return None
        return activation_qargs, tanh_input_qargs, tanh_output_qargs

    def _match(self, tanh: Node) -> Optional[_DyTMatch]:
        output_match = self._match_output_rescale_and_mul(tanh)
        if output_match is None:
            return None
        mul, output_rescale = output_match

        mul_match = self._match_mul_and_operands(mul)
        if mul_match is None:
            return None
        alpha_operand, activation_operand, activation_int32_qargs = mul_match

        qargs_match = self._match_qargs(
            tanh,
            activation_operand,
            activation_int32_qargs,
            output_rescale,
        )
        if qargs_match is None:
            return None
        activation_qargs, tanh_input_qargs, tanh_output_qargs = qargs_match

        alpha_code = alpha_operand.scalar_constant
        if alpha_code is None:
            return None

        return _DyTMatch(
            tanh=tanh,
            activation=activation_operand.source,
            activation_qargs=activation_qargs,
            activation_rescale=activation_operand.rescale,
            alpha_code=alpha_code,
            alpha_rescale=alpha_operand.rescale,
            mul_output_rescale=output_rescale,
            tanh_input_qargs=tanh_input_qargs,
            tanh_output_qargs=tanh_output_qargs,
        )

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            match = self._match(node)
            if match is None:
                continue

            table = _generate_dyt_lut(
                activation_qargs=match.activation_qargs,
                alpha_code=match.alpha_code,
                activation_rescale=match.activation_rescale,
                alpha_rescale=match.alpha_rescale,
                mul_output_rescale=match.mul_output_rescale,
                tanh_input_qargs=match.tanh_input_qargs,
                tanh_output_qargs=match.tanh_output_qargs,
            )
            insert_pos = next(iter(graph_module.graph.nodes))
            with graph_module.graph.inserting_before(insert_pos):
                table_constant = create_constant_placeholder(
                    exp_program=self.exported_program,
                    graph=graph_module.graph,
                    kind=InputKind.BUFFER,
                    name=f"b_{match.tanh.name}_dyt_table_constant",
                    data=table,
                    persistent_buffer=True,
                )
            with graph_module.graph.inserting_before(match.tanh):
                table_node = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.backend.tosa.TABLE.default,
                    args=(match.activation, table_constant),
                    from_node=match.tanh,
                )
            table_node.meta["input_qparams"] = {0: match.activation_qargs}
            table_node.meta["output_qparams"] = {0: match.tanh_output_qargs}
            match.tanh.replace_all_uses_with(table_node)
            graph_module.graph.erase_node(match.tanh)
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)
