# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch

from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm._passes.tosa_lowering_utils import create_const
from executorch.backends.transforms.aten_to_dialect_pass import (
    AtenToDialectPass,
    DialectNodeSpec,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node


def rewrite_rfft2(node: Node, pass_: AtenToDialectPass) -> DialectNodeSpec | None:
    fft_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("s")
    fft_dims = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", [-2, -1])
    norm = node.args[3] if len(node.args) > 3 else node.kwargs.get("norm")
    if fft_size is not None or fft_dims not in ([-2, -1], (-2, -1)) or norm is not None:
        return None

    return DialectNodeSpec(
        exir_ops.backend.tosa.RFFT2D.default,
        (node.args[0],),
        {},
    )


def rewrite_mul(node: Node, pass_: AtenToDialectPass) -> DialectNodeSpec:
    with node.graph.inserting_before(node):
        shift = create_const(node.graph, 0, torch.int8, node)
    return DialectNodeSpec(exir_ops.backend.tosa.MUL.default, (*node.args, shift), {})


def rewrite_negate(node: Node, pass_: AtenToDialectPass) -> DialectNodeSpec:
    input_node = cast(Node, node.args[0])
    dtype = input_node.meta["val"].dtype
    input_zp, output_zp = 0, 0
    if dtype == torch.int8:
        input_zp = get_input_qparams(node)[0].get_zp_per_tensor()
        output_zp = get_output_qparams(node)[0].get_zp_per_tensor()
    with node.graph.inserting_before(node):
        input_zero_point = create_const(node.graph, input_zp, dtype, input_node)
        output_zero_point = create_const(node.graph, output_zp, dtype, input_node)
    return DialectNodeSpec(
        exir_ops.backend.tosa.NEGATE.default,
        (input_node, input_zero_point, output_zero_point),
        {},
    )


def rewrite_binary_operator(
    node: Node, pass_: AtenToDialectPass
) -> DialectNodeSpec | None:
    match node.target:
        case exir_ops.edge.aten.add.Tensor:
            target = exir_ops.backend.tosa.ADD.default
        case exir_ops.edge.aten.bitwise_and.Tensor:
            target = exir_ops.backend.tosa.BITWISE_AND.default
        case exir_ops.edge.aten.bitwise_left_shift.Tensor:
            target = exir_ops.backend.tosa.LOGICAL_LEFT_SHIFT.default
        case exir_ops.edge.aten.bitwise_or.Tensor:
            target = exir_ops.backend.tosa.BITWISE_OR.default
        case exir_ops.edge.aten.bitwise_right_shift.Tensor:
            target = exir_ops.backend.tosa.ARITHMETIC_RIGHT_SHIFT.default
        case exir_ops.edge.aten.bitwise_xor.Tensor:
            target = exir_ops.backend.tosa.BITWISE_XOR.default
        case exir_ops.edge.aten.logical_and.default:
            target = exir_ops.backend.tosa.LOGICAL_AND.default
        case exir_ops.edge.aten.logical_or.default:
            target = exir_ops.backend.tosa.LOGICAL_OR.default
        case exir_ops.edge.aten.logical_xor.default:
            target = exir_ops.backend.tosa.LOGICAL_XOR.default
        case exir_ops.edge.aten.maximum.default:
            target = exir_ops.backend.tosa.MAXIMUM.default
        case exir_ops.edge.aten.minimum.default:
            target = exir_ops.backend.tosa.MINIMUM.default
        case exir_ops.edge.aten.pow.Tensor_Tensor:
            target = exir_ops.backend.tosa.POW.default
        case exir_ops.edge.aten.sub.Tensor:
            target = exir_ops.backend.tosa.SUB.default
        case _:
            return None

    return DialectNodeSpec(target, node.args, dict(node.kwargs))


def rewrite_unary_operator(
    node: Node, pass_: AtenToDialectPass
) -> DialectNodeSpec | None:
    match node.target:
        case exir_ops.edge.aten.abs.default:
            target = exir_ops.backend.tosa.ABS.default
        case exir_ops.edge.aten.bitwise_not.default:
            target = exir_ops.backend.tosa.BITWISE_NOT.default
        case exir_ops.edge.aten.ceil.default:
            target = exir_ops.backend.tosa.CEIL.default
        case exir_ops.edge.aten.cos.default:
            target = exir_ops.backend.tosa.COS.default
        case exir_ops.edge.aten.exp.default:
            target = exir_ops.backend.tosa.EXP.default
        case exir_ops.edge.aten.floor.default:
            target = exir_ops.backend.tosa.FLOOR.default
        case exir_ops.edge.aten.log.default:
            target = exir_ops.backend.tosa.LOG.default
        case exir_ops.edge.aten.logical_not.default:
            target = exir_ops.backend.tosa.LOGICAL_NOT.default
        case exir_ops.edge.aten.reciprocal.default:
            target = exir_ops.backend.tosa.RECIPROCAL.default
        case exir_ops.edge.aten.rsqrt.default:
            target = exir_ops.backend.tosa.RSQRT.default
        case exir_ops.edge.aten.sin.default:
            target = exir_ops.backend.tosa.SIN.default
        case _:
            return None

    return DialectNodeSpec(target, node.args, dict(node.kwargs))
