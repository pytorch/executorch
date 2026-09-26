# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Optional

import torch
from executorch.backends.fused_quant.graph_utils import (
    add_constant,
    get_constant,
    set_constant,
)
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch import fx
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

logger: logging.Logger = logging.getLogger(__name__)

_PASSTHROUGH_TARGETS: set[EdgeOpOverload] = {
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.view_copy.default,
}

_CONV_TARGETS: set[EdgeOpOverload] = {
    exir_ops.edge.fused_quant.convolution.default,
    exir_ops.edge.fused_quant.convolution_channels_last.default,
}

_AFFINE_TARGETS: set[EdgeOpOverload] = {
    exir_ops.edge.fused_quant.linear.default,
    *_CONV_TARGETS,
}


def _find_affine_through_passthrough(
    node: fx.Node,
) -> Optional[tuple[fx.Node, list[fx.Node]]]:
    chain: list[fx.Node] = []
    current = node
    while current.target in _PASSTHROUGH_TARGETS:
        if len(current.users) != 1:
            return None
        chain.append(current)
        prev = current.args[0]
        if not isinstance(prev, fx.Node):
            return None
        current = prev

    if current.target not in _AFFINE_TARGETS:
        return None
    if current.target in _CONV_TARGETS and get_arg(current, "transposed", bool):
        return None
    if len(current.users) != 1:
        return None

    return current, chain


def _has_per_tensor_output_qparams(ep: ExportedProgram, add_node: fx.Node) -> bool:
    for field in ("out_scale", "out_zero_point"):
        node = get_arg(add_node, field, Optional[fx.Node])
        if node is None:
            continue
        val = get_constant(ep, node)
        if val is None or val.numel() > 1:
            return False
    return True


def _get_bias_increment(add_node: fx.Node) -> float:
    other = get_arg(add_node, "other", float | int)
    alpha = get_arg(add_node, "alpha", float | int)
    return float(other) * alpha


def _get_new_bias(
    ep: ExportedProgram, affine_node: fx.Node, increment: float
) -> Optional[torch.Tensor]:
    bias_node = get_arg(affine_node, "bias", Optional[fx.Node])
    if bias_node is None:
        weight_node = get_arg(affine_node, "weight", fx.Node)
        out_channels = weight_node.meta["val"].shape[0]
        return torch.full((out_channels,), increment, dtype=torch.float32)

    if len(bias_node.users) != 1:
        return None
    if get_arg(affine_node, "bias_scale", Optional[fx.Node]) is not None:
        return None

    bias = get_constant(ep, bias_node)
    if bias is None:
        return None
    return bias + bias.new_tensor(increment)


def _output_qparams_match_input(
    ep: ExportedProgram, affine_node: fx.Node, add_node: fx.Node
) -> bool:
    for affine_field, add_field in (
        ("out_scale", "inp_scale"),
        ("out_zero_point", "inp_zero_point"),
    ):
        affine_qp = get_arg(affine_node, affine_field, Optional[fx.Node])
        add_qp = get_arg(add_node, add_field, Optional[fx.Node])
        if affine_qp is None or add_qp is None:
            assert affine_qp is None and add_qp is None
            continue
        if affine_qp is add_qp:
            continue
        affine_val = get_constant(ep, affine_qp)
        add_val = get_constant(ep, add_qp)
        if affine_val is None or add_val is None:
            return False
        if not torch.equal(affine_val, add_val):
            return False
    return True


class FuseAddIntoLinear(ExportedProgramPassBase):
    """Folds fused_quant.add.Scalar into linear or convolution bias."""

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        modified = False

        for add_node in graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.add.Scalar
        ):
            inp_node = get_arg(add_node, "inp", fx.Node)
            result = _find_affine_through_passthrough(inp_node)
            if result is None:
                continue
            affine_node, passthrough_chain = result

            if passthrough_chain and not _has_per_tensor_output_qparams(
                exported_program, add_node
            ):
                continue
            if not _output_qparams_match_input(exported_program, affine_node, add_node):
                continue
            if get_arg(add_node, "out_dtype") != get_arg(affine_node, "out_dtype"):
                continue

            new_bias = _get_new_bias(
                exported_program, affine_node, _get_bias_increment(add_node)
            )
            if new_bias is None:
                continue

            bias_node = get_arg(affine_node, "bias", Optional[fx.Node])
            if bias_node is None:
                bias_node = add_constant(
                    exported_program,
                    f"{affine_node.name}_bias",
                    new_bias,
                    affine_node,
                    InputKind.BUFFER,
                )
                set_arg(affine_node, "bias", bias_node)
            else:
                set_constant(exported_program, bias_node, new_bias)

            for field in (
                "out_zero_point",
                "out_scale",
                "out_quant_min",
                "out_quant_max",
            ):
                set_arg(affine_node, field, get_arg(add_node, field))

            add_node.replace_all_uses_with(inp_node)
            graph.erase_node(add_node)
            modified = True
            logger.info("Folded scalar add into affine bias: %s", affine_node.name)

        if modified:
            exported_program = constant_prop_pass(exported_program)

        return ExportedProgramPassResult(exported_program, modified)
