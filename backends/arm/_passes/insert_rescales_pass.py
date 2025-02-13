# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from copy import copy
from typing import cast

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm.tosa_quant_utils import dq_op, q_op, QuantArgs
from executorch.exir.pass_base import ExportPass, PassResult
from torch import Tensor
from torch.fx import GraphModule, Node
from torch.library import custom_op, register_fake

logger = logging.getLogger(__name__)


@custom_op("tosa::_rescale", mutates_args=())  # type: ignore[misc]
def rescale(
    x: Tensor, dtype: torch.dtype, scale: float, in_zp: int, out_zp: int
) -> Tensor:
    logger.warning(
        "Ran default implementation of tosa::_rescale."
        "This op is meant to always be inserted inside a partition and a correct default implementation is not implemented."
    )
    # Clone is needed to not return reference when rescaling to same dtype.
    # This is a neccessary requirement for non-mutating custom ops.
    return x.to(dtype=dtype).clone()


@register_fake("tosa::_rescale")  # type: ignore[misc]
def rescale_fake(
    x: Tensor, dtype: torch.dtype, scale: float, in_zp: int, out_zp: int
) -> Tensor:
    """Casts the input tensor to dtype `dtype` to produce the correct tensor meta for a _rescale op.
    Additionally validates TOSA constraints of a RESCALE op.
    """
    if not (dtype == torch.int32 or dtype == torch.int8):
        raise NotImplementedError(
            "tosa::rescale currently only supports int32 and int8."
        )
    if dtype == torch.int32 and out_zp != 0:
        raise ValueError(
            "TOSA requires output_zp to be zero when the output dtype is int32."
        )
    if x.dtype == torch.int32 and in_zp != 0:
        raise ValueError(
            "TOSA requires input_zp to be zero when the input dtype is int32."
        )
    if x.dtype == torch.int8 and not -128 <= in_zp <= 127:
        raise ValueError(f"{in_zp=} outside valid range (-128,127) for int8.")
    if dtype == torch.int8 and not -128 <= out_zp <= 127:
        raise ValueError(f"{out_zp=} outside valid range (-128,127) for int8.")

    return x.to(dtype=dtype).clone()


class InsertRescalePass(ExportPass):
    """Finds patterns of dq -> q, and replaces them
    with passthrough_to_tosa::rescales.

    Does not garantuee that the dtypes and zero points are valid
    in TOSA, that is the job of the quantization annotator that
    produced the dq and q nodes. The TOSA constraints are validated
    in the fake implementation of passthrough_to_tosa:rescale.
    """

    def fold_dq_q_to_rescale(self, node: Node, user: Node, graph_module: GraphModule):
        dq_args = QuantArgs.from_operator(node.target, node.args)
        q_args = QuantArgs.from_operator(user.target, user.args)
        new_scale = dq_args.scale / q_args.scale

        with graph_module.graph.inserting_before(node):
            rescale_node = create_node(
                graph_module.graph,
                torch.ops.tosa._rescale.default,
                (
                    node.all_input_nodes[0],
                    q_args.dtype,
                    new_scale,
                    dq_args.zp,
                    q_args.zp,
                ),
            )
            rescale_node.meta = copy(user.meta)
            user.replace_all_uses_with(rescale_node)
            graph_module.graph.erase_node(user)

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            node = cast(Node, node)

            if node.target is not dq_op:
                continue
            # Copy users since we remove them while iterating, modyfing the node.users list.
            for user in copy(node.users):
                if user.target is q_op:
                    self.fold_dq_q_to_rescale(node, user, graph_module)
                    modified = True
            if len(node.users) == 0:
                graph_module.graph.erase_node(node)

        graph_module = super().call(graph_module).graph_module
        graph_module.recompile()
        return PassResult(graph_module, modified)
