# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    adjust_pooling_pad_if_needed,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

edge_max_pool2d_ops = (exir_ops.edge.aten.max_pool2d.default,)

_NCHW_TO_NHWC = [0, 2, 3, 1]
_NHWC_TO_NCHW = [0, 3, 1, 2]


def _to_2tuple(value):
    if isinstance(value, int):
        return (value, value)
    if len(value) == 1:
        return (value[0], value[0])
    return tuple(value)


class RewriteMaxPool2dPass(ArmPass):
    """Rewrite max_pool2d ops to TOSA MAX_POOL2D with NHWC layout."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    @staticmethod
    def _insert_permute(graph_module, anchor_node, input_node, perm, before=True):
        ctx = (
            graph_module.graph.inserting_before(anchor_node)
            if before
            else graph_module.graph.inserting_after(anchor_node)
        )
        with ctx:
            return create_node(
                graph=graph_module.graph,
                op_target=exir_ops.edge.aten.permute_copy.default,
                args=(input_node, perm),
                from_node=input_node,
            )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target not in edge_max_pool2d_ops:
                continue

            x = node.args[0]
            kernel = _to_2tuple(node.args[1])

            if len(node.args) > 2 and node.args[2] is not None and len(node.args[2]) > 0:
                stride = _to_2tuple(node.args[2])
            else:
                stride = kernel

            padding = _to_2tuple(node.args[3]) if len(node.args) > 3 else (0, 0)
            dilation = _to_2tuple(node.args[4]) if len(node.args) > 4 else (1, 1)
            ceil_mode = node.args[5] if len(node.args) > 5 else False

            if dilation != (1, 1):
                continue

            modified = True

            input_fake = get_first_fake_tensor(x)

            # TOSA MAX_POOL2D pad order is [top, bottom, left, right]
            pad = [padding[0], padding[0], padding[1], padding[1]]
            pad[1] = adjust_pooling_pad_if_needed(
                input_fake.shape[2], kernel[0], stride[0], pad[1], ceil_mode
            )
            pad[3] = adjust_pooling_pad_if_needed(
                input_fake.shape[3], kernel[1], stride[1], pad[3], ceil_mode
            )

            # Insert NCHW → NHWC permute on input
            x_permuted = self._insert_permute(
                graph_module, node, x, _NCHW_TO_NHWC, before=True
            )

            tosa_args = (x_permuted, list(kernel), list(stride), pad)

            # Create TOSA MAX_POOL2D node
            with graph_module.graph.inserting_after(node):
                tosa_op = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.backend.tosa.MAX_POOL2D.default,
                    args=tosa_args,
                    from_node=node,
                    inherit_qparams=True,
                )

            # Compute correct NHWC FakeTensor
            input_fake_nhwc = input_fake.permute(_NCHW_TO_NHWC)
            tosa_node_fake = exir_ops.backend.tosa.MAX_POOL2D.default(
                input_fake_nhwc, list(kernel), list(stride), pad
            )
            tosa_op.meta["val"] = tosa_node_fake

            # Insert NHWC → NCHW permute on output
            output_permute = self._insert_permute(
                graph_module, tosa_op, tosa_op, _NHWC_TO_NCHW, before=False
            )

            node.replace_all_uses_with(output_permute)
            graph_module.graph.erase_node(node)

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
