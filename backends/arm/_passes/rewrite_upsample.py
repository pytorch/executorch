# Copyright 2025 Arm Limited and/or its affiliates.
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
from executorch.backends.arm.tosa.utils import get_resize_parameters
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RewriteUpsamplePass(ArmPass):
    """Rewrite upsample2d nodes to TOSA.RESIZE nodes."""

    targeted_ops = (
        exir_ops.edge.aten.upsample_nearest2d.vec,
        exir_ops.edge.aten.upsample_bilinear2d.vec,
    )

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module):
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target not in self.targeted_ops:
                continue
            modified = True

            if node.target == exir_ops.edge.aten.upsample_bilinear2d.vec:
                x, output_size, align_corners, scale_factors = node.args
                resize_mode = "bilinear"
            else:
                x, output_size, scale_factors = node.args
                align_corners = False
                resize_mode = "nearest"

            with graph_module.graph.inserting_before(node):
                tosa_resize_node = create_node(
                    graph_module.graph,
                    op_target=exir_ops.backend.tosa.RESIZE.default,
                    args=(x, output_size, align_corners, scale_factors),
                    kwargs={"resize_mode": resize_mode},
                    from_node=node,
                )
                node.replace_all_uses_with(tosa_resize_node)
                graph_module.graph.erase_node(node)
            input_dtype = get_first_fake_tensor(x).dtype
            if input_dtype == torch.int8 and resize_mode == "bilinear":
                input_size = get_first_fake_tensor(x).shape
                input_size_xy = input_size[2:]
                output_size = get_first_fake_tensor(node).shape
                output_size_xy = output_size[2:]
                scale_n_yx, _, _, _ = get_resize_parameters(
                    input_size_xy=input_size_xy,
                    output_size_xy=output_size_xy,
                    resize_mode=1,
                    align_corners=align_corners,
                )
                output_dtype = get_first_fake_tensor(node).dtype
                output_scale = float(1 / (scale_n_yx[0] * scale_n_yx[1]))
                with graph_module.graph.inserting_after(tosa_resize_node):
                    rescale_node = create_node(
                        graph_module.graph,
                        exir_ops.backend.tosa.RESCALE.default,
                    )
                    tosa_resize_node.replace_all_uses_with(rescale_node)
                    rescale_node.args = (
                        tosa_resize_node,
                        output_dtype,
                        output_scale,
                        0,  # zero point
                        0,  # zero point
                    )

        if modified:
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
