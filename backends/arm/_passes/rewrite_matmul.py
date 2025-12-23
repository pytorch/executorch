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
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RewriteMatmulPass(ArmPass):
    """Rewrites aten.bmm to tosa.MATMUL and inserts a tosa.RESCALE op if needed."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def _insert_output_rescale(self, graph_module, node, tosa_matmul_node, dtype):
        input_qparams = get_input_qparams(node)
        output_qparams = get_output_qparams(node)[0]
        scale = (
            input_qparams[0].get_scale_per_tensor()
            * input_qparams[1].get_scale_per_tensor()
        ) / output_qparams.get_scale_per_tensor()

        with graph_module.graph.inserting_after(tosa_matmul_node):
            # If the input is int8, we need to cast the output to int32
            rescale_node = create_node(
                graph_module.graph,
                op_target=exir_ops.backend.tosa.RESCALE.default,
                from_node=tosa_matmul_node,
            )
            tosa_matmul_node.replace_all_uses_with(rescale_node)
            rescale_node.args = (
                tosa_matmul_node,
                dtype,
                [scale],
                0,
                output_qparams.get_zp_per_tensor(),
            )

    def call(self, graph_module):
        modified = False
        for node in graph_module.graph.nodes:
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.bmm.default
            ):
                continue
            modified = True

            x1, x2 = node.args
            tosa_matmul_target = exir_ops.backend.tosa.MATMUL.default
            with graph_module.graph.inserting_before(node):
                tosa_matmul_node = create_node(
                    graph_module.graph,
                    op_target=tosa_matmul_target,
                    args=(x1, x2),
                    kwargs={},
                    from_node=node,
                    inherit_qparams=True,
                )
                node.replace_all_uses_with(tosa_matmul_node)
                graph_module.graph.erase_node(node)

            x1_fake_tensor = get_first_fake_tensor(x1)
            x2_fake_tensor = get_first_fake_tensor(x2)
            output_fake_tensor = tosa_matmul_target(x1_fake_tensor, x2_fake_tensor)
            node_output_fake_tensor = get_first_fake_tensor(node)
            if (
                output_fake_tensor.dtype == torch.int32
                and node_output_fake_tensor.dtype in (torch.int8, torch.int16)
            ):
                self._insert_output_rescale(
                    graph_module,
                    node,
                    tosa_matmul_node,
                    dtype=node_output_fake_tensor.dtype,
                )
                if x1_fake_tensor.dtype == torch.int16:
                    tosa_matmul_node.meta[TosaSpecialDtype.meta_key()] = (
                        TosaSpecialDtype.INT48
                    )

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
