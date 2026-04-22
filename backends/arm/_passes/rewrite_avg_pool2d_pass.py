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

from .fuse_constant_ops_pass import ComputeConstantOpsAOTPass

_NCHW_TO_NHWC = [0, 2, 3, 1]
_NHWC_TO_NCHW = [0, 3, 1, 2]


class RewriteAvgPool2dPass(ArmPass):
    """Rewrite aten.avg_pool2d calls to TOSA AVG_POOL2D op with NHWC layout."""

    targeted_ops = {exir_ops.edge.aten.avg_pool2d.default}
    _passes_required_after: Set[Type[ExportPass]] = {
        ComputeConstantOpsAOTPass,
    }

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
            if node.op != "call_function" or node.target not in self.targeted_ops:
                continue

            modified = True
            x = node.args[0]

            pad_h, pad_w = node.args[3]
            pad = [pad_h, pad_w, pad_h, pad_w]

            input_fake = get_first_fake_tensor(x)
            _, _, h, w = input_fake.shape
            kernel_h, kernel_w = node.args[1]
            stride_h, stride_w = node.args[2]

            ceil_mode = node.args[4] if len(node.args) > 4 else False

            pad[1] = adjust_pooling_pad_if_needed(h, kernel_h, stride_h, pad[1], ceil_mode)
            pad[3] = adjust_pooling_pad_if_needed(w, kernel_w, stride_w, pad[3], ceil_mode)

            # Determine zero-points and accumulator type
            in_qparams = node.meta.get("input_qparams", {})
            in_zp_val = in_qparams[0].get_zp_per_tensor() if 0 in in_qparams else 0

            out_qparams = node.meta.get("output_qparams", {})
            out_zp_val = out_qparams[0].get_zp_per_tensor() if 0 in out_qparams else 0

            if input_fake.dtype in (torch.int8, torch.int16):
                acc_type = torch.int32
            else:
                acc_type = torch.float32

            # Insert NCHW → NHWC permute on input
            x_permuted = self._insert_permute(
                graph_module, node, x, _NCHW_TO_NHWC, before=True
            )

            # Materialize zp scalars as graph constants using aten.full with
            # explicit dtype matching the input tensor.  This ensures the
            # pre-computed buffer placeholders carry the correct type for
            # INT-only TOSA profiles (avoids defaulting to float32).
            zp_kwargs = {"dtype": input_fake.dtype, "device": input_fake.device}
            with graph_module.graph.inserting_before(node):
                input_zp_node = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.full.default,
                    args=((1,), in_zp_val),
                    kwargs=zp_kwargs,
                    from_node=node,
                )
                output_zp_node = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.full.default,
                    args=((1,), out_zp_val),
                    kwargs=zp_kwargs,
                    from_node=node,
                )

            kernel = list(node.args[1])
            stride = list(node.args[2])

            tosa_args = (x_permuted, input_zp_node, output_zp_node, kernel, stride, pad, acc_type)

            # Create TOSA AVG_POOL2D node
            with graph_module.graph.inserting_after(node):
                tosa_op = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.backend.tosa.AVG_POOL2D.default,
                    args=tosa_args,
                    from_node=node,
                    inherit_qparams=True,
                )

            # Compute correct NHWC FakeTensor
            input_fake_nhwc = input_fake.permute(_NCHW_TO_NHWC)
            input_zp_fake = torch.tensor(in_zp_val, dtype=input_fake.dtype)
            output_zp_fake = torch.tensor(out_zp_val, dtype=input_fake.dtype)
            tosa_node_fake = exir_ops.backend.tosa.AVG_POOL2D.default(
                input_fake_nhwc, input_zp_fake, output_zp_fake, kernel, stride, pad, acc_type
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
