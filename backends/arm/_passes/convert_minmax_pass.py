# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class ConvertMinMaxPass(ExportPass):
    """
    Converts min/max to amin/amax and unrolls multi-dimensional reduction and keep-dims arg to be
    TOSA compliant.

    The difference between max/min and amax/amin is (from pytorch docs):
        - amax/amin supports reducing on multiple dimensions,
        - amax/amin does not return indices,
        - amax/amin evenly distributes gradient between equal values, while max(dim)/min(dim)
          propagates gradient only to a single index in the source tensor.
    Since we do not care about gradients post training, convert min/max ops to amin/amax as long as
    the indices are not used.

    Original:
        amax([dim1, dim2], keepdim = False)
    After pass:
        amax(dim1, keepdim = True)
        amax(dim2, keepdim = True)
        squeeze(dim = [dim1, dim2])
    """

    def check_argmax(self, node):
        """
        Raises a RuntimeError if the argmax value returned by the min/max op is used in the graph.
        """
        if node.target in [torch.ops.aten.max.dim, torch.ops.aten.min.dim]:
            no_argmax = len(node.users) == 1
            no_argmax_users = (len(node.users) == 2) and (
                len(list(node.users)[1].users) == 0
            )
            if not (no_argmax or no_argmax_users):
                raise RuntimeError("Argmax is not supported by the arm_quantizer")

    def get_variables(self, node):
        """Returns variables specific for each op handled by the pass."""
        if node.target in [
            exir_ops.edge.aten.amax.default,
            exir_ops.edge.aten.amin.default,
        ]:
            replace_node = node
            op = node.target
            squeeze_op = exir_ops.edge.aten.squeeze_copy.dims
        elif node.target == exir_ops.edge.aten.max.dim:
            replace_node = list(node.users)[0]
            op = exir_ops.edge.aten.amax.default
            squeeze_op = exir_ops.edge.aten.squeeze_copy.dims
        elif node.target == exir_ops.edge.aten.min.dim:
            replace_node = list(node.users)[0]
            op = exir_ops.edge.aten.amin.default
            squeeze_op = exir_ops.edge.aten.squeeze_copy.dims
        elif node.target == torch.ops.aten.max.dim:
            replace_node = list(node.users)[0]
            op = torch.ops.aten.amax.default
            squeeze_op = torch.ops.aten.squeeze.dims
        elif node.target == torch.ops.aten.min.dim:
            replace_node = list(node.users)[0]
            op = torch.ops.aten.amin.default
            squeeze_op = torch.ops.aten.squeeze.dims
        else:
            raise RuntimeError(
                f"{node.name} is not an accepted target for ConvertMinMaxPass()"
            )

        return (replace_node, op, squeeze_op)

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target not in [
                exir_ops.edge.aten.amax.default,
                exir_ops.edge.aten.amin.default,
                exir_ops.edge.aten.max.dim,
                exir_ops.edge.aten.min.dim,
                torch.ops.aten.max.dim,
                torch.ops.aten.min.dim,
            ]:
                continue

            self.check_argmax(
                node
            )  # TODO: MLETORCH-718 : Quantization of indices in arm_quantizer
            replace_node, op, squeeze_op = self.get_variables(node)

            # Unwrap args
            if len(node.args) == 2:
                input_node, dims = node.args
                keepdims = False
            elif len(node.args) == 3:
                input_node, dims, keepdims = node.args
            else:
                raise RuntimeError(f"Unexpected arg size in {node.name}")

            try:
                iter(dims)
            except:
                dims = [dims]
            else:
                dims = list(dims)

            # Unroll multi-dimensional reduction and keep-dims arg
            with graph_module.graph.inserting_before(node):

                for dim in dims:
                    args = (input_node, dim, True)
                    input_node = graph_module.graph.create_node(
                        "call_function", op, args, node.kwargs
                    )

                if not keepdims:
                    input_node = graph_module.graph.create_node(
                        "call_function",
                        squeeze_op,
                        (input_node, dims),
                    )

            replace_node.replace_all_uses_with(input_node)
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
