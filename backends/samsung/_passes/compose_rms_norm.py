# Copyright (c) Qualcomm Innovation Center, Inc
# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class RecomposeRmsNorm(ExportPass):
    """
    Merge decomposed operators back to one super node.
    """

    def __init__(self):
        super().__init__()

    _ADD_TARGETS = (
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.add.Scalar,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
    )

    _MEAN_TARGETS = (
        exir_ops.edge.aten.mean.dim,
        torch.ops.aten.mean.dim,
    )

    def _get_eps_node(self, nodes):
        # eps: one of inputs of add node
        add_nodes = [
            n
            for n in nodes
            if isinstance(n, torch.fx.Node)
            and n.op == "call_function"
            and n.target in self._ADD_TARGETS
        ]
        if not add_nodes:
            raise RuntimeError("Failed to locate add node in RMSNorm partition")
        add_node = add_nodes[0]
        for a in add_node.args:
            if isinstance(a, float) or (
                isinstance(a, torch.fx.Node) and a.op != "call_function"
            ):
                return a
        raise RuntimeError("Failed to locate eps argument in RMSNorm add node")

    def _get_gamma_node(self, src_partition, input_node):
        # gamma: the partition input that is not the activation. A half-precision
        # norm ends in a dtype cast, so it is not an argument of the output node.
        others = [n for n in src_partition.input_nodes if n is not input_node]
        return others[0] if others else None

    def _get_normalized_shape(self, input_node, nodes):
        # Derive normalized_shape from mean.dim's axis in the decomposed graph.
        # The RMSNorm decomposition always contains a mean.dim op.
        for n in nodes:
            if n.op == "call_function" and n.target in self._MEAN_TARGETS:
                dims = n.args[1]
                return [input_node.meta["val"].shape[d] for d in dims]
        raise RuntimeError(
            f"Failed to find mean.dim node in RMSNorm partition: {nodes}"
        )

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        partitions = get_source_partitions(
            graph, [torch.nn.RMSNorm, torch.ops.aten.rms_norm.default]
        )

        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:
                input_len = len(src_partition.input_nodes)
                if input_len == 1:
                    input_node = src_partition.input_nodes[0]
                elif input_len == 2:
                    inp_0, inp_1 = src_partition.input_nodes
                    input_node = inp_0 if inp_0.op == "call_function" else inp_1
                else:
                    raise RuntimeError(
                        f"Found unsupported case of rms_node {src_partition}, "
                        f"which has {input_len} inputs"
                    )

                output_node = src_partition.output_nodes[0]
                eps_node = self._get_eps_node(src_partition.nodes)
                gamma_node = self._get_gamma_node(src_partition, input_node)
                normalized_shape = self._get_normalized_shape(
                    input_node, src_partition.nodes
                )

                with graph.inserting_before(output_node):
                    # When elementwise_affine=False, RMSNorm has no weight.
                    # Create a fake all-ones gamma_node so the Samsung builder
                    # always receives a valid gamma argument.
                    if gamma_node is None:
                        const_name = output_node.name + "_fake_gamma"
                        weight_tensor = torch.ones(
                            normalized_shape, dtype=output_node.meta["val"].dtype
                        )
                        graph_module.register_buffer(const_name, weight_tensor)
                        gamma_node = graph.create_node("get_attr", const_name)
                        gamma_node.meta["val"] = weight_tensor

                    # args schema
                    # (Tensor input, int[] normalized_shape, Tensor?
                    # weight=None, float? eps=None) -> Tensor
                    rms_node = graph.create_node(
                        "call_function",
                        exir_ops.edge.aten.rms_norm.default,
                        (
                            input_node,
                            normalized_shape,
                            gamma_node,
                            eps_node,
                        ),
                    )
                    users = output_node.users.copy()
                    for user in users:
                        user.replace_input_with(output_node, rms_node)
                    # copy metadata
                    rms_node.meta = output_node.meta

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
