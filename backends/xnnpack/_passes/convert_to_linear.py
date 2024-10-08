# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List

import torch

from executorch.backends.transforms import get_shape
from executorch.backends.transforms.addmm_mm_to_linear import (
    apply_addmm_mm_to_linear_transform,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.utils.source_matcher_utils import (
    get_source_partitions,
    SourcePartition,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ConvertToLinearPass(ExportPass):
    linear_modules = [
        torch.nn.Linear,
        torch.nn.functional.linear,
    ]

    targets = [
        exir_ops.edge.aten.mm.default,
        exir_ops.edge.aten.addmm.default,
        exir_ops.edge.aten.bmm.default,
    ]

    @staticmethod
    def find(
        node: torch.fx.Node,
        args: List[torch.fx.Node],
        kind: str = "args",
        index: int = 0,
    ):
        # This is a hack to support lifted graphs.
        # TODO(T171263351) - fix source partitioning for lifted graphs
        if not node or node in args or node.op == "placeholder":
            return node
        if kind == "args":
            other = node.args[index]
        elif kind == "users":
            other = list(node.users.keys())[index]
        else:
            raise AssertionError(f"Unexpected kind: {kind}")
        return ConvertToLinearPass.find(other, args, kind)  # pyre-ignore[6]

    @staticmethod
    def get_arg(node: torch.fx.Node, arg: str):
        if node.target == exir_ops.edge.aten.addmm.default:
            map_ = {
                "bias": 0,
                "input": 1,
                "weight": 2,
            }
            return node.args[map_[arg]]
        else:
            map_ = {"input": 0, "weight": 1}
            return None if arg == "bias" else node.args[map_[arg]]

    def find_bias_for_mm(self, src_partition: SourcePartition, mm_node: torch.fx.Node):
        """
        For linear decomposed with mm + add, find bias in src partition
        """

        mm_users = list(mm_node.users.keys())
        if len(mm_users) != 1:
            return None

        add_node = mm_users[0]
        if add_node.target != exir_ops.edge.aten.add.Tensor:
            return None

        for arg in add_node.all_input_nodes:
            if arg != mm_node and arg in src_partition.input_nodes:
                return arg

        return None

    def create_linear(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        src_partition: SourcePartition,
    ):
        logger.debug(f"Source Partition: {src_partition}")
        linear_input = self.find(
            self.get_arg(node, "input"),
            src_partition.input_nodes,
        )
        logger.debug(f"Found input: {linear_input} from node {node}")

        linear_weight = self.find(
            self.get_arg(node, "weight"),
            src_partition.input_nodes
            + src_partition.params,  # non quant weight can be in params
        )
        logger.debug(f"Found weight: {linear_weight} from node {node}")

        linear_bias = self.find(
            self.get_arg(node, "bias"),
            src_partition.input_nodes + src_partition.params,  # bias can be in params
        )
        if linear_bias is None and node.target == exir_ops.edge.aten.mm.default:
            linear_bias = self.find_bias_for_mm(src_partition, node)

        logger.debug(f"Found bias(?): {linear_bias} from node {node}")

        # Ignore dynamic shape nodes
        outputs = [
            node
            for node in src_partition.output_nodes
            if node.target != torch.ops.aten.sym_size.int and node.op != "placeholder"
        ]
        assert (
            len(outputs) == 1
        ), f"Unexpected number of outputs for a torch.nn.Linear module, expecting 1 but got {outputs}"
        output = outputs[0]

        with graph_module.graph.inserting_before(output):
            args = (linear_input, linear_weight)
            if linear_bias is not None:
                args += (linear_bias,)
            linear_node = graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.aten.linear.default,  # HACK not edge_op/CATen
                args,
            )
        # TODO - calculate output even when dynamic_shape=True
        linear_node.meta["val"] = torch.zeros(get_shape(output))
        logger.debug(
            f"Replacing {output}{get_shape(output)} node with {linear_node}{get_shape(linear_node)}"
        )
        output.replace_all_uses_with(linear_node)
        graph_module.graph.eliminate_dead_code()

    # override
    def call(self, graph_module: torch.fx.GraphModule):
        logger.debug("ConvertToLinear Begin: ")
        logger.debug(graph_module.print_readable(print_output=False))

        processed_partitions = 0
        while True:
            src_partition_dict = get_source_partitions(
                graph_module.graph, self.linear_modules
            )

            src_node_dict = {
                node: src_partition
                for src_partitions in src_partition_dict.values()
                for src_partition in src_partitions
                for node in src_partition.nodes
                if node.target in self.targets
            }

            # No more [add]mm target in source partitions
            if len(src_node_dict) == 0:
                if processed_partitions == 0:
                    logger.debug(
                        "Did not find any [add]mm target in source partitions, skipping the pass."
                    )
                else:
                    logger.debug(
                        f"Converted {processed_partitions} [add]mm target(s) into Linear."
                    )
                break

            logger.debug("Converting [add]mm into Linear")
            for node in src_node_dict.keys():
                self.create_linear(graph_module, node, src_node_dict[node])
                processed_partitions += 1
                # Only convert the first [add]mm target
                break

        # fall back to linear transform
        graph_module.graph = apply_addmm_mm_to_linear_transform(graph_module.graph)

        graph_module.recompile()

        # Propagate metadata and retrace module
        graph_module = super().call(graph_module).graph_module

        logger.debug("ConvertToLinear End: ")
        logger.debug(graph_module.print_readable(print_output=False))

        return PassResult(graph_module, True)
