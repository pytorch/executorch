# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import executorch.backends.samsung.builders.node_visitor as node_visitor

import executorch.backends.samsung.python.PyEnnWrapperAdaptor as PyEnnWrapper

import torch

from executorch.backends.samsung._passes.customized_constant_prop import (
    get_nodes_in_const_subgraph,
)
from executorch.backends.samsung._passes.remove_useless_ops import can_remove

from executorch.backends.samsung.enn_preprocess import EnnBackend
from executorch.backends.samsung.serialization.compile_options import (
    ENN_COMPILE_OPTION_TITLE,
)
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph
from executorch.backends.samsung.utils.utils import get_compile_spec
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data

from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase

SUPPORTED_OPS = [
    # support because preprocess in backend
    exir_ops.edge.aten.addmm.default,
    exir_ops.edge.aten.add.Scalar,
    exir_ops.edge.aten.sub.Scalar,
    exir_ops.edge.aten.mul.Scalar,
    exir_ops.edge.aten.div.Scalar,
    exir_ops.edge.aten.alias_copy.default,
    exir_ops.edge.aten.clone.default,
    exir_ops.edge.aten.pow.Tensor_Scalar,
    exir_ops.edge.aten.as_strided_copy.default,
]


class EnnOperatorSupport(OperatorSupportBase):

    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        compile_specs: List[CompileSpec],
    ):
        self.edge_program = edge_program
        self.enn_wrapper = PyEnnWrapper.EnnWrapper()
        self.node_visitors = node_visitor.get_node_visitors(edge_program)
        option_spec = get_compile_spec(
            compile_specs, ENN_COMPILE_OPTION_TITLE, required=True
        )
        self.enn_wrapper.Init(option_spec.value)
        self.nodes_in_const_subgraph = get_nodes_in_const_subgraph(edge_program)
        for node in self.nodes_in_const_subgraph:
            node.meta["can_fold"] = True

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        if node.op != "call_function":
            return False

        if node.op in [
            "get_attr",
            "placeholder",
            "output",
        ]:
            return False

        if node.target in SUPPORTED_OPS:
            return True

        if node.target.__name__ in self.node_visitors:
            enn_graph = EnnGraph()
            vals_to_ids: Dict[torch.fx.Node, int] = {}
            return self.node_visitors[node.target.__name__].define_node(
                node, enn_graph, vals_to_ids
            )
        elif node in self.nodes_in_const_subgraph:
            return True

        supported = self.enn_wrapper.IsNodeSupportedByBackend()
        return supported

    def __del__(self):
        self.enn_wrapper.Destroy()


class EnnPartitioner(Partitioner):
    def __init__(self, compile_specs: List[CompileSpec]):
        # TODO(anyone): Add meaningful initialize
        self.delegation_spec = DelegationSpec(EnnBackend.__name__, compile_specs)
        self.partition_tags: Dict[str, DelegationSpec] = {}
        self.compile_specs = compile_specs

    def remove_fold_node(self, partition_list: list[Partition]):
        """
        Remove nodes marked with 'can_fold' from partitions if their users are not in the same partition.
        """
        for partition in partition_list:
            partition_nodes = set(partition.nodes.keys())

            nodes_to_remove = []
            no_user_fold_nodes = []
            for node in partition_nodes:
                if node.meta.get("can_fold", False):
                    has_external_user = False
                    for user in node.users:
                        if user not in partition_nodes:
                            has_external_user = True
                            break
                    if has_external_user:
                        no_user_fold_nodes.append(node)

            nodes_queue = list(no_user_fold_nodes)

            while nodes_queue:
                node = nodes_queue.pop(0)
                nodes_to_remove.append(node)
                for input_node in node.all_input_nodes:
                    if (
                        input_node in partition_nodes
                        and input_node not in nodes_to_remove
                    ):
                        nodes_queue.append(input_node)

            for node in nodes_to_remove:
                if node in partition.nodes:
                    del partition.nodes[node]

        partition_list = [p for p in partition_list if len(p.nodes) > 0]
        return partition_list

    def generate_partitions(
        self, edge_program: torch.export.ExportedProgram
    ) -> List[Any]:
        self.op_support_checker = EnnOperatorSupport(edge_program, self.compile_specs)
        partition_list = generate_partitions_from_list_of_nodes(
            edge_program.graph_module,
            op_support=self.op_support_checker,
        )
        if len(partition_list) == 1 and partition_list[0].size() == 1:
            first_node = list(partition_list[0].nodes.keys())[0]
            # If there is only one partition graph containing a single op that is a useless operation,
            # the RemoveUselessOpPass will remove this operation and cause a graph error.
            # Therefore, we delete this node to prevent this graph error.
            # For example, in the test_index_put_in_place_dtype case partition_list is [{aten_clone_default: 2}]
            if can_remove(first_node):
                del partition_list[0]

        partition_list = self.remove_fold_node(partition_list)
        return partition_list

    def tag_nodes(self, partitions: List[Partition]) -> None:
        for partition in partitions:
            # Add delegation tags
            for node in partition.nodes:
                delegation_tag = f"enn_{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                self.partition_tags[delegation_tag] = self.delegation_spec

    # override
    def partition(self, edge_program: torch.export.ExportedProgram) -> PartitionResult:
        partitions = self.generate_partitions(edge_program)
        logging.info(f"Find {len(partitions)} " "subgraphs to partition and lowering.")
        if len(partitions) != 0:
            self.tag_nodes(partitions)
            tag_constant_data(edge_program)
        del self.op_support_checker
        return PartitionResult(
            tagged_exported_program=edge_program, partition_tags=self.partition_tags
        )

    # override
    def ops_to_not_decompose(
        self, ep: torch.export.ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        ops_not_to_decompose = [
            torch.ops.aten.hardswish.default,
            torch.ops.aten.max_pool2d.default,
            torch.ops.aten.linear.default,
            torch.ops.aten._safe_softmax.default,
            torch.ops.aten.prelu.default,
            torch.ops.aten.layer_norm.default,
            torch.ops.aten.pixel_shuffle.default,
            torch.ops.aten.hardsigmoid.default,
            torch.ops.aten.silu.default,
            torch.ops.aten.pad.default,
        ]
        return (ops_not_to_decompose, None)
