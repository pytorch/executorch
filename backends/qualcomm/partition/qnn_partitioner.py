# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
from typing import Any, Dict, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import torch
from executorch.backends.qualcomm.builders import node_visitor
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.utils.utils import generate_qnn_executorch_option

from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from torch.export import ExportedProgram
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .common_defs import allow_list_operator, not_supported_operator, supported_modules


class QnnOperatorSupport(OperatorSupportBase):
    def __init__(self, graph_module: torch.fx.GraphModule, compiler_specs):
        self.node_visitors = node_visitor.get_node_visitors(
            graph_module, compile_mode=False
        )
        self.nodes_to_wrappers = {}
        self.qnn_manager = PyQnnManager.QnnManager(
            generate_qnn_executorch_option(compiler_specs)
        )
        self.qnn_manager.Init()

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        if node.op != "call_function":
            return False

        if node.target in not_supported_operator:
            return False

        op_wrapper = self.node_visitors[node.target.__name__].define_node(
            node, self.nodes_to_wrappers
        )
        if node.target in allow_list_operator:
            return True

        if op_wrapper is None:
            print(f"[QNN Partitioner Op Support]: {node.target.__name__} | False")
            return False

        supported = self.qnn_manager.IsNodeSupportedByBackend(
            [op_wrapper.GetOpWrapper()]
        )
        self.nodes_to_wrappers.clear()
        print(f"[QNN Partitioner Op Support]: {node.target.__name__} | {supported}")
        return supported


class QnnPartitioner(Partitioner):
    compiler_specs = []

    @classmethod
    def set_compiler_spec(cls, compiler_specs: List[CompileSpec]):
        # note: operations on compiler specs are not thread-safe
        # please pay attention to it
        QnnPartitioner.compiler_specs = compiler_specs

    def __init__(self):
        self.supported_modules = set(supported_modules)
        self.compiler_specs_snapshot = copy.deepcopy(QnnPartitioner.compiler_specs)
        self.delegation_spec = DelegationSpec(
            QnnBackend.__name__, self.compiler_specs_snapshot
        )
        self.partition_tags: Dict[str, DelegationSpec] = {}

    def get_module_partitions(
        self, graph_module: torch.fx.GraphModule
    ) -> List[List[torch.fx.Node]]:
        def filter_fn(node):
            if node.op == "call_function":
                return self.op_support_checker.is_node_supported(None, node)
            return True

        src_partition_dict = get_source_partitions(
            graph_module.graph,
            self.supported_modules,
            filter_fn,
        )
        all_partitions = src_partition_dict.values()

        module_partitions = []
        for src_partitions in all_partitions:
            for src_partition in src_partitions:
                module_partitions.append(src_partition.nodes)

        return module_partitions

    def generate_partitions(self, graph_module: torch.fx.GraphModule) -> List[Any]:
        self.op_support_checker = QnnOperatorSupport(
            graph_module, self.compiler_specs_snapshot
        )
        return generate_partitions_from_list_of_nodes(
            graph_module,
            pattern_list=self.get_module_partitions(graph_module),
            op_support=self.op_support_checker,
        )

    def tag_nodes(self, partitions: List[Partition]) -> None:
        for partition in partitions:
            for node in partition.nodes:
                delegation_tag = f"qnn_{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                self.partition_tags[delegation_tag] = self.delegation_spec

    # override
    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        graph_module = exported_program.graph_module
        partitions = self.generate_partitions(graph_module)
        if len(partitions) != 0:
            self.tag_nodes(partitions)
        for node in graph_module.graph.nodes:
            if hasattr(node, "meta"):
                # pop certain keys in meta for not affecting the passes in compilation
                # TODO: need to put property name in common definitions
                node.meta.pop("axis_order", "")
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=self.partition_tags
        )
