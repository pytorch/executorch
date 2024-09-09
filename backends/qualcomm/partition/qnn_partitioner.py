# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
from collections import defaultdict
from typing import Any, Dict, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import torch
from executorch.backends.qualcomm.builders import node_visitor
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.utils.constants import QCOM_AXIS_ORDER
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
from executorch.exir.backend.utils import tag_constant_data
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase

from .common_defs import (
    allow_list_operator,
    not_supported_operator,
    to_be_implemented_operator,
)


class QnnOperatorSupport(OperatorSupportBase):
    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        compiler_specs,
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
    ):
        self.node_visitors = node_visitor.get_node_visitors(edge_program)

        self.skip_node_op_set = skip_node_op_set
        self.skip_node_id_set = skip_node_id_set
        self.nodes_to_wrappers = defaultdict(dict)
        self.qnn_manager = PyQnnManager.QnnManager(
            generate_qnn_executorch_option(compiler_specs)
        )

        self.qnn_manager.Init()

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        if node.op != "call_function" or node.target in not_supported_operator:
            return False

        if node.target in to_be_implemented_operator:
            print(
                f"[QNN Partitioner Op Support]: {node.target.__name__} | Skipped, this op can be supported, please report an issue in https://github.com/pytorch/executorch/issues"
            )
            return False

        if node.target in allow_list_operator:
            return True

        if (
            node.name in self.skip_node_id_set
            or node.target.__name__ in self.skip_node_op_set
        ):
            print(f"[QNN Partitioner Op Support]: {node.target.__name__} | Skipped")
            return False

        supported = False
        op_wrapper = self.node_visitors[node.target.__name__].define_node(
            node, self.nodes_to_wrappers
        )

        op_wrapper_list = []
        if isinstance(op_wrapper, List):
            op_wrapper_list.extend(op_wrapper)
        else:
            op_wrapper_list.append(op_wrapper)

        if op_wrapper is not None:
            supported = self.qnn_manager.IsNodeSupportedByBackend(
                [op_wrapper.GetOpWrapper() for op_wrapper in op_wrapper_list]
            )

        self.nodes_to_wrappers.clear()
        print(f"[QNN Partitioner Op Support]: {node.target.__name__} | {supported}")
        return supported

    def __del__(self):
        self.qnn_manager.Destroy()


class QnnPartitioner(Partitioner):
    def __init__(
        self,
        compiler_specs: List[CompileSpec],
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
    ):
        self.compiler_specs_snapshot = copy.deepcopy(compiler_specs)

        self.delegation_spec = DelegationSpec(
            QnnBackend.__name__, self.compiler_specs_snapshot
        )
        self.partition_tags: Dict[str, DelegationSpec] = {}
        self.skip_node_id_set = set() if skip_node_id_set is None else skip_node_id_set
        self.skip_node_op_set = set() if skip_node_op_set is None else skip_node_op_set

    def generate_partitions(
        self, edge_program: torch.export.ExportedProgram
    ) -> List[Any]:
        self.op_support_checker = QnnOperatorSupport(
            edge_program,
            self.compiler_specs_snapshot,
            self.skip_node_id_set,
            self.skip_node_op_set,
        )
        return generate_partitions_from_list_of_nodes(
            edge_program.graph_module,
            op_support=self.op_support_checker,
        )

    def tag_nodes(
        self, partitions: List[Partition], edge_program: torch.export.ExportedProgram
    ) -> None:
        for partition in partitions:
            for node in partition.nodes:
                delegation_tag = f"qnn_{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                self.partition_tags[delegation_tag] = self.delegation_spec

        # need to take care of consumed constants
        consumed_constants = (
            *edge_program.graph_signature.inputs_to_buffers,
            *edge_program.graph_signature.inputs_to_parameters,
        )
        for node in edge_program.graph_module.graph.nodes:
            # find placeholders as lifted_constants
            if node.op != "placeholder" or len(node.users) != 0:
                continue

            if node.name in consumed_constants:
                # does no harm to merge them into last partition,
                # since they will all be removed in following stage
                node.meta["delegation_tag"] = delegation_tag

    # override
    def partition(self, edge_program: torch.export.ExportedProgram) -> PartitionResult:
        partitions = self.generate_partitions(edge_program)
        if len(partitions) != 0:
            self.tag_nodes(partitions, edge_program)
            tag_constant_data(edge_program)
        for node in edge_program.graph_module.graph.nodes:
            if hasattr(node, "meta"):
                # pop certain keys in meta for not affecting the passes in compilation
                # TODO: need to put property name in common definitions
                node.meta.pop(QCOM_AXIS_ORDER, "")
        del self.op_support_checker
        return PartitionResult(
            tagged_exported_program=edge_program, partition_tags=self.partition_tags
        )
