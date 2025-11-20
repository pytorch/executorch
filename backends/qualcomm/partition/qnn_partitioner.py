# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from executorch.backends.qualcomm.builders import node_visitor_manager
from executorch.backends.qualcomm.builders.qnn_constants import OpContextLoader
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS_ORDER,
    QCOM_BYPASS_NODE,
)

from executorch.backends.qualcomm.utils.qnn_manager_lifecycle import (
    get_current_qnn_manager,
)
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase

from .common_defs import (
    allow_list_operator,
    constant_operator,
    not_supported_operator,
    to_be_implemented_operator,
)
from .utils import filter_fn, generate_qnn_executorch_option, get_skip_decomp_table

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QnnOperatorSupport(OperatorSupportBase):
    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        compiler_specs,
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
    ):
        option = generate_qnn_executorch_option(compiler_specs)
        python_options = flatbuffer_to_option(option)
        self.node_visitors = node_visitor_manager.get_node_visitors(
            edge_program,
            op_package_infos=python_options.op_package_options.op_package_infos,
        )

        self.skip_node_op_set = skip_node_op_set
        self.skip_node_id_set = skip_node_id_set
        self.nodes_to_wrappers = defaultdict(dict)
        self.qnn_manager = get_current_qnn_manager(
            python_options.backend_options.backend_type, compiler_specs
        )

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        if node.op != "call_function" or node.target in not_supported_operator:
            return False

        if node.target in to_be_implemented_operator:
            print(
                f"[QNN Partitioner Op Support]: {node.target.__name__} | Skipped, this op can be supported, please report an issue in https://github.com/pytorch/executorch/issues"
            )
            return False

        if (
            node.target in allow_list_operator
            # bypass if custom op appears
            or OpContextLoader.namespace == node.target.namespace
            # bypass dequantize op for parameters & buffers
            or node.meta.get(QCOM_BYPASS_NODE, False)
        ):
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
        if node.target in constant_operator:
            return True

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


class QnnPartitioner(Partitioner):
    """
    QnnPartitioner identifies subgraphs that can be lowered to QNN backend, by tagging nodes for delegation,
    and manages special cases such as mutable buffers and consumed constants.
    """

    def __init__(
        self,
        compiler_specs: List[CompileSpec],
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
        skip_mutable_buffer: bool = False,
    ):
        """
        Args:
            compiler_specs (List[CompileSpec]): Backend compiler specifications.
            skip_node_id_set (set, optional): Set of node IDs to exclude from partitioning.
            skip_node_op_set (set, optional): Set of OpOverload to exclude from partitioning.
            skip_mutable_buffer (bool, optional): If True, mutable buffers are not delegated to QNN.
        """
        self.compiler_specs_snapshot = copy.deepcopy(compiler_specs)

        self.delegation_spec = DelegationSpec(
            QnnBackend.__name__, self.compiler_specs_snapshot
        )
        self.partition_tags: Dict[str, DelegationSpec] = {}
        self.skip_node_id_set = set() if skip_node_id_set is None else skip_node_id_set
        self.skip_node_op_set = set() if skip_node_op_set is None else skip_node_op_set
        self.skip_mutable_buffer = skip_mutable_buffer

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
        """
        Tags nodes in the given partitions and the edge program's graph with delegation tags for QNN partitioning.
        """
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
        # Generate partitions by QNN op_support checker
        partitions = self.generate_partitions(edge_program)
        del self.op_support_checker

        # If partitions are found, handle tagging of nodes, constant data, and mutated buffers for delegation
        if len(partitions) != 0:
            self.tag_nodes(partitions, edge_program)
            tag_constant_data(edge_program)
            if not self.skip_mutable_buffer:
                logger.info(
                    "Qnn partitioner will delegate torch mutable buffer with the same I/O address during the runtime, "
                    "so if your model contains mutable buffer, "
                    "then you can get the better performance with skip_mutable_buffer=False. "
                    "If you encounter accuracy issue during the runtime, "
                    "then please set `skip_mutable_buffer=True` and try again."
                )
                tag_mutated_buffer(edge_program)

        # pop certain keys in meta for not affecting the passes in compilation
        for node in edge_program.graph_module.graph.nodes:
            if hasattr(node, "meta"):
                # TODO: need to put property name in common definitions
                node.meta.pop(QCOM_AXIS_ORDER, "")
        return PartitionResult(
            tagged_exported_program=edge_program, partition_tags=self.partition_tags
        )

    # override
    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """
        Determines which op should not be decomposed during partitioning.
        The list of operators is obtained from `get_skip_decomp_table()`.
        The filter function (`filter_fn`) can be used to further refine which nodes are not decomposed. (advanced use case)
        """
        do_not_decompose = get_skip_decomp_table()
        return (do_not_decompose, filter_fn)
