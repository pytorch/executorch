# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code=import-not-found

from typing import Callable, final, List, Optional, Tuple

import torch
from executorch.backends.openvino.preprocess import OpenvinoBackend
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from openvino.frontend.pytorch.torchdynamo.op_support import (  # type: ignore[import-untyped]
    OperatorSupport,
)

from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase


class PatternNode:
    op_types: dict[str, Optional[list]] = {}

    def __init__(self):
        self.op_types = {}


class OpenvinoOperatorsSupport(OperatorSupportBase):
    extended_support_dict = {
        "torch.ops.dim_order_ops._clone_dim_order.default": None,
        "torch.ops.dim_order_ops._to_dim_order_copy.default": None,
    }

    def __init__(
        self,
        op_types_to_skip: Optional[set] = None,
        op_names_to_skip: Optional[set] = None,
        enabled_ops_by_name: Optional[set] = None,
    ) -> None:
        """
        Initializes the OpenvinoOperatorsSupport class.

        :param op_types_to_skip: A set of operator types to skip during support checking.
        :param op_names_to_skip: A set of operator names to skip during support checking.
        """
        if op_types_to_skip is None:
            op_types_to_skip = set()
        if op_names_to_skip is None:
            op_names_to_skip = set()
        if enabled_ops_by_name is None:
            enabled_ops_by_name = set()

        self._op_types_to_skip = op_types_to_skip
        self._op_names_to_skip = op_names_to_skip
        self._enabled_ops_by_name = enabled_ops_by_name

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        """
        Checks if a given node is supported by OpenVINO.

        :param node: The FX graph node representing an operation.
        :return: True if the node is supported, otherwise False.
        """
        if node.op != "call_function":
            return False

        options: list[str] = []
        if not isinstance(node.target, str):
            op_type = node.target.__name__
        else:
            op_type = str(node.target)

        if node.name in self._enabled_ops_by_name:
            return True

        supported_ops = (
            OperatorSupport(options)._support_dict | self.extended_support_dict
        )
        if op_type == "getitem":
            return True

        if "torch.ops." + str(op_type) in supported_ops:
            return True
        else:
            print("Op not supported: ", "torch.ops." + str(op_type))

        if op_type in self._op_types_to_skip or node.name in self._op_names_to_skip:
            print(
                f"[OpenVINO Backend] The {op_type} operator with name '{node.name}' is skipped."
            )
            return False

        return False


@final
class OpenvinoPartitioner(Partitioner):

    def __init__(
        self,
        compile_spec: List[CompileSpec],
        op_types_to_skip: Optional[set] = None,
        op_names_to_skip: Optional[set] = None,
    ) -> None:
        """
        Initializes the OpenvinoPartitioner class.

        :param compile_spec: A list of compile specifications for OpenVINO.
        :param op_types_to_skip: A set of operator types to skip during partitioning.
        :param op_names_to_skip: A set of operator names to skip during partitioning.
        """
        self.delegation_spec = DelegationSpec(OpenvinoBackend.__name__, compile_spec)
        self._op_types_to_skip = op_types_to_skip
        self._op_names_to_skip = op_names_to_skip
        self._enabled_ops_by_name: set = set()

    def ops_to_not_decompose(
        self,
        ep: ExportedProgram,
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """
        Returns a tuple containing a list of operations that should not be decomposed
        and an optional function to filter nodes.

        :param ep: The exported program.
        :return: A tuple consisting of a list of ops to keep and an optional filtering function.
        """
        ops_not_decompose = [
            torch.ops.aten.pixel_shuffle.default,
            torch.ops.aten.upsample_bilinear2d.default,
            torch.ops.aten.upsample_bilinear2d.vec,
            torch.ops.aten.upsample_nearest2d.default,
            torch.ops.aten.upsample_nearest2d.vec,
            torch.ops.aten.stack.default,
        ]
        return (ops_not_decompose, None)

    def check_pattern(
        self, node: torch.fx.Node, pattern: type[PatternNode], enabled_ops: list
    ) -> bool:
        if node.op == "call_function":
            if ("call_function" + ":" + str(node.target.__name__)) in pattern.op_types:  # type: ignore[union-attr]
                pt_input_nodes = node.all_input_nodes
                pattern_input_ops = pattern.op_types[
                    "call_function" + ":" + str(node.target.__name__)  # type: ignore[union-attr]
                ]
                if pattern_input_ops is None:
                    enabled_ops.append(node)
                    return True
                if len(pt_input_nodes) != len(pattern_input_ops):
                    return False
                for i in range(len(pt_input_nodes)):
                    if not self.check_pattern(
                        pt_input_nodes[i], pattern_input_ops[i], enabled_ops
                    ):
                        return False
                enabled_ops.append(node)
                return True
        elif node.op == "get_attr":
            if "get_attr" in pattern.op_types:
                return True
            else:
                return False
        elif node.op == "placeholder":
            if "placeholder" in pattern.op_types:
                return True
            else:
                return False
        return False

    def capture_nncf_patterns(self, graph_module: torch.fx.GraphModule):
        const_node = PatternNode
        const_node.op_types["get_attr"] = None
        const_node.op_types["placeholder"] = None
        bitwise_right_shift_node = PatternNode
        bitwise_right_shift_node.op_types[
            "call_function:aten.bitwise_right_shift.Tensor_Scalar"
        ] = [const_node]
        bitwise_and_node = PatternNode
        bitwise_and_node.op_types["call_function:aten.bitwise_and.Scalar"] = [
            const_node
        ]
        stack_node = PatternNode
        stack_node.op_types["call_function:aten.stack.default"] = [
            bitwise_and_node,
            bitwise_right_shift_node,
        ]

        for node in graph_module.graph.nodes:
            if (
                str(node.op) == "call_function"
                and str(node.target.__name__) == "aten.stack.default"
            ):
                enabled_ops: list = []
                pattern_match = self.check_pattern(node, stack_node, enabled_ops)
                if pattern_match:
                    for pattern_op in enabled_ops:
                        self._enabled_ops_by_name.add(pattern_op.name)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """
        Partitions an exported program into supported and unsupported segments.

        :param exported_program: The exported program.
        :return: A PartitionResult containing the partitioned graph and delegation tags.
        """
        self.capture_nncf_patterns(exported_program.graph_module)
        partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            OpenvinoOperatorsSupport(
                self._op_types_to_skip,
                self._op_names_to_skip,
                self._enabled_ops_by_name,
            ),
            allows_single_node_partition=True,
        )
        partition_list = partitioner.propose_partitions()

        partition_tags = {}
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
