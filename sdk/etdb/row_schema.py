# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from executorch.sdk.edir.base_schema import OperatorGraph, OperatorNode, ValueNode
from executorch.sdk.edir.et_schema import RESERVED_METADATA_ARG


@dataclass
class AbstractInstanceRow:
    @staticmethod
    def get_schema_header(verbose: bool = False) -> List[str]:
        pass

    def to_row_format(self, verbose=False) -> Tuple[Any, ...]:
        pass


@dataclass
class AbstractNodeInstanceRow(AbstractInstanceRow):
    def get_name(self) -> str:
        pass

    def get_input_nodes(self) -> List[str]:
        pass

    def get_output_nodes(self) -> List[str]:
        pass


@dataclass
class GraphInstanceRow(AbstractNodeInstanceRow):
    name: str
    elements: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    parent_graph: Optional[str] = None

    # Element counts
    graph_count: int = 0
    operator_count: int = 0
    constant_count: int = 0

    # Name identified rows
    input_nodes: List[str] = field(default_factory=list)
    output_nodes: List[str] = field(default_factory=list)

    @staticmethod
    def gen_from_operator_node(
        op_graph: OperatorGraph, parent_graph: Optional[str] = None
    ) -> "GraphInstanceRow":
        graph_count = 0
        operator_count = 0
        constant_count = 0
        elements = []
        for e in op_graph.elements:
            if isinstance(e, OperatorGraph):
                elements.append(e.graph_name)
                graph_count += 1
            elif isinstance(e, OperatorNode):
                elements.append(e.name)
                operator_count += 1
            elif isinstance(e, ValueNode):
                elements.append(e.name)
                constant_count += 1

        return GraphInstanceRow(
            op_graph.graph_name,
            elements,
            op_graph.metadata,
            parent_graph,
            graph_count,
            operator_count,
            constant_count,
        )

    @staticmethod
    def get_schema_header(verbose: bool = False, count_format=False) -> List[str]:
        # (!!) Format to coincide with OpInstanceRow format
        if not count_format:
            return OpInstanceRow.get_schema_header(verbose)

        if verbose:
            return [
                "Parent Graph",
                "Name",
                "Element Count",
                "Graph Count",
                "Operator Count",
                "Constant Count",
            ]
        return ["Name", "Graph Count", "Operator Count"]

    def get_module_type(self) -> str:
        return "[Sub Module] " + (
            self.metadata.get("module_type", "") if self.metadata is not None else ""
        )

    def to_row_format(self, verbose=False, count_format=False) -> Tuple[Any, ...]:
        if count_format:
            element_count = len(self.elements) if self.elements else 0
            if verbose:
                return (
                    self.parent_graph,
                    self.name,
                    element_count,
                    self.graph_count,
                    self.operator_count,
                    self.constant_count,
                )
            return (self.name, self.graph_count, self.operator_count)

        # (!!) Format to coincide with OpInstanceRow format
        module_type = self.get_module_type()
        input_node_str = "\n".join(self.input_nodes)
        output_node_str = "\n".join(self.output_nodes)

        if verbose:
            return (
                self.parent_graph,
                module_type,
                self.name,
                len(self.input_nodes),
                len(self.output_nodes),
                "",
                input_node_str,
                output_node_str,
            )
        return (module_type, self.name, input_node_str, output_node_str)

    def get_name(self) -> str:
        return self.name

    def get_input_nodes(self) -> List[str]:
        return self.input_nodes

    def get_output_nodes(self) -> List[str]:
        return self.output_nodes


@dataclass
class ValueInstanceRow(AbstractNodeInstanceRow):
    dtype: str
    name: str
    val: Any
    metadata: Optional[Dict[str, Any]] = None
    parent_graph: Optional[str] = None

    # Name identified rows
    input_nodes: List[str] = field(default_factory=list)
    output_nodes: List[str] = field(default_factory=list)

    # Note: This does not populate output nodes
    @staticmethod
    def gen_from_operator_node(
        value_node: ValueNode, parent_graph: Optional[str] = None
    ) -> "ValueInstanceRow":
        input_nodes = [e.name for e in value_node.inputs] if value_node.inputs else []
        return ValueInstanceRow(
            value_node.dtype,
            value_node.name,
            value_node.val,
            value_node.metadata,
            parent_graph,
            input_nodes,
            [],
        )

    @staticmethod
    def get_schema_header(verbose: bool = False) -> List[str]:
        if verbose:
            return [
                "Parent Graph",
                "Dtype",
                "Name",
                "Value (Shape if Tensor)",
                "Input Count",
                "Output Count",
                "Input Nodes",
                "Output Nodes",
            ]
        return [
            "Dtype",
            "Name",
            "Value (Shape if Tensor)",
            "Input Nodes",
            "Output Nodes",
        ]

    def to_row_format(self, verbose=False) -> Tuple[Any, ...]:
        row = (self.dtype, self.name, self.val)
        if verbose:
            row = (self.parent_graph,) + row
            row += (len(self.input_nodes), len(self.output_nodes))

        return row + ("\n".join(self.input_nodes), "\n".join(self.output_nodes))

    def get_name(self) -> str:
        return self.name

    def get_input_nodes(self) -> List[str]:
        return self.input_nodes

    def get_output_nodes(self) -> List[str]:
        return self.output_nodes


@dataclass
class OpInstanceRow(AbstractNodeInstanceRow):
    op: str
    name: str
    output_shapes: Optional[List[List[int]]] = None
    metadata: Optional[Dict[str, Any]] = None
    parent_graph: Optional[str] = None

    # Name identified rows
    input_nodes: List[str] = field(default_factory=list)
    output_nodes: List[str] = field(default_factory=list)

    @staticmethod
    def gen_from_operator_node(
        operator_node: OperatorNode, parent_graph: Optional[str] = None
    ) -> "OpInstanceRow":
        op = operator_node.op if operator_node.op is not None else "Unknown"
        input_nodes = (
            [e.name for e in operator_node.inputs] if operator_node.inputs else []
        )
        return OpInstanceRow(
            op,
            operator_node.name,
            operator_node.output_shapes,
            operator_node.metadata,
            parent_graph,
            input_nodes,
        )

    @staticmethod
    def get_schema_header(verbose: bool = False) -> List[str]:
        if verbose:
            return [
                "Parent Group",
                "Op (Module if annotated)",
                "Name",
                "Input Count",
                "Output Count",
                "Output Shapes",
                "Input Nodes",
                "Output Nodes",
                "Coldstart (ms)",
                "Mean (ms)",
                "Min (ms)",
                "P10 (ms)",
                "P90 (ms)",
                "Max (ms)",
            ]
        return [
            "Op (Module if annotated)",
            "Name",
            "Input Nodes",
            "Output Nodes",
        ]

    def to_row_format(self, verbose=False) -> Tuple[Any, ...]:
        input_node_str = "\n".join(self.input_nodes)
        output_node_str = "\n".join(self.output_nodes)

        output_shape_str = (
            "\n".join([str(shape) for shape in self.output_shapes])
            if self.output_shapes is not None
            else ""
        )

        if verbose:
            row = (
                self.parent_graph,
                self.op,
                self.name,
                len(self.input_nodes),
                len(self.output_nodes),
                output_shape_str,
                input_node_str,
                output_node_str,
            )

            # Note: These fields can be opaquely extracted from a keyed metadta field
            metadata = self.metadata
            if (
                metadata is not None
                and RESERVED_METADATA_ARG.METRICS_KEYWORD.value in metadata
            ):
                metrics = metadata[RESERVED_METADATA_ARG.METRICS_KEYWORD.value]
                coldstart_ms = metrics.get(
                    RESERVED_METADATA_ARG.PROFILE_SUMMARY_COLDSTART.value, None
                )
                mean_ms = metrics.get(
                    RESERVED_METADATA_ARG.PROFILE_SUMMARY_AVERAGE.value, None
                )
                min_ms = metrics.get(
                    RESERVED_METADATA_ARG.PROFILE_SUMMARY_MIN.value, None
                )
                p10_ms = metrics.get(
                    RESERVED_METADATA_ARG.PROFILE_SUMMARY_P10.value, None
                )
                p90_ms = metrics.get(
                    RESERVED_METADATA_ARG.PROFILE_SUMMARY_P90.value, None
                )
                max_ms = metrics.get(
                    RESERVED_METADATA_ARG.PROFILE_SUMMARY_MAX.value, None
                )
                row += (coldstart_ms, mean_ms, min_ms, p10_ms, p90_ms, max_ms)

            return row

        return (self.op, self.name, input_node_str, output_node_str)

    def get_name(self) -> str:
        return self.name

    def get_input_nodes(self) -> List[str]:
        return self.input_nodes

    def get_output_nodes(self) -> List[str]:
        return self.output_nodes


@dataclass
class OpSummaryRow(AbstractInstanceRow):
    op: str
    elements: List[AbstractNodeInstanceRow]

    # Summary Stats
    mean_ms: Optional[float] = None
    min_ms: Optional[float] = None
    p10_ms: Optional[float] = None
    p90_ms: Optional[float] = None
    max_ms: Optional[float] = None

    @staticmethod
    def get_schema_header(verbose: bool = False) -> List[str]:
        return [
            "Op (Module if annotated)",
            "Instance Count",
            "Mean (ms)",
            "Min (ms)",
            "P10 (ms)",
            "P90 (ms)",
            "Max (ms)",
        ]

    def to_row_format(self, verbose=False) -> Tuple[Any, ...]:
        return (
            self.op,
            len(self.elements),
            self.mean_ms,
            self.min_ms,
            self.p10_ms,
            self.p90_ms,
            self.max_ms,
        )
