from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

from executorch.sdk.edir.base_schema import Node, ValueNode

from executorch.sdk.edir.et_schema import (
    OperatorGraph,
    OperatorNode,
    RESERVED_METADATA_ARG,
)

from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.step_stats_pb2 import (
    DeviceStepStats,
    NodeExecStats,
    StepStats,
)
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto

# TODO: these strings are shared between this and et_schema.py, move them to a shared file
class RESERVED_GRAPH_NAME(Enum):
    INPUTS_GRAPH_NAME = "inputs"
    OUTPUTS_GRAPH_NAME = "outputs"


class RESERVED_OP_NAME(Enum):
    CONSTANT_OP_STR = "prim::Constant"
    UNKNOWN_OP_STR = "UnSpecified"
    IO_OP_STR = "IO Node"


class Converter:
    """
    To convert from EDIR to TensorBoard objects
    """

    def __init__(self, op_graph: OperatorGraph) -> None:
        self.op_graph = op_graph

    def convert(
        self,
    ) -> Tuple[GraphDef, RunMetadata]:
        """
        API to convert an OperatorGraph object to a TB GraphDef object and a TB RunMetadata object
        """

        # We ignore the top level scope name (thus parent_scope_name is an empty string), otherwise TB
        # puts all the nodes into a single module at startup
        nodes_with_scoped_names = self._flatten_nodes(
            parent_scope_name="", op_graph=self.op_graph
        )

        graph_def_nodes = []
        node_stats_aggregated = defaultdict(list)
        for node in nodes_with_scoped_names:
            node, tagged_stats = self._convert_on_node_level(node=node)
            graph_def_nodes.append(node)
            for tag, stats in tagged_stats.items():
                node_stats_aggregated[tag].append(stats)

        run_metadata = {}
        for tag, aggregated_stats in node_stats_aggregated.items():
            dev_stats = DeviceStepStats(
                device="/device:CPU:0", node_stats=aggregated_stats
            )
            run_metadata[tag] = RunMetadata(step_stats=StepStats(dev_stats=[dev_stats]))

        graph_def = GraphDef(node=graph_def_nodes)

        # pyre-fixme[7]: Expected `Tuple[GraphDef, RunMetadata]` but got
        #  `Tuple[GraphDef, Dict[typing.Any, typing.Any]]`.
        return (graph_def, run_metadata)

    def _flatten_nodes(
        self, parent_scope_name: str, op_graph: OperatorGraph
    ) -> List[Node]:
        """
        Recursively search all nodes in the op_graph. Prepend scope to the name of each node.
        The hierarchy of nodes is kept through the scope.
        """

        nodes_with_scoped_name: List[Node] = []
        for element in op_graph.elements:
            if isinstance(element, Node):
                # If the element is a Node, prepend scope name to node name
                element.name = "/".join([parent_scope_name, element.name])
                # Append the node to the nodes_with_scoped_name list
                nodes_with_scoped_name.append(element)
            elif isinstance(element, OperatorGraph):
                # If the element is an OperatorGraph, flatten it recursively
                sub_nodes_with_scoped_name = self._flatten_nodes(
                    parent_scope_name="/".join(
                        filter(None, [parent_scope_name, element.graph_name])
                    ),
                    op_graph=element,
                )
                # Add the sub-nodes to the nodes_with_scoped_name list
                nodes_with_scoped_name += sub_nodes_with_scoped_name
            else:
                raise Exception("Unexpected element type {}".format(type(element)))

        return nodes_with_scoped_name

    def _convert_on_node_level(
        self, node: Node
    ) -> Tuple[NodeDef, Dict[str, NodeExecStats]]:
        # Adjustments based on operation types
        if isinstance(node, OperatorNode):
            op = node.op
        elif isinstance(node, ValueNode):
            if node.name.startswith(
                RESERVED_GRAPH_NAME.INPUTS_GRAPH_NAME.value + "/"
            ) or node.name.startswith(
                RESERVED_GRAPH_NAME.OUTPUTS_GRAPH_NAME.value + "/"
            ):
                op = RESERVED_OP_NAME.IO_OP_STR.value
            else:
                op = RESERVED_OP_NAME.CONSTANT_OP_STR.value
        else:
            op = RESERVED_OP_NAME.UNKNOWN_OP_STR.value

        node_metadata = node.metadata or {}
        stack_trace = node_metadata.get(RESERVED_METADATA_ARG.STACK_TRACE.value)

        # Create a node object. The object itself has no run stats
        node_def = NodeDef(
            name=node.name,
            op=op,
            input=self._parse_inputs(node.inputs),
            attr=self._parse_attr(
                output_shapes=node.output_shapes, stack_trace=stack_trace
            ),
        )

        # Then create node execution stats object, which is associated with the node object through node_name
        debug_handle = node_metadata.get(RESERVED_METADATA_ARG.DEBUG_HANDLE.value)
        node_stats = {}
        if debug_handle is not None:
            metrics = node_metadata.get(RESERVED_METADATA_ARG.METRICS_KEYWORD.value)
            if metrics is not None:
                for key, value in metrics.items():
                    if value is None:
                        continue
                    node_stats[key] = NodeExecStats(
                        node_name=node.name,
                        all_start_micros=1,
                        all_end_rel_micros=1
                        if int(value * 1000) == 0
                        else int(value * 1000),
                    )

        return node_def, node_stats

    def _parse_inputs(self, inputs: Optional[List[Node]]) -> List[str]:
        if inputs is None:
            return []
        return [node.name for node in inputs]

    def _parse_attr(
        self,
        output_shapes: Optional[List[List[int]]] = None,
        stack_trace: Optional[str] = None,
    ) -> Dict[str, AttrValue]:
        """Creates a dict of objects matching
        https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/attr_value.proto
        specifically designed for a NodeDef. The values have been
        reverse engineered from standard TensorBoard logged data.
        """
        attr_values_dict = {}
        # Parse out output shapes
        if output_shapes is not None and len(output_shapes) > 0:
            shape_protos = []
            for output_shape in output_shapes:
                shape_protos += [self._tensor_shape_proto(output_shape)]
            attr_values_dict["_output_shapes"] = AttrValue(
                list=AttrValue.ListValue(shape=shape_protos)
            )

        # Parse out stack trace
        if stack_trace:
            attr_values_dict[RESERVED_METADATA_ARG.STACK_TRACE.value] = AttrValue(
                s=stack_trace.encode(encoding="utf_8")
            )

        return attr_values_dict

    def _tensor_shape_proto(self, outputsize: List[int]) -> TensorShapeProto:
        """Creates an object matching
        https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/tensor_shape.proto
        """
        return TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in outputsize])
