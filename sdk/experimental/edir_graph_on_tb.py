# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.step_stats_pb2 import DeviceStepStats, StepStats
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.versions_pb2 import VersionDef
from torch.utils.tensorboard import SummaryWriter

############ 1. Define the executorch debug intermediate representation (EDIR) ############
# Representation of a single "op" or node within a ModelGraph
@dataclass
class OperatorNode:
    name: str
    op: str
    inputs: Optional[List["OperatorNode"]] = None
    # Generic Op level metadata
    metadata: Optional[Dict[str, str]] = None


# Generic Representation of a operator graph with metadata
@dataclass
class OperatorGraph:
    nodes: List[OperatorNode]
    # Graph Level Metadata
    metadata: Optional[Dict[str, str]] = None


############ 2. Create a simple EDIR graph to be visualized ############
# Node input/x
input_node_x = OperatorNode(
    name="input/x",
    op="IO Node",
    metadata={"_output_shapes": "3,4", "attr": ""},
)

# Node input/y
input_node_y = OperatorNode(
    name="input/y",
    op="IO Node",
    metadata={
        "_output_shapes": "3,4",
        "attr": "",
    },
)

# Node input/alpha
input_node_alpha = OperatorNode(
    name="input/alpha",
    op="IO Node",
    metadata={
        "_output_shapes": "1",
        "attr": "",
    },
)

# Node AddModule/6
s = "{}"
addmodule_node_6 = OperatorNode(
    name="AddModule/6",
    op="aten::mul",
    inputs=[input_node_y, input_node_alpha],
    metadata={
        "_output_shapes": "3,4",
        "attr": s,
    },
)

# Node AddModule/7
s = "{ value : 1}"
addmodule_node_7 = OperatorNode(
    name="AddModule/7",
    op="prim::Constant",
    metadata={"attr": s},
)

# Node AddModule/8
s = "{}"
addmodule_node_8 = OperatorNode(
    name="AddModule/8",
    op="aten::add",
    inputs=[input_node_x, addmodule_node_6, addmodule_node_7],
    metadata={
        "_output_shapes": "3,4",
        "attr": s,
    },
)

# Node output/output.1
s = ""
output_node_1 = OperatorNode(
    name="output/output.1",
    op="IO Node",
    inputs=[addmodule_node_8],
    metadata={"attr": s},
)

operator_graph = OperatorGraph(
    nodes=[
        input_node_x,
        input_node_y,
        input_node_alpha,
        addmodule_node_6,
        addmodule_node_7,
        addmodule_node_8,
        output_node_1,
    ]
)

############ 3. Parse the EDIR graph to TB format ############
def parse_inputs(inputs: Optional[List["OperatorNode"]]) -> List[str]:
    if inputs is None:
        inputs = []
    return [node.name for node in inputs]


def parse_metadata(metadata: Optional[Dict[str, str]]) -> Dict[str, Any]:
    attr = {}
    if metadata is None:
        return attr

    for k, v in metadata.items():
        if k == "_output_shapes":
            shapeproto = TensorShapeProto(
                dim=[
                    TensorShapeProto.Dim(size=d)
                    for d in [int(num) for num in v.split(",")]
                ]
            )
            attr[k] = AttrValue(list=AttrValue.ListValue(shape=[shapeproto]))
        else:
            attr[k] = AttrValue(s=v.encode(encoding="utf_8"))

    return attr


tb_nodes = []
for node in operator_graph.nodes:
    tb_nodes.append(
        NodeDef(
            name=node.name,
            op=node.op,
            input=parse_inputs(node.inputs),
            attr=parse_metadata(node.metadata),
        )
    )

stepstats = RunMetadata(
    step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0")])
)
hardcoded_graph = (
    GraphDef(node=tb_nodes, versions=VersionDef(producer=22)),
    stepstats,
)

############ 4. Write to TensorBoard ############
writer = SummaryWriter()
# Call the add_graph function that takes a TB format object describing a graph
writer._get_file_writer().add_graph(graph_profile=hardcoded_graph)
writer.close()
