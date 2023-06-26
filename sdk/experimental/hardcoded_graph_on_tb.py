# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.step_stats_pb2 import DeviceStepStats, StepStats
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.versions_pb2 import VersionDef

from torch.utils.tensorboard import SummaryWriter

# Hard-code a graph in TB proto buffer format that just has one aten add operator in it

# Node input/x
s = ""
shapeproto = TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in [3, 4]])
input_node_x = NodeDef(
    name="input/x",
    op="IO Node",
    input=[],
    attr={
        "_output_shapes": AttrValue(list=AttrValue.ListValue(shape=[shapeproto])),
        "attr": AttrValue(s=s.encode(encoding="utf_8")),
    },
)

# Node input/y
s = ""
shapeproto = TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in [3, 4]])
input_node_y = NodeDef(
    name="input/y",
    op="IO Node",
    input=[],
    attr={
        "_output_shapes": AttrValue(list=AttrValue.ListValue(shape=[shapeproto])),
        "attr": AttrValue(s=s.encode(encoding="utf_8")),
    },
)

# Node input/alpha
s = ""
shapeproto = TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in [1]])
input_node_alpha = NodeDef(
    name="input/alpha",
    op="IO Node",
    input=[],
    attr={
        "_output_shapes": AttrValue(list=AttrValue.ListValue(shape=[shapeproto])),
        "attr": AttrValue(s=s.encode(encoding="utf_8")),
    },
)

# Node AddModule/6
s = "{}"
shapeproto = TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in [3, 4]])
addmodule_node_6 = NodeDef(
    name="AddModule/6",
    op="aten::mul",
    input=["input/y", "input/alpha"],
    attr={
        "_output_shapes": AttrValue(list=AttrValue.ListValue(shape=[shapeproto])),
        "attr": AttrValue(s=s.encode(encoding="utf_8")),
    },
)

# Node AddModule/7
s = "{ value : 1}"
addmodule_node_7 = NodeDef(
    name="AddModule/7",
    op="prim::Constant",
    attr={"attr": AttrValue(s=s.encode(encoding="utf_8"))},
)

# Node AddModule/8
s = "{}"
shapeproto = TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in [3, 4]])
addmodule_node_8 = NodeDef(
    name="AddModule/8",
    op="aten::add",
    input=["input/x", "AddModule/6", "AddModule/7"],
    attr={
        "_output_shapes": AttrValue(list=AttrValue.ListValue(shape=[shapeproto])),
        "attr": AttrValue(s=s.encode(encoding="utf_8")),
    },
)

# Node output/output.1
s = ""
output_node_1 = NodeDef(
    name="output/output.1",
    op="IO Node",
    input=["AddModule/8"],
    attr={"attr": AttrValue(s=s.encode(encoding="utf_8"))},
)

nodes = [
    input_node_x,
    input_node_y,
    input_node_alpha,
    addmodule_node_6,
    addmodule_node_7,
    addmodule_node_8,
    output_node_1,
]

stepstats = RunMetadata(
    step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0")])
)
hardcoded_graph = (
    GraphDef(node=nodes, versions=VersionDef(producer=22)),
    stepstats,
)

# Write to TensorBoard
writer = SummaryWriter()
# Call the add_graph function that takes a TB format object describing a graph
writer._get_file_writer().add_graph(graph_profile=hardcoded_graph)
writer.close()
