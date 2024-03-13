#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import logging
from typing import Dict, final, List

import torch

from executorch.backends.apple.mps.operators.node_visitor import (
    get_node_visitors,
    NodeVisitor,
    process_output_node,
    process_placeholder_nodes,
)

from executorch.backends.apple.mps.serialization.mps_graph_schema import (
    MPSGraph,
    MPSTensor,
)

from executorch.backends.apple.mps.serialization.mps_graph_serialize import (
    convert_to_flatbuffer,
)
from executorch.backends.apple.mps.utils.mps_utils import is_parameter

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch._export.exported_program import ExportedProgram

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


@final
class MPSBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        # The EdgeIR nodes are processed in the following order:
        # 1. Process first the input feeds to the graph (in the same
        #    order as args from forward(*args)), and generate a unique
        #    id for each input placeholder. Each input id is appended to
        #    `input_ids` array from the FlatBuffer schema.
        # 2. Process the nodes the graph (e.g `call_function`). For each
        #    EdgeIR node, create an equivalent MPS node in the FlatBuffer,
        #    based on which the MPSGraph is constructed at runtime. During
        #    this process, any visited constant in the EdgeIR is added to the
        #    final MPS FlatBuffer schema. Each constant id is appended to the
        #    `constant_ids` FlatBuffer schema.
        # 3. After all the inputs, nodes and constants are added to the
        #    FlatBuffer graph, process the `output` nodes and add their id to
        #    the `output_ids` array in the schema.

        mps_graph = MPSGraph(
            version="0",
            mps_nodes=[],
            mps_values=[],
            input_ids=[],
            output_ids=[],
            constant_ids=[],
        )

        convert_model_to_fp16 = True
        for spec in compile_specs:
            if spec.key == "use_fp16":
                convert_model_to_fp16 = bool(list(bytes(spec.value))[0])

        logging.debug(f"Convert model to FP16: {convert_model_to_fp16}")

        node_visitors = get_node_visitors(edge_program, convert_model_to_fp16)
        if logging.DEBUG >= logging.root.level:
            edge_program.graph.print_tabular()

        process_placeholder_nodes(
            edge_program,
            edge_program.graph_module,
            mps_graph,
            node_visitors["placeholder"],
        )

        op_handler = {
            "call_function": MPSBackend.handle_call_function,
            "placeholder": MPSBackend.handle_placeholder,
            "output": MPSBackend.handle_output,
            "get_attr": MPSBackend.handle_get_attr,
        }

        for node in edge_program.graph_module.graph.nodes:
            if node.op not in op_handler:
                raise RuntimeError(f"{node.op} is not supported in MPS")
            else:
                op_handler[node.op](edge_program, node_visitors, node, mps_graph)

        if logging.DEBUG >= logging.root.level:
            pretty_print(mps_graph)

        return PreprocessResult(processed_bytes=convert_to_flatbuffer(mps_graph))

    @staticmethod
    def handle_call_function(
        _: ExportedProgram,
        node_visitors: Dict[str, NodeVisitor],
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        logging.info(f"Visiting: {node}, {node.target.__name__}")
        if node.target.__name__ in node_visitors:
            node_visitors[node.target.__name__].define_node(node, mps_graph)
        else:
            pretty_print(mps_graph)
            raise RuntimeError(
                f"For {node}, {node.op}:{node.target.__name__} is not supported in MPS delegate"
            )

    @staticmethod
    def handle_placeholder(
        edge_program: ExportedProgram,
        node_visitors: Dict[str, NodeVisitor],
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        # Handle only constants. Placeholders have already
        # been visited in `process_input_placeholders`
        if is_parameter(edge_program, node):
            node_visitors[node.op].define_tensor(node, mps_graph)

    @staticmethod
    def handle_output(
        edge_program: ExportedProgram,
        node_visitors: Dict[str, NodeVisitor],
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        for output_nodes in node.args:
            for output_node in output_nodes:
                process_output_node(output_node, mps_graph, node_visitors[node.op])

    @staticmethod
    def handle_get_attr(
        edge_program: ExportedProgram,
        node_visitors: Dict[str, NodeVisitor],
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        pass


def tensor_to_str(mps_tensor: MPSTensor):
    tensor_str = "MPSTensor("
    tensor_str += "datatype=" + str(mps_tensor.datatype) + ", "
    tensor_str += "num_dims=" + str(mps_tensor.num_dims) + ", "
    tensor_str += "dims=" + str(mps_tensor.dims) + ", "
    tensor_str += "constant_buffer_size=" + str(mps_tensor.constant_buffer_size)
    tensor_str += ")"

    return tensor_str


def pretty_print(mps_graph: MPSGraph):
    logging.info("Serialized MPSGraph:")
    logging.info(f" Version: {mps_graph.version}")
    logging.info(" MPS nodes: ")
    for i in range(len(mps_graph.mps_nodes)):
        logging.info(f"   [{i}]: {mps_graph.mps_nodes[i]}")
    logging.info(" MPS values: ")
    for i in range(len(mps_graph.mps_values)):
        logging.info(f"   [{i}]: {tensor_to_str(mps_graph.mps_values[i])}")
    logging.info(" Input ids:")
    for in_id in mps_graph.input_ids:
        logging.info(f"   {in_id}")
    logging.info(" Constant ids:")
    for constant_id in mps_graph.constant_ids:
        logging.info(f"   {constant_id}")
    logging.info(" Output ids:")
    for out_id in mps_graph.output_ids:
        logging.info(f"   {out_id}")
