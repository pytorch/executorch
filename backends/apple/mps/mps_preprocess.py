#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#
import logging
from typing import ClassVar, Dict, final, List, Tuple

import torch

from executorch.backends.apple.mps.operators.node_visitor import (
    get_node_visitors,
    NodeVisitor,
    process_output_node,
    process_placeholder_nodes,
)

from executorch.backends.apple.mps.serialization.mps_graph_schema import (
    Buffer,
    DataSegment,
    MPSGraph,
    MPSTensor,
    OpType,
)

from executorch.backends.apple.mps.serialization.mps_graph_serialize import (
    convert_to_flatbuffer,
)
from executorch.exir._serialize._program import Cord

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch.export.exported_program import ExportedProgram

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


@final
class MPSBackend(BackendDetails):
    @staticmethod
    def slice_len_max(s):
        assert s.start is not None
        assert s.stop is not None
        step = 1
        if s.step is not None:
            step = s.step
        return max((s.stop - s.start) // step, 1)

    MAGIC_IX: ClassVar[slice] = slice(4, 8)
    DATA_SEGMENT_OFFSET_IX: ClassVar[slice] = slice(8, 16)
    DATA_SEGMENT_SIZE_IX: ClassVar[slice] = slice(16, 24)

    # magic bytes that should be at the beginning of the header
    EXPECTED_MAGIC: ClassVar[bytes] = b"MP00"
    # The length of the header in bytes
    EXPECTED_LENGTH: ClassVar[int] = (
        4
        + slice_len_max(MAGIC_IX)
        + slice_len_max(DATA_SEGMENT_OFFSET_IX)
        + slice_len_max(DATA_SEGMENT_SIZE_IX)
    )

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
            graph_type=OpType.mps_graph,
            constant_segment=DataSegment(0, 0),
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

        segment_data, mps_graph = _extract_constant_segment(mps_graph)
        if logging.DEBUG >= logging.root.level:
            pretty_print(mps_graph)

        # Add to aggregate segments cord with padding.
        padding_length = _padding_required(len(segment_data), 16)
        if padding_length > 0:
            segment_data.append(b"\x00" * padding_length)

        # Combine mps_graph with segment data
        combined = Cord()
        graph_bytes = convert_to_flatbuffer(mps_graph)

        data_segment_offset: int = MPSBackend.EXPECTED_LENGTH
        data_segment_offset = data_segment_offset + len(graph_bytes)

        graph_padding_length = _padding_required(data_segment_offset, 16)
        data_segment_offset = data_segment_offset + graph_padding_length
        data_segment_size = len(segment_data)

        data: bytes = (
            b"\x00\x00\x00\x00"
            + MPSBackend.EXPECTED_MAGIC
            + data_segment_offset.to_bytes(8, byteorder="little")
            + data_segment_size.to_bytes(8, byteorder="little")
        )
        assert len(data) == MPSBackend.EXPECTED_LENGTH

        combined.append(data)
        combined.append(graph_bytes)

        if graph_padding_length > 0:
            combined.append(b"\x00" * graph_padding_length)
        # Append the segment data to the end of the mps graph
        combined.append(segment_data)

        return PreprocessResult(processed_bytes=bytes(combined))

    @staticmethod
    def handle_call_function(
        _: ExportedProgram,
        node_visitors: Dict[str, NodeVisitor],
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        logging.info(f"Visiting: {node}, {node.target.__name__}")

        if (
            "delegation_tag" in node.meta
            and "metal_kernel" in node.meta["delegation_tag"]
        ):
            logging.info(
                f"Node '{node.target.__name__}' was marked as a Metal kernel by the MPSPartitioner!"
            )
            mps_graph.graph_type = OpType.metal_kernel

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
        # Constants are handled directly when visiting the nodes.
        pass

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


def _padding_required(offset: int, alignment: int) -> int:
    """Returns the padding required to align `offset` to `alignment`."""
    remainder: int = offset % alignment
    if remainder != 0:
        return alignment - remainder
    return 0


def _extract_constant_segment(mps_graph: MPSGraph) -> Tuple[Cord, MPSGraph]:
    """Extracts the constant segment from the MPSGraph and returns the updated MPSGraph along with the segment data."""
    # Note that the beginning of the segment data is not aligned. Need to handle out of this call.
    segment_data = Cord()
    offset = 0
    for i in range(len(mps_graph.mps_values)):
        tensor = mps_graph.mps_values[i]
        if tensor.constant_buffer_size > 0:
            # Notice that buffer is already force aligned so we don't need to pad it
            segment_data.append(tensor.constant_buffer.storage)

            # Reset buffer to empty
            tensor.constant_buffer = Buffer(storage=b"")
            # Update segment offset
            tensor.segment_offset = offset
            offset += tensor.constant_buffer_size

    return segment_data, mps_graph


def tensor_to_str(mps_tensor: MPSTensor):
    tensor_str = "MPSTensor("
    tensor_str += "datatype=" + str(mps_tensor.datatype) + ", "
    tensor_str += "num_dims=" + str(mps_tensor.num_dims) + ", "
    tensor_str += "dims=" + str(mps_tensor.dims) + ", "
    tensor_str += "constant_buffer_size=" + str(mps_tensor.constant_buffer_size) + ", "
    tensor_str += "segment_offset=" + str(mps_tensor.segment_offset)
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
    logging.info(f" Constant segment: {mps_graph.constant_segment}")
