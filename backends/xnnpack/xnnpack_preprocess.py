# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import logging
from dataclasses import dataclass
from typing import Dict, final, List

import torch

from executorch.backends.transforms import get_shape
from executorch.backends.xnnpack.operators.node_visitor import get_node_visitors

from executorch.backends.xnnpack.passes import XNNPACKPassManager

from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    Buffer,
    PerChannelQuant,
    PerTensorQuant,
    XNNDatatype,
    XNNGraph,
    XNNQuantizedTensorValue,
    XNNTensorValue,
    XValue,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_serialize import (
    convert_to_flatbuffer,
)
from executorch.backends.xnnpack.utils.utils import is_param_node

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier
from torch._export.exported_program import ExportedProgram

XNN_VALUE_FLAG_NON_EXTERNAL = 0
XNN_VALUE_FLAG_EXTERNAL_INPUT = 1
XNN_VALUE_FLAG_EXTERNAL_OUTPUT = 2
XNN_FLAG_TRANSPOSE_WEIGHTS = 1
XNN_INVALID_VALUE_ID = 2**32 - 1
XNN_TYPE_MAP = {
    torch.float32: XNNDatatype.xnn_datatype_fp32,
    torch.uint8: XNNDatatype.xnn_datatype_quint8,
    torch.int8: XNNDatatype.xnn_datatype_qint8,
    torch.int32: XNNDatatype.xnn_datatype_qint32,
}
DEFAULT_DEBUG_HANDLE = 65535

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@dataclass
class ExternalMeta:
    external_id: int
    io_type: int


def node_to_xvalue(
    node: torch.fx.Node,
    constant_buffer_idx: int,
    external_id: int,
    flags: int,
    id_out: int,
    dq_datatype=XNNDatatype.xnn_datatype_invalid,
) -> XValue:
    node_val = node.meta["val"]
    node_value = XValue(
        xvalue_union=XNNTensorValue(
            datatype=XNN_TYPE_MAP[node_val.dtype],
            num_dims=node_val.dim(),
            dims=get_shape(node),
            constant_buffer_idx=constant_buffer_idx,
            external_id=external_id,
            flags=flags,
            id_out=id_out,
            dq_datatype=dq_datatype,
        )
    )
    return node_value


def node_to_per_tensor_quantized_xvalue(
    node: torch.fx.Node,
    dtype: torch.dtype,
    constant_buffer_idx: int,
    external_id: int,
    flags: int,
    id_out: int,
    scale: float,
    zero_point: int,
) -> XValue:
    node_val = node.meta["val"]
    node_xvalue = XNNTensorValue(
        datatype=XNN_TYPE_MAP[dtype],
        num_dims=node_val.dim(),
        dims=get_shape(node),
        constant_buffer_idx=constant_buffer_idx,
        external_id=external_id,
        flags=flags,
        id_out=id_out,
        dq_datatype=XNNDatatype.xnn_datatype_invalid,  # always invalid
    )

    per_tensor_quantized_params = PerTensorQuant(scale=scale, zero_point=zero_point)
    quantized_node_val = XValue(
        xvalue_union=XNNQuantizedTensorValue(
            tensor_value=node_xvalue,
            quant_params=per_tensor_quantized_params,
        )
    )
    return quantized_node_val


def node_to_per_channel_quantized_xvalue(
    node: torch.fx.Node,
    dtype: torch.dtype,
    constant_buffer_idx: int,
    external_id: int,
    flags: int,
    id_out: int,
    channel_dim: int,
    scale: torch.Tensor,
) -> XValue:
    node_val = node.meta["val"]
    assert dtype == torch.torch.int8
    node_xvalue = XNNTensorValue(
        datatype=XNNDatatype.xnn_datatype_qcint8,  # HACK: XNN_TYPE_MAP[dtype],
        num_dims=node_val.dim(),
        dims=get_shape(node),
        constant_buffer_idx=constant_buffer_idx,
        external_id=external_id,
        flags=flags,
        id_out=id_out,
        dq_datatype=XNNDatatype.xnn_datatype_invalid,  # always invalid
    )

    per_channel_quantized_params = PerChannelQuant(
        scale=scale.tolist(), channel_dim=channel_dim
    )
    quantized_node_val = XValue(
        xvalue_union=XNNQuantizedTensorValue(
            tensor_value=node_xvalue,
            quant_params=per_channel_quantized_params,
        )
    )
    return quantized_node_val


def generate_node_to_external_map(
    exported_program: ExportedProgram,
    edge_graph_module: torch.fx.GraphModule,
) -> Dict[torch.fx.Node, ExternalMeta]:
    node_to_external_map = {}
    for node in edge_graph_module.graph.nodes:
        # The order in which we visit the placeholder node is same as the *args
        # order for the forward(*args) signature for this gm. Using the order of
        # the nodes as external_id to extract the right arg from *args at runtime
        #
        # Removing parameters/buffers since they will disappear from the signature
        # at runtime
        if node.op == "placeholder" and not is_param_node(exported_program, node):
            node_to_external_map[node] = ExternalMeta(
                external_id=len(node_to_external_map),
                io_type=XNN_VALUE_FLAG_EXTERNAL_INPUT,
            )
    for node in edge_graph_module.graph.nodes:
        if node.op == "output":
            for output_nodes in node.args:
                for output_node in output_nodes:
                    node_to_external_map[output_node] = ExternalMeta(
                        external_id=len(node_to_external_map),
                        io_type=XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                    )
    return node_to_external_map


@final
class XnnpackBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        ep = copy.deepcopy(edge_program)
        # Need to wrap EP here because xnnpack does addmm to linear
        # transforms. This makes resulting graph not aten comliant
        # as aten.linear is not a core aten op.
        # Ideal fix would be to have XNNPACK verifier that bypass
        # most checks but the base Verifier itself has some strict changes
        # and to bypass those, we would basicallyy copy what EdgeDialectVerifier
        # does. So for now instead of copy pasting that, just instantiate
        # EdgeDialectVerifier, but disable it.
        # TODO (task link) to implement NullVerifier or something similar
        ep = ExportedProgram(
            ep.graph_module,
            ep.graph,
            ep.graph_signature,
            ep.state_dict,
            ep.range_constraints,
            ep.equality_constraints,
            copy.deepcopy(ep.module_call_graph),
            ep.example_inputs,
            verifier=EXIREdgeDialectVerifier(
                check_edge_ops=False, enable=False, class_only=True
            ),
        )

        # XNNPACK Delegate Specific Passes
        ep = XNNPACKPassManager(ep).transform()
        graph_module = ep.graph_module

        node_to_external_map = generate_node_to_external_map(ep, graph_module)

        # TODO retrace the graph module to lift the new params may have
        # been added to the graph in passes

        vals_to_ids = {}
        xnnpack_graph = XNNGraph(
            version="0",
            xnodes=[],
            xvalues=[],
            num_externs=len(node_to_external_map),
            input_ids=[],
            output_ids=[],
            constant_buffer=[Buffer(storage=b"")],
            mem_buffer_sizes=[0],
        )

        node_visitors = get_node_visitors(ep, node_to_external_map)

        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                logger.info(f"Visiting: {node}, {node.target.__name__}")
                if node.target.__name__ in node_visitors:
                    node_visitors[node.target.__name__].define_node(
                        node,
                        xnnpack_graph,
                        vals_to_ids,
                        node.meta.get("debug_handle", DEFAULT_DEBUG_HANDLE),
                    )
                else:
                    raise RuntimeError(
                        f"For {node}, {node.op}:{node.target.__name__} is not supported in XNNPACK Delegate"
                    )
            elif node.op in [
                "get_attr",
                "placeholder",
                "output",
            ]:
                continue
            else:
                raise RuntimeError(f"{node.op} is not supported in XNNPACK")
        return PreprocessResult(processed_bytes=convert_to_flatbuffer(xnnpack_graph))
