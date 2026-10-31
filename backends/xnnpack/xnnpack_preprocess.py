# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, replace
from typing import Dict, final, List, Tuple

import torch

from executorch.backends.xnnpack._passes import XNNPACKPassManager
from executorch.backends.xnnpack._passes.channels_last_tagged_reshape_pass import (
    ChannelsLastTaggedReshapePass,
)
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack.operators.node_visitor import get_node_visitors

from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    ConstantDataOffset,
    PerTensorQuant,
    XNNGraph,
    XNNQuantizedTensorValue,
    XNNStaticTranspose,
    XNode,
    XValue,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_serialize import (
    serialize_xnnpack_binary,
)
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.backends.xnnpack.utils.utils import is_param_node

from executorch.backends.xnnpack.utils.xnnpack_constants import (
    XNN_INVALID_VALUE_ID,
    XNN_VALUE_FLAG_EXTERNAL_INPUT,
    XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
)
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)

from executorch.exir.tensor import dim_order_from_stride
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier
from torch.export.exported_program import ExportedProgram

DEFAULT_DEBUG_HANDLE = 65535

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@dataclass
class ExternalMeta:
    external_id: int
    io_type: int


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
                    if output_node in node_to_external_map:
                        raise RuntimeError(
                            f"Output node '{output_node}' is already in the inputs. "
                            "This is likely due to pass through arguments, which are not supported in XNNPACK Delegate."
                        )
                    node_to_external_map[output_node] = ExternalMeta(
                        external_id=len(node_to_external_map),
                        io_type=XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                    )
    return node_to_external_map


def get_external_dim_orders(
    exported_program: ExportedProgram,
) -> List[Tuple[int, ...]]:
    """
    The dim order the runtime gives each external, in external id order (inputs,
    then outputs). Read before the XNNPACK passes, which can change an output's
    layout, and from strides like the emitter. NHWC externals are left to
    ChannelsLastTaggedReshapePass, so they get the default order here.
    """
    graph = exported_program.graph_module.graph
    nodes = [
        node
        for node in graph.nodes
        if node.op == "placeholder" and not is_param_node(exported_program, node)
    ]
    nodes += list(graph.output_node().args[0])
    dim_orders = []
    for node in nodes:
        val = node.meta.get("val", None)
        if not isinstance(val, torch.Tensor):
            raise RuntimeError(f"XNNPACK external {node} has no tensor value")
        if ChannelsLastTaggedReshapePass.is_nhwc_tensor(val):
            dim_orders.append(tuple(range(val.dim())))
        else:
            dim_orders.append(
                tuple(int(d) for d in dim_order_from_stride(val.stride(), val.shape))
            )
    return dim_orders


def define_externals_in_memory_order(
    xnn_graph: XNNGraph, dim_orders: List[Tuple[int, ...]]
) -> None:
    """
    XNNExecutor shapes external tensors in dim order, so declare the ones whose
    dim order is not the default in memory order, transposed to and from the
    logical layout the graph is built in.
    """
    prologue, epilogue = [], []
    for xvalue in list(xnn_graph.xvalues):
        union = xvalue.xvalue_union
        quantized = isinstance(union, XNNQuantizedTensorValue)
        logical = union.tensor_value if quantized else union
        is_input = logical.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT
        if not is_input and not logical.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT:
            continue
        dim_order = dim_orders[logical.external_id]
        if dim_order == tuple(range(len(dim_order))):
            continue

        # Memory dim j is logical dim dim_order[j].
        memory_id = len(xnn_graph.xvalues)
        memory = replace(
            logical, dims=[logical.dims[d] for d in dim_order], id_out=memory_id
        )
        if quantized:
            # Activations are quantized per tensor, so the transpose keeps the
            # quant params as they are.
            if not isinstance(union.quant_params, PerTensorQuant):
                raise RuntimeError(
                    f"XNNPACK external {logical.external_id} with dim order "
                    f"{dim_order} must be quantized per tensor"
                )
            memory = XNNQuantizedTensorValue(
                tensor_value=memory, quant_params=union.quant_params
            )
        xnn_graph.xvalues.append(XValue(xvalue_union=memory))
        logical.external_id = XNN_INVALID_VALUE_ID
        logical.flags = 0

        if is_input:
            perm = [dim_order.index(i) for i in range(len(dim_order))]
            input_id, output_id, nodes = memory_id, logical.id_out, prologue
        else:
            perm = list(dim_order)
            input_id, output_id, nodes = logical.id_out, memory_id, epilogue
        transpose = XNNStaticTranspose(
            num_dims=len(dim_order),
            perm=perm,
            input_id=input_id,
            output_id=output_id,
            flags=0,
        )
        nodes.append(XNode(xnode_union=transpose, debug_handle=DEFAULT_DEBUG_HANDLE))
    xnn_graph.xnodes = prologue + xnn_graph.xnodes + epilogue


@final
class XnnpackBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        named_data_store = NamedDataStore()
        xnnpack_edge_compile_config = get_xnnpack_edge_compile_config()

        # Need to wrap EP here because xnnpack does addmm to linear
        # transforms. This makes resulting graph not aten compliant
        # as aten.linear is not a core aten op.
        # Ideal fix would be to have XNNPACK verifier that bypass
        # most checks but the base Verifier itself has some strict changes
        # and to bypass those, we would basically copy what EdgeDialectVerifier
        # does. So for now instead of copy pasting that, just instantiate
        # EdgeDialectVerifier, but disable it.
        # TODO (task link) to implement NullVerifier or something similar
        ep = ExportedProgram(
            root=edge_program.graph_module,
            graph=edge_program.graph,
            graph_signature=edge_program.graph_signature,
            state_dict=edge_program.state_dict,
            range_constraints=edge_program.range_constraints,
            module_call_graph=edge_program.module_call_graph,
            example_inputs=edge_program.example_inputs,
            constants=edge_program.constants,
            verifiers=[
                EXIREdgeDialectVerifier(
                    edge_compile_config=xnnpack_edge_compile_config, class_only=True
                )
            ],
        )

        passes = []
        for spec in compile_specs:
            if spec.key == "dqlinear_partitioner":
                passes.append(ConvertToLinearPass)

        passes = passes if len(passes) > 0 else None
        external_dim_orders = get_external_dim_orders(ep)
        # XNNPACK Delegate Specific Passes
        ep = XNNPACKPassManager(ep, passes=passes).transform()
        graph_module = ep.graph_module

        node_to_external_map = generate_node_to_external_map(ep, graph_module)
        assert len(external_dim_orders) == len(node_to_external_map), (
            f"{len(external_dim_orders)} externals before the XNNPACK passes, "
            f"{len(node_to_external_map)} after"
        )

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
            constant_data=[ConstantDataOffset(0, 0)],
        )

        constant_data_bytes = bytearray()
        node_visitors = get_node_visitors(ep, node_to_external_map, named_data_store)

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

        define_externals_in_memory_order(xnnpack_graph, external_dim_orders)
        return PreprocessResult(
            processed_bytes=serialize_xnnpack_binary(
                xnnpack_graph, constant_data_bytes
            ),
            debug_handle_map={},
            data_store_output=named_data_store.get_named_data_store_output(),
        )
