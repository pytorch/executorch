# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Dict, final, List

import torch

from executorch.backends.xnnpack._passes import XNNPACKPassManager
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack._passes.tag_implicit_q_dq_pass import (
    TagImplicitQDqPass,
)
from executorch.backends.xnnpack.operators.node_visitor import get_node_visitors

from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    ConstantDataOffset,
    XNNGraph,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_serialize import (
    serialize_xnnpack_binary,
)
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.backends.xnnpack.utils.utils import is_param_node

from executorch.backends.xnnpack.utils.xnnpack_constants import (
    XNN_VALUE_FLAG_EXTERNAL_INPUT,
    XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
)

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
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
                    node_to_external_map[output_node] = ExternalMeta(
                        external_id=len(node_to_external_map),
                        io_type=XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                    )
    return node_to_external_map


def assert_default_dim_order(edge_graph_module: torch.fx.GraphModule) -> None:
    for node in edge_graph_module.graph.nodes:
        if node.op != "placeholder":
            continue

        # We expect the default dim order for all tensor-like inputs i.e. inputs, buffers, and params
        t = node.meta.get("val", None)
        if t is not None and getattr(t, "dim_order", None) is not None:
            default_dim_order = tuple(range(t.dim()))
            if t.dim_order() != default_dim_order:
                raise RuntimeError(
                    f"XNNPACK backend only supports contiguous memory format for inputs."
                    f"Expecting dim_order: {default_dim_order}, but got {node.meta['val'].dim_order()} for a placeholder node {node}."
                )


@final
class XnnpackBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:

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
                passes.append(TagImplicitQDqPass)

        passes = passes if len(passes) > 0 else None
        # XNNPACK Delegate Specific Passes
        ep = XNNPACKPassManager(ep, passes=passes).transform()
        graph_module = ep.graph_module

        node_to_external_map = generate_node_to_external_map(ep, graph_module)

        # Make sure all inputs are contiguous_format or NCHW or default dim order
        assert_default_dim_order(graph_module)

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
        node_visitors = get_node_visitors(ep, node_to_external_map, constant_data_bytes)

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
        return PreprocessResult(
            processed_bytes=serialize_xnnpack_binary(
                xnnpack_graph, constant_data_bytes
            ),
            debug_handle_map={},
        )
