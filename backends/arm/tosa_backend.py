# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

#
# Main implementation of AoT flow to partition and preprocess for Arm target
# backends. Converts via TOSA as an intermediate form supported by AoT and
# JIT compiler flows.
#
import logging
import os
from typing import cast, final, List

import serializer.tosa_serializer as ts  # type: ignore

from executorch.backends.arm.arm_backend import get_tosa_spec
from executorch.backends.arm.operators.node_visitor import get_node_visitors
from executorch.backends.arm._passes.arm_pass_manager import (
    ArmPassManager,
)  # usort: skip
from executorch.backends.arm.process_node import (
    process_call_function,
    process_output,
    process_placeholder,
)
from executorch.backends.arm.tosa_utils import dbg_fail, dbg_tosa_dump
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram
from torch.fx import Node

# TOSA backend debug functionality
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
TOSA_DBG_VERBOSE = os.environ.get("TOSA_DBG_VERBOSE") == "1"
if TOSA_DBG_VERBOSE:
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)


def _get_first_delegation_tag(graph_module) -> str | None:
    """Get the first delegation tag from the graph_module or return None."""
    for node in graph_module.graph.nodes:
        tag = node.meta.get("delegation_tag")
        if tag:
            return tag

    logger.debug("No delegation tag found in partition.")
    return None


@final
class TOSABackend(BackendDetails):
    """
    BackendDetails subclass for lowering to TOSA.
    Is used either by itself to get to a TOSA representation, or with composition
    to be used as a separate step to target TOSA compliant hardware.
    """

    @staticmethod
    def preprocess(  # noqa: C901
        edge_program: ExportedProgram,
        compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        # if a debug/test build capture output files from TOSA stage
        artifact_path = None
        output_format = ""
        compile_flags = []
        input_order = []
        for spec in compile_spec:
            if spec.key == "debug_artifact_path":
                artifact_path = spec.value.decode()
            if spec.key == "output_format":
                output_format = spec.value.decode()
            if spec.key == "compile_flags":
                compile_flags.append(spec.value.decode())
            if spec.key == "input_order":
                input_order = list(map(int, spec.value.decode().split(",")))

        # Check that the output format is set correctly in the compile spec
        assert output_format == "tosa", "output format must be tosa"

        tosa_spec = get_tosa_spec(compile_spec)
        assert (
            tosa_spec is not None
        ), "TOSA backend needs a TOSA version specified in the CompileSpec!"

        logger.info(f"Converting ExportedProgram to TOSA: {tosa_spec}")

        # Converted output for this subgraph, serializer needs path early as it emits
        # const data directly. Path created and data written only in debug builds.
        tosa_graph = ts.TosaSerializer(artifact_path)
        graph_module = ArmPassManager(tosa_spec).transform_to_backend_pipeline(  # type: ignore
            exported_program=edge_program
        )

        node_visitors = get_node_visitors(edge_program, tosa_spec)
        input_count = 0
        for node in graph_module.graph.nodes:
            node = cast(Node, node)
            if node.op == "call_function":
                process_call_function(node, tosa_graph, node_visitors, tosa_spec)
            elif node.op == "placeholder":
                process_placeholder(node, tosa_graph, edge_program, tosa_spec)
                if node.name in edge_program.graph_signature.user_inputs:
                    input_count += 1
            elif node.op == "output":
                process_output(node, tosa_graph)
            else:
                # This will only happen if an unpartitioned graph is passed without
                # any checking of compatibility.
                dbg_fail(node, tosa_graph, artifact_path)

        if len(input_order) > 0:
            if input_count != len(input_order):
                raise RuntimeError(
                    "The rank of the input order is not equal to amount of input tensors"
                )

        if artifact_path:
            tag = _get_first_delegation_tag(graph_module)
            dbg_tosa_dump(
                tosa_graph,
                artifact_path,
                suffix="{}".format(f"_{tag}" if tag else ""),
            )

        # Serialize and return the TOSA flatbuffer.
        binary = bytes(tosa_graph.serialize())

        return PreprocessResult(processed_bytes=binary)

    @staticmethod
    def filter_tosa_compile_specs(
        compile_spec: List[CompileSpec],
    ) -> List[CompileSpec]:
        """
        Filter out the CompileSpec elements relevant for the TOSA backend.
        This is needed to compose a backend targetting hardware IP with the
        TOSABackend, since we first want to use the TOSABackend to generate
        the TOSA flatbuffer representation as an intermediate step. The TOSA
        flatbuffer can then be consumed by the backend targetting specific
        hardware.
        """
        tosa_compile_spec = []
        tosa_compile_spec.append(CompileSpec("output_format", "tosa".encode()))

        # Copy everything that's TOSA generic
        tosa_backend_compile_spec_keys = [
            "tosa_spec",
            "debug_artifact_path",
        ]

        for spec in compile_spec:
            if spec.key in tosa_backend_compile_spec_keys:
                tosa_compile_spec.append(CompileSpec(spec.key, spec.value))

        return tosa_compile_spec
