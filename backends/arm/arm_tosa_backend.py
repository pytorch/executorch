# Copyright 2023-2024 Arm Limited and/or its affiliates.
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
from typing import final, List

import serializer.tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import get_node_visitors
from executorch.backends.arm.operators.op_output import process_output
from executorch.backends.arm.operators.op_placeholder import process_placeholder
from executorch.backends.arm._passes.arm_pass_manager import (
    ArmPassManager,
)  # usort: skip
from executorch.backends.arm.tosa_utils import (
    dbg_fail,
    dbg_tosa_dump,
    process_call_function,
)
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram

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
class ArmTOSABackend(BackendDetails):
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
        logger.info(f"{ArmTOSABackend.__name__} preprocess")

        # if a debug/test build capture output files from TOSA stage
        artifact_path = None
        output_format = ""
        for spec in compile_spec:
            if spec.key == "debug_artifact_path":
                artifact_path = spec.value.decode()
            if spec.key == "output_format":
                output_format = spec.value.decode()

        # Check that the output format is set in the compile spec
        if output_format != "tosa":
            raise RuntimeError("TOSA compile spec is missing")

        # Converted output for this subgraph, serializer needs path early as it emits
        # const data directly. Path created and data written only in debug builds.

        tosa_graph = ts.TosaSerializer(artifact_path)

        graph_module = ArmPassManager().transform_to_backend_pipeline(
            exported_program=edge_program, compile_spec=compile_spec
        )

        node_visitors = get_node_visitors(edge_program)

        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                process_call_function(node, tosa_graph, node_visitors)
            elif node.op == "placeholder":
                process_placeholder(node, tosa_graph, edge_program)
            elif node.op == "output":
                process_output(node, tosa_graph)
            else:
                # This will only happen if an unpartitioned graph is passed without
                # any checking of compatibility.
                dbg_fail(node, tosa_graph, artifact_path)

        # TODO: It would be awesome if this dump could somehow be done on top level and not here.
        # Problem is that the desc.json has to be created on the tosa_graph object, which we can't
        # access from top level.
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
        ArmTOSABackend, since we first want to use the ArmTOSABackend to generate
        the TOSA flatbuffer representation as an intermediate step. The TOSA
        flatbuffer can then be consumed by the backend targetting specific
        hardware.
        """
        tosa_compile_spec = []
        tosa_compile_spec.append(CompileSpec("output_format", "tosa".encode()))

        # Copy everything that's TOSA generic
        tosa_backend_compile_spec_keys = [
            "debug_artifact_path",
            "permute_memory_format",
        ]

        for spec in compile_spec:
            if spec.key in tosa_backend_compile_spec_keys:
                tosa_compile_spec.append(CompileSpec(spec.key, spec.value))

        return tosa_compile_spec
