# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Main implementation of AoT flow to partition and preprocess for Arm target
# backends. Converts via TOSA as an intermediate form supported by AoT and
# JIT compiler flows.
#

import logging
import os
from typing import final, List, Optional

import serializer.tosa_serializer as ts
from executorch.backends.arm.arm_vela import vela_compile
from executorch.backends.arm.operators.node_visitor import get_node_visitors
from executorch.backends.arm.operators.op_placeholder import process_placeholder
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import is_quant_node
from executorch.backends.arm.tosa_utils import (
    dbg_fail,
    dbg_tosa_dump,
    is_permute_node_before_addmm,
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


def generate_ethosu_compile_spec(
    config: str,
    system_config: Optional[str] = None,
    memory_mode: Optional[str] = None,
    extra_flags: Optional[str] = None,
    config_ini: Optional[str] = "Arm/vela.ini",
) -> List[CompileSpec]:
    """
    Generate compile spec for Ethos-U NPU
    """
    compiler_flags = [f"--accelerator-config={config}", f"--config={config_ini}"]
    if system_config is not None:
        compiler_flags.append(f"--system-config={system_config}")
    if memory_mode is not None:
        compiler_flags.append(f"--memory-mode={memory_mode}")
    if extra_flags is not None:
        compiler_flags.append(extra_flags)

    compile_spec = [
        CompileSpec("output_format", "vela".encode()),
        CompileSpec("compile_flags", " ".join(compiler_flags).encode()),
    ]

    return compile_spec


def generate_tosa_compile_spec(
    output_path: Optional[str] = None,
) -> List[CompileSpec]:
    """
    Generate compile spec for TOSA flatbuffer output
    """
    compile_spec = [
        CompileSpec("output_format", "tosa".encode()),
    ]

    if output_path is not None:
        compile_spec.append(CompileSpec("debug_tosa_path", output_path.encode()))

    return compile_spec


@final
class ArmBackend(BackendDetails):
    @staticmethod
    def preprocess(  # noqa: C901
        edge_program: ExportedProgram,
        compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        logger.info("ArmBackend::preprocess")

        # if a debug/test build capture output files from TOSA stage
        path = None
        debug_output = False
        output_format = ""
        compile_flags = []
        for spec in compile_spec:
            if spec.key == "debug_tosa_path":
                path = spec.value.decode()
                debug_output = True
            if spec.key == "output_format":
                output_format = spec.value.decode()
            if spec.key == "compile_flags":
                compile_flags.append(spec.value.decode())

        # Check that the output format is set in the compile spec
        if not output_format:
            raise RuntimeError("output format is required")

        if output_format == "vela" and len(compile_flags) == 0:
            # Not testing for compile_flags correctness here, just that they are
            # present. The compiler will give errors if they are not valid.
            raise RuntimeError("compile flags are required for vela output format")

        # Converted output for this subgraph, serializer needs path early as it emits
        # const data directly. Path created and data written only in debug builds.
        tosa_graph = ts.TosaSerializer(path)

        node_visitors = get_node_visitors(edge_program)

        for node in edge_program.graph.nodes:
            if node.op == "call_function":
                # Unpack arguments and convert
                inputs = []
                for arg in node.args:
                    inputs.append(TosaArg(arg))

                # Convert output (this node itself)
                output = TosaArg(node)
                # Add output to TOSA graph
                tosa_graph.currRegion.currBasicBlock.addTensor(
                    output.name,
                    (
                        inputs[0].shape
                        if is_permute_node_before_addmm(node)
                        else output.shape
                    ),
                    ts.DType.INT8 if is_quant_node(node) else output.dtype,
                )

                # Visiting each Node
                if node.target.__name__ in node_visitors:
                    node_visitors[node.target.__name__].define_node(
                        node, tosa_graph, inputs, output, is_quant_node(node)
                    )
                else:
                    raise RuntimeError(f"Unknown operator {node.target}")
            elif node.op == "placeholder":
                process_placeholder(node, tosa_graph, edge_program)
            elif node.op == "output":
                for output in node.args[0]:
                    tosa_graph.addOutputTensor(
                        tosa_graph.currRegion.currBasicBlock.tensors[output.name]
                    )
            else:
                # This will only happen if an unpartitioned graph is passed without
                # any checking of compatibility.
                dbg_fail(node, tosa_graph, path)

        # TODO: It would be awesome if this dump could somehow be done on top level and not here.
        # Problem is that the desc.json has to be created on the tosa_graph object, which we can't
        # access from top level.
        if debug_output is True:
            dbg_tosa_dump(tosa_graph, path)

        # Serialize and return the program. While we have always produced TOSA
        # output as an intermediate, some flows compile to device binaries in
        # preprocess and some consume TOSA fb directly.
        if output_format == "vela":
            # Emit vela_bin_stream format
            binary = vela_compile(tosa_graph, compile_flags)
        elif output_format == "tosa":
            # Emit TOSA flatbuffer
            binary = bytes(tosa_graph.serialize())
        else:
            raise RuntimeError(f"Unknown format {output_format}")

        # Continueing from above. Can I put tosa_graph into this function?
        # debug_handle_map = ...
        return PreprocessResult(processed_bytes=binary)
