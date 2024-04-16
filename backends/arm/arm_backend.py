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
    is_consumer_node_depthwise_conv2d,
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


class ArmCompileSpecBuilder:
    def __init__(self):
        self.compile_spec: List[CompileSpec] = []
        self.compiler_flags = []
        self.output_format = None
        self.path_for_intermediates = None
        self.permute_nhwc = False

    def ethosu_compile_spec(
        self,
        config: str,
        system_config: Optional[str] = None,
        memory_mode: Optional[str] = None,
        extra_flags: Optional[str] = None,
        config_ini: Optional[str] = "Arm/vela.ini",
    ):
        """
        Generate compile spec for Ethos-U NPU

        Args:
            config: Ethos-U accelerator configuration, e.g. ethos-u55-128
            system_config: System configuration to select from the Vel
                configuration file
            memory_mode: Memory mode to select from the Vela configuration file
            extra_flags: Extra flags for the Vela compiler
            config_ini: Vela configuration file(s) in Python ConfigParser .ini
                file format
        """
        assert (
            self.output_format is None
        ), f"Output format already set to f{self.output_format}"
        self.output_format = "vela"
        self.compiler_flags = [
            f"--accelerator-config={config}",
            f"--config={config_ini}",
        ]
        if system_config is not None:
            self.compiler_flags.append(f"--system-config={system_config}")
        if memory_mode is not None:
            self.compiler_flags.append(f"--memory-mode={memory_mode}")
        if extra_flags is not None:
            self.compiler_flags.append(extra_flags)

        return self

    def tosa_compile_spec(self):
        """
        Generate compile spec for TOSA flatbuffer output
        """
        assert (
            self.output_format is None
        ), f"Output format already set: {self.output_format}"
        self.output_format = "tosa"
        return self

    def dump_intermediate_tosa(self, output_path: str):
        """
        Output intermediate .tosa file
        """
        self.path_for_intermediates = output_path
        return self

    def set_permute_memory_format(self, set_nhwc_permutation: bool = True):
        self.permute_nhwc = set_nhwc_permutation
        return self

    def build(self):
        """
        Generate a list of compile spec objects from the builder
        """
        if self.output_format == "vela":
            self.compile_spec += [
                CompileSpec("output_format", "vela".encode()),
                CompileSpec("compile_flags", " ".join(self.compiler_flags).encode()),
            ]
        elif self.output_format == "tosa":
            self.compile_spec.append(CompileSpec("output_format", "tosa".encode()))

        if self.path_for_intermediates is not None:
            self.compile_spec.append(
                CompileSpec("debug_tosa_path", self.path_for_intermediates.encode())
            )

        if self.permute_nhwc:
            self.compile_spec.append(
                CompileSpec("permute_memory_format", "nhwc".encode())
            )

        return self.compile_spec


def is_permute_memory(compile_spec: List[CompileSpec]) -> bool:
    for spec in compile_spec:
        if spec.key == "permute_memory_format":
            return spec.value.decode() == "nhwc"
    return False


def is_tosa(compile_spec: List[CompileSpec]) -> bool:
    for spec in compile_spec:
        if spec.key == "output_format":
            return spec.value.decode() == "tosa"
    return False


def get_intermediate_path(compile_spec: List[CompileSpec]) -> str:
    for spec in compile_spec:
        if spec.key == "debug_tosa_path":
            return spec.value.decode()
    return None


def generate_ethosu_compile_spec(
    config: str,
    permute_memory_to_nhwc: Optional[bool] = None,
    system_config: Optional[str] = None,
    memory_mode: Optional[str] = None,
    extra_flags: Optional[str] = None,
    config_ini: Optional[str] = "Arm/vela.ini",
) -> List[CompileSpec]:
    return (
        ArmCompileSpecBuilder()
        .ethosu_compile_spec(
            config,
            system_config=system_config,
            memory_mode=memory_mode,
            extra_flags=extra_flags,
            config_ini=config_ini,
        )
        .set_permute_memory_format(permute_memory_to_nhwc)
        .build()
    )


def generate_tosa_compile_spec(
    permute_memory_to_nhwc: Optional[bool] = None,
    output_path: Optional[str] = None,
) -> List[CompileSpec]:
    return (
        ArmCompileSpecBuilder()
        .tosa_compile_spec()
        .set_permute_memory_format(permute_memory_to_nhwc)
        .dump_intermediate_tosa(output_path)
        .build()
    )


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
        permute_memory_to_nhwc = False
        for spec in compile_spec:
            if spec.key == "debug_tosa_path":
                path = spec.value.decode()
                debug_output = True
            if spec.key == "output_format":
                output_format = spec.value.decode()
            if spec.key == "compile_flags":
                compile_flags.append(spec.value.decode())
            if spec.key == "permute_memory_format":
                memory_format = spec.value.decode()
                if memory_format == "nhwc":
                    permute_memory_to_nhwc = True

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

                # TODO: fragile code for temporary fix, not all outputs will be
                # rank 4
                if permute_memory_to_nhwc and len(output.shape) == 4:
                    # TODO: remove this if check
                    # this is added because we need to align the quant node
                    # output shape before the depthwise_conv2d node. The output
                    # shape between TOSA conv2d and depthwise_conv2d are different.
                    if (
                        node.all_input_nodes[0].op
                        == "placeholder"  # check its parent is a placeholder
                        and is_quant_node(node)
                        and is_consumer_node_depthwise_conv2d(node)
                    ):
                        NHWC_Order = [2, 3, 0, 1]
                    else:
                        NHWC_Order = [0, 2, 3, 1]
                    output.shape = [output.shape[i] for i in NHWC_Order]

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
                    if node.target.__name__ in [
                        "aten.add.Tensor",
                        "aten._native_batch_norm_legit_no_training.default",
                    ]:
                        node_visitors[node.target.__name__].define_node(
                            node,
                            tosa_graph,
                            inputs,
                            output,
                            is_quant_node(node),
                            permute_memory_to_nhwc,
                        )
                    else:
                        node_visitors[node.target.__name__].define_node(
                            node, tosa_graph, inputs, output, is_quant_node(node)
                        )
                else:
                    raise RuntimeError(f"Unknown operator {node.target}")
            elif node.op == "placeholder":
                process_placeholder(
                    node, tosa_graph, edge_program, permute_memory_to_nhwc
                )
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
