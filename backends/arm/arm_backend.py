# Copyright 2023-2025 Arm Limited and/or its affiliates.
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
from typing import cast, final, List, Optional

import serializer.tosa_serializer as ts  # type: ignore
from executorch.backends.arm.arm_vela import vela_compile
from executorch.backends.arm.operators.node_visitor import get_node_visitors

from executorch.backends.arm.tosa_specification import TosaSpecification
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


class ArmCompileSpecBuilder:
    def __init__(self):
        self.compile_spec: List[CompileSpec] = []
        self.compiler_flags = []
        self.output_format = None
        self.path_for_intermediates = None
        self.tosa_version = None
        self.tosa_spec = None
        self.input_order = None

    def ethosu_compile_spec(
        self,
        config: str,
        system_config: str,
        memory_mode: str,
        extra_flags: Optional[str] = None,
        config_ini: Optional[str] = "Arm/vela.ini",
    ) -> "ArmCompileSpecBuilder":
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

        base_tosa_version = "TOSA-0.80+BI"
        if "u55" in config:
            # Add the Ethos-U55 extension marker
            base_tosa_version += "+u55"
        self.tosa_spec = TosaSpecification.create_from_string(base_tosa_version)

        return self

    def tosa_compile_spec(
        self, tosa_spec: str | TosaSpecification
    ) -> "ArmCompileSpecBuilder":
        """
        Generate compile spec for TOSA flatbuffer output
        """
        assert (
            self.output_format is None
        ), f"Output format already set: {self.output_format}"
        self.output_format = "tosa"
        if isinstance(tosa_spec, TosaSpecification):
            self.tosa_spec = tosa_spec
        elif isinstance(tosa_spec, str):
            self.tosa_spec = TosaSpecification.create_from_string(tosa_spec)
        else:
            raise RuntimeError(f"Invalid type for {tosa_spec}!")
        return self

    def dump_intermediate_artifacts_to(
        self, output_path: str
    ) -> "ArmCompileSpecBuilder":
        """
        Sets a path for dumping intermediate results during such as tosa and pte.
        """
        self.path_for_intermediates = output_path
        return self

    def build(self) -> List[CompileSpec]:
        """
        Generate a list of compile spec objects from the builder
        """
        assert self.tosa_spec

        # Always supply a TOSA version
        self.compile_spec = [CompileSpec("tosa_version", str(self.tosa_spec).encode())]

        if self.output_format == "vela":
            self.compile_spec += [
                CompileSpec("output_format", "vela".encode()),
                CompileSpec("compile_flags", " ".join(self.compiler_flags).encode()),
            ]
        elif self.output_format == "tosa":
            self.compile_spec.append(CompileSpec("output_format", "tosa".encode()))

        if self.path_for_intermediates is not None:
            self.compile_spec.append(
                CompileSpec("debug_artifact_path", self.path_for_intermediates.encode())
            )

        if self.input_order:
            self.compile_spec.append(
                CompileSpec(
                    "input_order", " ".join(map(str, self.input_order)).encode()
                )
            )

        return self.compile_spec


def is_tosa(compile_spec: List[CompileSpec]) -> bool:
    for spec in compile_spec:
        if spec.key == "output_format":
            return spec.value.decode() == "tosa"
    return False


def get_tosa_version(compile_spec: List[CompileSpec]) -> TosaSpecification:
    for spec in compile_spec:
        if spec.key == "tosa_version":
            return TosaSpecification.create_from_string(spec.value.decode())
    raise RuntimeError("Could not find TOSA version in CompileSpec")


def get_intermediate_path(compile_spec: List[CompileSpec]) -> Optional[str]:
    for spec in compile_spec:
        if spec.key == "debug_artifact_path":
            return spec.value.decode()
    return None


def _get_first_delegation_tag(graph_module) -> str | None:
    """Get the first delegation tag from the graph_module or return None."""
    for node in graph_module.graph.nodes:
        tag = node.meta.get("delegation_tag")
        if tag:
            return tag

    logger.debug("No delegation tag found in partition.")
    return None


@final
class ArmBackend(BackendDetails):
    @staticmethod
    def preprocess(  # noqa: C901
        edge_program: ExportedProgram,
        compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        logger.info("ArmBackend::preprocess")

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

        # Check that the output format is set in the compile spec
        if not output_format:
            raise RuntimeError("output format is required")

        tosa_spec = TosaSpecification.create_from_compilespecs(compile_spec)
        assert (
            tosa_spec is not None
        ), "TOSA backend needs a TOSA version specified in the CompileSpec!"

        if output_format == "vela" and len(compile_flags) == 0:
            # Not testing for compile_flags correctness here, just that they are
            # present. The compiler will give errors if they are not valid.
            raise RuntimeError("compile flags are required for vela output format")

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

        # Serialize and return the program. While we have always produced TOSA
        # output as an intermediate, some flows compile to device binaries in
        # preprocess and some consume TOSA fb directly.
        if output_format == "vela":
            # Emit vela_bin_stream format
            binary = vela_compile(tosa_graph, compile_flags, input_order)
        elif output_format == "tosa":
            # Emit TOSA flatbuffer
            binary = bytes(tosa_graph.serialize())
        else:
            raise RuntimeError(f"Unknown format {output_format}")

        return PreprocessResult(processed_bytes=binary)
