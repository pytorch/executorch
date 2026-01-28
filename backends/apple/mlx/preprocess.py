#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
MLX Backend preprocessing - converts EdgeIR to MLX delegate payload.

This module implements the BackendDetails.preprocess() method which:
1. Takes an ExportedProgram (edge dialect)
2. Builds an MLXGraph using MLXProgramBuilder
3. Serializes to FlatBuffer with constant data segment
4. Returns PreprocessResult with the combined binary
"""

from __future__ import annotations

import logging
from typing import ClassVar, final, List

from executorch.backends.apple.mlx.program_builder import MLXProgramBuilder
from executorch.backends.apple.mlx.serialization.mlx_graph_serialize import (
    HEADER_LENGTH,
    MAGIC,
    serialize_mlx_graph,
)

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)

from torch.export.exported_program import ExportedProgram

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def _padding_required(offset: int, alignment: int) -> int:
    """Returns padding needed to align offset to alignment boundary."""
    remainder = offset % alignment
    return (alignment - remainder) % alignment


@final
class MLXBackend(BackendDetails):
    """
    ExecuTorch backend for MLX (Apple Silicon GPU compute framework).

    This backend compiles EdgeIR programs to a custom bytecode format
    that can be executed by the MLX C++ runtime.
    """

    MAGIC_IX: ClassVar[slice] = slice(4, 8)
    DATA_SEGMENT_OFFSET_IX: ClassVar[slice] = slice(8, 16)
    DATA_SEGMENT_SIZE_IX: ClassVar[slice] = slice(16, 24)

    EXPECTED_MAGIC: ClassVar[bytes] = MAGIC
    EXPECTED_LENGTH: ClassVar[int] = HEADER_LENGTH

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        """
        Convert an ExportedProgram to MLX delegate payload.

        Args:
            edge_program: The ExportedProgram in edge dialect to compile.
            compile_specs: List of compilation options.

        Returns:
            PreprocessResult containing the serialized MLX program.
        """
        logging.info("MLXBackend.preprocess() called")

        # Parse compile specs
        use_fp16 = True
        for spec in compile_specs:
            if spec.key == "use_fp16":
                use_fp16 = bool(list(bytes(spec.value))[0])

        logging.debug(f"MLX compile options: use_fp16={use_fp16}")

        if logging.DEBUG >= logging.root.level:
            edge_program.graph.print_tabular()

        # Build MLXGraph from ExportedProgram
        builder = MLXProgramBuilder(edge_program)
        mlx_graph = builder.build()

        # Extract constant data
        constant_data, name_to_offset = builder.get_constant_data()

        # Update constant segment info in the graph
        mlx_graph.constant_segment.size = len(constant_data)

        # Log graph info
        logging.info(f"MLX Graph: {len(mlx_graph.instructions)} instructions")
        logging.info(f"  num_constant_tensors: {mlx_graph.num_constant_tensors}")
        logging.info(
            f"  num_non_constant_tensors: {mlx_graph.num_non_constant_tensors}"
        )
        logging.info(f"  num_non_constant_values: {mlx_graph.num_non_constant_values}")
        logging.info(f"  constant_data_size: {len(constant_data)} bytes")

        # Serialize to bytes
        serialized = serialize_mlx_graph(mlx_graph, constant_data)

        logging.info(f"MLXBackend.preprocess() complete: {len(serialized)} bytes")

        return PreprocessResult(processed_bytes=serialized)


def pretty_print_mlx_graph(mlx_graph) -> None:
    """Debug utility to print MLXGraph contents."""
    logging.info("MLXGraph:")
    logging.info(f"  version: {mlx_graph.version}")
    logging.info(f"  num_constant_tensors: {mlx_graph.num_constant_tensors}")
    logging.info(f"  num_non_constant_tensors: {mlx_graph.num_non_constant_tensors}")
    logging.info(f"  num_non_constant_values: {mlx_graph.num_non_constant_values}")
    logging.info(f"  instructions ({len(mlx_graph.instructions)}):")
    for i, instr in enumerate(mlx_graph.instructions):
        logging.info(f"    [{i}]: {type(instr.op).__name__}")
    logging.info(f"  input_map: {mlx_graph.input_map}")
    logging.info(f"  output_map: {mlx_graph.output_map}")
    logging.info(f"  mutable_buffer_map: {mlx_graph.mutable_buffer_map}")
    logging.info(f"  named_slots ({len(mlx_graph.named_slots)}):")
    for ns in mlx_graph.named_slots:
        logging.info(f"    {ns.name}: {ns.slot}")
    logging.info(f"  tensor_meta ({len(mlx_graph.tensor_meta)}):")
    for i, tm in enumerate(mlx_graph.tensor_meta):
        if tm is not None:
            logging.info(f"    [{i}]: shape={tm.shape}, dtype={tm.dtype}")
    logging.info(
        f"  constant_segment: offset={mlx_graph.constant_segment.offset}, size={mlx_graph.constant_segment.size}"
    )
