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
3. Serializes to FlatBuffer (no embedded constants - those come via named_data_map)
4. Returns PreprocessResult with the binary and data_store_output for constants
"""

from __future__ import annotations

import hashlib
import logging
import os
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

_MLX_DEBUG = os.environ.get("ET_MLX_DEBUG", "") not in ("", "0")


@final
class MLXBackend(BackendDetails):
    """
    ExecuTorch backend for MLX (Apple Silicon GPU compute framework).

    This backend compiles EdgeIR programs to a custom bytecode format
    that can be executed by the MLX C++ runtime.

    Constants (weights) are stored in ExecuTorch's named_data_map rather than
    embedded in the delegate payload. This allows ExecuTorch to own the constant
    data and provide it to the backend at runtime.
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
            PreprocessResult containing the serialized MLX program and
            data_store_output with constant tensor data.
        """
        if _MLX_DEBUG:
            logging.info("MLXBackend.preprocess() called")
            logging.info(f"Edge program:\n{edge_program}")
            edge_program.graph.print_tabular()

        # Build MLXGraph from ExportedProgram
        # Use a deterministic 4-hex prefix derived from the edge program to
        # namespace named_data keys, avoiding collisions in multi-method
        # programs where different methods may have lifted tensor constants
        # with the same auto-generated name.
        prefix = hashlib.sha256(str(edge_program).encode()).hexdigest()[:4]
        builder = MLXProgramBuilder(edge_program, named_data_key_prefix=prefix)
        mlx_graph = builder.build()

        # Get constant data as NamedDataStore (ET will own this data)
        named_data_store = builder.get_named_data_store()

        if _MLX_DEBUG:
            logging.info(
                f"  named_data_store entries: {len(named_data_store.pte_data)}"
            )
            _log_mlx_graph(mlx_graph)

        # Serialize to bytes (no constant data embedded)
        serialized = serialize_mlx_graph(mlx_graph)

        if _MLX_DEBUG:
            logging.info(f"MLXBackend.preprocess() complete: {len(serialized)} bytes")

        return PreprocessResult(
            processed_bytes=serialized,
            data_store_output=named_data_store.get_named_data_store_output(),
        )


def _format_tensor_meta(meta) -> str:
    """Format a TensorMeta for display."""
    shape_parts = []
    for dim in meta.shape:
        if dim.is_vid:
            shape_parts.append(f"v{dim.vid.idx}")
        else:
            shape_parts.append(str(dim.literal))
    shape_str = f"[{', '.join(shape_parts)}]"
    dtype_str = f"dtype={meta.scalar_type}" if meta.scalar_type is not None else ""
    dim_order_str = f"dim_order={meta.dim_order}" if meta.dim_order is not None else ""
    parts = [shape_str]
    if dtype_str:
        parts.append(dtype_str)
    if dim_order_str:
        parts.append(dim_order_str)
    return ", ".join(parts)


def _log_mlx_graph(mlx_graph) -> None:  # noqa: C901
    """Log MLXGraph contents at INFO level for debugging."""
    logging.info("MLXGraph:")
    logging.info(f"  version: {mlx_graph.version}")
    logging.info(f"  num_constant_tensors: {mlx_graph.num_constant_tensors}")
    logging.info(f"  num_input_tensors: {mlx_graph.num_input_tensors}")
    logging.info(f"  num_output_tensors: {mlx_graph.num_output_tensors}")
    logging.info(
        f"  num_mutable_buffer_tensors: {mlx_graph.num_mutable_buffer_tensors}"
    )
    logging.info(f"  num_temp_tensors: {mlx_graph.num_temp_tensors}")
    logging.info(f"  num_values: {mlx_graph.num_values}")
    if mlx_graph.init_instructions:
        logging.info(f"  init_instructions ({len(mlx_graph.init_instructions)}):")
        for i, instr in enumerate(mlx_graph.init_instructions):
            logging.info(f"    [{i}]: {type(instr.op).__name__}")
    logging.info(f"  instructions ({len(mlx_graph.instructions)}):")
    for i, instr in enumerate(mlx_graph.instructions):
        logging.info(f"    [{i}]: {type(instr.op).__name__}")
    if mlx_graph.input_map:
        logging.info(f"  input_map ({len(mlx_graph.input_map)}):")
        for i, slot in enumerate(mlx_graph.input_map):
            logging.info(f"    [{i}]: {slot}")
    if mlx_graph.output_map:
        logging.info(f"  output_map ({len(mlx_graph.output_map)}):")
        for i, slot in enumerate(mlx_graph.output_map):
            logging.info(f"    [{i}]: {slot}")
    if mlx_graph.mutable_buffer_map:
        logging.info(f"  mutable_buffer_map ({len(mlx_graph.mutable_buffer_map)}):")
        for i, slot in enumerate(mlx_graph.mutable_buffer_map):
            logging.info(f"    [{i}]: {slot}")
    if mlx_graph.named_slots:
        logging.info(f"  named_slots ({len(mlx_graph.named_slots)}):")
        for ns in mlx_graph.named_slots:
            logging.info(f"    {ns.name}: {ns.slot}")
    if mlx_graph.tensor_meta:
        logging.info(f"  tensor_meta ({len(mlx_graph.tensor_meta)}):")
        for i, meta in enumerate(mlx_graph.tensor_meta):
            logging.info(f"    t{i}: {_format_tensor_meta(meta)}")
