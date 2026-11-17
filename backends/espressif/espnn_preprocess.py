# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import List

import torch

from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec

from .serialization.xts3 import build_graph, serialize_graph

logger = logging.getLogger(__name__)

BACKEND_ID = "EspNNBackend"
INT8_FORMAT_SPEC = CompileSpec("espnn_format", b"int8")
_IO_CONVERSION_SPEC_KEY = "espnn_io_conversion_slots"


def _io_conversion_slots(program, graph) -> int:
    signature = program.graph_signature
    nodes = {node.name: node for node in program.graph.nodes}
    io_nodes = [
        node
        for node in program.graph.nodes
        if node.op == "placeholder" and node.name in signature.user_inputs
    ] + [nodes[name] for name in signature.user_outputs]
    io_ids = graph.inputs + graph.outputs
    if len(io_nodes) != len(io_ids):
        raise ValueError("ESP-NN boundary tensors do not match the serialized I/O")
    slots = 0
    seen = set()
    for slot, (node, tensor_id) in enumerate(zip(io_nodes, io_ids)):
        if tensor_id in seen:
            continue
        seen.add(tensor_id)
        tensor = node.meta["val"]
        # Returned inputs need a protected copy even when their layout matches.
        returned_input = slot < len(graph.inputs) and tensor_id in graph.outputs
        if (
            returned_input
            or tensor.dtype != torch.int8
            or (
                tensor.dim() == 4
                and tensor.shape[1] != 1
                and tensor.shape[2] * tensor.shape[3] != 1
                and not tensor.is_contiguous(memory_format=torch.channels_last)
            )
        ):
            slots |= 1 << slot
    return slots


class EspNNBackend(BackendDetails):
    """
    Ahead-of-time half of the ESP-NN delegate.

    Each partition produced by EspNNPartitioner is lowered to an XTS3 int8
    graph: ESP-NN operator records, per-channel requantization parameters,
    16-byte aligned int8/int32 constants and an input/output table. The
    on-device EspNNBackend (backends/espressif/espnn_runtime) parses the blob
    and executes it with ESP-NN kernels.

    Compile specs: `espnn_format` selects the payload format; only `int8` is
    defined, and any other value is rejected.
    """

    @staticmethod
    def preprocess(edge_program, compile_specs: List[CompileSpec]) -> PreprocessResult:
        for spec in compile_specs:
            if (
                spec.key == INT8_FORMAT_SPEC.key
                and spec.value != INT8_FORMAT_SPEC.value
            ):
                raise ValueError(
                    f"Unsupported espnn_format {spec.value!r}; only b'int8' is defined"
                )
        graph = build_graph(edge_program)
        blob = serialize_graph(graph)
        compile_specs[:] = [
            spec for spec in compile_specs if spec.key != _IO_CONVERSION_SPEC_KEY
        ]
        compile_specs.append(
            CompileSpec(
                _IO_CONVERSION_SPEC_KEY,
                _io_conversion_slots(edge_program, graph).to_bytes(4, "little"),
            )
        )
        logger.info(
            "EspNN delegate: %d tensors, %d operations, %d constant bytes, %d byte blob",
            len(graph.tensors),
            len(graph.operations),
            len(graph.constant_data),
            len(blob),
        )
        return PreprocessResult(processed_bytes=blob, debug_handle_map={})


__all__ = ["BACKEND_ID", "EspNNBackend", "INT8_FORMAT_SPEC"]
