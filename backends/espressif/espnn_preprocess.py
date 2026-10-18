# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec

from .serialization.xts3 import build_graph, serialize_graph

logger = logging.getLogger(__name__)

BACKEND_ID = "EspNNBackend"
INT8_FORMAT_SPEC = CompileSpec("espnn_format", b"int8")
DUMP_ENV = "ESPNN_DELEGATE_OUTPUT"


class EspNNBackend(BackendDetails):
    """
    Ahead-of-time half of the ESP-NN delegate.

    Each partition produced by EspNNPartitioner is lowered to an XTS3 int8
    graph: ESP-NN operator records, per-channel requantization parameters,
    16-byte aligned int8/int32 constants and an input/output table. The
    on-device EspNNBackend (backends/espressif/runtime) parses the blob and
    executes it with ESP-NN kernels.

    Set ESPNN_DELEGATE_OUTPUT=<path> to also write the last blob to a file.
    """

    @staticmethod
    def preprocess(edge_program, compile_specs: List[CompileSpec]) -> PreprocessResult:
        graph = build_graph(edge_program)
        blob = serialize_graph(graph)
        logger.info(
            "EspNN delegate: %d tensors, %d operations, %d constant bytes, %d byte blob",
            len(graph.tensors),
            len(graph.operations),
            len(graph.constant_data),
            len(blob),
        )
        output_path = os.environ.get(DUMP_ENV)
        if output_path:
            Path(output_path).write_bytes(blob)
        return PreprocessResult(processed_bytes=blob, debug_handle_map={})


__all__ = ["BACKEND_ID", "EspNNBackend", "INT8_FORMAT_SPEC"]
