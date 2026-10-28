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

from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec

from .serialization.xts3 import build_graph, serialize_graph

logger = logging.getLogger(__name__)

BACKEND_ID = "EspNNBackend"
INT8_FORMAT_SPEC = CompileSpec("espnn_format", b"int8")


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
        logger.info(
            "EspNN delegate: %d tensors, %d operations, %d constant bytes, %d byte blob",
            len(graph.tensors),
            len(graph.operations),
            len(graph.constant_data),
            len(blob),
        )
        return PreprocessResult(processed_bytes=blob, debug_handle_map={})


__all__ = ["BACKEND_ID", "EspNNBackend", "INT8_FORMAT_SPEC"]
