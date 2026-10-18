# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import executorch.kernels.quantized  # noqa: F401  registers quantize/dequantize out variants

from executorch.backends.espressif.partition.espnn_partitioner import (
    DuplicateQuantizationFanoutPass,
    EspNNPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower


def to_espnn_edge(program, allow_cpu_fallback: bool = True):
    """
    Lower a PT2E-quantized torch.export program for the ESP32 ExecuTorch
    runtime: int8 regions become ESP-NN delegates, anything else stays in
    the program for the portable CPU kernels. Returns (edge_manager, report).
    """
    partitioner = EspNNPartitioner(allow_cpu_fallback=allow_cpu_fallback)
    edge = to_edge_transform_and_lower(
        program,
        transform_passes=[DuplicateQuantizationFanoutPass()],
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    return edge, partitioner.report


__all__ = ["to_espnn_edge"]
