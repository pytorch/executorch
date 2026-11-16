# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import executorch.kernels.quantized  # noqa: F401  registers quantize/dequantize out variants

from executorch.backends.espressif.partition.espnn_partitioner import EspNNPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.backend.canonical_partitioners.duplicate_dequant_node_pass import (
    DuplicateDequantNodePass,
)


def to_espnn_edge(
    program,
    allow_cpu_fallback: bool = True,
    *,
    delegate_mutable_buffers: bool = False,
):
    """Lower a PT2E-quantized program to EspNNBackend delegates and portable ops.

    Returns (edge_manager, report). For the Fused Quant operator-library flow,
    use get_esp_fused_quant_recipe() with executorch.export.export().

    With delegate_mutable_buffers=True, the delegates own the quantized mutable
    buffers (streaming state) they read and update; see EspNNPartitioner.
    """
    partitioner = EspNNPartitioner(
        allow_cpu_fallback=allow_cpu_fallback,
        delegate_mutable_buffers=delegate_mutable_buffers,
    )
    edge = to_edge_transform_and_lower(
        program,
        transform_passes=[DuplicateDequantNodePass()],
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    return edge, partitioner.report


__all__ = ["to_espnn_edge"]
