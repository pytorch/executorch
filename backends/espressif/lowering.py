# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator

import executorch.kernels.quantized  # noqa: F401  registers quantize/dequantize out variants

from executorch.backends.espressif.partition.espnn_partitioner import EspNNPartitioner
from executorch.backends.espressif.passes import EspPassManager
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.canonical_partitioners.duplicate_dequant_node_pass import (
    DuplicateDequantNodePass,
)
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch._ops import OpOverload


def _operator_library_report(edge: EdgeProgramManager) -> dict:
    esp_nn_ops = 0
    cpu_ops = []
    for method in sorted(edge.methods):
        for node in edge.exported_program(method).graph.nodes:
            if node.op != "call_function" or node.target == operator.getitem:
                continue
            target = node.target
            if isinstance(target, (EdgeOpOverload, OpOverload)):
                if target._schema.name in {
                    "quantized_decomposed::quantize_per_tensor",
                    "quantized_decomposed::dequantize_per_tensor",
                    "quantized_decomposed::quantize_per_channel",
                    "quantized_decomposed::dequantize_per_channel",
                }:
                    continue
                if target.namespace == "esp_nn":
                    esp_nn_ops += 1
                    continue
                target = target.name()
            cpu_ops.append({"method": method, "name": node.name, "target": str(target)})
    return {"esp_nn_ops": esp_nn_ops, "cpu_ops": cpu_ops}


def to_espnn_edge(
    program, allow_cpu_fallback: bool = True, *, use_operator_library: bool = False
):
    """
    Lower a PT2E-quantized torch.export program for the ESP32 ExecuTorch
    runtime: int8 regions become ESP-NN delegates, anything else stays in
    the program for the portable CPU kernels. Returns (edge_manager, report).

    With use_operator_library=True, run EspPassManager without partitioning.
    Supported patterns become direct ESP-NN operators; others retain portable
    operators and executable Q/DQ.
    Its report counts esp_nn_ops and lists remaining cpu_ops by method,
    excluding Q/DQ and structural getitem nodes.
    """
    if use_operator_library:
        edge = to_edge_transform_and_lower(
            program,
            transform_passes=EspPassManager(),
            compile_config=EdgeCompileConfig(),
        )
        report = _operator_library_report(edge)
        if not allow_cpu_fallback and report["cpu_ops"]:
            raise RuntimeError(
                "The ESP-NN operator library left these operators on the CPU: "
                + ", ".join(
                    f"{op['method']}.{op['name']} ({op['target']})"
                    for op in report["cpu_ops"]
                )
            )
        return edge, report

    partitioner = EspNNPartitioner(allow_cpu_fallback=allow_cpu_fallback)
    edge = to_edge_transform_and_lower(
        program,
        transform_passes=[DuplicateDequantNodePass()],
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    return edge, partitioner.report


__all__ = ["to_espnn_edge"]
