# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable

import executorch.kernels.quantized  # noqa: F401  registers quantize/dequantize out variants

from executorch.backends.espressif.partition.espnn_partitioner import EspNNPartitioner
from executorch.backends.espressif.passes import EspPassManager
from executorch.backends.espressif.quantizer import get_esp_fused_quant_quantizer
from executorch.backends.espressif.target import EspKernelMode, EspTarget
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.canonical_partitioners.duplicate_dequant_node_pass import (
    DuplicateDequantNodePass,
)
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.memory import alloc, view
from executorch.export import ExportRecipe, LoweringRecipe, StageType
from executorch.frontends.fused_quant.quantizer.quantizer import FusedQuantQuantizer
from executorch.frontends.fused_quant.recipe import (
    get_fused_quant_edge_compile_config,
    get_fused_quant_quantization_recipe,
)
from torch._ops import OpOverload


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


def get_operator_library_report(edge: EdgeProgramManager) -> dict:
    """Report ESP-NN compute, portable compute and layout copies by method."""
    esp_nn_ops = 0
    cpu_ops = []
    layout_ops = []
    for method in sorted(edge.methods):
        for node in edge.exported_program(method).graph.nodes:
            if node.op != "call_function" or node.target in (operator.getitem, alloc):
                continue
            target = node.target
            ops = cpu_ops
            if target is view:
                ops = layout_ops
                target = "executorch.exir.memory.view"
            elif isinstance(target, (EdgeOpOverload, OpOverload)):
                if target.namespace == "esp_nn":
                    esp_nn_ops += 1
                    continue
                if target._schema.name in {
                    "aten::alias_copy",
                    "aten::clone",
                    "aten::permute_copy",
                    "aten::squeeze_copy",
                    "aten::transpose_copy",
                    "aten::unsqueeze_copy",
                    "aten::view_copy",
                }:
                    ops = layout_ops
                elif target._schema.name in {
                    "quantized_decomposed::quantize_per_tensor",
                    "quantized_decomposed::dequantize_per_tensor",
                    "quantized_decomposed::quantize_per_channel",
                    "quantized_decomposed::dequantize_per_channel",
                }:
                    continue
                target = target.name()
            ops.append({"method": method, "name": node.name, "target": str(target)})
    return {"esp_nn_ops": esp_nn_ops, "cpu_ops": cpu_ops, "layout_ops": layout_ops}


def _require_no_portable_compute(edge: EdgeProgramManager) -> list:
    portable_ops = get_operator_library_report(edge)["cpu_ops"]
    if portable_ops:
        raise RuntimeError(
            "The ESP-NN operator library left these operators on portable kernels: "
            + ", ".join(
                f"{op['method']}.{op['name']} ({op['target']})" for op in portable_ops
            )
        )
    return []


def get_esp_fused_quant_recipe(
    calibration_inputs_fn: Callable[[], Iterable[tuple]] | None = None,
    *,
    quantizer: FusedQuantQuantizer | None = None,
    allow_cpu_fallback: bool = True,
    target: EspTarget = EspTarget.ESP32S3,
    kernel_mode: EspKernelMode = EspKernelMode.OPTIMIZED,
) -> ExportRecipe:
    """Quantize and lower a float model for the Espressif operator library.

    Use with executorch.export.export(). Supported Fused Quant operators become
    direct ESP-NN calls; other operators decompose to portable kernels. Setting
    allow_cpu_fallback=False rejects remaining portable compute. Q/DQ and
    structural operations such as views and layout copies remain allowed.
    Set target and kernel_mode to match the device's ESP-NN configuration
    when planning scratch.
    Use EspNNQuantizer and to_espnn_edge for the EspNNBackend delegate.
    """
    quantizer = quantizer if quantizer is not None else get_esp_fused_quant_quantizer()
    quantization_recipe = get_fused_quant_quantization_recipe(quantizer)
    quantization_recipe.calibration_inputs_fn = calibration_inputs_fn

    transforms = [lambda _: EspPassManager(target=target, kernel_mode=kernel_mode)]
    if not allow_cpu_fallback:
        transforms.append(_require_no_portable_compute)
    return ExportRecipe(
        strict=False,
        quantization_recipe=quantization_recipe,
        lowering_recipe=LoweringRecipe(
            edge_compile_config=get_fused_quant_edge_compile_config(()),
            edge_manager_transform_passes=transforms,
        ),
        pipeline_stages=[
            StageType.QUANTIZE,
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE,
            StageType.EDGE_PROGRAM_MANAGER_TRANSFORM,
            StageType.TO_EXECUTORCH,
        ],
    )


__all__ = ["to_espnn_edge", "get_esp_fused_quant_recipe", "get_operator_library_report"]
