# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.backends.cortex_m.fused_quant.lowering import get_cortex_m_passes
from executorch.backends.cortex_m.fused_quant.quantizer import (
    get_cortex_m_fused_quant_quantizer,
)
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
)
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.backends.transforms.fused_quant.passes import get_fused_quant_passes
from executorch.exir import ExportedProgram
from executorch.exir._warnings import experimental
from executorch.exir.pass_manager import PassType
from executorch.export.recipe import ExportRecipe, LoweringRecipe
from executorch.frontends.fused_quant.recipe import (
    get_fused_quant_edge_compile_config,
    get_fused_quant_quantization_recipe,
)
from torchao.quantization.pt2e.quantizer import QuantizationSpec

# Kept whole through quantization, for the quantizer to decompose, and
# decomposed by to_edge where it did not.
_DECOMPOSED_AT_EDGE = (torch.ops.aten.scaled_dot_product_attention.default,)


@experimental("The fused_quant recipes are experimental and may change without notice.")
def get_cortex_m_fused_quant_recipe(
    target_config: Optional[CortexMTargetConfig] = None,
    *,
    per_channel_linear: bool = False,
    activation: QuantizationSpec = INT8_ACTIVATION_PER_TENSOR_QSPEC,
) -> ExportRecipe:
    """Quantize for CMSIS-NN through fused_quant, all the way to ``to_executorch``.

    The fused_quant quantize stage with the Cortex-M quantizer, then the
    fused_quant passes with ``channels_last`` and :func:`get_cortex_m_passes`:
    every op a CMSIS-NN kernel can run becomes its cortex_m op, and the rest
    runs on the portable kernels. There is no partitioner; the cortex_m ops are
    ordinary kernels. ``target_config``
    picks the kernel variants and scratch sizes; it defaults to an M55.
    ``per_channel_linear`` and ``activation`` configure the quantizer, as in
    :func:`get_cortex_m_fused_quant_quantizer`.
    """
    quantizer = get_cortex_m_fused_quant_quantizer(
        per_channel_linear=per_channel_linear, activation=activation
    )

    def edge_passes(method_name: str, program: ExportedProgram) -> list[PassType]:
        return [
            *get_fused_quant_passes(channels_last=True),
            *get_cortex_m_passes(program, target_config),
        ]

    return ExportRecipe(
        name="cortex_m_fused_quant",
        strict=False,
        quantization_recipe=get_fused_quant_quantization_recipe(quantizer),
        lowering_recipe=LoweringRecipe(
            edge_transform_passes=[edge_passes],
            edge_compile_config=get_fused_quant_edge_compile_config(
                op for op in quantizer.preserved_ops() if op not in _DECOMPOSED_AT_EDGE
            ),
        ),
    )
