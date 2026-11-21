# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.frontends.fused_quant.quantizer.quantizer import (
    FusedQuantQuantizer,
    get_core_aten_quantizers,
    OpQuantizer,
)
from torchao.quantization.pt2e.quantizer import SharedQuantizationSpec
from torchao.quantization.pt2e.quantizer.utils import annotate_output_qspec


class _AvgPoolQuantizer(OpQuantizer):
    def annotate_node(self, node: torch.fx.Node) -> None:
        super().annotate_node(node)
        annotate_output_qspec(node, SharedQuantizationSpec((node.args[0], node)))


class _EspFusedQuantizer(FusedQuantQuantizer):
    """Fused Quant quantizer for the Espressif operator-library backend.

    Uses int8 per-tensor affine activations and symmetric weights, with
    per-channel weights for convolution. AvgPool2d without divisor_override
    shares input and output qparams. EspPassManager lowers supported
    Fused Quant operators to ESP-NN and decomposes the rest for portable
    execution. Use EspNNQuantizer for the EspNNBackend delegate instead.
    """

    def __init__(self) -> None:
        config = get_symmetric_quantization_config(is_per_channel=False)
        super().__init__(
            [
                _AvgPoolQuantizer(
                    torch.ops.aten.avg_pool2d.default,
                    torch.ops.fused_quant.avg_pool2d.default,
                    config,
                    node_filter=lambda node: get_arg(node, "divisor_override") is None,
                ),
                *get_core_aten_quantizers(
                    config,
                    get_symmetric_quantization_config(is_per_channel=True),
                ),
            ]
        )


def get_esp_fused_quant_quantizer() -> FusedQuantQuantizer:
    """Create the quantizer for the Espressif Fused Quant operator-library flow."""
    return _EspFusedQuantizer()
