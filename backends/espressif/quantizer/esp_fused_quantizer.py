# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.frontends.fused_quant.quantizer.quantizer import (
    FusedQuantQuantizer,
    get_core_aten_quantizers,
)


class _EspFusedQuantizer(FusedQuantQuantizer):
    """Fused Quant quantizer for the Espressif operator-library backend.

    Uses int8 per-tensor affine activations and symmetric weights, with
    per-channel weights for convolution. EspPassManager lowers supported
    Fused Quant operators to ESP-NN and decomposes the rest for portable
    execution. Use EspNNQuantizer for the EspNNBackend delegate instead.
    """

    def __init__(self) -> None:
        super().__init__(
            get_core_aten_quantizers(
                get_symmetric_quantization_config(is_per_channel=False),
                get_symmetric_quantization_config(is_per_channel=True),
            )
        )


def get_esp_fused_quant_quantizer() -> FusedQuantQuantizer:
    """Create the quantizer for the Espressif Fused Quant operator-library flow."""
    return _EspFusedQuantizer()
