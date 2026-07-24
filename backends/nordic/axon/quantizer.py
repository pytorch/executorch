# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AXON NPU quantizer for ExecuTorch.

Provides a quantizer configured for the AXON NPU's INT8 requirements.
Wraps the ARM backend's quantizer infrastructure with AXON-specific
defaults:

- Symmetric INT8 quantization (AXON requirement)
- Per-channel weights for FC/Conv (better accuracy)
- TOSA-1.0+INT profile (matches the AXON compilation pipeline)

Usage::

    from executorch.backends.nordic.axon import AxonQuantizer

    quantizer = AxonQuantizer()
    prepared = prepare_pt2e(exported.module(), quantizer)
    prepared(*calibration_data)
    quantized = convert_pt2e(prepared)
"""
from __future__ import annotations

from executorch.backends.arm.quantizer import (
    TOSAQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.tosa.specification import TosaSpecification


class AxonQuantizer(TOSAQuantizer):
    """Quantizer configured for the AXON NPU.

    Defaults to symmetric INT8 quantization with per-channel weights,
    targeting the TOSA-1.0+INT profile that the AXON compilation
    pipeline requires.

    Args:
        per_channel: Use per-channel quantization for weights (default True).
            Per-channel gives better accuracy; per-tensor gives smaller
            command buffers (single shared shift instruction).
        quantize_io: Also quantize the model's input and output tensors
            (default False). When False, the model accepts fp32 input
            and produces fp32 output, with q/dq ops at the AXON
            delegation boundaries.
    """

    def __init__(
        self,
        per_channel: bool = True,
        quantize_io: bool = False,
    ):
        tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
        super().__init__(tosa_spec)

        config = get_symmetric_quantization_config(
            is_per_channel=per_channel,
        )
        self.set_global(config)

        if quantize_io:
            self.set_io(config)
