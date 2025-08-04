# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from .. import QuantType


def quantize(
    model, example_inputs, quant_type: QuantType = QuantType.STATIC_PER_TENSOR
):
    """This is the official recommended flow for quantization in pytorch 2.0 export"""
    logging.info(f"Original model: {model}")
    quantizer = XNNPACKQuantizer()
    # if we set is_per_channel to True, we also need to add out_variant of quantize_per_channel/dequantize_per_channel
    is_per_channel = (
        quant_type == QuantType.STATIC_PER_CHANNEL
        or quant_type == QuantType.DYNAMIC_PER_CHANNEL
    )
    is_dynamic = quant_type == QuantType.DYNAMIC_PER_CHANNEL
    operator_config = get_symmetric_quantization_config(
        is_per_channel=is_per_channel,
        is_dynamic=is_dynamic,
    )
    quantizer.set_global(operator_config)
    m = prepare_pt2e(model, quantizer)
    # calibration
    m(*example_inputs)
    m = convert_pt2e(m)
    logging.info(f"Quantized model: {m}")
    # make sure we can export to flat buffer
    return m
