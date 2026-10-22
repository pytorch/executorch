# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`et_vk.linear_dq8ca_q4gsw` module + config: dynamic per-row int8 activation
quant x 4-bit-group symmetric weight (the 8da4w path).

Reached by running a plain `nn.Linear` through torchao's
`Int8DynamicActivationIntxWeightConfig(weight_dtype=int4, PerGroup(gs))`: the
Vulkan partitioner fuses the dynamic-quant + linear into `choose_qparams_affine`
(per-row activation scale/zp) feeding `et_vk.linear_dq8ca_q4gsw` (fp32 out). The
factory returns the CONVERTED module so the op-test framework goldens the WebGPU
output against the converted eager forward (fp32 fake-quant reference). q4gsw
requires K % group_size == 0, K % 8 == 0, N % 8 == 0.
"""

import torch
import torch.nn as nn

from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    quantize_,
)


def make_linear_dq8ca_q4gsw_module(k, n, m, group_size=32, bias=False, seed=0):
    torch.manual_seed(seed)  # fixes the weights the golden derives from
    lin = nn.Linear(k, n, bias=bias).eval()
    quantize_(
        lin,
        Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4, weight_granularity=PerGroup(group_size)
        ),
    )
    return lin
