# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`et_vk.q8ta_conv2d_pw` module + configs: int8 pointwise (1x1) conv.

Produced by running a plain 1x1 `nn.Conv2d` through XNNPACK static PT2E: the
converted module lowers to a delegated `quantize_per_tensor -> q8ta_conv2d_pw ->
dequantize_per_tensor` subgraph (quantize/dequantize = the landed C0 ops). The
`module_factory` returns the CONVERTED module, so the op-test framework goldens
the WebGPU output against the converted eager (fp32 fake-quant), e2e.
"""

import unittest

import torch
import torch.nn as nn

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401 registers et_vk ops

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def make_q8ta_conv2d_pw_module(ic, oc, h, w, n=1, bias=True, seed=0):
    torch.manual_seed(seed)
    conv = nn.Conv2d(ic, oc, 1, bias=bias).eval()
    ex = (torch.randn(n, ic, h, w),)
    q = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    )
    prepared = prepare_pt2e(torch.export.export(conv, ex).module(), q)
    prepared(*ex)  # calibrate
    return convert_pt2e(prepared)


class Q8taConv2dPwTest(unittest.TestCase):
    def test_delegates(self) -> None:
        m = make_q8ta_conv2d_pw_module(4, 8, 6, 8)
        et = to_edge_transform_and_lower(
            torch.export.export(m, (torch.randn(1, 4, 6, 8),)),
            partitioner=[VulkanPartitioner()],
        ).to_executorch()
        self.assertTrue(
            any(
                d.id == "VulkanBackend"
                for plan in et.executorch_program.execution_plan
                for d in plan.delegates
            )
        )
        self.assertIn(b"q8ta_conv2d_pw", et.buffer)
