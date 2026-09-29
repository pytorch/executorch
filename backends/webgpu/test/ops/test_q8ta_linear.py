# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`et_vk.q8ta_linear` module + configs: int8-activation x int8-weight linear.

The op is produced by running a plain `nn.Linear` through XNNPACK static PT2E
(per-channel weight, static per-tensor activation): the converted module lowers
to a delegated `quantize_per_tensor -> q8ta_linear -> dequantize_per_tensor`
subgraph (the quantize/dequantize are the landed C0 ops). The `module_factory`
returns the CONVERTED module, so the op-test framework goldens the WebGPU output
against the converted eager (fp32, the fake-quant reference), exercising all three
ops end-to-end. `activation` is scoped to "none" (the XNNPACK-static default).
"""

import unittest

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401 registers et_vk ops

import torch
import torch.nn as nn

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def make_q8ta_linear_module(k, n, m, bias=True, seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(k, n, bias=bias).eval()
    ex = (torch.randn(m, k),)
    q = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    )
    prepared = prepare_pt2e(torch.export.export(lin, ex).module(), q)
    prepared(*ex)  # calibrate
    return convert_pt2e(prepared)


class Q8taLinearTest(unittest.TestCase):
    def test_delegates(self) -> None:
        m = make_q8ta_linear_module(16, 8, 4)
        et = to_edge_transform_and_lower(
            torch.export.export(m, (torch.randn(4, 16),)),
            partitioner=[VulkanPartitioner()],
        ).to_executorch()
        self.assertTrue(
            any(
                d.id == "VulkanBackend"
                for plan in et.executorch_program.execution_plan
                for d in plan.delegates
            )
        )
        # The delegate must contain the new op (+ the landed C0 quant ops).
        self.assertIn(b"q8ta_linear", et.buffer)
