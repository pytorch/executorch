# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`et_vk.linear_qcs4w` module + configs: 4-bit channels-symmetric weight linear.

The op is produced by running a plain `nn.Linear` through the `VulkanQuantizer`
weight-only 4-bit path (per-output-channel symmetric weight, no activation
quant): the converted module lowers to a delegated `et_vk.linear_qcs4w`. The
`module_factory` returns the CONVERTED module, so the op-test framework goldens
the WebGPU output against the converted eager (fp32, the fake-quant reference).
Bias is out of the op's args (a `nn.Linear` bias would lower to a separate
`aten.add`); the cases use `bias=False` to keep the golden focused on qcs4w.
"""

import unittest

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401 registers et_vk ops

import torch
import torch.nn as nn

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
    get_symmetric_quantization_config,
    VulkanQuantizer,
)
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def make_qcs4w_linear_module(k, n, m, bias=False, seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(k, n, bias=bias).eval()
    ex = (torch.randn(m, k),)
    q = VulkanQuantizer().set_global(
        get_symmetric_quantization_config(is_dynamic=False, weight_bits=4)
    )
    prepared = prepare_pt2e(torch.export.export(lin, ex).module(), q)
    prepared(*ex)  # calibrate
    return convert_pt2e(prepared)


class LinearQcs4wTest(unittest.TestCase):
    def test_delegates(self) -> None:
        m = make_qcs4w_linear_module(32, 16, 4)
        et = to_edge_transform_and_lower(
            torch.export.export(m, (torch.randn(4, 32),)),
            partitioner=[VulkanPartitioner()],
        ).to_executorch()
        self.assertTrue(
            any(
                d.id == "VulkanBackend"
                for plan in et.executorch_program.execution_plan
                for d in plan.delegates
            )
        )
        self.assertIn(b"linear_qcs4w", et.buffer)
