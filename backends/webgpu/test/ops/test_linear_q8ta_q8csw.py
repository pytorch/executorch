# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`et_vk.linear_q8ta_q8csw` module + configs: int8-act x int8-channelwise-weight
linear with FP32 output (no output requant).

Reached by running a plain `nn.Linear` through XNNPACK static PT2E with the
activation config's `output_activation` nulled (`dataclasses.replace(..., output_
activation=None)`): the linear's INPUT is statically per-tensor quantized but its
OUTPUT is left fp32, so the Vulkan fusion routes to `linear_q8ta_q8csw` (fp32 out)
instead of `q8ta_linear` (int8 out). The `module_factory` returns the CONVERTED
module, so the op-test framework goldens the WebGPU output against the converted
eager (fp32 fake-quant reference); the served subgraph is `quantize_per_tensor`
(landed C0) -> `linear_q8ta_q8csw`.
"""

import dataclasses
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


def make_linear_q8ta_q8csw_module(k, n, m, bias=True, seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(k, n, bias=bias).eval()
    ex = (torch.randn(m, k),)
    cfg = dataclasses.replace(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False),
        output_activation=None,
    )
    q = XNNPACKQuantizer().set_global(cfg)
    prepared = prepare_pt2e(torch.export.export(lin, ex).module(), q)
    prepared(*ex)  # calibrate
    return convert_pt2e(prepared)


class LinearQ8taQ8cswTest(unittest.TestCase):
    def test_delegates(self) -> None:
        m = make_linear_q8ta_q8csw_module(32, 16, 4, bias=True)
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
        self.assertIn(b"linear_q8ta_q8csw", et.buffer)
