# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`et_vk.q8ta_add` module + configs: int8 elementwise add.

Both int8 operands are baked constants (only the scalar qparams vary), so the op
is exercised alone with a byte-exact int8 golden vs the CPU eager op. The kernel
implements `a + alpha*b` (the CPU reference semantics); an `alpha != 1` case pins
the alpha term, which the Vulkan glsl buffer path drops.
"""

import unittest

import torch

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401 registers et_vk ops

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class Q8taAddModule(torch.nn.Module):
    def __init__(
        self,
        a_vals,
        b_vals,
        a_scale=0.05,
        a_zp=0,
        b_scale=0.1,
        b_zp=3,
        out_scale=0.08,
        out_zp=-2,
        alpha=1.0,
    ):
        super().__init__()
        self.register_buffer("a", torch.tensor(a_vals, dtype=torch.int8))
        self.register_buffer("b", torch.tensor(b_vals, dtype=torch.int8))
        self.a_scale, self.a_zp = a_scale, a_zp
        self.b_scale, self.b_zp = b_scale, b_zp
        self.out_scale, self.out_zp, self.alpha = out_scale, out_zp, alpha

    def forward(self) -> torch.Tensor:
        return torch.ops.et_vk.q8ta_add(
            self.a,
            self.b,
            self.a_scale,
            self.a_zp,
            self.b_scale,
            self.b_zp,
            self.out_scale,
            self.out_zp,
            self.alpha,
        )


class Q8taAddTest(unittest.TestCase):
    def test_delegates(self) -> None:
        m = Q8taAddModule(list(range(-8, 8)), list(range(7, -9, -1)))
        et = to_edge_transform_and_lower(
            torch.export.export(m, ()), partitioner=[VulkanPartitioner()]
        ).to_executorch()
        delegate_ids = [
            d.id
            for plan in et.executorch_program.execution_plan
            for d in plan.delegates
        ]
        self.assertIn("VulkanBackend", delegate_ids)
