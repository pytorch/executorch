# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`et_vk.conv_with_clamp` module + configs for the WebGPU op-test framework.

`ConvWithClampModule` is a plain `nn.Conv2d` followed by `F.relu6`, which the
Vulkan fusion rewrites to a delegated `et_vk.conv_with_clamp` (fp32 general conv
+ output clamp[0,6]). The conv weight/bias are baked params; only the float
tensor `x` is a runtime input. Goldened vs the module's fp32 eager.
`ConvWithClampTest` is the export-delegation smoke test.
"""

import unittest

import torch
import torch.nn as nn

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (input_shape, ic, oc, k, stride, padding, dilation, bias)
CONFIGS = {
    "k3p1": ((1, 4, 8, 8), 4, 8, 3, 1, 1, 1, True),
    "stride2": ((1, 3, 10, 10), 3, 6, 3, 2, 1, 1, True),
    "dil2": ((2, 3, 9, 9), 3, 5, 3, 1, 2, 2, True),
    "no_bias": ((1, 4, 8, 8), 4, 8, 3, 1, 1, 1, False),
    "asym": ((1, 3, 7, 9), 3, 5, (2, 3), (1, 2), (1, 0), (2, 1), True),
}


class ConvWithClampModule(nn.Module):
    def __init__(self, ic, oc, k, stride, padding, dilation, bias) -> None:
        super().__init__()
        self.c = nn.Conv2d(
            ic, oc, k, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu6(self.c(x))


def _det(shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(shape, ic, oc, k, stride, padding, dilation, bias):
    mod = ConvWithClampModule(ic, oc, k, stride, padding, dilation, bias).eval()
    ep = torch.export.export(mod, (_det(shape),))
    return to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _op_delegated(edge, op_substr: str) -> bool:
    # The op must be absorbed into a delegate: absent from the top-level graph AND
    # present inside a lowered submodule reached by an executorch_call_delegate node
    # (a bare absence check also passes for an empty graph or a renamed op).
    from executorch.exir.lowered_backend_module import get_lowered_submodules

    gm = edge.exported_program().graph_module
    if any(op_substr in str(getattr(n, "target", "")) for n in gm.graph.nodes):
        return False
    return any(
        op_substr in str(getattr(dn, "target", ""))
        for _, lowered, _ in get_lowered_submodules(gm)
        for dn in lowered.original_module.graph_module.graph.nodes
    )


class ConvWithClampTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (shape, ic, oc, k, s, p, d, b) in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(shape, ic, oc, k, s, p, d, b)
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (conv_with_clamp {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "conv_with_clamp"),
                    f"conv_with_clamp not delegated (CPU fallback) for {name}",
                )
