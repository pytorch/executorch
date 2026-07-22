# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Depthwise conv1d module + configs for the WebGPU op-test framework.

`Conv1dDWModule` is a depthwise `nn.Conv1d` (groups=C); it serializes as
`aten.convolution.default`, and the handler runs the depthwise-conv1d kernel.
`Conv1dDWTest` is the export-delegation smoke test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> C, L, kernel, stride, padding, dilation, bias
CONFIGS = {
    "k3s1p1": (4, 8, 3, 1, 1, 1, True),
    "k3s2p1": (4, 8, 3, 2, 1, 1, True),
    "dil2": (3, 10, 3, 1, 2, 2, True),
    "k5_nobias": (5, 7, 5, 1, 0, 1, False),
}


class Conv1dDWModule(torch.nn.Module):
    def __init__(self, C, kernel, stride, padding, dilation, bias) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.conv = torch.nn.Conv1d(
            C, C, kernel, stride=stride, padding=padding,
            dilation=dilation, groups=C, bias=bias,
        )
        with torch.no_grad():
            self.conv.weight.normal_(generator=g)
            if bias:
                self.conv.bias.normal_(generator=g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _det_input(shape):
    g = torch.Generator().manual_seed(1)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(cfg):
    C, L, kernel, stride, padding, dilation, bias = cfg
    m = Conv1dDWModule(C, kernel, stride, padding, dilation, bias).eval()
    ep = torch.export.export(m, (_det_input((1, C, L)),))
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


class Conv1dDWTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, cfg in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(cfg)
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (conv1d_dw {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "convolution"),
                    f"conv1d not delegated (fell back to CPU) for {name}",
                )
