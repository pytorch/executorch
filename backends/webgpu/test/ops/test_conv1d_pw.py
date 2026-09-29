# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pointwise conv1d module + configs for the WebGPU op-test framework.

`Conv1dPwModule` is a pointwise `nn.Conv1d` (K=1, groups=1); it serializes as
`aten.convolution.default`, and the handler runs the pointwise-conv1d (matmul)
kernel. `Conv1dPwTest` is the export-delegation smoke test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> batch, in_channels, out_channels, L, bias
CONFIGS = {
    "ic4_oc6": (1, 4, 6, 5, True),
    "square": (1, 3, 3, 7, True),
    "oc2_nobias": (1, 5, 2, 4, False),
    "ic8_oc8": (1, 8, 8, 3, True),
    "batch2": (2, 3, 4, 5, True),
}

# name -> N, C_in, C_out, L, K, stride, padding, dilation, bias
GENERAL_CONFIGS = {
    "voxtral_stride1": (1, 4, 6, 10, 3, 1, 0, 1, True),
    "voxtral_stride2": (1, 6, 5, 10, 3, 2, 0, 1, True),
    "no_bias": (1, 3, 2, 9, 3, 1, 0, 1, False),
    "padded": (1, 3, 4, 9, 3, 1, 1, 1, True),
    "dilated": (1, 2, 3, 11, 3, 1, 2, 2, True),
    "batch2": (2, 3, 4, 8, 3, 2, 1, 1, True),
}


class Conv1dPwModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        with torch.no_grad():
            self.conv.weight.normal_(generator=g)
            if bias:
                self.conv.bias.normal_(generator=g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dModule(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
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
    n, ic, oc, L, bias = cfg
    m = Conv1dPwModule(ic, oc, bias).eval()
    ep = torch.export.export(m, (_det_input((n, ic, L)),))
    return to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])


def _lower_general(cfg, dynamic_shapes=None):
    n, ic, oc, length, kernel, stride, padding, dilation, bias = cfg
    module = Conv1dModule(ic, oc, kernel, stride, padding, dilation, bias).eval()
    inputs = (_det_input((n, ic, length)),)
    ep = torch.export.export(module, inputs, dynamic_shapes=dynamic_shapes)
    return to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _op_delegated(edge, op_substr: str) -> bool:
    # Require the op in a delegate, not merely absent from the host graph.
    from executorch.exir.lowered_backend_module import get_lowered_submodules

    gm = edge.exported_program().graph_module
    if any(op_substr in str(getattr(n, "target", "")) for n in gm.graph.nodes):
        return False
    return any(
        op_substr in str(getattr(dn, "target", ""))
        for _, lowered, _ in get_lowered_submodules(gm)
        for dn in lowered.original_module.graph_module.graph.nodes
    )


class Conv1dPwTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, cfg in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(cfg)
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (conv1d_pw {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "convolution"),
                    f"conv1d not delegated (fell back to CPU) for {name}",
                )


class Conv1dTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, cfg in GENERAL_CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower_general(cfg)
                self.assertTrue(
                    _delegated(edge.to_executorch()),
                    f"Expected a VulkanBackend delegate (conv1d {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "convolution"),
                    f"conv1d not delegated (fell back to CPU) for {name}",
                )

    def test_dynamic_length_export_delegates(self) -> None:
        length = torch.export.Dim("conv1d_length", min=5, max=16)
        edge = _lower_general(GENERAL_CONFIGS["padded"], dynamic_shapes=({2: length},))
        self.assertTrue(_delegated(edge.to_executorch()))
        self.assertTrue(_op_delegated(edge, "convolution"))
