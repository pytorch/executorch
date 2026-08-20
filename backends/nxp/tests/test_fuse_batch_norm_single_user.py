# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.nxp.aten_passes.fuse_batch_norm_with_conv_pass import (
    FuseBatchNormWithConvPass,
)
from executorch.backends.nxp.aten_passes.fuse_batch_norm_with_linear_pass import (
    FuseBatchNormWithLinearPass,
)


def _graph_has_batch_norm(graph_module: torch.fx.GraphModule) -> bool:
    return any(
        node.op == "call_function" and node.target == torch.ops.aten.batch_norm.default
        for node in graph_module.graph.nodes
    )


class ConvBatchNorm(torch.nn.Module):
    def __init__(self, share_conv_output: bool):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3)
        self.bn = torch.nn.BatchNorm2d(4)
        self.share_conv_output = share_conv_output

    def forward(self, x):
        y = self.conv(x)
        out = self.bn(y)
        if self.share_conv_output:
            # The conv output now feeds another consumer, so the BatchNorm must
            # not be fused into the conv weights (that would corrupt this branch).
            out = out + torch.relu(y)
        return out


class LinearBatchNorm(torch.nn.Module):
    def __init__(self, share_linear_output: bool):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.bn = torch.nn.BatchNorm1d(4)
        self.share_linear_output = share_linear_output

    def forward(self, x):
        y = self.linear(x)
        out = self.bn(y)
        if self.share_linear_output:
            out = out + torch.relu(y)
        return out


def _exported_module(module, example_input):
    module.eval()
    return torch.export.export(module, example_input, strict=True).module()


def test_conv_batch_norm_fused_when_single_user():
    gm = _exported_module(
        ConvBatchNorm(share_conv_output=False), (torch.randn(1, 3, 8, 8),)
    )
    assert _graph_has_batch_norm(gm)
    FuseBatchNormWithConvPass().call(gm)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    # Conv has a single user (the BatchNorm), so the fusion removes the BatchNorm.
    assert not _graph_has_batch_norm(gm)


def test_conv_batch_norm_not_fused_when_conv_has_multiple_users():
    example_input = (torch.randn(1, 3, 8, 8),)
    module = ConvBatchNorm(share_conv_output=True)
    reference = _exported_module(module, example_input)
    gm = _exported_module(module, example_input)

    assert _graph_has_batch_norm(gm)
    FuseBatchNormWithConvPass().call(gm)
    gm.graph.eliminate_dead_code()
    gm.recompile()

    # Conv feeds two consumers, so the BatchNorm must be left untouched.
    assert _graph_has_batch_norm(gm)
    x = torch.randn(1, 3, 8, 8)
    torch.testing.assert_close(reference(x), gm(x))


def test_linear_batch_norm_fused_when_single_user():
    gm = _exported_module(
        LinearBatchNorm(share_linear_output=False), (torch.randn(4, 4),)
    )
    assert _graph_has_batch_norm(gm)
    FuseBatchNormWithLinearPass().call(gm)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    assert not _graph_has_batch_norm(gm)


def test_linear_batch_norm_not_fused_when_linear_has_multiple_users():
    example_input = (torch.randn(4, 4),)
    module = LinearBatchNorm(share_linear_output=True)
    reference = _exported_module(module, example_input)
    gm = _exported_module(module, example_input)

    assert _graph_has_batch_norm(gm)
    FuseBatchNormWithLinearPass().call(gm)
    gm.graph.eliminate_dead_code()
    gm.recompile()

    assert _graph_has_batch_norm(gm)
    x = torch.randn(4, 4)
    torch.testing.assert_close(reference(x), gm(x))
