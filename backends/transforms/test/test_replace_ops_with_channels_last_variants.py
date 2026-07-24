# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import executorch.backends.transforms.channels_last_ops  # noqa: F401
import pytest
import torch
from executorch.backends.transforms.replace_ops_with_channels_last_variants import (
    _NCHW_TO_NHWC_PERM,
    _NHWC_TO_NCHW_PERM,
    ChannelsLastOpSpec,
    ReplaceOpsWithChannelsLastVariants,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torch.fx.node import Target


# ── helpers ────────────────────────────────────────────────────────────────────


def _export_to_edge(module: torch.nn.Module, inputs: tuple) -> ExportedProgram:
    ep = torch.export.export(module.eval(), inputs)
    return to_edge(ep).exported_program()


def _run_pass(ep: ExportedProgram) -> tuple[GraphModule, bool]:
    result = ReplaceOpsWithChannelsLastVariants(ep)(ep.graph_module)
    return result.graph_module, result.modified


def _find_nodes(gm: GraphModule, target: Target) -> list[torch.fx.Node]:
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target == target]


def _count(gm: GraphModule, target: Target) -> int:
    return len(_find_nodes(gm, target))


# ── modules ────────────────────────────────────────────────────────────────────


class Conv2dModule(torch.nn.Module):
    def __init__(self, groups: int = 1, bias: bool = True):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            4, 4, kernel_size=3, padding=1, groups=groups, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AvgPool2dModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)


class AdaptiveAvgPool2dModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))


class Conv2dAvgPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(self.conv(x), kernel_size=2, stride=2)


class GridSampler2DModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.grid_sample(x, torch.ones(1, 2, 2, 2))


class Conv1dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)


class MaxPool2DModule(torch.nn.Module):
    def __init__(self, kernel_size: int | tuple[int, ...] = 3, **kwargs):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size, **kwargs)

    def forward(self, x):
        return self.max_pool2d(x)


class UpsampleBilinearModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        return self.upsample(x)


class UpsampleNearestModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        return self.upsample(x)


# ── tests ──────────────────────────────────────────────────────────────────────


class TestReplaceOpsWithChannelsLastVariants:

    def test_conv2d(self):
        ep = _export_to_edge(Conv2dModule(bias=True), (torch.randn(1, 4, 8, 8),))
        assert _count(ep.graph_module, exir_ops.edge.aten.convolution.default) == 1

        gm, modified = _run_pass(ep)

        assert modified
        assert _count(gm, exir_ops.edge.aten.convolution.default) == 0
        assert _count(gm, exir_ops.edge.channels_last.convolution.default) == 1
        assert _count(gm, exir_ops.edge.channels_last.permute_copy.default) == 2

        dialect_node = _find_nodes(gm, exir_ops.edge.channels_last.convolution.default)[
            0
        ]

        # Input (arg 0) must be wrapped in an NCHW→NHWC permute.
        input_permute = dialect_node.args[0]
        assert input_permute.target == exir_ops.edge.channels_last.permute_copy.default
        assert list(input_permute.args[1]) == _NCHW_TO_NHWC_PERM

        # Weight (arg 1) and bias (arg 2) must not be permuted.
        for arg in (dialect_node.args[1], dialect_node.args[2]):
            if isinstance(arg, torch.fx.Node):
                assert arg.target != exir_ops.edge.channels_last.permute_copy.default

        # Output must be wrapped in an NHWC→NCHW permute.
        users = list(dialect_node.users)
        assert len(users) == 1
        output_permute = users[0]
        assert output_permute.target == exir_ops.edge.channels_last.permute_copy.default
        assert list(output_permute.args[1]) == _NHWC_TO_NCHW_PERM

    def test_conv2d__implicit_batch(self):
        input_shape = (4, 8, 8)  # Use implicit batch size of `1`.
        ep = _export_to_edge(Conv2dModule(bias=True), (torch.randn(*input_shape),))
        assert _count(ep.graph_module, exir_ops.edge.aten.convolution.default) == 1

        gm, modified = _run_pass(ep)

        assert modified
        assert _count(gm, exir_ops.edge.aten.convolution.default) == 0
        assert _count(gm, exir_ops.edge.channels_last.convolution.default) == 1
        assert _count(gm, exir_ops.edge.channels_last.permute_copy.default) == 2

        # These ops are added automatically during lowering to edge.
        assert _count(gm, exir_ops.edge.aten.unsqueeze_copy.default) == 1
        assert _count(gm, exir_ops.edge.aten.squeeze_copy.dims) == 1

        dialect_node = _find_nodes(gm, exir_ops.edge.channels_last.convolution.default)[
            0
        ]

        # Input (arg 0) must be wrapped in an NCHW→NHWC permute.
        input_permute = dialect_node.args[0]
        assert input_permute.target == exir_ops.edge.channels_last.permute_copy.default
        assert list(input_permute.args[1]) == _NCHW_TO_NHWC_PERM
        assert input_permute.args[0].target == exir_ops.edge.aten.unsqueeze_copy.default

        # Weight (arg 1) and bias (arg 2) must not be permuted.
        for arg in (dialect_node.args[1], dialect_node.args[2]):
            if isinstance(arg, torch.fx.Node):
                assert arg.target != exir_ops.edge.channels_last.permute_copy.default

        # Output must be wrapped in an NHWC→NCHW permute.
        users = list(dialect_node.users)
        assert len(users) == 1
        output_permute = users[0]
        assert output_permute.target == exir_ops.edge.channels_last.permute_copy.default
        assert list(output_permute.args[1]) == _NHWC_TO_NCHW_PERM
        assert (
            list(output_permute.users)[0].target == exir_ops.edge.aten.squeeze_copy.dims
        )

    def test_depthwise_conv2d(self):
        ep = _export_to_edge(Conv2dModule(groups=4), (torch.randn(1, 4, 8, 8),))
        gm, modified = _run_pass(ep)

        assert modified
        assert _count(gm, exir_ops.edge.channels_last.convolution.default) == 1
        assert _count(gm, exir_ops.edge.channels_last.permute_copy.default) == 2

    def test_max_pool2d(self):
        ep = _export_to_edge(MaxPool2DModule(), (torch.randn(1, 4, 8, 8),))
        assert (
            _count(ep.graph_module, exir_ops.edge.aten.max_pool2d_with_indices.default)
            == 1
        )

        gm, modified = _run_pass(ep)

        assert modified
        assert _count(gm, exir_ops.edge.aten.max_pool2d_with_indices.default) == 0
        assert (
            _count(gm, exir_ops.edge.channels_last.max_pool2d_with_indices.default) == 1
        )
        assert _count(gm, exir_ops.edge.channels_last.permute_copy.default) == 2
        assert _count(gm, operator.getitem) == 1

        dialect_node = _find_nodes(
            gm, exir_ops.edge.channels_last.max_pool2d_with_indices.default
        )[0]

        # Input (arg 0) must be wrapped in an NCHW→NHWC permute.
        input_permute = dialect_node.args[0]
        assert input_permute.target == exir_ops.edge.channels_last.permute_copy.default
        assert list(input_permute.args[1]) == _NCHW_TO_NHWC_PERM

        users = list(dialect_node.users)
        assert len(users) == 1
        output_getitem = users[0]
        assert output_getitem.target == operator.getitem

        # Output must be wrapped in an NHWC→NCHW permute.
        getitem_users = list(output_getitem.users)
        assert len(getitem_users) == 1
        output_permute = getitem_users[0]
        assert output_permute.target == exir_ops.edge.channels_last.permute_copy.default
        assert list(output_permute.args[1]) == _NHWC_TO_NCHW_PERM

    def test_max_pool2d__implicit_batch(self):
        input_shape = (4, 8, 8)  # Use implicit batch size of `1`.
        ep = _export_to_edge(MaxPool2DModule(), (torch.randn(*input_shape),))
        assert (
            _count(ep.graph_module, exir_ops.edge.aten.max_pool2d_with_indices.default)
            == 1
        )

        gm, modified = _run_pass(ep)

        assert modified
        assert _count(gm, exir_ops.edge.aten.max_pool2d_with_indices.default) == 0
        assert (
            _count(gm, exir_ops.edge.channels_last.max_pool2d_with_indices.default) == 1
        )
        assert _count(gm, exir_ops.edge.channels_last.permute_copy.default) == 2
        assert _count(gm, exir_ops.edge.aten.unsqueeze_copy.default) == 1
        assert _count(gm, exir_ops.edge.aten.squeeze_copy.dims) == 1

        dialect_node = _find_nodes(
            gm, exir_ops.edge.channels_last.max_pool2d_with_indices.default
        )[0]

        # Input (arg 0) must be wrapped in an NCHW→NHWC permute + an unsqueeze.
        input_permute = dialect_node.args[0]
        assert input_permute.target == exir_ops.edge.channels_last.permute_copy.default
        assert list(input_permute.args[1]) == _NCHW_TO_NHWC_PERM
        assert input_permute.args[0].target == exir_ops.edge.aten.unsqueeze_copy.default

        users = list(dialect_node.users)
        assert len(users) == 1
        output_getitem = users[0]
        assert output_getitem.target == operator.getitem

        # Output must be wrapped in an NHWC→NCHW permute + a squeeze.
        getitem_users = list(output_getitem.users)
        assert len(getitem_users) == 1
        output_permute = getitem_users[0]
        assert output_permute.target == exir_ops.edge.channels_last.permute_copy.default
        assert list(output_permute.args[1]) == _NHWC_TO_NCHW_PERM
        assert (
            list(output_permute.users)[0].target == exir_ops.edge.aten.squeeze_copy.dims
        )

    @pytest.mark.parametrize(
        "module_cls, aten_op, channels_last_op",
        [
            pytest.param(
                AvgPool2dModule,
                exir_ops.edge.aten.avg_pool2d.default,
                exir_ops.edge.channels_last.avg_pool2d.default,
                id="AvgPool2D",
            ),
            pytest.param(
                AdaptiveAvgPool2dModule,
                exir_ops.edge.aten._adaptive_avg_pool2d.default,
                exir_ops.edge.channels_last.adaptive_avg_pool2d.default,
                id="AdaptiveAvgPool2D",
            ),
            pytest.param(
                GridSampler2DModule,
                exir_ops.edge.aten.grid_sampler_2d.default,
                exir_ops.edge.channels_last.grid_sampler_2d.default,
                id="GridSampler2D",
            ),
            pytest.param(
                UpsampleBilinearModule,
                exir_ops.edge.aten.upsample_bilinear2d.vec,
                exir_ops.edge.channels_last.upsample_bilinear2d.default,
                id="UpsampleBilinear2D",
            ),
            pytest.param(
                UpsampleNearestModule,
                exir_ops.edge.aten.upsample_nearest2d.vec,
                exir_ops.edge.channels_last.upsample_nearest2d.default,
                id="UpsampleNearest2D",
            ),
        ],
    )
    def test_ops_without_weights(self, module_cls, aten_op, channels_last_op):
        ep = _export_to_edge(module_cls(), (torch.randn(1, 4, 8, 8),))
        assert _count(ep.graph_module, aten_op) == 1

        gm, modified = _run_pass(ep)

        assert modified
        assert _count(gm, aten_op) == 0
        assert _count(gm, channels_last_op) == 1
        assert _count(gm, exir_ops.edge.channels_last.permute_copy.default) == 2

        dialect_node = _find_nodes(gm, channels_last_op)[0]

        input_permute = dialect_node.args[0]
        assert input_permute.target == exir_ops.edge.channels_last.permute_copy.default
        assert list(input_permute.args[1]) == _NCHW_TO_NHWC_PERM

        users = list(dialect_node.users)
        assert len(users) == 1
        output_permute = users[0]
        assert output_permute.target == exir_ops.edge.channels_last.permute_copy.default
        assert list(output_permute.args[1]) == _NHWC_TO_NCHW_PERM

    @pytest.mark.parametrize(
        "module_cls, aten_op, channels_last_op",
        [
            # GridSampler2D, UpsampleBilinear2D and UpsampleNearest2D don't support implicit batch.
            pytest.param(
                AvgPool2dModule,
                exir_ops.edge.aten.avg_pool2d.default,
                exir_ops.edge.channels_last.avg_pool2d.default,
                id="AvgPool2D",
            ),
            pytest.param(
                AdaptiveAvgPool2dModule,
                exir_ops.edge.aten._adaptive_avg_pool2d.default,
                exir_ops.edge.channels_last.adaptive_avg_pool2d.default,
                id="AdaptiveAvgPool2D",
            ),
        ],
    )
    def test_ops_without_weights__implicit_batch(
        self, module_cls, aten_op, channels_last_op
    ):
        input_shape = (4, 8, 8)  # Use implicit batch size of `1`.
        ep = _export_to_edge(module_cls(), (torch.randn(*input_shape),))
        assert _count(ep.graph_module, aten_op) == 1

        gm, modified = _run_pass(ep)

        assert modified
        assert _count(gm, aten_op) == 0
        assert _count(gm, channels_last_op) == 1
        assert _count(gm, exir_ops.edge.channels_last.permute_copy.default) == 2
        assert _count(gm, exir_ops.edge.aten.unsqueeze_copy.default) == 1
        assert _count(gm, exir_ops.edge.aten.squeeze_copy.dims) == 1

        dialect_node = _find_nodes(gm, channels_last_op)[0]

        input_permute = dialect_node.args[0]
        assert input_permute.target == exir_ops.edge.channels_last.permute_copy.default
        assert list(input_permute.args[1]) == _NCHW_TO_NHWC_PERM
        assert input_permute.args[0].target == exir_ops.edge.aten.unsqueeze_copy.default

        users = list(dialect_node.users)
        assert len(users) == 1
        output_permute = users[0]
        assert output_permute.target == exir_ops.edge.channels_last.permute_copy.default
        assert list(output_permute.args[1]) == _NHWC_TO_NCHW_PERM
        assert (
            list(output_permute.users)[0].target == exir_ops.edge.aten.squeeze_copy.dims
        )

    def test_conv2d_and_avg_pool2d(self):
        # 2 permutes per op. The redundant adjacent pair (NHWC→NCHW then NCHW→NHWC) between the two ops will be
        #  optimized out by a different pass.
        ep = _export_to_edge(Conv2dAvgPool2dModule(), (torch.randn(1, 4, 8, 8),))
        gm, modified = _run_pass(ep)

        assert modified
        assert _count(gm, exir_ops.edge.aten.convolution.default) == 0
        assert _count(gm, exir_ops.edge.aten.avg_pool2d.default) == 0
        assert _count(gm, exir_ops.edge.channels_last.convolution.default) == 1
        assert _count(gm, exir_ops.edge.channels_last.avg_pool2d.default) == 1
        assert _count(gm, exir_ops.edge.channels_last.permute_copy.default) == 4

    def test_modified_false_when_no_matching_ops(self):
        ep = _export_to_edge(LinearModule(), (torch.randn(2, 8),))
        _, modified = _run_pass(ep)
        assert not modified

    def test_empty_op_map_leaves_graph_unchanged(self):
        ep = _export_to_edge(Conv2dModule(), (torch.randn(1, 4, 8, 8),))

        result = ReplaceOpsWithChannelsLastVariants(ep, op_map={})(ep.graph_module)

        assert not result.modified
        assert _count(result.graph_module, exir_ops.edge.aten.convolution.default) == 1
        assert (
            _count(result.graph_module, exir_ops.edge.channels_last.convolution.default)
            == 0
        )

    def test_custom_op_map_only_replaces_specified_op(self):
        ep = _export_to_edge(Conv2dAvgPool2dModule(), (torch.randn(1, 4, 8, 8),))

        custom_map = {
            exir_ops.edge.aten.avg_pool2d.default: ChannelsLastOpSpec(
                target=exir_ops.edge.channels_last.avg_pool2d.default,
                input_indices=[0],
                output_indices=[0],
            )
        }
        result = ReplaceOpsWithChannelsLastVariants(ep, op_map=custom_map)(
            ep.graph_module
        )
        gm = result.graph_module

        assert result.modified
        assert _count(gm, exir_ops.edge.aten.convolution.default) == 1
        assert _count(gm, exir_ops.edge.channels_last.convolution.default) == 0
        assert _count(gm, exir_ops.edge.aten.avg_pool2d.default) == 0
        assert _count(gm, exir_ops.edge.channels_last.avg_pool2d.default) == 1

    def test_conv1d__not_replaced(self):
        ep = _export_to_edge(Conv1dModule(), (torch.randn(1, 4, 16),))
        gm, modified = _run_pass(ep)

        assert not modified
        assert _count(gm, exir_ops.edge.aten.convolution.default) == 1
        assert _count(gm, exir_ops.edge.channels_last.convolution.default) == 0

    def test_convolution__numerical_correctness(self):
        torch.manual_seed(2)
        x = torch.randn(1, 4, 8, 8)
        model = Conv2dModule(bias=False)
        reference_output = model(x)

        ep = _export_to_edge(model, (x,))
        gm, modified = _run_pass(ep)
        assert modified
        assert _count(gm, exir_ops.edge.channels_last.convolution.default) == 1

        channels_last_output = gm(ep.state_dict["conv.weight"], x)[0]
        assert torch.allclose(channels_last_output, reference_output)

    def test_avg_pool2d__numerical_correctness(self):
        torch.manual_seed(2)
        x = torch.randn(1, 4, 8, 8)
        model = AvgPool2dModule()
        reference_output = model(x)

        ep = _export_to_edge(model, (x,))
        gm, modified = _run_pass(ep)
        assert modified
        assert _count(gm, exir_ops.edge.channels_last.avg_pool2d.default) == 1

        channels_last_output = gm(x)[0]
        assert torch.allclose(channels_last_output, reference_output)

    def test_avg_pool2d__implicit_batch__numerical_correctness(self):
        torch.manual_seed(2)
        x = torch.randn(4, 8, 8)
        model = AvgPool2dModule()
        reference_output = model(x)

        ep = _export_to_edge(model, (x,))
        gm, modified = _run_pass(ep)
        assert modified
        assert _count(gm, exir_ops.edge.channels_last.avg_pool2d.default) == 1

        channels_last_output = gm(x)[0]
        assert torch.allclose(channels_last_output, reference_output)
