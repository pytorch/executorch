# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from typing import Any, Tuple

import pytest

import torch
from executorch.backends.transforms.to_contiguous_channels_last_pass import (
    ToContiguousChannelsLastPass,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops

InputT = Tuple[Any, ...]


@dataclass(frozen=True)
class PermuteCountTestCase:
    module: torch.nn.Module
    inputs: InputT
    expected_initial_permutes: int = 0
    expected_initial_views: int = 0
    expected_final_permutes: int = 0
    expected_final_views: int = 0


class Conv1dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 4, kernel_size=3)

    def forward(self, x):
        return self.conv(x)


class Conv2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, kernel_size=3)

    def forward(self, x):
        return self.conv(x)


class Conv3dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(2, 4, kernel_size=3)

    def forward(self, x):
        return self.conv(x)


class LinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)


class MatmulModule(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


class IndexPutModule(torch.nn.Module):
    def forward(self, x, indices, values, acc: bool):
        return torch.index_put(x, indices=indices, values=values, accumulate=acc)


class PixelShuffleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pixel_shuffle = torch.nn.PixelShuffle(2)

    def forward(self, x):
        return self.pixel_shuffle(x)


class IndexSelectModule(torch.nn.Module):
    def forward(self, x, dim: int, index: torch.Tensor):
        return torch.index_select(x, dim=dim, index=index)


class GroupedConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, kernel_size=3, groups=2)

    def forward(self, x):
        return self.conv(x)


class TransposeConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(2, 4, kernel_size=3)

    def forward(self, x):
        return self.conv(x)


class ViewsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(1, 1)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.view(1, 4, 2, 2)
        x = x * 2
        x = x.view(1, 2, 4, 2)
        x = x * 2
        x = self.maxpool(x)
        return x


class TransposesModule(torch.nn.Module):
    def forward(self, x):
        x = torch.transpose(x, 2, 3)
        x = x.permute(0, 3, 1, 2)
        return x


class MaxPool2dDilatedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=2)

    def forward(self, x):
        return self.pool(x)


class LstmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=8, hidden_size=4, num_layers=1, batch_first=True
        )

    def forward(self, x):
        y, _ = self.lstm(x)
        return y


class GroupNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.group_norm = torch.nn.GroupNorm(num_groups=2, num_channels=4)

    def forward(self, x):
        return self.group_norm(x)


class MultiheadAttentionModule(torch.nn.Module):
    def __init__(self, embed_dim: int = 8, num_heads: int = 2):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        out, _ = self.mha(x, x, x, need_weights=False)
        return out


class CumsumModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, dim: int):
        return torch.cumsum(x, dim)


class Model1ConvMaxPoolResidualLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 8, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.linear = torch.nn.Linear(8, 6)

    def forward(self, x):
        residual = self.pool(x)
        x = self.pool(self.conv(x))
        x = x + residual
        x = x.transpose(1, 2)
        return self.linear(x)


class Model2ConvMhaLinearLayerNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 8, kernel_size=3, padding=1)
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=8, num_heads=2, batch_first=True
        )
        self.linear = torch.nn.Linear(8, 8)
        self.layernorm = torch.nn.LayerNorm(8)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.mha(x, x, x, need_weights=False)
        x = self.linear(x)
        return self.layernorm(x)


class Model3LstmLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=8, hidden_size=8, num_layers=1, batch_first=True
        )
        self.linear = torch.nn.Linear(8, 6)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x)


class Model4ConvLstmLinearLayerNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 8, kernel_size=3, padding=1)
        self.lstm = torch.nn.LSTM(
            input_size=8, hidden_size=6, num_layers=1, batch_first=True
        )
        self.linear = torch.nn.Linear(6, 4)
        self.layernorm = torch.nn.LayerNorm(4)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return self.layernorm(x)


class Model5DwConvGeluLayerNormAvgPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(
            8, 8, kernel_size=3, padding=1, groups=8, bias=False
        )
        self.gelu = torch.nn.GELU()
        self.layernorm = torch.nn.LayerNorm(8)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.gelu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        x = x.permute(0, 3, 1, 2)
        return self.avgpool(x)


class Model6GruLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=8, hidden_size=6, num_layers=1, batch_first=True
        )
        self.linear = torch.nn.Linear(6, 4)

    def forward(self, x):
        x, _ = self.gru(x)
        return self.linear(x)


class Model7DwConvBatchNormLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = torch.nn.Conv1d(
            8, 8, kernel_size=3, padding=1, groups=8, bias=False
        )
        self.bn = torch.nn.BatchNorm1d(8)
        self.linear = torch.nn.Linear(8, 4)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = x.transpose(1, 2)
        return self.linear(x)


class Model8ConvBatchNormMaxPoolResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        residual = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        return x + residual


class Model9DilatedConvBatchNormAvgPoolResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, kernel_size=3, padding=2, dilation=2)
        self.bn = torch.nn.BatchNorm2d(8)
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        residual = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        return x + residual


class Model10DwConvBatchNormLinearCat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = torch.nn.Conv1d(
            8, 8, kernel_size=3, padding=1, groups=8, bias=False
        )
        self.bn = torch.nn.BatchNorm1d(8)
        self.linear_a = torch.nn.Linear(8, 4)
        self.linear_b = torch.nn.Linear(8, 4)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = x.transpose(1, 2)
        a = self.linear_a(x)
        b = self.linear_b(x)
        return torch.cat((a, b), dim=-1)


class PermuteSiluPermute(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = torch.nn.SiLU()

    def forward(self, x: torch.Tensor):
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.silu(x)
        return torch.permute(x, [0, 3, 1, 2])


cases = {
    "conv1d_rank2": PermuteCountTestCase(
        Conv1dModule(), (torch.randn(2, 8),), 0, 2, 0, 2
    ),
    "conv1d_rank3": PermuteCountTestCase(Conv1dModule(), (torch.randn(1, 2, 8),), 0),
    "conv2d_rank3": PermuteCountTestCase(
        Conv2dModule(), (torch.randn(2, 8, 8),), 0, 2, 2, 2
    ),
    "conv2d_rank4": PermuteCountTestCase(
        Conv2dModule(), (torch.randn(1, 2, 8, 8),), 0, 0, 2, 0
    ),
    "conv3d_rank4": PermuteCountTestCase(
        Conv3dModule(), (torch.randn(2, 6, 6, 6),), 0, 2, 0, 2
    ),
    "conv3d_rank5": PermuteCountTestCase(
        Conv3dModule(), (torch.randn(1, 2, 6, 6, 6),), 0
    ),
    "linear_rank2": PermuteCountTestCase(
        LinearModule(), (torch.randn(2, 8),), 1, 0, 1, 0
    ),
    "linear_rank3": PermuteCountTestCase(
        LinearModule(), (torch.randn(2, 2, 8),), 1, 2, 1, 2
    ),
    "linear_rank4": PermuteCountTestCase(
        LinearModule(), (torch.randn(1, 2, 2, 8),), 1, 2, 1, 2
    ),
    "matmul_rank2": PermuteCountTestCase(
        MatmulModule(),
        (torch.randn(2, 3), torch.randn(3, 4)),
        0,
    ),
    "matmul_rank4": PermuteCountTestCase(
        MatmulModule(),
        (torch.randn(2, 2, 2, 3), torch.randn(2, 2, 3, 4)),
        0,
        3,
        0,
        3,
    ),
    "index_put": PermuteCountTestCase(
        IndexPutModule(),
        (
            torch.zeros((2, 4), dtype=torch.float32),
            (
                torch.tensor([0, 1]),
                torch.tensor([2, 3]),
            ),
            torch.ones((2,), dtype=torch.float32),
            False,
        ),
        0,
    ),
    "pixel_shuffle": PermuteCountTestCase(
        PixelShuffleModule(),
        (torch.randn(1, 8, 2, 2),),
        1,
        2,
        1,
        2,
    ),
    "index_select": PermuteCountTestCase(
        IndexSelectModule(),
        (torch.randn(2, 4, 3), 1, torch.tensor([0, 2])),
        0,
    ),
    "grouped_conv": PermuteCountTestCase(
        GroupedConvModule(),
        (torch.randn(1, 4, 8, 8),),
        0,
        0,
        2,
        0,
    ),
    "transpose_conv": PermuteCountTestCase(
        TransposeConvModule(),
        (torch.randn(1, 2, 8, 8),),
        0,
        0,
        2,
        0,
    ),
    "views": PermuteCountTestCase(ViewsModule(), (torch.rand(1, 2, 2, 4),), 0, 2, 4, 2),
    "transposes": PermuteCountTestCase(
        TransposesModule(),
        (torch.randn(1, 2, 3, 4),),
        2,
        0,
        1,
        0,
    ),
    "maxpool2d_dilation": PermuteCountTestCase(
        MaxPool2dDilatedModule(),
        (torch.randn(1, 2, 8, 8),),
        0,
        0,
        2,
        0,
    ),
    "lstm": PermuteCountTestCase(
        LstmModule(),
        (torch.randn(2, 4, 8),),
        7,
        19,
        7,
        16,
    ),
    "groupnorm": PermuteCountTestCase(
        GroupNormModule(),
        (torch.randn(1, 4, 4, 4),),
        0,
    ),
    "multihead_attention_rank2": PermuteCountTestCase(
        MultiheadAttentionModule(),
        (torch.randn(4, 8),),
        11,
        24,
        8,
        14,
    ),
    "multihead_attention_rank3": PermuteCountTestCase(
        MultiheadAttentionModule(),
        (torch.randn(2, 4, 8),),
        12,
        20,
        10,
        18,
    ),
    "cumsum_rank3_dim0": PermuteCountTestCase(
        CumsumModule(),
        (torch.randn(2, 3, 4), 1),
        0,
    ),
    "cumsum_rank4_dim3": PermuteCountTestCase(
        CumsumModule(),
        (torch.randn(1, 2, 3, 4), 3),
        0,
    ),
    "model_1_conv_maxpool_residual_linear": PermuteCountTestCase(
        Model1ConvMaxPoolResidualLinear(), (torch.randn(2, 8, 64),), 2, 7, 6, 7
    ),
    "model_2_conv_mha_linear_layernorm": PermuteCountTestCase(
        Model2ConvMhaLinearLayerNorm(), (torch.randn(2, 8, 32),), 14, 23, 11, 21
    ),
    "model_3_lstm_linear": PermuteCountTestCase(
        Model3LstmLinear(), (torch.randn(2, 16, 8),), 20, 58, 20, 55
    ),
    "model_4_conv_lstm_linear_layernorm": PermuteCountTestCase(
        Model4ConvLstmLinearLayerNorm(), (torch.randn(2, 8, 32),), 37, 106, 36, 103
    ),
    "model_5_dwconv_gelu_layernorm_avgpool": PermuteCountTestCase(
        Model5DwConvGeluLayerNormAvgPool(), (torch.randn(1, 8, 16, 16),), 2, 0, 4, 0
    ),
    "model_6_gru_linear": PermuteCountTestCase(
        Model6GruLinear(), (torch.randn(2, 16, 8),), 20, 56, 20, 55
    ),
    "model_7_dwconv_batchnorm_linear": PermuteCountTestCase(
        Model7DwConvBatchNormLinear(), (torch.randn(2, 8, 64),), 2, 3, 2, 3
    ),
    "model_8_conv_batchnorm_maxpool_residual": PermuteCountTestCase(
        Model8ConvBatchNormMaxPoolResidual(),
        (torch.randn(1, 8, 16, 16),),
        0,
        0,
        5,
        0,
    ),
    "model_9_dilated_conv_batchnorm_avgpool_residual": PermuteCountTestCase(
        Model9DilatedConvBatchNormAvgPoolResidual(),
        (torch.randn(1, 8, 16, 16),),
        0,
        0,
        5,
        0,
    ),
    "model_10_dwconv_batchnorm_linear_cat": PermuteCountTestCase(
        Model10DwConvBatchNormLinearCat(), (torch.randn(2, 8, 64),), 3, 6, 3, 6
    ),
    "permute_silu_permute": PermuteCountTestCase(
        PermuteSiluPermute(),
        (torch.randn(1, 2, 3, 4),),
        2,
        0,
        0,
        0,
    ),
}


# Channels-last inputs are left alone: the replacement pass only converts
# contiguous anchors, so these pin the fallback rather than a conversion.
cases_channels_last = {
    "conv2d_rank4_channels_last": PermuteCountTestCase(
        Conv2dModule(),
        (torch.randn(1, 2, 8, 8).to(memory_format=torch.channels_last),),
        0,
    ),
    "conv3d_rank4_channels_last": PermuteCountTestCase(
        Conv3dModule(),
        (torch.randn(2, 6, 6, 6).to(memory_format=torch.channels_last),),
        0,
        2,
        0,
        2,
    ),
    "conv3d_rank5_channels_last": PermuteCountTestCase(
        Conv3dModule(),
        (torch.randn(1, 2, 6, 6, 6).to(memory_format=torch.channels_last_3d),),
        0,
    ),
    "linear_rank4_channels_last": PermuteCountTestCase(
        LinearModule(),
        (torch.randn(1, 2, 2, 8).to(memory_format=torch.channels_last),),
        1,
        3,
        1,
        3,
    ),
    "matmul_rank4_channels_last": PermuteCountTestCase(
        MatmulModule(),
        (
            torch.randn(2, 2, 2, 3).to(memory_format=torch.channels_last),
            torch.randn(2, 2, 3, 4).to(memory_format=torch.channels_last),
        ),
        0,
        3,
        0,
        3,
    ),
    "pixel_shuffle_channels_last": PermuteCountTestCase(
        PixelShuffleModule(),
        (torch.randn(1, 8, 2, 2).to(memory_format=torch.channels_last),),
        1,
        2,
        1,
        2,
    ),
    "grouped_conv_channels_last": PermuteCountTestCase(
        GroupedConvModule(),
        (torch.randn(1, 4, 8, 8).to(memory_format=torch.channels_last),),
        0,
    ),
    "transpose_conv_channels_last": PermuteCountTestCase(
        TransposeConvModule(),
        (torch.randn(1, 2, 8, 8).to(memory_format=torch.channels_last),),
        0,
    ),
    "views_channels_last": PermuteCountTestCase(
        ViewsModule(),
        (torch.rand(1, 2, 2, 4).to(memory_format=torch.channels_last),),
        -1,  # The test crashes before reaching the transpose count
    ),
    "transposes_channels_last": PermuteCountTestCase(
        TransposesModule(),
        (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),),
        2,
        0,
        1,
        0,
    ),
    "maxpool2d_dilation_channels_last": PermuteCountTestCase(
        MaxPool2dDilatedModule(),
        (torch.randn(1, 2, 8, 8).to(memory_format=torch.channels_last),),
        0,
    ),
    "groupnorm_channels_last": PermuteCountTestCase(
        GroupNormModule(),
        (torch.randn(1, 4, 4, 4).to(memory_format=torch.channels_last),),
        0,
    ),
    "cumsum_rank4_dim3_channels_last": PermuteCountTestCase(
        CumsumModule(),
        (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last), 3),
        0,
    ),
}

_CHANNELS_LAST_XFAILS = {
    "views_channels_last": "Views are not supported for channels last tensors",
}

_PERMUTE_TARGETS = {
    exir_ops.edge.aten.permute.default,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.transpose.int,
    exir_ops.edge.aten.transpose_copy.int,
    exir_ops.edge.channels_last.permute_copy.default,
}
_VIEW_TARGETS = {
    exir_ops.edge.aten._unsafe_view.default,
    exir_ops.edge.aten.reshape.default,
    exir_ops.edge.aten.squeeze.default,
    exir_ops.edge.aten.squeeze.dim,
    exir_ops.edge.aten.squeeze.dims,
    exir_ops.edge.aten.squeeze_copy.default,
    exir_ops.edge.aten.squeeze_copy.dim,
    exir_ops.edge.aten.squeeze_copy.dims,
    exir_ops.edge.aten.unsqueeze.default,
    exir_ops.edge.aten.unsqueeze_copy.default,
    exir_ops.edge.aten.view.default,
    exir_ops.edge.aten.view_copy.default,
}


def _count_ops(graph_module: torch.fx.GraphModule, targets: set) -> int:
    return sum(
        node.op == "call_function" and node.target in targets
        for node in graph_module.graph.nodes
    )


def run_test(case: PermuteCountTestCase) -> None:
    case.module.eval()
    with torch.no_grad():
        exported_program = torch.export.export(case.module, case.inputs)
        edge_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
        )
        initial_graph = edge_program.exported_program().graph_module
        initial_permutes = _count_ops(initial_graph, _PERMUTE_TARGETS)
        initial_views = _count_ops(initial_graph, _VIEW_TARGETS)

        layout_pass = ToContiguousChannelsLastPass(edge_program.exported_program())
        transformed = edge_program.transform([layout_pass])
        final_graph = transformed.exported_program().graph_module
        final_permutes = _count_ops(final_graph, _PERMUTE_TARGETS)
        final_views = _count_ops(final_graph, _VIEW_TARGETS)

        assert initial_permutes == case.expected_initial_permutes
        assert initial_views == case.expected_initial_views
        assert final_permutes == case.expected_final_permutes
        assert final_views == case.expected_final_views
        ref_result = exported_program.module()(*case.inputs)
        edge_result = transformed.exported_program().module()(*case.inputs)
        assert torch.allclose(ref_result, edge_result, atol=1e-6)


class TestToContiguousChannelsLastPass(unittest.TestCase):
    def test_permute_view_counts(self) -> None:
        for name, case in cases.items():
            with self.subTest(name=name):
                run_test(case)

    def test_permute_view_counts_channels_last(self) -> None:
        for name, case in cases_channels_last.items():
            if name in _CHANNELS_LAST_XFAILS:
                continue
            with self.subTest(name=name):
                run_test(case)


class ConvChain(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(torch.relu(self.conv1(x)))


class ConvOnly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DynamicViewConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]))


class ConvThenLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.linear = torch.nn.Linear(4 * 8 * 8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.conv(x).flatten(1))


class UserPermute(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)


class ConvChannelBias(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.register_buffer("channel_bias", torch.randn(1, 4, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.channel_bias


class PadConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.nn.functional.pad(x, (1, 1, 1, 1)))


class ConvSoftmax(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.conv(x), dim=-1)


class ConvRuntimeBiasConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x) + bias)


class ConvSpatialBufferConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.register_buffer("bias", torch.randn(4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x) + self.bias)


def _edge(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
    exported = torch.export.export(module.eval(), inputs)
    return to_edge(
        exported,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )


def _count(graph_module: torch.fx.GraphModule, target: object) -> int:
    return sum(
        node.op == "call_function" and node.target == target
        for node in graph_module.graph.nodes
    )


def test_conv_chain_folds_to_boundary_copies() -> None:
    torch.manual_seed(0)
    module = ConvChain().eval()
    inputs = (torch.randn(1, 4, 8, 8),)
    expected = module(*inputs)
    edge = _edge(module, inputs)
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program())

    transformed = edge.transform([layout_pass])
    graph_module = transformed.exported_program().graph_module

    assert _count(graph_module, exir_ops.edge.channels_last.convolution.default) == 2
    assert _count(graph_module, exir_ops.edge.channels_last.permute_copy.default) == 2
    assert layout_pass.report.candidate_anchor_count == 2
    assert layout_pass.report.converted_anchor_count == 2
    assert layout_pass.report.inserted_copy_count == 4
    assert layout_pass.report.eliminated_copy_count == 2
    assert layout_pass.report.boundary_copy_count == 2
    assert layout_pass.report.internal_copy_count == 0
    assert layout_pass.report.unknown_copy_count == 0
    assert layout_pass.report.boundary_copy_bytes == 2048
    assert layout_pass.report.internal_copy_bytes == 0
    assert layout_pass.report.unknown_copy_bytes == 0
    assert layout_pass.report.copies_with_unknown_size == 0
    actual = transformed.exported_program().module()(*inputs)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_strict_rejects_internal_layout_copy() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    edge = _edge(ConvThenLinear().eval(), inputs)

    with pytest.raises(RuntimeError, match="left .* internal"):
        ToContiguousChannelsLastPass(edge.exported_program(), strict=True).call(
            edge.exported_program().graph_module
        )


def test_strict_rejects_supported_anchor_with_noncontiguous_dim_order() -> None:
    module = ConvOnly().eval().to(memory_format=torch.channels_last)
    inputs = (torch.randn(1, 4, 8, 8).to(memory_format=torch.channels_last),)
    exported = torch.export.export(module, inputs)
    edge = to_edge(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    layout_pass = ToContiguousChannelsLastPass(
        edge.exported_program(),
        strict=True,
    )

    with pytest.raises(RuntimeError, match="0 converted of 1 candidate anchors"):
        layout_pass.call(edge.exported_program().graph_module)

    assert layout_pass.report.candidate_anchor_count == 1
    assert layout_pass.report.converted_anchor_count == 0


def test_strict_rejects_unknown_boundary_copy_size() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    exported = torch.export.export(
        ConvOnly().eval(),
        inputs,
        dynamic_shapes={"x": {2: torch.export.Dim("height", min=4, max=16)}},
    )
    edge = to_edge(
        exported,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program(), strict=True)

    with pytest.raises(RuntimeError, match="unknown sizes"):
        layout_pass.call(edge.exported_program().graph_module)

    assert layout_pass.report.boundary_copy_count == 2
    assert layout_pass.report.boundary_copy_bytes == 0
    assert layout_pass.report.copies_with_unknown_size == 2


def test_dynamic_conv_chain_eliminates_internal_layout_copies() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    exported = torch.export.export(
        ConvChain().eval(),
        inputs,
        dynamic_shapes={"x": {2: torch.export.Dim("height", min=4, max=16)}},
    )
    edge = to_edge(
        exported,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program())

    edge.transform([layout_pass])

    assert layout_pass.report.boundary_copy_count == 2
    assert layout_pass.report.internal_copy_count == 0
    assert layout_pass.report.copies_with_unknown_size == 2


def test_dynamic_view_does_not_hide_input_boundary_copy() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    exported = torch.export.export(
        DynamicViewConv().eval(),
        inputs,
        dynamic_shapes={"x": {2: torch.export.Dim("height", min=4, max=16)}},
    )
    edge = to_edge(
        exported,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program())

    edge.transform([layout_pass])

    assert layout_pass.report.boundary_copy_count == 2
    assert layout_pass.report.internal_copy_count == 0
    assert layout_pass.report.copies_with_unknown_size == 2


def test_backend_can_block_layout_propagation_at_a_node() -> None:
    torch.manual_seed(0)
    module = ConvChain().eval()
    inputs = (torch.randn(1, 4, 8, 8),)
    expected = module(*inputs)
    edge = _edge(module, inputs)
    layout_pass = ToContiguousChannelsLastPass(
        edge.exported_program(),
        can_propagate=lambda node: node.target != exir_ops.edge.aten.relu.default,
    )

    transformed = edge.transform([layout_pass])

    assert layout_pass.report.boundary_copy_count == 2
    assert layout_pass.report.internal_copy_count == 2
    actual = transformed.exported_program().module()(*inputs)
    torch.testing.assert_close(actual, expected)


def test_backend_barrier_blocks_view_reordering() -> None:
    module = DynamicViewConv().eval()
    inputs = (torch.randn(1, 4, 8, 8),)
    expected = module(*inputs)
    edge = _edge(module, inputs)
    layout_pass = ToContiguousChannelsLastPass(
        edge.exported_program(),
        can_propagate=lambda node: node.target
        not in (
            exir_ops.edge.aten.view.default,
            exir_ops.edge.aten.view_copy.default,
        ),
    )

    transformed = edge.transform([layout_pass])

    assert layout_pass.report.boundary_copy_count == 1
    assert layout_pass.report.internal_copy_count == 1
    actual = transformed.exported_program().module()(*inputs)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("module", [ConvChannelBias(), PadConv()])
def test_one_sided_propagation_reaches_graph_boundary(
    module: torch.nn.Module,
) -> None:
    torch.manual_seed(0)
    module.eval()
    inputs = (torch.randn(1, 4, 8, 8),)
    expected = module(*inputs)
    edge = _edge(module, inputs)
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program())

    transformed = edge.transform([layout_pass])

    assert layout_pass.report.boundary_copy_count == 2
    assert layout_pass.report.internal_copy_count == 0
    assert layout_pass.report.unknown_copy_count == 0
    actual = transformed.exported_program().module()(*inputs)
    assert torch.allclose(actual, expected, atol=1e-6)

    if isinstance(module, PadConv):
        assert (
            _count(
                transformed.exported_program().graph_module,
                exir_ops.edge.aten.constant_pad_nd.default,
            )
            == 1
        )


def test_softmax_blocks_layout_propagation() -> None:
    module = ConvSoftmax().eval()
    inputs = (torch.randn(1, 4, 8, 8),)
    expected = module(*inputs)
    edge = _edge(module, inputs)
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program())

    transformed = edge.transform([layout_pass])

    assert layout_pass.report.boundary_copy_count == 1
    assert layout_pass.report.internal_copy_count == 1
    softmax = next(
        node
        for node in transformed.exported_program().graph.nodes
        if node.target
        in (exir_ops.edge.aten._softmax.default, exir_ops.edge.aten.softmax.int)
    )
    assert softmax.args[1] in (-1, 3)
    actual = transformed.exported_program().module()(*inputs)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "module, inputs",
    [
        (
            ConvRuntimeBiasConv(),
            (torch.randn(1, 4, 8, 8), torch.randn(4, 1, 1)),
        ),
        (ConvSpatialBufferConv(), (torch.randn(1, 4, 8, 8),)),
    ],
)
def test_boundary_propagation_rejects_unsafe_broadcast_rewrites(
    module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
) -> None:
    module.eval()
    expected = module(*inputs)
    edge = _edge(module, inputs)
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program())

    transformed = edge.transform([layout_pass])

    assert layout_pass.report.boundary_copy_count == 2
    assert layout_pass.report.internal_copy_count == 2
    actual = transformed.exported_program().module()(*inputs)
    torch.testing.assert_close(actual, expected)


def test_layout_copy_report_is_idempotent() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    edge = _edge(ConvChain().eval(), inputs)
    first_pass = ToContiguousChannelsLastPass(edge.exported_program())
    transformed = edge.transform([first_pass])
    second_pass = ToContiguousChannelsLastPass(transformed.exported_program())

    transformed.transform([second_pass])

    assert second_pass.report.inserted_copy_count == 0
    assert second_pass.report.eliminated_copy_count == 0
    assert second_pass.report.candidate_anchor_count == 0
    assert second_pass.report.converted_anchor_count == 0


def test_user_permute_is_not_reported_as_layout_copy() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    edge = _edge(UserPermute(), inputs)
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program(), op_map={})

    transformed = edge.transform([layout_pass])
    permutes = [
        node
        for node in transformed.exported_program().graph.nodes
        if node.target == exir_ops.edge.aten.permute_copy.default
    ]

    assert len(permutes) == 1
    assert (
        _count(
            transformed.exported_program().graph_module,
            exir_ops.edge.channels_last.permute_copy.default,
        )
        == 0
    )
    assert layout_pass.report.inserted_copy_count == 0
    assert layout_pass.report.boundary_copy_count == 0
    assert layout_pass.report.internal_copy_count == 0
