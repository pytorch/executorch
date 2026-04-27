# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineFP


InputT = Tuple[Any, ...]


@dataclass(frozen=True)
class TransposeCountCase:
    module: torch.nn.Module
    inputs: InputT
    expected_transposes: int


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


cases = {
    "conv1d_rank2": TransposeCountCase(Conv1dModule(), (torch.randn(2, 8),), 2),
    "conv1d_rank3": TransposeCountCase(Conv1dModule(), (torch.randn(1, 2, 8),), 2),
    "conv2d_rank3": TransposeCountCase(Conv2dModule(), (torch.randn(2, 8, 8),), 2),
    "conv2d_rank4": TransposeCountCase(Conv2dModule(), (torch.randn(1, 2, 8, 8),), 2),
    "conv3d_rank4": TransposeCountCase(Conv3dModule(), (torch.randn(2, 6, 6, 6),), 5),
    "conv3d_rank5": TransposeCountCase(
        Conv3dModule(), (torch.randn(1, 2, 6, 6, 6),), 2
    ),
    "linear_rank2": TransposeCountCase(LinearModule(), (torch.randn(2, 8),), 0),
    "linear_rank3": TransposeCountCase(LinearModule(), (torch.randn(2, 2, 8),), 0),
    "linear_rank4": TransposeCountCase(LinearModule(), (torch.randn(1, 2, 2, 8),), 3),
    "matmul_rank2": TransposeCountCase(
        MatmulModule(),
        (torch.randn(2, 3), torch.randn(3, 4)),
        0,
    ),
    "matmul_rank4": TransposeCountCase(
        MatmulModule(),
        (torch.randn(2, 2, 2, 3), torch.randn(2, 2, 3, 4)),
        5,
    ),
    "index_put": TransposeCountCase(
        IndexPutModule(),
        (
            torch.zeros((2, 4), dtype=torch.float32),
            (
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([2, 3], dtype=torch.int32),
            ),
            torch.ones((2,), dtype=torch.float32),
            False,
        ),
        0,
    ),
    "pixel_shuffle": TransposeCountCase(
        PixelShuffleModule(),
        (torch.randn(1, 8, 2, 2),),
        7,
    ),
    "index_select": TransposeCountCase(
        IndexSelectModule(),
        (torch.randn(2, 4, 3), 1, torch.tensor([0, 2], dtype=torch.int32)),
        0,
    ),
    "grouped_conv": TransposeCountCase(
        GroupedConvModule(),
        (torch.randn(1, 4, 8, 8),),
        2,
    ),
    "transpose_conv": TransposeCountCase(
        TransposeConvModule(),
        (torch.randn(1, 2, 8, 8),),
        2,
    ),
    "views": TransposeCountCase(ViewsModule(), (torch.rand(1, 2, 2, 4),), 6),
    "transposes": TransposeCountCase(
        TransposesModule(),
        (torch.randn(1, 2, 3, 4),),
        4,
    ),
    "maxpool2d_dilation": TransposeCountCase(
        MaxPool2dDilatedModule(),
        (torch.randn(1, 2, 8, 8),),
        8,
    ),
    "lstm": TransposeCountCase(
        LstmModule(),
        (torch.randn(2, 4, 8),),
        2,
    ),
    "groupnorm": TransposeCountCase(
        GroupNormModule(),
        (torch.randn(1, 4, 4, 4),),
        5,
    ),
    "multihead_attention_rank2": TransposeCountCase(
        MultiheadAttentionModule(),
        (torch.randn(4, 8),),
        14,
    ),
    "multihead_attention_rank3": TransposeCountCase(
        MultiheadAttentionModule(),
        (torch.randn(2, 4, 8),),
        22,
    ),
    "cumsum_rank3_dim0": TransposeCountCase(
        CumsumModule(),
        (torch.randn(2, 3, 4), 1),
        0,
    ),
    "cumsum_rank4_dim3": TransposeCountCase(
        CumsumModule(),
        (torch.randn(1, 2, 3, 4), 3),
        3,
    ),
    "model_1_conv_maxpool_residual_linear": TransposeCountCase(
        Model1ConvMaxPoolResidualLinear(), (torch.randn(2, 8, 64),), 5
    ),
    "model_2_conv_mha_linear_layernorm": TransposeCountCase(
        Model2ConvMhaLinearLayerNorm(), (torch.randn(2, 8, 32),), 27
    ),
    "model_3_lstm_linear": TransposeCountCase(
        Model3LstmLinear(), (torch.randn(2, 16, 8),), 2
    ),
    "model_4_conv_lstm_linear_layernorm": TransposeCountCase(
        Model4ConvLstmLinearLayerNorm(), (torch.randn(2, 8, 32),), 7
    ),
    "model_5_dwconv_gelu_layernorm_avgpool": TransposeCountCase(
        Model5DwConvGeluLayerNormAvgPool(), (torch.randn(1, 8, 16, 16),), 4
    ),
    "model_6_gru_linear": TransposeCountCase(
        Model6GruLinear(), (torch.randn(2, 16, 8),), 2
    ),
    "model_7_dwconv_batchnorm_linear": TransposeCountCase(
        Model7DwConvBatchNormLinear(), (torch.randn(2, 8, 64),), 3
    ),
    "model_8_conv_batchnorm_maxpool_residual": TransposeCountCase(
        Model8ConvBatchNormMaxPoolResidual(), (torch.randn(1, 8, 16, 16),), 2
    ),
    "model_9_dilated_conv_batchnorm_avgpool_residual": TransposeCountCase(
        Model9DilatedConvBatchNormAvgPoolResidual(), (torch.randn(1, 8, 16, 16),), 2
    ),
    "model_10_dwconv_batchnorm_linear_cat": TransposeCountCase(
        Model10DwConvBatchNormLinearCat(), (torch.randn(2, 8, 64),), 3
    ),
}


cases_channels_last = {
    "conv2d_rank4_channels_last": TransposeCountCase(
        Conv2dModule(),
        (torch.randn(1, 2, 8, 8).to(memory_format=torch.channels_last),),
        0,
    ),
    "conv3d_rank4_channels_last": TransposeCountCase(
        Conv3dModule(),
        (torch.randn(2, 6, 6, 6).to(memory_format=torch.channels_last),),
        4,
    ),
    "conv3d_rank5_channels_last": TransposeCountCase(
        Conv3dModule(),
        (torch.randn(1, 2, 6, 6, 6).to(memory_format=torch.channels_last_3d),),
        1,
    ),
    "linear_rank4_channels_last": TransposeCountCase(
        LinearModule(),
        (torch.randn(1, 2, 2, 8).to(memory_format=torch.channels_last),),
        -1,  # The test crashes before reaching the transpose count
    ),
    "matmul_rank4_channels_last": TransposeCountCase(
        MatmulModule(),
        (
            torch.randn(2, 2, 2, 3).to(memory_format=torch.channels_last),
            torch.randn(2, 2, 3, 4).to(memory_format=torch.channels_last),
        ),
        -1,  # The test crashes before reaching the transpose count
    ),
    "pixel_shuffle_channels_last": TransposeCountCase(
        PixelShuffleModule(),
        (torch.randn(1, 8, 2, 2).to(memory_format=torch.channels_last),),
        5,
    ),
    "grouped_conv_channels_last": TransposeCountCase(
        GroupedConvModule(),
        (torch.randn(1, 4, 8, 8).to(memory_format=torch.channels_last),),
        0,
    ),
    "transpose_conv_channels_last": TransposeCountCase(
        TransposeConvModule(),
        (torch.randn(1, 2, 8, 8).to(memory_format=torch.channels_last),),
        0,
    ),
    "views_channels_last": TransposeCountCase(
        ViewsModule(),
        (torch.rand(1, 2, 2, 4).to(memory_format=torch.channels_last),),
        -1,  # The test crashes before reaching the transpose count
    ),
    "transposes_channels_last": TransposeCountCase(
        TransposesModule(),
        (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),),
        3,
    ),
    "maxpool2d_dilation_channels_last": TransposeCountCase(
        MaxPool2dDilatedModule(),
        (torch.randn(1, 2, 8, 8).to(memory_format=torch.channels_last),),
        6,
    ),
    "groupnorm_channels_last": TransposeCountCase(
        GroupNormModule(),
        (torch.randn(1, 4, 4, 4).to(memory_format=torch.channels_last),),
        4,
    ),
    "cumsum_rank4_dim3_channels_last": TransposeCountCase(
        CumsumModule(),
        (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last), 3),
        -1,  # The test crashes before reaching the transpose count
    ),
}


@common.parametrize("case", cases)
def test_transpose_counts_tosa_FP(case: TransposeCountCase) -> None:
    pipeline = TosaPipelineFP[InputT](case.module, case.inputs, aten_op=[])
    pipeline.count_tosa_ops({"TRANSPOSE": case.expected_transposes})
    pipeline.run()


xfails = {
    "conv3d_rank5_channels_last": "Numerical error",
    "linear_rank4_channels_last": "DecomposeLinearPass: Tries inserting a view not supported in channels last format",
    "matmul_rank4_channels_last": "ToTosaMemoryFormatPass: Tries inserting view not supported in channels last format",
    "views_channels_last": "Torch.export: View not supported by torch.export in channels last format",
    "cumsum_rank4_dim3_channels_last": "DecomposeCumssumPass: Tries inserting a view not supported in channels last format",
}


@common.parametrize("case", cases_channels_last, xfails=xfails)  # type: ignore[arg-type]
def test_transpose_counts_tosa_FP_channels_last(case: TransposeCountCase) -> None:
    pipeline = TosaPipelineFP[InputT](
        case.module,
        case.inputs,
        aten_op=[],
    )
    pipeline.count_tosa_ops({"TRANSPOSE": case.expected_transposes})
    pipeline.run()
