# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
from executorch.examples.qualcomm.oss_scripts.moshi.model.static_conv import (
    StaticStreamingConv1d,
)
from executorch.examples.qualcomm.oss_scripts.moshi.model.static_convtr import (
    StaticStreamingConvTranspose1d,
)
from moshi.modules.seanet import SEANetDecoder, SEANetResnetBlock
from moshi.modules.streaming import StreamingAdd
from moshi.utils.compile import torch_compile_lazy
from torch import nn


class StaticSEANetResnetBlock(SEANetResnetBlock):
    """Residual block from SEANet model.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection.
    """

    # Static Mimi Changes:
    # 1) Replace Conv with Static Conv
    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],  # noqa: B006
        dilations: tp.List[int] = [1, 1],  # noqa: B006
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},  # noqa: B006
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},  # noqa: B006
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super(SEANetResnetBlock, self).__init__()
        assert len(kernel_sizes) == len(
            dilations
        ), "Number of kernel sizes should match number of dilations"
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                StaticStreamingConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                    ignore_previous=(i == 1),
                ),
            ]
        self.block = nn.Sequential(*block)
        self.add = StreamingAdd()
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = StaticStreamingConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

    # Static Mimi Changes:
    # 1) Pass in previous to Conv. Return the output and updated previous.
    def forward(self, x, previous):
        block_list = list(self.block.children())
        assert (
            len(block_list) == 4
        ), "Expect block list to have 4 modules, check if model is changed"
        assert isinstance(block_list[1], StaticStreamingConv1d)
        assert isinstance(block_list[3], StaticStreamingConv1d)

        u = self.shortcut(x)

        x = block_list[0](x)
        x, previous = block_list[1](x, previous)
        x = block_list[2](x)
        v = block_list[3](x)
        return self.add(u, v), previous


class StaticSEANetDecoder(SEANetDecoder):
    """SEANet decoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple.
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the decoder, it corresponds to the N last blocks.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """

    # Static Mimi Changes:
    # 1) Replace Conv and Conv Transpose with Static Conv and Static Conv Transpose. Main difference is that Static version does not store any states in the model. Instead, pass state as output and feed it back in during the next execution.
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],  # noqa: B006
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},  # noqa: B006
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},  # noqa: B006
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        disable_norm_outer_blocks: int = 0,
        trim_right_ratio: float = 1.0,
    ):
        super(SEANetDecoder, self).__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert (
            self.disable_norm_outer_blocks >= 0
            and self.disable_norm_outer_blocks <= self.n_blocks
        ), (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model = [
            StaticStreamingConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm=(
                    "none" if self.disable_norm_outer_blocks == self.n_blocks else norm
                ),
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = (
                "none"
                if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1)
                else norm
            )
            # Add upsampling layers
            model += [
                act(**activation_params),
                StaticStreamingConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    StaticSEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=block_norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            mult //= 2

        # Add final layers
        model += [
            act(**activation_params),
            StaticStreamingConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        self.model = nn.Sequential(*model)

    # Static Mimi Changes
    # 1) Pass in all states related variables(e.g., partial, previous) as input and output.
    @torch_compile_lazy
    def forward(
        self,
        z,
        partial_convtr_1,
        partial_convtr_2,
        partial_convtr_3,
        partial_convtr_4,
        previous_conv_0,
        previous_conv_1,
        previous_conv_3,
        previous_conv_5,
        previous_conv_7,
        previous_conv_9,
    ):
        model_list = list(self.model.children())
        assert (
            len(model_list) == 15
        ), "Expect to have 15 submodules, check if model is changed."
        z, previous_conv_0 = model_list[0](z, previous_conv_0)
        z = model_list[1](z)
        z, partial_convtr_1 = model_list[2](z, partial_convtr_1)
        z, previous_conv_1 = model_list[3](z, previous_conv_1)
        z = model_list[4](z)
        z, partial_convtr_2 = model_list[5](z, partial_convtr_2)
        z, previous_conv_3 = model_list[6](z, previous_conv_3)
        z = model_list[7](z)
        z, partial_convtr_3 = model_list[8](z, partial_convtr_3)
        z, previous_conv_5 = model_list[9](z, previous_conv_5)
        z = model_list[10](z)
        z, partial_convtr_4 = model_list[11](z, partial_convtr_4)
        z, previous_conv_7 = model_list[12](z, previous_conv_7)
        z = model_list[13](z)
        z, previous_conv_9 = model_list[14](z, previous_conv_9)

        return (
            z,
            partial_convtr_1,
            partial_convtr_2,
            partial_convtr_3,
            partial_convtr_4,
            previous_conv_0,
            previous_conv_1,
            previous_conv_3,
            previous_conv_5,
            previous_conv_7,
            previous_conv_9,
        )
