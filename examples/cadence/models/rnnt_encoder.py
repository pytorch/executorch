# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

import torch

from executorch.backends.cadence.aot.ops_registrations import *  # noqa

from typing import List, Optional, Tuple

from executorch.backends.cadence.aot.export_example import export_model
from torchaudio.prototype.models import ConvEmformer


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


if __name__ == "__main__":

    class _TimeReduction(torch.nn.Module):
        def __init__(self, stride: int) -> None:
            super().__init__()
            self.stride = stride

        def forward(
            self, input: torch.Tensor, lengths: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            B, T, D = input.shape
            num_frames = T - (T % self.stride)
            input = input[:, :num_frames, :]
            lengths = lengths.div(self.stride, rounding_mode="trunc")
            T_max = num_frames // self.stride

            output = input.reshape(B, T_max, D * self.stride)
            output = output.contiguous()
            return output, lengths

    class ConvEmformerEncoder(torch.nn.Module):
        def __init__(
            self,
            *,
            input_dim: int,
            output_dim: int,
            segment_length: int,
            kernel_size: int,
            right_context_length: int,
            time_reduction_stride: int,
            transformer_input_dim: int,
            transformer_num_heads: int,
            transformer_ffn_dim: int,
            transformer_num_layers: int,
            transformer_left_context_length: int,
            transformer_dropout: float = 0.0,
            transformer_activation: str = "relu",
            transformer_max_memory_size: int = 0,
            transformer_weight_init_scale_strategy: str = "depthwise",
            transformer_tanh_on_mem: bool = False,
        ) -> None:
            super().__init__()
            self.time_reduction = _TimeReduction(time_reduction_stride)
            self.input_linear = torch.nn.Linear(
                input_dim * time_reduction_stride,
                transformer_input_dim,
                bias=False,
            )
            self.transformer = ConvEmformer(
                transformer_input_dim,
                transformer_num_heads,
                transformer_ffn_dim,
                transformer_num_layers,
                segment_length // time_reduction_stride,
                kernel_size=kernel_size,
                dropout=transformer_dropout,
                ffn_activation=transformer_activation,
                left_context_length=transformer_left_context_length,
                right_context_length=right_context_length // time_reduction_stride,
                max_memory_size=transformer_max_memory_size,
                weight_init_scale_strategy=transformer_weight_init_scale_strategy,
                tanh_on_mem=transformer_tanh_on_mem,
                conv_activation="silu",
            )
            self.output_linear = torch.nn.Linear(transformer_input_dim, output_dim)
            self.layer_norm = torch.nn.LayerNorm(output_dim)

        def forward(
            self, input: torch.Tensor, lengths: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            time_reduction_out, time_reduction_lengths = self.time_reduction(
                input, lengths
            )
            input_linear_out = self.input_linear(time_reduction_out)
            transformer_out, transformer_lengths = self.transformer(
                input_linear_out, time_reduction_lengths
            )
            output_linear_out = self.output_linear(transformer_out)
            layer_norm_out = self.layer_norm(output_linear_out)
            return layer_norm_out, transformer_lengths

        @torch.jit.export
        def infer(
            self,
            input: torch.Tensor,
            lengths: torch.Tensor,
            states: Optional[List[List[torch.Tensor]]],
        ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
            time_reduction_out, time_reduction_lengths = self.time_reduction(
                input, lengths
            )
            input_linear_out = self.input_linear(time_reduction_out)
            (
                transformer_out,
                transformer_lengths,
                transformer_states,
            ) = self.transformer.infer(input_linear_out, time_reduction_lengths, states)
            output_linear_out = self.output_linear(transformer_out)
            layer_norm_out = self.layer_norm(output_linear_out)
            return layer_norm_out, transformer_lengths, transformer_states

    # Instantiate model
    time_reduction_stride = 4
    encoder = ConvEmformerEncoder(
        input_dim=80,
        output_dim=256,
        segment_length=4 * time_reduction_stride,
        kernel_size=7,
        right_context_length=1 * time_reduction_stride,
        time_reduction_stride=time_reduction_stride,
        transformer_input_dim=128,
        transformer_num_heads=4,
        transformer_ffn_dim=512,
        transformer_num_layers=1,
        transformer_left_context_length=10,
        transformer_tanh_on_mem=True,
    )

    # Batch size
    batch_size = 1

    max_input_length = 100
    input_dim = 80
    right_context_length = 4

    # Dummy inputs
    transcriber_input = torch.rand(
        batch_size, max_input_length + right_context_length, input_dim
    )
    transcriber_lengths = torch.randint(1, max_input_length + 1, (batch_size,))

    example_inputs = (
        transcriber_input,
        transcriber_lengths,
    )

    export_model(encoder, example_inputs)
