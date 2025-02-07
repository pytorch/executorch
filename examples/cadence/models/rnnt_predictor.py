# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

import torch

from executorch.backends.cadence.aot.ops_registrations import *  # noqa

from typing import Tuple

from executorch.backends.cadence.aot.export_example import export_model


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


if __name__ == "__main__":

    class Predictor(torch.nn.Module):
        def __init__(
            self,
            num_symbols: int,
            symbol_embedding_dim: int,
        ) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(num_symbols, symbol_embedding_dim)
            self.relu = torch.nn.ReLU()
            self.linear = torch.nn.Linear(symbol_embedding_dim, symbol_embedding_dim)
            self.layer_norm = torch.nn.LayerNorm(symbol_embedding_dim)

        def forward(
            self,
            input: torch.Tensor,
            lengths: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            input_tb = input.permute(1, 0)
            embedding_out = self.embedding(input_tb)
            relu_out = self.relu(embedding_out)
            linear_out = self.linear(relu_out)
            layer_norm_out = self.layer_norm(linear_out)
            return layer_norm_out.permute(1, 0, 2), lengths

    # Predictor
    model = Predictor(128, 256)
    model.eval()

    # Batch size
    batch_size = 1

    num_symbols = 128
    max_target_length = 10

    # Dummy inputs
    predictor_input = torch.randint(0, num_symbols, (batch_size, max_target_length))
    predictor_lengths = torch.randint(1, max_target_length + 1, (batch_size,))

    example_inputs = (
        predictor_input,
        predictor_lengths,
    )

    export_model(model, example_inputs)
