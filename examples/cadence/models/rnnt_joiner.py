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

    class Joiner(torch.nn.Module):
        def __init__(
            self, input_dim: int, output_dim: int, activation: str = "relu"
        ) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
            if activation == "relu":
                # pyre-fixme[4]: Attribute must be annotated.
                self.activation = torch.nn.ReLU()
            elif activation == "tanh":
                self.activation = torch.nn.Tanh()
            else:
                raise ValueError(f"Unsupported activation {activation}")

        def forward(
            self,
            source_encodings: torch.Tensor,
            target_encodings: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            joint_encodings = (
                source_encodings.unsqueeze(2).contiguous()
                + target_encodings.unsqueeze(1).contiguous()
            )
            activation_out = self.activation(joint_encodings)
            output = self.linear(activation_out)
            return output

    # Joiner
    model = Joiner(256, 128)

    # Get dummy joiner inputs
    source_encodings = torch.randn(1, 25, 256)
    target_encodings = torch.randn(1, 10, 256)

    example_inputs = (
        source_encodings,
        target_encodings,
    )

    export_model(model, example_inputs)
