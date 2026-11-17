# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export an add/multiply model with bundled inputs and reference outputs."""

import argparse
from pathlib import Path

import torch
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import to_edge_transform_and_lower


class SmokeModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y, x * y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = SmokeModel().eval()
    inputs = (
        torch.tensor([[-4.0, -1.5, 0.0, 0.25, 1.0, 2.5, 7.0, 16.0]]),
        torch.tensor([[1.0, 3.0, -2.0, 0.25, -0.5, 4.0, -9.0, 0.125]]),
    )
    program = to_edge_transform_and_lower(
        torch.export.export(model, inputs)
    ).to_executorch()
    bundle = BundledProgram(
        program,
        [MethodTestSuite("forward", [MethodTestCase(inputs, model(*inputs))])],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(serialize_from_bundled_program_to_flatbuffer(bundle))
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()
