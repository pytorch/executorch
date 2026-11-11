# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a quantized mul that runs on the esp_nn:: op library, as a bundled program."""

import argparse
from pathlib import Path

import torch
from executorch.backends.espressif.test.tester import EspTester
from executorch.backends.test.harness.stages import StageType
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from torch.utils._pytree import tree_flatten


class Mul(torch.nn.Module):
    def forward(self, x, y):
        return x * y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = (torch.linspace(-1.5, 2.5, 17), torch.linspace(3, -4, 17))
    tester = EspTester(Mul(), inputs)
    tester.quantize().export().to_edge().run_passes()
    expected, _ = tree_flatten(
        tester.get_artifact().exported_program().module()(*inputs)
    )
    program = tester.to_executorch().get_artifact(StageType.TO_EXECUTORCH)
    operators = program.executorch_program.execution_plan[0].operators
    if "esp_nn::quantized_mul_int8" not in {op.name for op in operators}:
        raise SystemExit("the mul did not lower to esp_nn::quantized_mul_int8")
    bundle = BundledProgram(
        program, [MethodTestSuite("forward", [MethodTestCase(inputs, expected)])]
    )
    args.output.write_bytes(serialize_from_bundled_program_to_flatbuffer(bundle))


if __name__ == "__main__":
    main()
