# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AOT export for the RISC-V Phase 1.0 smoke test.

Exports a trivial ``torch.add`` module to a BundledProgram (.bpte) that the
portable executor_runner can load on a riscv64 target and verify against the
embedded reference output, emitting ``Test_result: PASS`` on success.
"""

import argparse
from pathlib import Path

import torch
from executorch.devtools import BundledProgram
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import to_edge_transform_and_lower
from torch.export import export


class AddModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("add_riscv.bpte"),
        help="Output .bpte path",
    )
    args = parser.parse_args()

    model = AddModule().eval()
    example_inputs = (torch.ones(1, 4), torch.full((1, 4), 2.0))

    exported = export(model, example_inputs)
    et_program = to_edge_transform_and_lower(exported).to_executorch()

    test_inputs = [
        (torch.ones(1, 4), torch.full((1, 4), 2.0)),
        (torch.full((1, 4), 3.0), torch.full((1, 4), 4.0)),
    ]
    test_suite = MethodTestSuite(
        method_name="forward",
        test_cases=[
            MethodTestCase(inputs=inp, expected_outputs=(model(*inp),))
            for inp in test_inputs
        ],
    )

    bundled = BundledProgram(et_program, [test_suite])
    serialized = serialize_from_bundled_program_to_flatbuffer(bundled)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(serialized)
    print(f"Wrote {args.output} ({len(serialized)} bytes)")


if __name__ == "__main__":
    main()
