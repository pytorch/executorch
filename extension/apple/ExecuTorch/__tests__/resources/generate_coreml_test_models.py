"""Generate CoreML-delegated test fixtures for the Swift/ObjC bindings.

Currently produces:
  - add_coreml.pte: a CoreML-delegated tensor-add model whose forward(x, y)
    returns x + y. Used by ModuleTest.testLoadWithBackendOptionsThenExecuteOnCoreML
    to exercise the BackendOptionsMap lifetime path end-to-end against a
    delegated model (add.pte has no delegates, so the per-delegate option
    lookup path is only exercised with a delegated fixture like this one).

Usage:
    python extension/apple/ExecuTorch/__tests__/resources/generate_coreml_test_models.py

This script is invoked by scripts/build_apple_frameworks.sh in CI before
`swift test` so the fixture is always present in CI runs. The output .pte is
gitignored; local developers who want to run the CoreML-dependent tests should
run this script once.
"""

import os

import torch

from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch import nn


class AddModule(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


def main() -> None:
    model = AddModule().eval()
    example_inputs = (torch.tensor([1.0]), torch.tensor([1.0]))

    ep = torch.export.export(model, example_inputs)
    lowered = to_edge_transform_and_lower(
        ep,
        partitioner=[CoreMLPartitioner()],
    )
    exec_program = lowered.to_executorch()

    out_path = os.path.join(os.path.dirname(__file__), "add_coreml.pte")
    with open(out_path, "wb") as f:
        exec_program.write_to_file(f)
    print(f"Wrote {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main()
