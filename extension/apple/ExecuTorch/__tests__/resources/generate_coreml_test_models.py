"""Generate CoreML-delegated test fixtures for the Swift/ObjC bindings.

Currently produces:
  - add_coreml.pte: a single-method CoreML-delegated tensor-add model whose
    forward(x, y) returns x + y.
  - add_mul_coreml.pte: a two-method CoreML-delegated model exposing
    forward(x, y) = x + y and mul(x, y) = x * y. Used to exercise mixed
    load(options:) / load(_:options:) sequences where one method is loaded
    explicitly with its own options and another is loaded lazily, so the
    C++ Module's stored backend_options_ is consulted during the lazy path.
    A non-delegated or single-method fixture does not reach that code path.

Usage:
    python extension/apple/ExecuTorch/__tests__/resources/generate_coreml_test_models.py

This script is invoked by scripts/build_apple_frameworks.sh in CI before
`swift test` so the fixtures are always present in CI runs. The output .pte
files are gitignored; local developers who want to run the CoreML-dependent
tests should run this script once.
"""

import os

import torch
from executorch.backends.apple.coreml.compiler.coreml_preprocess import (
    CoreMLBackend,
    MULTIMETHOD_WEIGHT_SHARING_STRATEGY,
)
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch import nn


class AddModule(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class MulModule(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


def _write_pte(exec_program, filename: str) -> None:
    out_path = os.path.join(os.path.dirname(__file__), filename)
    with open(out_path, "wb") as f:
        exec_program.write_to_file(f)
    print(f"Wrote {out_path} ({os.path.getsize(out_path)} bytes)")


def main() -> None:
    example_inputs = (torch.tensor([1.0]), torch.tensor([1.0]))

    # Single-method add model.
    ep_add = torch.export.export(AddModule().eval(), example_inputs)
    add_only = to_edge_transform_and_lower(
        ep_add,
        partitioner=[CoreMLPartitioner()],
    ).to_executorch()
    _write_pte(add_only, "add_coreml.pte")

    # Two-method model: forward (add) and mul. Both are CoreML-delegated so
    # each has its own per-delegate option set to query at load time.
    # Disable multi-method weight sharing so the two methods produce
    # independent CoreML programs
    multi_method_compile_specs = CoreMLBackend.generate_compile_specs()
    multi_method_compile_specs.append(
        CoreMLBackend.generate_multimethod_weight_sharing_strategy_compile_spec(
            MULTIMETHOD_WEIGHT_SHARING_STRATEGY.DISABLED
        )
    )
    ep_mul = torch.export.export(MulModule().eval(), example_inputs)
    add_mul = to_edge_transform_and_lower(
        {"forward": ep_add, "mul": ep_mul},
        partitioner=[CoreMLPartitioner(compile_specs=multi_method_compile_specs)],
    ).to_executorch()
    _write_pte(add_mul, "add_mul_coreml.pte")


if __name__ == "__main__":
    main()
