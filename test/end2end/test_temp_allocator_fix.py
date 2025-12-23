#!/usr/bin/env python3
"""
Test to verify the fix for temp memory allocation issue in torch.topk operations.

This test specifically checks that the MallocMemoryAllocator fix in pybindings.cpp
resolves the "Memory allocation failed" error when executing operations that
require temporary memory allocation.
"""

import os
import tempfile
from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.runtime import Runtime, Verification
from torch.export import export


class TopKModel(torch.nn.Module):
    """Model that uses torch.topk operation which requires temp memory allocation."""

    def __init__(self, k=3) -> None:
        super().__init__()
        self.k = k

    def forward(self, x) -> "tuple[torch.Tensor, torch.Tensor]":
        # This operation requires temporary memory allocation
        top_values, top_indices = torch.topk(x, self.k)
        return top_values, top_indices


class TopKModelWithOut(torch.nn.Module):
    """Model that uses torch.topk with out parameter which also requires temp memory."""

    def __init__(self, k=3) -> None:
        super().__init__()
        self.k = k

    def forward(self, x) -> "tuple[torch.Tensor, torch.Tensor]":
        top_values = torch.ones(x.shape[0], self.k, dtype=torch.float32)
        top_indices = torch.ones(x.shape[0], self.k, dtype=torch.long)
        torch.topk(x.contiguous(), self.k, out=(top_values, top_indices))
        return top_values, top_indices


def test_topk_without_out_parameter():
    """Test torch.topk without out parameter."""
    print("Testing torch.topk without out parameter...")

    model = TopKModel(k=5)
    example_input = (torch.randn(3, 100),)

    # Export and compile the model
    with torch.no_grad():
        aten_dialect = export(model, example_input)

        backend_dialect = to_edge_transform_and_lower(
            aten_dialect,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            partitioner=[XnnpackPartitioner()],
        )

        executorch_dialect = backend_dialect.to_executorch()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            temp_path = f.name

        try:
            executorch_dialect.save(temp_path)

            # Load and execute with ExecuTorch runtime
            et_runtime = Runtime.get()
            program = et_runtime.load_program(
                Path(temp_path),
                verification=Verification.Minimal,
            )

            forward = program.load_method("forward")
            outputs = forward.execute(example_input)

            print(
                f"✓ Successfully executed topk model: {example_input[0].shape} -> {outputs[0].shape}"
            )
            return True

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def test_topk_with_out_parameter():
    """Test torch.topk with out parameter (original failing case)."""
    print("Testing torch.topk with out parameter...")

    model = TopKModelWithOut(k=3)
    example_input = (torch.randn(2, 256),)

    # Export and compile the model
    with torch.no_grad():
        aten_dialect = export(model, example_input)

        backend_dialect = to_edge_transform_and_lower(
            aten_dialect,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            partitioner=[XnnpackPartitioner()],
        )

        executorch_dialect = backend_dialect.to_executorch()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            temp_path = f.name

        try:
            executorch_dialect.save(temp_path)

            # Load and execute with ExecuTorch runtime
            et_runtime = Runtime.get()
            program = et_runtime.load_program(
                Path(temp_path),
                verification=Verification.Minimal,
            )

            forward = program.load_method("forward")
            outputs = forward.execute(example_input)

            print(
                f"✓ Successfully executed topk model with out parameter: {example_input[0].shape} -> {outputs[0].shape}"
            )
            return True

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def test_larger_topk_operation():
    """Test larger topk operation that would require more temporary memory."""
    print("Testing larger topk operation...")

    model = TopKModel(k=50)
    example_input = (torch.randn(5, 1000),)

    # Export and compile the model
    with torch.no_grad():
        aten_dialect = export(model, example_input)

        backend_dialect = to_edge_transform_and_lower(
            aten_dialect,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            partitioner=[XnnpackPartitioner()],
        )

        executorch_dialect = backend_dialect.to_executorch()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            temp_path = f.name

        try:
            executorch_dialect.save(temp_path)

            # Load and execute with ExecuTorch runtime
            et_runtime = Runtime.get()
            program = et_runtime.load_program(
                Path(temp_path),
                verification=Verification.Minimal,
            )

            forward = program.load_method("forward")
            outputs = forward.execute(example_input)

            print(
                f"✓ Successfully executed large topk model: {example_input[0].shape} -> {outputs[0].shape}"
            )
            return True

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def main():
    """Run all tests to verify the temp memory allocation fix."""
    print("Testing temp memory allocation fix for torch.topk operations")
    print("=" * 60)

    tests = [
        test_topk_without_out_parameter,
        test_topk_with_out_parameter,
        test_larger_topk_operation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print(
            "✓ All tests passed! The temp memory allocation fix is working correctly."
        )
        return True
    else:
        print("✗ Some tests failed. The fix may not be working correctly.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
