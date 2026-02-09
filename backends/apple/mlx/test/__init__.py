# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MLX delegate op tests.

This package provides a framework for systematically testing individual ops
on the MLX delegate.

Workflow:
1. Generate test files (Python): Creates .pte, input.bin, expected_output.bin
2. Run C++ binary: Executes the .pte with inputs and produces actual_output.bin
3. Compare outputs (Python): Verifies actual outputs match expected outputs

Usage example:
    # Run full test for linear op
    python -m executorch.backends.apple.mlx.test.test_linear run

    # Or step by step:
    python -m executorch.backends.apple.mlx.test.test_linear generate
    ./cmake-out-mlx/backends/apple/mlx/test/op_test_runner \\
        --pte backends/apple/mlx/test/op_tests/linear/model.pte \\
        --input backends/apple/mlx/test/op_tests/linear/input.bin \\
        --output backends/apple/mlx/test/op_tests/linear/actual_output.bin
    python -m executorch.backends.apple.mlx.test.test_linear compare
"""
