# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot.program_builder import ProgramBuilder
from executorch.backends.cadence.aot.to_out_var_pass import CadenceToOutVarPass
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from later.unittest import TestCase


class TestCadenceToOutVarPass(TestCase):
    def test_serialize_with_slice_scatter_inplace(self) -> None:
        """Test that a graph with slice_scatter_ can be serialized after CadenceToOutVarPass."""
        builder = ProgramBuilder()
        # Create input tensor placeholder
        x = builder.placeholder("x", torch.randn(10, dtype=torch.float32))
        # Create source tensor placeholder
        src = builder.placeholder("src", torch.randn(3, dtype=torch.float32))

        # Call slice_scatter_ inplace op
        result = builder.call_operator(
            op=exir_ops.edge.cadence.slice_scatter_.default,
            args=(x, src, 0, 2, 5, 1),
        )
        builder.output([result])

        # Get the edge program
        edge_program = builder.get_edge_program()

        # Apply CadenceToOutVarPass and serialize
        exec_program = edge_program.to_executorch(
            ExecutorchBackendConfig(
                to_out_var_pass=CadenceToOutVarPass(),
            )
        )

        # Verify serialization succeeded by checking the buffer is non-empty
        buffer = exec_program.buffer
        self.assertIsNotNone(buffer)
        self.assertGreater(len(buffer), 0)

    def test_serialize_with_mixed_ops(self) -> None:
        """Test that a graph with mixed ops including slice_scatter_ can be serialized."""
        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(10, dtype=torch.float32))
        y = builder.placeholder("y", torch.randn(10, dtype=torch.float32))
        src = builder.placeholder("src", torch.randn(3, dtype=torch.float32))

        # Add operation
        add_result = builder.call_operator(exir_ops.edge.aten.add.Tensor, (x, y))

        # Slice scatter inplace operation
        scatter_result = builder.call_operator(
            op=exir_ops.edge.cadence.slice_scatter_.default,
            args=(add_result, src, 0, 2, 5, 1),
        )

        builder.output([scatter_result])

        edge_program = builder.get_edge_program()

        exec_program = edge_program.to_executorch(
            ExecutorchBackendConfig(
                to_out_var_pass=CadenceToOutVarPass(),
            )
        )

        buffer = exec_program.buffer
        self.assertIsNotNone(buffer)
        self.assertGreater(len(buffer), 0)

    def test_serialize_with_add_tensor(self) -> None:
        """Test that a simple add graph can be serialized with CadenceToOutVarPass."""
        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(3, 5, dtype=torch.float32))
        y = builder.placeholder("y", torch.randn(3, 5, dtype=torch.float32))

        add = builder.call_operator(exir_ops.edge.aten.add.Tensor, (x, y))
        builder.output([add])

        edge_program = builder.get_edge_program()

        exec_program = edge_program.to_executorch(
            ExecutorchBackendConfig(
                to_out_var_pass=CadenceToOutVarPass(),
            )
        )

        buffer = exec_program.buffer
        self.assertIsNotNone(buffer)
        self.assertGreater(len(buffer), 0)
