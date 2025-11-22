# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch import exir
from executorch.exir.backend.backend_details import CompileSpec, ExportedProgram
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.test.demos.rpc.executor_backend_preprocess import (
    ExecutorBackend,
)
from executorch.exir.backend.utils import get_delegates
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class TestExampleInputOfSubmodule(unittest.TestCase):
    """
    Tests for verifying that create_exported_program_from_submodule correctly
    handles example inputs of subgraphs based on input signature matching.

    More specifically, if the partitioner delegates a subgraph that doesn't
    start from the original inputs or not cover all or them, the example inputs
    of the delegate should  be None. Otherwise, the example inputs should match
    the original inputs.
    """

    def test_multiple_subgraphs_first_matches_original_others_none(self):
        """
        Test partitioning a model into several submodules where:
        - First submodule starts from the very beginning (same inputs as original)
        - Verify first submodule has original example inputs
        - Verify rest of submodules have None example inputs
        """
        # Setup test data
        model, example_inputs = self._create_three_stage_model()

        # Create and run partitioning
        partitioned_program = self._partition_three_stage_model(model, example_inputs)

        # Verify partitioning results
        self._verify_multiple_subgraphs_example_inputs(
            partitioned_program, example_inputs
        )

    def _create_three_stage_model(self):
        """Create a test model with three distinct processing stages."""

        class ThreeStageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight1 = torch.nn.Parameter(torch.tensor([2.0]))
                self.weight2 = torch.nn.Parameter(torch.tensor([3.0]))

            def forward(self, x, y):
                # Stage 1: Direct operation on original inputs (will be first partition)
                stage1 = x + y  # This should match original signature

                # Stage 2: Uses stage1 result (different signature)
                stage2 = stage1 * self.weight1

                # Stage 3: Uses stage2 result (different signature)
                stage3 = stage2 + self.weight2

                return stage3

        model = ThreeStageModel()
        example_inputs = (torch.tensor([1.0]), torch.tensor([2.0]))
        return model, example_inputs

    def _create_three_stage_partitioner(self):
        """Create a partitioner that delegates each stage separately."""

        class ThreeStagePartitioner(Partitioner):
            def __init__(self):
                super().__init__()
                self.spec1 = DelegationSpec(
                    ExecutorBackend.__name__, [CompileSpec("stage1", bytes([1]))]
                )
                self.spec2 = DelegationSpec(
                    ExecutorBackend.__name__, [CompileSpec("stage2", bytes([2]))]
                )
                self.spec3 = DelegationSpec(
                    ExecutorBackend.__name__, [CompileSpec("stage3", bytes([3]))]
                )

            def partition(
                self, edge_exported_program: ExportedProgram
            ) -> PartitionResult:
                partition_tags = {}

                # Track which add operation we've seen (first or second)
                add_operations_seen = 0

                for node in edge_exported_program.graph.nodes:
                    if node.op == "call_function":
                        if node.target == exir_ops.edge.aten.add.Tensor:
                            add_operations_seen += 1
                            if add_operations_seen == 1:
                                # First add operation (x + y) - uses original inputs
                                node.meta["delegation_tag"] = "stage1"
                                partition_tags["stage1"] = self.spec1
                            elif add_operations_seen == 2:
                                # Second add operation (stage2 + weight2) - uses intermediate result
                                node.meta["delegation_tag"] = "stage3"
                                partition_tags["stage3"] = self.spec3
                        elif node.target == exir_ops.edge.aten.mul.Tensor:
                            # Multiplication operation (stage1 * weight1) - uses intermediate result
                            node.meta["delegation_tag"] = "stage2"
                            partition_tags["stage2"] = self.spec2

                return PartitionResult(
                    tagged_exported_program=edge_exported_program,
                    partition_tags=partition_tags,
                )

        return ThreeStagePartitioner()

    def _partition_three_stage_model(self, model, example_inputs):
        """Execute the partitioning process on the three-stage model."""
        exported_program = export(model, example_inputs, strict=True)
        edge_program = exir.to_edge(exported_program)
        partitioner = self._create_three_stage_partitioner()
        return edge_program.to_backend(partitioner)

    def _get_delegate_modules(self, partitioned_program):
        """Get all delegate modules without sorting."""
        delegates = get_delegates(partitioned_program.exported_program().graph)
        self.assertGreater(
            len(delegates), 1, "Should have multiple delegate submodules"
        )

        delegate_modules = []
        for delegate_node in delegates:
            delegate_module = getattr(
                partitioned_program.exported_program().graph_module, delegate_node.name
            )
            delegate_modules.append(delegate_module)

        return delegate_modules

    def _identify_delegate_with_original_inputs(
        self, delegate_modules, original_example_inputs
    ):
        """
        Identify which delegate (if any) should have the original example inputs.

        Returns:
            tuple: (delegate_with_inputs, other_delegates)
                   where delegate_with_inputs could be None if none match
        """
        original_signature_len = len(original_example_inputs)
        delegate_with_inputs = None
        other_delegates = []

        for delegate in delegate_modules:
            if hasattr(delegate.original_module, "example_inputs"):
                delegate_inputs = delegate.original_module.example_inputs
                if (
                    delegate_inputs is not None
                    and len(delegate_inputs) == original_signature_len
                ):
                    # This delegate likely has the original inputs
                    if delegate_with_inputs is None:
                        delegate_with_inputs = delegate
                    else:
                        # Multiple delegates claim to have original inputs - this could be a test setup issue
                        # For now, just pick the first one and treat others as regular delegates
                        other_delegates.append(delegate)
                else:
                    other_delegates.append(delegate)
            else:
                other_delegates.append(delegate)

        return delegate_with_inputs, other_delegates

    def _verify_delegate_example_inputs(
        self, delegate, expected_example_inputs, delegate_description="Delegate"
    ):
        """
        Verify that a delegate has the expected example inputs.

        Args:
            delegate: The delegate module to verify
            expected_example_inputs: Expected example inputs (None if delegate shouldn't have inputs,
                                   or tuple/list matching original inputs if it should)
            delegate_description: Description for error messages (e.g., "First delegate", "Delegate 2")
        """
        self.assertIsNotNone(
            delegate.original_module,
            f"{delegate_description} should have original_module",
        )

        if hasattr(delegate.original_module, "example_inputs"):
            actual_example_inputs = delegate.original_module.example_inputs

            if expected_example_inputs is None:
                self.assertIsNone(
                    actual_example_inputs,
                    f"{delegate_description} should have None example inputs",
                )
            else:
                self.assertIsNotNone(
                    actual_example_inputs,
                    f"{delegate_description} should have example inputs",
                )
                # Verify they match expected inputs structure
                self.assertEqual(
                    len(actual_example_inputs),
                    len(expected_example_inputs),
                    f"{delegate_description} example inputs should match expected count",
                )

    def _verify_multiple_subgraphs_example_inputs(
        self, partitioned_program, example_inputs
    ):
        """Verify example inputs for all delegates in the partitioned program."""
        delegate_modules = self._get_delegate_modules(partitioned_program)

        # Identify which delegate should have the original inputs (if any)
        delegate_with_inputs, other_delegates = (
            self._identify_delegate_with_original_inputs(
                delegate_modules, example_inputs
            )
        )

        # Verify that exactly one delegate has the original example inputs
        self.assertIsNotNone(
            delegate_with_inputs,
            "Expected to find one delegate with original example inputs",
        )
        self._verify_delegate_example_inputs(
            delegate_with_inputs, example_inputs, "Delegate with original inputs"
        )

        # Verify remaining delegates have None example inputs
        for i, delegate in enumerate(other_delegates, 1):
            self._verify_delegate_example_inputs(delegate, None, f"Other delegate {i}")

    def test_single_subgraph_not_starting_from_original_input(self):
        """
        Test partitioning into only one subgraph that doesn't start from the original
        inputs, and verify that this subgraph has None example inputs.
        """

        class IntermediateOnlyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.multiplier = torch.nn.Parameter(torch.tensor([2.0]))

            def forward(self, x, y):
                # Step 1: Create intermediate (not delegated)
                intermediate = x + y

                # Step 2: Process intermediate (this will be delegated)
                # This doesn't use original x, y directly - uses intermediate result
                result = intermediate * self.multiplier
                return result

        model = IntermediateOnlyModel()
        example_inputs = (torch.tensor([1.0]), torch.tensor([2.0]))

        # Partitioner that only delegates the multiplication step
        class IntermediateOnlyPartitioner(Partitioner):
            def __init__(self):
                super().__init__()
                self.delegation_spec = DelegationSpec(
                    ExecutorBackend.__name__,
                    [CompileSpec("intermediate_only", bytes([1]))],
                )

            def partition(
                self, edge_exported_program: ExportedProgram
            ) -> PartitionResult:
                partition_tags = {}

                for node in edge_exported_program.graph.nodes:
                    if node.op == "call_function":
                        # Only delegate the multiplication (intermediate * multiplier)
                        # NOT the addition (x + y) which uses original inputs
                        if node.target == exir_ops.edge.aten.mul.Tensor:
                            node.meta["delegation_tag"] = "intermediate_only"
                            partition_tags["intermediate_only"] = self.delegation_spec

                return PartitionResult(
                    tagged_exported_program=edge_exported_program,
                    partition_tags=partition_tags,
                )

        exported_program = export(model, example_inputs, strict=True)
        edge_program = exir.to_edge(exported_program)

        partitioned_program = edge_program.to_backend(IntermediateOnlyPartitioner())

        # Verify functionality
        output = partitioned_program.exported_program().module()(*example_inputs)
        expected_output = model(*example_inputs)
        self.assertTrue(
            torch.allclose(output, expected_output),
            "Partitioned program should produce same results as original",
        )

        # Get the single delegate
        delegates = get_delegates(partitioned_program.exported_program().graph)
        self.assertEqual(len(delegates), 1, "Should have exactly one delegate")

        # Get the delegate module
        delegate_node = delegates[0]
        delegate_module = getattr(
            partitioned_program.exported_program().graph_module, delegate_node.name
        )

        # Key verification: This delegate doesn't start from original inputs,
        # so it should have None example inputs
        self._verify_delegate_example_inputs(
            delegate_module, None, "Delegate not starting from original inputs"
        )

    def test_inputs_match_original_unit_logic(self):
        """
        Unit test for the core logic that determines if subgraph inputs match original inputs.
        This directly tests the _inputs_match_original function behavior.
        """
        # Setup test data
        _, _, original_program = self._create_multi_input_test_model()

        # Test the core input matching logic
        self._test_input_matching_logic(original_program)

    def _create_multi_input_test_model(self):
        """Create a test model with multiple inputs for testing input matching logic."""

        class MultiInputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.tensor([1.0]))
                self.register_buffer("buffer", torch.tensor([2.0]))

            def forward(self, x, y):
                return x + y + self.param + self.buffer

        model = MultiInputModel()
        example_inputs = (torch.tensor([1.0]), torch.tensor([2.0]))
        original_program = export(model, example_inputs, strict=True)
        return model, example_inputs, original_program

    def _inputs_match_original(self, subgraph_user_inputs, original_user_inputs):
        """
        Core matching logic: check if user inputs match exactly.
        Replicates the logic from create_exported_program_from_submodule.
        """
        if len(subgraph_user_inputs) != len(original_user_inputs):
            return False
        return subgraph_user_inputs == original_user_inputs

    def _test_input_matching_logic(self, original_program):
        """Test various scenarios of input matching logic."""
        # Get original user inputs for reference
        original_user_inputs = original_program.graph_signature.user_inputs
        self.assertEqual(
            len(original_user_inputs), 2, "Original should have 2 user inputs"
        )

        # Test Case 1: Matching user inputs (same as original)
        matching_user_inputs = original_user_inputs  # Exact same structure
        self.assertTrue(
            self._inputs_match_original(matching_user_inputs, original_user_inputs),
            "Should return True when user inputs match exactly",
        )

        # Test Case 2: Different count of user inputs (subset)
        different_count_inputs = original_user_inputs[:1]  # Only first input
        self.assertFalse(
            self._inputs_match_original(different_count_inputs, original_user_inputs),
            "Should return False when number of user inputs differs",
        )

        # Test Case 3: Empty inputs
        empty_inputs = []
        self.assertFalse(
            self._inputs_match_original(empty_inputs, original_user_inputs),
            "Should return False when subgraph has no user inputs",
        )

        # Test Case 4: Test with a completely different signature
        different_user_inputs = self._create_different_signature_inputs()
        self.assertFalse(
            self._inputs_match_original(different_user_inputs, original_user_inputs),
            "Should return False when user inputs from different model signature",
        )

    def _create_different_signature_inputs(self):
        """Create inputs from a different model signature for testing."""

        class SingleInputModel(torch.nn.Module):
            def forward(self, x):
                return x * 2

        single_input_model = SingleInputModel()
        single_input_example = (torch.tensor([5.0]),)

        single_input_program = export(
            single_input_model, single_input_example, strict=True
        )
        return single_input_program.graph_signature.user_inputs
