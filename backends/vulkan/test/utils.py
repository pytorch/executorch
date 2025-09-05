# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from collections import OrderedDict
from typing import List, Optional, Tuple

import executorch.backends.vulkan.utils as utils

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.vulkan_preprocess import VulkanBackend
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.xnnpack_preprocess import XnnpackBackend
from executorch.devtools import BundledProgram
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import ExecutorchProgramManager, to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from executorch.extension.pytree import tree_flatten
from torch.export import export, export_for_training


def export_model_to_vulkan(
    model,
    sample_inputs,
    dynamic_shapes=None,
    operator_blocklist=None,
    operator_allowlist=None,
):
    """Helper to export a model to Vulkan backend."""
    compile_options = {}
    export_training_graph = export_for_training(
        model, sample_inputs, strict=True
    ).module()
    program = export(
        export_training_graph,
        sample_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )
    edge_program = to_edge_transform_and_lower(
        program,
        partitioner=[
            VulkanPartitioner(
                compile_options,
                operator_blocklist=operator_blocklist,
                operator_allowlist=operator_allowlist,
            )
        ],
        transform_passes=None,
        compile_config=None,
    )

    executorch_program = edge_program.to_executorch()

    # Check if the delegate ID matches VulkanBackend
    if (
        executorch_program.executorch_program.execution_plan[0].delegates[0].id
        != VulkanBackend.__name__
    ):
        raise RuntimeError(
            f"Expected delegate ID {VulkanBackend.__name__}, but got {executorch_program.executorch_program.execution_plan[0].delegates[0].id}"
        )

    return executorch_program


def export_model_to_xnnpack(model, sample_inputs, dynamic_shapes=None):
    """Helper to export a model to XNNPACK backend."""
    compile_options = {}
    export_training_graph = export_for_training(
        model, sample_inputs, strict=True
    ).module()
    program = export(
        export_training_graph,
        sample_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )
    edge_program = to_edge_transform_and_lower(
        program,
        partitioner=[XnnpackPartitioner(compile_options)],
        transform_passes=None,
        compile_config=None,
    )

    executorch_program = edge_program.to_executorch()

    # Check if the delegate ID matches XnnpackBackend
    if (
        executorch_program.executorch_program.execution_plan[0].delegates[0].id
        != XnnpackBackend.__name__
    ):
        raise RuntimeError(
            f"Expected delegate ID {XnnpackBackend.__name__}, but got {executorch_program.executorch_program.execution_plan[0].delegates[0].id}"
        )

    return executorch_program


def check_outputs_equal(
    model_output, ref_output, atol=1e-03, rtol=1e-03, first_output_only=False
):
    """
    Helper function that checks if model output and reference output are equal with some tolerance.
    Returns True if equal, False otherwise.
    """
    # Convert OrderedDict to list if needed
    if isinstance(ref_output, OrderedDict):
        ref_output = list(ref_output.values())

    # Compare the result from executor and eager mode directly
    if isinstance(ref_output, tuple) or isinstance(ref_output, list):
        # Multiple outputs executor always returns tuple, even if there is one output
        if len(ref_output) != len(model_output):
            return False
        if first_output_only:
            return torch.allclose(model_output[0], ref_output[0], atol=atol, rtol=rtol)
        else:
            for i in range(len(ref_output)):
                if not torch.allclose(
                    model_output[i], ref_output[i], atol=atol, rtol=rtol
                ):
                    return False
            return True
    else:
        # If one output, eager returns tensor while executor tuple of size 1
        return torch.allclose(model_output[0], ref_output, atol=atol, rtol=rtol)


def run_and_check_output(
    reference_model: torch.nn.Module,
    executorch_program: ExecutorchProgramManager,
    sample_inputs: Tuple[torch.Tensor],
    atol=1e-03,
    rtol=1e-01,
    first_output_only=False,
) -> bool:
    """
    Utility function that accepts an already lowered ExecuTorch program, executes it with
    the provided sample input, and checks the output for correctness.

    Args:
        executorch_program: Already lowered ExecutorchProgramManager
        sample_inputs: Sample inputs to run the program with
        reference_model: Reference model to generate reference outputs for comparison
        atol: Absolute tolerance for output comparison
        rtol: Relative tolerance for output comparison
        first_output_only: Whether to compare only the first output

    Returns:
        bool: True if outputs match within tolerance, False otherwise
    """
    # Load the ExecutorTorch program
    executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)

    # Flatten inputs for execution
    inputs_flattened, _ = tree_flatten(sample_inputs)

    # Run the ExecutorTorch program
    model_output = executorch_module.run_method("forward", tuple(inputs_flattened))

    # Generate reference outputs using the reference model
    ref_output = reference_model(*sample_inputs)

    # Check if outputs are equal
    return check_outputs_equal(
        model_output,
        ref_output,
        atol=atol,
        rtol=rtol,
        first_output_only=first_output_only,
    )


def lower_module_and_test_output(
    model: torch.nn.Module,
    sample_inputs: Tuple[torch.Tensor],
    atol=1e-03,
    rtol=1e-01,
    dynamic_shapes=None,
    test_inputs=None,
    first_output_only=False,
    operator_blocklist=None,
    operator_allowlist=None,
) -> bool:
    """
    Helper testing function that takes a torch.nn.Module and lowers it to Vulkan with
    the given sample inputs. It then runs the lowered module and compares its
    outputs with the outputs of the eager module.

    Returns:
        bool: True if all comparisons pass, False otherwise.
    """
    # Export model to Vulkan using the helper function
    executorch_program = export_model_to_vulkan(
        model, sample_inputs, dynamic_shapes, operator_blocklist, operator_allowlist
    )

    executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)

    inputs_flattened, _ = tree_flatten(sample_inputs)

    model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
    ref_output = model(*sample_inputs)

    if not check_outputs_equal(
        model_output,
        ref_output,
        atol=atol,
        rtol=rtol,
        first_output_only=first_output_only,
    ):
        return False

    if test_inputs is not None:
        for test_input in test_inputs:
            test_inputs_flattened, _ = tree_flatten(test_input)
            model_output = executorch_module.run_method(
                "forward", tuple(test_inputs_flattened)
            )
            ref_output = model(*test_input)

            if not check_outputs_equal(
                model_output,
                ref_output,
                atol=atol,
                rtol=rtol,
                first_output_only=first_output_only,
            ):
                return False

    return True


def save_bundled_program(
    model: torch.nn.Module,
    sample_inputs: Tuple[torch.Tensor],
    output_path: str,
    method_name: str = "forward",
    et_program: Optional[ExecutorchProgramManager] = None,
    dynamic_shapes=None,
) -> str:
    """
    Export a bundled .pte file containing the model and test cases.

    Args:
        model: The PyTorch model to export
        sample_inputs: Sample inputs for the model
        output_path: Path where the bundled .pte file should be saved (should end with .bpte)
        method_name: Name of the method to test (default: "forward")
        et_program: Optional pre-exported ExecutorchProgramManager. If None, will export to Vulkan
        dynamic_shapes: Optional dynamic shapes for export

    Returns:
        str: Path to the saved bundled program file
    """
    # If no ExecutorchProgramManager provided, export to Vulkan
    if et_program is None:
        et_program = export_model_to_vulkan(model, sample_inputs, dynamic_shapes)

    # Generate expected outputs by running the model
    expected_outputs = [getattr(model, method_name)(*sample_inputs)]

    # Flatten sample inputs to match expected format
    inputs_flattened, _ = tree_flatten(sample_inputs)

    # Create test suite with the sample inputs and expected outputs
    test_suites = [
        MethodTestSuite(
            method_name=method_name,
            test_cases=[
                MethodTestCase(
                    inputs=inputs_flattened,
                    expected_outputs=expected_outputs,
                )
            ],
        )
    ]

    # Create bundled program
    bp = BundledProgram(et_program, test_suites)

    # Serialize to flatbuffer
    bp_buffer = serialize_from_bundled_program_to_flatbuffer(bp)

    # Ensure output path has correct extension
    if not output_path.endswith(".bpte"):
        output_path = output_path + ".bpte"

    # Write to file
    with open(output_path, "wb") as file:
        file.write(bp_buffer)
    return output_path


def save_executorch_program(
    executorch_program: ExecutorchProgramManager,
    output_path: str,
) -> str:
    """
    Save an ExecutorchProgramManager as a .pte file.

    Args:
        executorch_program: The ExecutorchProgramManager to save
        output_path: Path where the .pte file should be saved (should end with .pte)

    Returns:
        str: Path to the saved .pte file
    """
    # Ensure output path has correct extension
    if not output_path.endswith(".pte"):
        output_path = output_path + ".pte"

    # Write to file
    with open(output_path, "wb") as file:
        executorch_program.write_to_file(file)

    return output_path


def print_occurrences(edge_program, operator_list: List):
    """
    Print the input/output information for all occurrences of specified operators in the edge program.

    Args:
        edge_program: The edge program created by to_edge_transform_and_lower
        operator_list: List of operators to search for in the graph
    """
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    logger.info(
        f"Searching for occurrences of {len(operator_list)} operators in the graph..."
    )

    occurrence_count = 0

    for node in edge_program.exported_program().graph.nodes:
        if utils.is_torch_op_node(node):
            target = node.target
            # Handle auto_functionalized nodes
            if node.target == torch.ops.higher_order.auto_functionalized:
                first_arg = node.args[0]
                if hasattr(first_arg, "name"):
                    target = first_arg.name()
                elif hasattr(first_arg, "__name__"):
                    target = first_arg.__name__

            # Check if this operator is in our list
            if target in operator_list:
                occurrence_count += 1
                logger.info(f"Occurrence {occurrence_count}: {node.format_node()}")

                # Get the node I/O string using the utils function
                try:
                    io_str = utils.node_io_str(node)
                    logger.info(f"  {io_str}")
                except Exception as e:
                    logger.info(f"  Error getting I/O string: {e}")

    if occurrence_count == 0:
        logger.info("No occurrences of the specified operators found in the graph.")
    else:
        logger.info(
            f"Found {occurrence_count} total occurrences of the specified operators."
        )


def op_ablation_test(  # noqa: C901
    model: torch.nn.Module,
    sample_inputs: Tuple[torch.Tensor],
    atol=1e-03,
    rtol=1e-01,
    dynamic_shapes=None,
    test_inputs=None,
    first_output_only=False,
) -> dict:
    """
    Fast binary search utility function to determine which operators work correctly when delegated to Vulkan.

    This function uses a binary search approach to efficiently find bad operators:
    1. Split operators into two halves (least frequent first, most frequent second)
    2. Test each half to see if it produces correct output
    3. Add good halves to known_good_ops and recursively search bad halves
    4. Continue until all operators are classified

    Args:
        model: The PyTorch model to test
        sample_inputs: Sample inputs for the model
        atol: Absolute tolerance for output comparison
        rtol: Relative tolerance for output comparison
        dynamic_shapes: Optional dynamic shapes for export
        test_inputs: Optional additional test inputs
        first_output_only: Whether to compare only the first output

    Returns:
        dict: Dictionary with keys:
            - 'good_operators': List of operators that work correctly
            - 'bad_operators': List of operators that cause failures
            - 'operator_frequencies': Dictionary mapping operators to their occurrence count
            - 'all_operators': List of all unique operators found in the graph
            - 'test_count': Number of tests performed
    """
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    logger.info("Starting fast binary search operator ablation test...")

    # Step 1: Export model to get edge_program and extract operators
    export_training_graph = export_for_training(
        model, sample_inputs, strict=True
    ).module()
    program = export(
        export_training_graph,
        sample_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )
    edge_program = to_edge_transform_and_lower(
        program,
        partitioner=[],  # No partitioner to get the full graph
        transform_passes=None,
        compile_config=None,
    )

    # Step 2: Scan edge_program.graph_module to obtain unique operators and their frequencies
    operator_frequencies = {}
    for node in edge_program.exported_program().graph.nodes:
        if utils.is_torch_op_node(node):
            target = node.target
            # Handle auto_functionalized nodes
            if node.target == torch.ops.higher_order.auto_functionalized:
                first_arg = node.args[0]
                if hasattr(first_arg, "name"):
                    target = first_arg.name()
                elif hasattr(first_arg, "__name__"):
                    target = first_arg.__name__

            if target in operator_frequencies:
                operator_frequencies[target] += 1
            else:
                operator_frequencies[target] = 1

    all_operators = list(operator_frequencies.keys())
    logger.info(f"Found {len(all_operators)} unique operators in the graph")

    # Sort operators by frequency (least frequent first for binary search)
    operators_by_frequency = sorted(
        all_operators, key=lambda op: operator_frequencies[op]
    )

    logger.info("Operator frequencies (sorted by occurrence, least frequent first):")
    for op in operators_by_frequency:
        logger.info(f"  {op}: {operator_frequencies[op]} occurrences")

    # Global test counter
    test_count = 0

    def test_operator_set(ops_to_test: List, known_good_ops: List) -> bool:
        """Test if a set of operators works correctly when combined with known good operators."""
        nonlocal test_count
        test_count += 1

        test_allowlist = known_good_ops + ops_to_test
        logger.info(
            f"Test {test_count}: Testing {len(ops_to_test)} operators with {len(known_good_ops)} known good"
        )

        try:
            success = lower_module_and_test_output(
                model=model,
                sample_inputs=sample_inputs,
                atol=atol,
                rtol=rtol,
                dynamic_shapes=dynamic_shapes,
                test_inputs=test_inputs,
                first_output_only=first_output_only,
                operator_allowlist=test_allowlist,
            )
            logger.info(f"  {'✓ PASS' if success else '✗ FAIL'}")
            return success
        except Exception as e:
            logger.info(f"  ! Error: {e}")
            return False

    def find_bad_operators(
        ops_to_test: List, known_good_ops: List
    ) -> Tuple[List, List]:
        """
        Recursively find bad operators using binary search.

        Returns:
            Tuple of (good_operators, bad_operators) from ops_to_test
        """
        if not ops_to_test:
            return [], []

        if len(ops_to_test) == 1:
            # Base case: single operator
            op = ops_to_test[0]
            if test_operator_set([op], known_good_ops):
                logger.info(f"  Single operator {op} is GOOD")
                return [op], []
            else:
                logger.info(f"  Single operator {op} is BAD")
                return [], [op]

        # Split ops_to_test into two halves
        mid = len(ops_to_test) // 2
        first_half = ops_to_test[:mid]  # Least frequent operators
        second_half = ops_to_test[mid:]  # Most frequent operators

        logger.info(
            f"Splitting {len(ops_to_test)} operators: {len(first_half)} + {len(second_half)}"
        )

        # Test each half
        first_half_good = test_operator_set(first_half, known_good_ops)
        second_half_good = test_operator_set(second_half, known_good_ops)

        good_ops = []
        bad_ops = []

        # Process first half
        if first_half_good:
            logger.info(
                f"First half ({len(first_half)} ops) is good - adding to known good"
            )
            good_ops.extend(first_half)
            known_good_ops.extend(first_half)
        if second_half_good:
            logger.info(
                f"Second half ({len(second_half)} ops) is good - adding to known good"
            )
            good_ops.extend(second_half)

        if not first_half_good:
            logger.info(f"First half ({len(first_half)} ops) is bad - recursing")
            sub_good, sub_bad = find_bad_operators(first_half, known_good_ops)
            good_ops.extend(sub_good)
            bad_ops.extend(sub_bad)
            known_good_ops.extend(sub_good)
        if not second_half_good:
            logger.info(f"Second half ({len(second_half)} ops) is bad - recursing")
            sub_good, sub_bad = find_bad_operators(second_half, known_good_ops)
            good_ops.extend(sub_good)
            bad_ops.extend(sub_bad)

        return good_ops, bad_ops

    # Start the binary search
    logger.info(
        f"\n=== Starting binary search on {len(operators_by_frequency)} operators ==="
    )
    good_operators, bad_operators = find_bad_operators(operators_by_frequency, [])

    # Summary of results
    logger.info(f"\n=== Binary search complete after {test_count} tests ===")
    logger.info(f"Good operators ({len(good_operators)}):")
    for op in good_operators:
        logger.info(f"  ✓ {op} (frequency: {operator_frequencies[op]})")

    logger.info(f"Bad operators ({len(bad_operators)}):")
    for op in bad_operators:
        logger.info(f"  ✗ {op} (frequency: {operator_frequencies[op]})")

    print_occurrences(edge_program, bad_operators)

    efficiency_gain = len(all_operators) - test_count
    logger.info(
        f"Efficiency: {test_count} tests instead of {len(all_operators)} (saved {efficiency_gain} tests)"
    )

    return {
        "good_operators": good_operators,
        "bad_operators": bad_operators,
        "operator_frequencies": operator_frequencies,
        "all_operators": all_operators,
        "test_count": test_count,
    }
