# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from collections import OrderedDict
from copy import deepcopy
from enum import auto, Enum
from typing import Any, List, Optional, Tuple

import executorch.backends.vulkan.utils as utils
import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.vulkan_preprocess import VulkanBackend
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.backends.xnnpack.xnnpack_preprocess import XnnpackBackend
from executorch.devtools import BundledProgram
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import ExecutorchProgramManager, to_edge_transform_and_lower

from executorch.exir.backend.backend_api import _get_node_list_with_same_tag

from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)

from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer

from executorch.exir.lowered_backend_module import (
    create_exported_program_from_submodule,
    create_submodule_from_nodes,
)
from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from executorch.extension.pytree import tree_flatten
from torch.export import export

from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class NodeFlagIsSetChecker(OperatorSupportBase):
    """
    Check if a node is marked with a given field in node.meta["custom"]
    """

    def __init__(self, field: str) -> None:
        super().__init__()
        self.field = field

    def check_field(self, node: torch.fx.Node) -> bool:
        if "custom" not in node.meta:
            return False

        custom_map = node.meta["custom"]
        if self.field not in custom_map:
            return False

        return custom_map[self.field]

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        if node.op == "placeholder" or node.op == "output":
            return False

        # Check if the node itself is tagged
        if self.check_field(node):
            return True

        # Check if any direct user of this node is tagged
        for user in node.users:
            if self.check_field(user):
                return True

        return False


class FlagBasedPartitioner(Partitioner):
    """
    Partitioner that partitions based on whether node.meta["custom"][field] is set to
    True.
    """

    def __init__(self, field: str) -> None:
        super().__init__()
        self.field = field
        self.delegation_spec = DelegationSpec("custom_partition", [])

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            NodeFlagIsSetChecker(self.field),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()

        partition_tags = {}
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)
        tag_mutated_buffer(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )


def mark_node_range(
    graph_module: torch.fx.GraphModule,
    end_idx: int = (2**31 - 1),
    start_idx: int = 0,
    field: str = "_in_target_subgraph",
):
    call_fn_count = 0
    for node in graph_module.graph.nodes:
        if "custom" not in node.meta:
            node.meta["custom"] = {}

        node.meta["custom"][field] = False

        if node.op != "call_function":
            continue

        call_fn_count += 1
        if call_fn_count >= start_idx and call_fn_count < end_idx:
            node.meta["custom"][field] = True


def extract_submodule_program(
    tagged_graph_module: torch.fx.GraphModule,
    owning_program: ExportedProgram,
    field: str = "_in_target_subgraph",
) -> ExportedProgram:
    tagged_graph_module_output_node = tagged_graph_module.graph.output_node()

    partitioner = FlagBasedPartitioner(field)
    partition_result = partitioner.partition(owning_program)

    tag, delegation_spec = next(iter(partition_result.partition_tags.items()))
    node_list = _get_node_list_with_same_tag(tagged_graph_module, tag, owning_program)

    replace_ctx = tagged_graph_module._set_replace_hook(
        owning_program.graph_signature.get_replace_hook()
    )
    with replace_ctx:
        submodule, call_module_node = create_submodule_from_nodes(
            tagged_graph_module, node_list, tag
        )

    submodule_output_node = submodule.graph.output_node()
    # Copy the output node meta from the original output node, because
    # create_submodule_from_nodes doesn't cover the meta field
    submodule_output_node.meta = tagged_graph_module_output_node.meta

    (
        submodule_program,
        _,
        _,
    ) = create_exported_program_from_submodule(
        submodule,
        owning_program,
        tag,
        call_module_node,
        False,
    )

    return submodule_program


class QuantizationMode(Enum):
    """Enum to describe how a model should be quantized."""

    NONE = auto()
    INT8_STATIC_PER_CHANNEL = auto()


def get_exported_graph(
    model,
    sample_inputs,
    sample_kwargs=None,
    dynamic_shapes=None,
    qmode=QuantizationMode.NONE,
) -> torch.fx.GraphModule:
    export_training_graph = export(
        model,
        sample_inputs,
        kwargs=sample_kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    ).module()

    if qmode == QuantizationMode.NONE:
        return export_training_graph

    quantizer = XNNPACKQuantizer()

    operator_config = get_symmetric_quantization_config(is_per_channel=True)
    quantizer.set_global(operator_config)

    prepared_graph = prepare_pt2e(export_training_graph, quantizer)
    prepared_graph(*sample_inputs)
    converted_graph = convert_pt2e(prepared_graph)

    return converted_graph


def random_uniform_tensor(shape, low=0.0, high=1.0, device=None, dtype=None):
    if dtype is None:
        dtype = torch.float32

    # Handle integer types using randint
    if dtype in (
        torch.int,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
        torch.short,
    ):
        low_int = int(low)
        high_int = int(high)
        # randint requires high > low, so ensure at least a range of 1
        if high_int <= low_int:
            high_int = low_int + 1
        return torch.randint(low_int, high_int, shape, device=device, dtype=dtype)

    # Handle unsigned integer types
    if dtype in (torch.uint8,):
        low_int = max(0, int(low))
        high_int = int(high)
        if high_int <= low_int:
            high_int = low_int + 1
        return torch.randint(low_int, high_int, shape, device=device, dtype=dtype)

    # Handle boolean type
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device, dtype=torch.int8).bool()

    # Handle floating-point types (float16, float32, float64, bfloat16)
    return torch.empty(shape, device=device, dtype=dtype).uniform_(low, high)


def generate_sample_inputs(
    exported_program: ExportedProgram,
    low: float = -1.0,
    high: float = 1.0,
) -> Tuple[torch.Tensor, ...]:
    """
    Analyze the exported program graph to determine input shapes and dtypes,
    then generate random sample inputs.

    Uses the graph signature to identify only user inputs (excluding parameters,
    buffers, and other non-input placeholders).

    Args:
        exported_program: The exported program to analyze
        low: Lower bound for random uniform values (default: -1.0)
        high: Upper bound for random uniform values (default: 1.0)

    Returns:
        Tuple of randomly generated tensors matching the input specs
    """
    sample_inputs = []

    # Get the set of user input names by filtering input_specs for USER_INPUT kind
    user_input_names = set()
    for spec in exported_program.graph_signature.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            if hasattr(spec.arg, "name"):
                user_input_names.add(spec.arg.name)

    for node in exported_program.graph.nodes:
        if node.op != "placeholder":
            continue

        # Only process nodes that are user inputs (not parameters, buffers, etc.)
        if node.name not in user_input_names:
            continue

        if "val" in node.meta:
            val = node.meta["val"]
            shape = None
            dtype = None

            if isinstance(val, torch.Tensor):
                shape = tuple(val.shape)
                dtype = val.dtype
            elif hasattr(val, "shape") and hasattr(val, "dtype"):
                # Handle FakeTensor or similar
                shape = tuple(val.shape)
                dtype = val.dtype

            if shape is not None and dtype is not None:
                tensor = random_uniform_tensor(shape, low=low, high=high, dtype=dtype)
                sample_inputs.append(tensor)

    inputs_flattened, _ = tree_flatten(sample_inputs)
    return inputs_flattened


def export_model_to_vulkan(
    model,
    sample_inputs,
    sample_kwargs=None,
    dynamic_shapes=None,
    operator_blocklist=None,
    operator_allowlist=None,
    nn_module_blocklist=None,
    nn_module_allowlist=None,
    qmode=QuantizationMode.NONE,
):
    compile_options = {}
    exported_graph = get_exported_graph(
        model,
        sample_inputs,
        sample_kwargs=sample_kwargs,
        dynamic_shapes=dynamic_shapes,
        qmode=qmode,
    )
    program = export(
        exported_graph,
        sample_inputs,
        kwargs=sample_kwargs,
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
                nn_module_blocklist=nn_module_blocklist,
                nn_module_allowlist=nn_module_allowlist,
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


def export_model_to_xnnpack(
    model,
    sample_inputs,
    dynamic_shapes=None,
    operator_blocklist=None,
    operator_allowlist=None,
    nn_module_blocklist=None,
    nn_module_allowlist=None,
    qmode=QuantizationMode.NONE,
):
    compile_options = {}
    exported_graph = get_exported_graph(model, sample_inputs, qmode=qmode)
    program = export(
        exported_graph,
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


def print_tensor_comparison_errors(
    tensor1, tensor2, atol=1e-03, rtol=1e-03, max_errors=10
):
    """
    Print the first max_errors tensor indexes that exceed the absolute/relative tolerance
    and the error at each of those locations.

    Args:
        tensor1: First tensor to compare
        tensor2: Second tensor to compare
        atol: Absolute tolerance
        rtol: Relative tolerance
        max_errors: Maximum number of errors to print (default: 10)
    """
    # Handle lists/tuples of tensors
    if isinstance(tensor1, (list, tuple)) and isinstance(tensor2, (list, tuple)):
        if len(tensor1) != len(tensor2):
            print(f"Tensor count mismatch: {len(tensor1)} vs {len(tensor2)}")
            return

        for i, (t1, t2) in enumerate(zip(tensor1, tensor2)):
            print(f"\n=== Tensor {i} comparison ===")
            print_tensor_comparison_errors(t1, t2, atol, rtol, max_errors)
        return

    # Handle single tensor comparison
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        print("Error: Both inputs must be torch.Tensor objects")
        return

    if tensor1.shape != tensor2.shape:
        print(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
        return

    # Calculate absolute and relative errors
    abs_diff = torch.abs(tensor1 - tensor2)
    rel_diff = abs_diff / (
        torch.abs(tensor2) + 1e-8
    )  # Add small epsilon to avoid division by zero

    # Find locations where tolerance is exceeded
    tolerance_mask = (abs_diff > atol) & (rel_diff > rtol)

    if not tolerance_mask.any():
        print("All values are within tolerance")
        return

    # Get indices where tolerance is exceeded
    error_indices = torch.nonzero(tolerance_mask, as_tuple=False)
    total_errors = error_indices.shape[0]

    print(f"Found {total_errors} values exceeding tolerance (atol={atol}, rtol={rtol})")
    print(f"Showing first {min(max_errors, total_errors)} errors:")
    print("Index -> tensor1_value, tensor2_value, abs_error, rel_error")

    # Print first max_errors locations
    for i in range(min(max_errors, total_errors)):
        idx = tuple(error_indices[i].tolist())
        val1 = tensor1[idx].item()
        val2 = tensor2[idx].item()
        abs_err = abs_diff[idx].item()
        rel_err = rel_diff[idx].item()

        print(
            f"{idx} -> {val1:.6f}, {val2:.6f}, abs_err={abs_err:.6f}, rel_err={rel_err:.6f}"
        )


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
            print_tensor_comparison_errors(model_output, ref_output, atol, rtol)
            return False
        if first_output_only:
            result = torch.allclose(
                model_output[0], ref_output[0], atol=atol, rtol=rtol
            )
            if not result:
                print_tensor_comparison_errors(
                    model_output[0], ref_output[0], atol, rtol
                )
            return result
        else:
            result = True
            for i in range(len(ref_output)):
                if isinstance(ref_output[i], torch.Tensor):
                    if not torch.allclose(
                        model_output[i], ref_output[i], atol=atol, rtol=rtol
                    ):
                        print(f"\n=== Output {i} comparison failed ===")
                        print_tensor_comparison_errors(
                            model_output[i], ref_output[i], atol, rtol
                        )
                        result = False
                elif isinstance(ref_output[i], int):
                    if not model_output[i] == ref_output[i]:
                        print(f"\n=== Output {i} comparison failed ===")
                        print(f"{model_output[i]} vs {ref_output[[i]]}")
                        result = False
                else:
                    print(f"WARNING: Output {i} has type {type(ref_output[i])}")
            return result
    else:
        # If one output, eager returns tensor while executor tuple of size 1
        result = torch.allclose(model_output[0], ref_output, atol=atol, rtol=rtol)
        if not result:
            print_tensor_comparison_errors(model_output[0], ref_output, atol, rtol)
        return result


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
    # Load the ExecuTorch program
    executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)

    # Flatten inputs for execution
    inputs_flattened, _ = tree_flatten(sample_inputs)

    # Run the ExecuTorch program
    model_output = executorch_module.run_method("forward", tuple(inputs_flattened))

    # Generate reference outputs using the reference model
    ref_output, _ = tree_flatten(reference_model(*sample_inputs))

    # Check if outputs are equal
    return check_outputs_equal(
        model_output,
        ref_output,
        atol=atol,
        rtol=rtol,
        first_output_only=first_output_only,
    )


def make_copy_of_inputs(sample_inputs: Tuple[Any]) -> Tuple[Any]:
    sample_inputs_copy = []
    for input_val in sample_inputs:
        if isinstance(input_val, torch.Tensor):
            sample_inputs_copy.append(input_val.clone())
        else:
            sample_inputs_copy.append(deepcopy(input_val))
    return tuple(sample_inputs_copy)


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
    nn_module_allowlist=None,
    nn_module_blocklist=None,
    xnnpack=False,
) -> bool:
    """
    Helper testing function that takes a torch.nn.Module and lowers it to Vulkan with
    the given sample inputs. It then runs the lowered module and compares its
    outputs with the outputs of the eager module.

    Returns:
        bool: True if all comparisons pass, False otherwise.
    """
    # Export model to Vulkan using the helper function
    if xnnpack:
        executorch_program = export_model_to_xnnpack(
            model,
            make_copy_of_inputs(sample_inputs),
            dynamic_shapes,
            operator_blocklist,
            operator_allowlist,
            nn_module_blocklist,
            nn_module_allowlist,
        )
    else:
        executorch_program = export_model_to_vulkan(
            model,
            make_copy_of_inputs(sample_inputs),
            dynamic_shapes,
            operator_blocklist=operator_blocklist,
            operator_allowlist=operator_allowlist,
            nn_module_blocklist=nn_module_blocklist,
            nn_module_allowlist=nn_module_allowlist,
        )

    executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)

    inputs_flattened, _ = tree_flatten(sample_inputs)

    model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
    ref_output = model(*make_copy_of_inputs(sample_inputs))

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


def create_bundled_program(
    executorch_program: ExecutorchProgramManager,
    sample_inputs: Tuple[torch.Tensor, ...],
    expected_outputs: List[Any],
    method_name: str = "forward",
) -> bytes:
    """
    Create a bundled program containing the model and test cases for correctness testing.

    Args:
        executorch_program: The ExecutorchProgramManager to bundle
        sample_inputs: Sample inputs for the model
        expected_outputs: Expected outputs from running the model with sample_inputs
        method_name: Name of the method to test (default: "forward")

    Returns:
        Serialized bundled program as bytes
    """
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
    bundled_program = BundledProgram(executorch_program, test_suites)

    # Serialize to flatbuffer
    bundled_buffer = serialize_from_bundled_program_to_flatbuffer(bundled_program)

    return bundled_buffer


def save_bundled_program(
    model: torch.nn.Module,
    sample_inputs: Tuple[torch.Tensor],
    output_path: str,
    method_name: str = "forward",
    sample_kwargs=None,
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
        et_program = export_model_to_vulkan(
            model,
            sample_inputs,
            sample_kwargs=sample_kwargs,
            dynamic_shapes=dynamic_shapes,
        )

    if sample_kwargs is None:
        sample_kwargs = {}

    # Generate expected outputs by running the model
    expected_outputs = [getattr(model, method_name)(*sample_inputs, **sample_kwargs)]

    # Flatten sample inputs with kwargs to match expected format
    inputs_flattened, _ = tree_flatten((sample_inputs, sample_kwargs))

    # Create bundled program
    bp_buffer = create_bundled_program(
        et_program,
        tuple(inputs_flattened),
        expected_outputs,
        method_name,
    )

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
            if (
                node.target == torch.ops.higher_order.auto_functionalized
                or node.target == torch.ops.higher_order.auto_functionalized_v2
            ):
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
    export_training_graph = export(model, sample_inputs, strict=True).module()
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
            if (
                node.target == torch.ops.higher_order.auto_functionalized
                or node.target == torch.ops.higher_order.auto_functionalized_v2
            ):
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

    # Sort operators by frequency (most frequent first for binary search)
    operators_by_frequency = sorted(
        all_operators, key=lambda op: operator_frequencies[op], reverse=True
    )

    logger.info("Operator frequencies (sorted by occurrence, most frequent first):")
    for op in operators_by_frequency:
        logger.info(f"  {op.name()}: {operator_frequencies[op]} occurrences")

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

            # Log known good ops
            logger.info("  Known good:")
            for op in known_good_ops:
                logger.info(f"  * {op.name()}")

            # Log tested ops
            logger.info("  Tested ops:")
            for op in ops_to_test:
                logger.info(f"  * {op.name()}")

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
                logger.info(f"  Single operator {op.name()} is GOOD")
                return [op], []
            else:
                logger.info(f"  Single operator {op.name()} is BAD")
                return [], [op]

        # Split ops_to_test into two halves
        mid = len(ops_to_test) // 2
        first_half = ops_to_test[:mid]  # Least frequent operators
        second_half = ops_to_test[mid:]  # Most frequent operators

        logger.info(
            f"Splitting {len(ops_to_test)} operators: {len(first_half)} + {len(second_half)}"
        )

        # Log known good ops
        logger.info("  Known good:")
        for op in known_good_ops:
            logger.info(f"  * {op.name()}")

        # Log first half ops
        logger.info("  First half ops:")
        for op in first_half:
            logger.info(f"  * {op.name()}")

        # Log second half ops
        logger.info("  Second half ops:")
        for op in second_half:
            logger.info(f"  * {op.name()}")

        good_ops = []
        bad_ops = []

        first_half_good = test_operator_set(first_half, known_good_ops)
        if first_half_good:
            logger.info(
                f"First half ({len(first_half)} ops) is good - adding to known good"
            )
            good_ops.extend(first_half)
            known_good_ops.extend(first_half)

        second_half_good = test_operator_set(second_half, known_good_ops)
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
        logger.info(f"  ✓ {op.name()} (frequency: {operator_frequencies[op]})")

    logger.info(f"Bad operators ({len(bad_operators)}):")
    for op in bad_operators:
        logger.info(f"  ✗ {op.name()} (frequency: {operator_frequencies[op]})")

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


def make_indent(indent_level):
    indent_str = ""
    for _ in range(indent_level):
        indent_str += " "
    return indent_str


def print_output(outputs, n: int = 0, indent_level: int = 0):
    if isinstance(outputs, (list, tuple)):
        print(f"{make_indent(indent_level)}output_{n} = {type(outputs)}")
        new_indent_level = indent_level + 2
        for n, test_out in enumerate(outputs):
            print_output(test_out, n, new_indent_level)
    elif isinstance(outputs, torch.Tensor):
        print(
            f"{make_indent(indent_level)}output_{n} = test_utils.random_uniform_tensor({outputs.shape}, low={outputs.min().item()}, high={outputs.max().item()},  dtype={outputs.dtype})"
        )
    elif isinstance(outputs, int):
        print(f"{make_indent(indent_level)}output_{n} = {outputs}")
    else:
        print(f"{make_indent(indent_level)}output_{n} = {type(outputs)}")
