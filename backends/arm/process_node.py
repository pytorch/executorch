# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import operator
from typing import Any, cast, Dict

import numpy as np
import torch
import torch.fx
import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import NodeVisitor
from executorch.backends.arm.tosa.dialect.shape import is_shape_op_node
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.backends.arm.tosa.utils import normalize_symint
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch.export.exported_program import ExportedProgram


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype in (
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e8m0fnu,
    ):
        try:
            import ml_dtypes  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                f"ml_dtypes is required to serialize {tensor.dtype} tensors for TOSA. "
                "Have you run setup.sh?"
            ) from e
        ml_dtype_map = {
            torch.bfloat16: (torch.uint16, ml_dtypes.bfloat16),
            torch.float8_e4m3fn: (torch.uint8, ml_dtypes.float8_e4m3fn),
            torch.float8_e5m2: (torch.uint8, ml_dtypes.float8_e5m2),
            torch.float8_e8m0fnu: (torch.uint8, ml_dtypes.float8_e8m0fnu),
        }
        storage_dtype, ml_dtype = ml_dtype_map[tensor.dtype]
        return tensor.view(storage_dtype).numpy().view(ml_dtype)
    else:
        return tensor.numpy()


def _prepare_const_values_for_tosa_dtype(
    values: np.ndarray, tosa_arg: TosaArg
) -> np.ndarray:
    """Normalize constant storage to the expected TOSA serializer dtype."""
    if tosa_arg.dtype == ts.DType.INT48 and values.dtype != np.int64:
        return values.astype(np.int64)
    if tosa_arg.dtype in (ts.DType.FP6E2M3, ts.DType.FP6E3M2):
        if values.dtype == np.uint8:
            try:
                import ml_dtypes  # type: ignore[import-not-found]
            except ImportError as e:
                raise RuntimeError(
                    "ml_dtypes is required to serialize FP6 tensors for TOSA. "
                    "Have you run setup.sh?"
                ) from e
            ml_dtype = {
                ts.DType.FP6E2M3: ml_dtypes.float6_e2m3fn,
                ts.DType.FP6E3M2: ml_dtypes.float6_e3m2fn,
            }[tosa_arg.dtype]
            return values.view(ml_dtype)
    return values


def _get_const_shape(values: np.ndarray, tosa_arg: TosaArg) -> list[int]:
    """Return the TOSA logical shape for a serialized constant."""
    if tosa_arg.dtype == ts.DType.FP4E2M1:
        return normalize_symint(tosa_arg.shape)
    return normalize_symint(values.shape)


def _is_packed_fp4_const(values: np.ndarray, tosa_arg: TosaArg) -> bool:
    """FP4 elements are pairwise in each byte of a uint8 tensor.

    This function checks if the given values and TOSA argument represent a
    packed FP4 constant.

    """

    return (
        tosa_arg.dtype == ts.DType.FP4E2M1
        and values.dtype == np.uint8
        and values.shape[-1] * 2 == tosa_arg.shape[-1]
    )


def _add_const(
    tosa_graph: Any,
    values: np.ndarray,
    tosa_arg: TosaArg,
    name: str,
) -> None:
    """Add a graph-owned constant under its exact name.

    Parameters, buffers, and lifted constants are referenced by their FX names,
    so pooling them could leave those names undefined. Preserve packed FP4
    storage when required.

    """
    if _is_packed_fp4_const(values, tosa_arg):
        # TOSA FP4 tensors have logical FP4 shape, but constants are stored as
        # packed bytes (two values per byte). Add the raw bytes as INT8 first
        # then set TOSA dtype and shape correctly on the tensor metadata.
        tosa_graph.addUnpooledConst(
            normalize_symint(values.shape),
            ts.DType.INT8,
            values,
            name=name,
        )
        tensor = tosa_graph.currRegion.currBasicBlock.tensors[name]
        tensor.setDtype(ts.DType.FP4E2M1)
        for dim, size in enumerate(normalize_symint(tosa_arg.shape)):
            tensor.SetDimSize(dim, size)
        return

    prepared_values = _prepare_const_values_for_tosa_dtype(values, tosa_arg)
    tosa_graph.addUnpooledConst(
        _get_const_shape(prepared_values, tosa_arg),
        tosa_arg.dtype,
        prepared_values,
        name=name,
    )


def process_call_function(
    node: torch.fx.Node,
    tosa_graph: Any,
    node_visitors: Dict[str, NodeVisitor],
    tosa_spec: TosaSpecification,
):
    # Unpack arguments and convert
    try:
        inputs = [TosaArg(arg, tosa_spec) for arg in node.args]
    except ValueError as e:
        raise ValueError(f"Failed processing args to op:\n{node}") from e

    # Convert output (this node itself)
    try:
        output = TosaArg(node, tosa_spec)
    except ValueError as e:
        raise ValueError(
            f"Failed processing call_function: {node.name}. "
            "Is the original torch function supported?"
        ) from e

    tosa_graph = cast(ts.TosaSerializer, tosa_graph)
    if not output.multiple_output_names and not is_shape_op_node(node):
        tosa_graph.currRegion.currBasicBlock.addTensor(
            output.name, normalize_symint(output.shape), output.dtype
        )

    # Get item nodes just add tensors, no node visitor is needed.
    if node.target == operator.getitem:
        return

    # Visiting each Node
    if node.target.__name__ in node_visitors:  # type: ignore[union-attr]
        node_visitors[node.target.__name__].define_node(  # type: ignore[union-attr]
            node,
            tosa_graph,
            inputs,
            output,
        )
    else:
        raise RuntimeError(f"Unknown operator {node.target} for TOSA : {tosa_spec}")


def process_inputs(
    node: torch.fx.Node,
    tosa_graph: Any,
    tosa_spec: TosaSpecification,
):
    """Serialize an input node."""

    try:
        tosa_arg = TosaArg(node, tosa_spec)
    except ValueError as e:
        raise ValueError(
            f"Failed processing input placeholder: {node.name}. "
            "Is the original torch function supported?"
        ) from e

    input_shape = tosa_arg.shape
    tensor = ts.TosaSerializerTensor(
        tosa_arg.name,
        normalize_symint(input_shape),
        tosa_arg.dtype,
        data=None,
    )
    tosa_graph.addInputTensor(tensor)


def process_inputs_to_parameters(
    node: torch.fx.Node,
    tosa_graph: Any,
    edge_program: ExportedProgram,
    tosa_spec: TosaSpecification,
):
    """Serialize bias and non-quantized weights."""
    try:
        tosa_arg = TosaArg(node, tosa_spec)
    except ValueError as e:
        raise ValueError(
            f"Failed processing parameter placeholder: {node.name}. "
            "Is the original torch function supported?"
        ) from e
    parameter_data = get_param(edge_program, node)

    if not isinstance(parameter_data, torch.Tensor):
        raise TypeError(
            f"Expected parameter '{node.name}' to be a torch.Tensor, got "
            f"{type(parameter_data).__name__}"
        )
    parameter_values = _tensor_to_numpy(parameter_data)
    _add_const(tosa_graph, parameter_values, tosa_arg, name=tosa_arg.name)


def is_mutable_buffer(node: torch.fx.Node, edge_program: ExportedProgram) -> bool:
    """Is this placeholder a buffer the graph mutates (streaming state)?"""
    signature = edge_program.graph_signature
    target = signature.inputs_to_buffers.get(node.name)
    if target is None:
        return False
    return target in set(signature.buffers_to_mutate.values())


def _variable_tensor_name(buffer_name: str) -> str:
    """Name of the TOSA variable tensor that holds a mutable buffer."""
    return f"{buffer_name}_variable"


def process_inputs_to_buffers(
    node: torch.fx.Node,
    tosa_graph: Any,
    edge_program: ExportedProgram,
    tosa_spec: TosaSpecification,
):
    """Serialize quantized weights."""
    try:
        tosa_arg = TosaArg(node, tosa_spec)
    except ValueError as e:
        raise ValueError(
            f"Failed processing buffer placeholder: {node.name}. "
            "Is the original torch function supported?"
        ) from e
    if is_mutable_buffer(node, edge_program):
        # Streaming state. A CONST would freeze it, so the buffer becomes a TOSA
        # variable that the delegate owns and keeps between invocations. The
        # variable carries the buffer's contents as its initial value: the
        # memory belongs to the delegate, so ExecuTorch cannot initialise it.
        # Consumers read the variable through VARIABLE_READ into a tensor named
        # after the placeholder, so they need no special handling.
        initial = get_buffer(edge_program, node)
        initial_values = (
            _tensor_to_numpy(initial) if isinstance(initial, torch.Tensor) else None
        )
        shape = normalize_symint(tosa_arg.shape)
        block = tosa_graph.currRegion.currBasicBlock
        variable = _variable_tensor_name(tosa_arg.name)
        block.addTensor(
            variable,
            shape,
            tosa_arg.dtype,
            data=initial_values,
            variable=True,
            variable_name=tosa_arg.name,
        )
        block.addTensor(tosa_arg.name, shape, tosa_arg.dtype)
        attr = ts.TosaSerializerAttribute()
        attr.setAttribute(ts.Op.VARIABLE_READ)
        tosa_graph.addOperator(ts.Op.VARIABLE_READ, [variable], [tosa_arg.name], attr)
        return

    buffer_data = get_buffer(edge_program, node)

    if not isinstance(buffer_data, torch.Tensor):
        raise TypeError(
            f"Expected buffer '{node.name}' to be a torch.Tensor, got "
            f"{type(buffer_data).__name__}"
        )
    buffer_values = _tensor_to_numpy(buffer_data)
    _add_const(tosa_graph, buffer_values, tosa_arg, name=tosa_arg.name)


def process_inputs_to_lifted_tensor_constants(
    node: torch.fx.Node,
    tosa_graph: Any,
    edge_program: ExportedProgram,
    tosa_spec: TosaSpecification,
):
    try:
        tosa_arg = TosaArg(node, tosa_spec)
    except ValueError as e:
        raise ValueError(
            f"Failed processing lifted tensor constant placeholder: {node.name}. "
            "Is the original torch function supported?"
        ) from e
    tensor = get_lifted_tensor_constant(edge_program, node)
    assert isinstance(tensor, torch.Tensor), (
        f"Expected lifted tensor constant '{node.name}' to be a torch.Tensor, got "
        f"{type(tensor).__name__}"
    )
    tensor_values = _tensor_to_numpy(tensor)
    _add_const(tosa_graph, tensor_values, tosa_arg, name=tosa_arg.name)


def _is_submodule_input(
    node: torch.fx.Node, containing_graph_module: torch.fx.GraphModule
) -> bool:
    """Determines whether 'node' is an input to a submodule of
    'containing_graph_module'.
    """
    if node.op != "placeholder":
        return False
    return node.meta.get("is_input", False)


def process_placeholder(
    node: torch.fx.Node,
    tosa_graph: Any,
    edge_program: ExportedProgram,
    containing_graph_module: torch.fx.GraphModule | None,
    tosa_spec: TosaSpecification,
):
    """Wrapper for processing and serializing all types of placeholders."""
    if node.name != node.target:
        raise ValueError(
            f"Placeholder name '{node.name}' does not match target '{node.target}'"
        )
    if len(node.args) != 0:
        raise ValueError(f"Placeholder '{node.name}' must not have default values")

    if node.name in edge_program.graph_signature.user_inputs:
        process_inputs(node, tosa_graph, tosa_spec)
    elif containing_graph_module and _is_submodule_input(node, containing_graph_module):
        process_inputs(node, tosa_graph, tosa_spec)
    elif is_param(edge_program, node):
        process_inputs_to_parameters(node, tosa_graph, edge_program, tosa_spec)
    elif is_buffer(edge_program, node):
        process_inputs_to_buffers(node, tosa_graph, edge_program, tosa_spec)
    elif is_lifted_tensor_constant(edge_program, node):
        process_inputs_to_lifted_tensor_constants(
            node, tosa_graph, edge_program, tosa_spec
        )
    elif node.name in edge_program.graph_signature.inputs_to_lifted_custom_objs:
        raise NotImplementedError(
            "Placeholder is of type 'lifted custom object' which is not supported."
        )
    else:
        raise RuntimeError(f"Placeholder '{node.name}' is of unknown type.")


def process_output(
    node: torch.fx.Node,
    tosa_graph: Any,
    tosa_spec: TosaSpecification,
    edge_program: ExportedProgram | None = None,
):
    mutated = _buffer_mutation_targets(edge_program)
    for output in cast(tuple[torch.fx.Node, ...], node.args[0]):
        output_arg = TosaArg(output, tosa_spec)
        variable_name = mutated.get(output.name)
        if variable_name is not None:
            # A buffer mutation is a write into the variable, not a graph output,
            # so the update happens on the NPU rather than as an aten.copy_ on
            # the MCU.
            attr = ts.TosaSerializerAttribute()
            attr.setAttribute(ts.Op.VARIABLE_WRITE)
            tosa_graph.addOperator(
                ts.Op.VARIABLE_WRITE,
                [output_arg.name],
                [_variable_tensor_name(variable_name)],
                attr,
            )
            continue
        tosa_graph.addOutputTensor(
            tosa_graph.currRegion.currBasicBlock.tensors[output_arg.name]
        )


def _buffer_mutation_targets(
    edge_program: ExportedProgram | None,
) -> dict[str, str]:
    """Map producing-node name to the buffer placeholder it writes into."""
    if edge_program is None:
        return {}
    signature = edge_program.graph_signature
    buffer_to_placeholder = {
        target: name for name, target in signature.inputs_to_buffers.items()
    }
    targets: dict[str, str] = {}
    for output_name, buffer_target in signature.buffers_to_mutate.items():
        placeholder = buffer_to_placeholder.get(buffer_target)
        if placeholder is not None:
            targets[output_name] = placeholder
    return targets
