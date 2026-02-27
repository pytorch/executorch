# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT Converter Registry for ExecuTorch."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

import torch

# Standard converter signature: (node, network, input_map) -> tensor
ConverterFn = Callable[[torch.fx.Node, Any, Dict[torch.fx.Node, Any]], Any]

# Extended converter signature: (node, network, input_map, edge_program) -> tensor
# Used by converters that need access to model weights (conv2d, batch_norm, etc.)
ExtendedConverterFn = Callable[
    [torch.fx.Node, Any, Dict[torch.fx.Node, Any], Optional[Any]], Any
]

ValidatorFn = Callable[[torch.fx.Node], bool]


@dataclass
class ConverterEntry:
    """Entry in the converter registry.

    Attributes:
        converter: The converter function that transforms FX nodes to TensorRT layers.
        validator: Optional function to check if a node can be converted.
        op_names: Set of operation names this converter handles.
        needs_edge_program: Whether converter needs access to ExportedProgram
            for weight extraction (e.g., conv2d, batch_norm).
        supports_dynamic_shapes: Whether the converter supports dynamic input shapes.
            When False, the converter only works with static shapes. When True,
            the converter can handle inputs with dynamic dimensions (symbolic sizes).
    """

    converter: ConverterFn
    validator: Optional[ValidatorFn]
    op_names: Set[str]
    needs_edge_program: bool = False
    supports_dynamic_shapes: bool = False


_CONVERTER_REGISTRY: Dict[str, ConverterEntry] = {}


def register_converter(
    op_name: str,
    converter_fn: ConverterFn,
    validator_fn: Optional[ValidatorFn] = None,
    needs_edge_program: bool = False,
    supports_dynamic_shapes: bool = False,
) -> None:
    """Register a converter function for a specific operation.

    Args:
        op_name: Operation name (e.g., "aten.conv2d.default").
        converter_fn: Converter function that converts FX node to TensorRT layer.
        validator_fn: Optional validator function to check if node can be converted.
        needs_edge_program: Whether the converter needs access to ExportedProgram
                           for weight extraction (e.g., conv2d, batch_norm).
        supports_dynamic_shapes: Whether the converter supports dynamic input shapes.
                                When True, the converter can handle inputs with
                                dynamic dimensions (symbolic sizes).
    """
    if not op_name:
        raise ValueError("op_name cannot be empty")
    if converter_fn is None:
        raise ValueError("converter_fn cannot be None")

    normalized_name = op_name.replace("::", ".")

    entry = ConverterEntry(
        converter=converter_fn,
        validator=validator_fn,
        op_names={normalized_name},
        needs_edge_program=needs_edge_program,
        supports_dynamic_shapes=supports_dynamic_shapes,
    )

    _CONVERTER_REGISTRY[normalized_name] = entry


def lookup_converter(op_name: str) -> Optional[ConverterFn]:
    """Lookup a converter function by operation name."""
    normalized_name = op_name.replace("::", ".")
    entry = _CONVERTER_REGISTRY.get(normalized_name)
    if entry is None:
        return None
    return entry.converter


def lookup_validator(op_name: str) -> Optional[ValidatorFn]:
    """Lookup a validator function by operation name."""
    normalized_name = op_name.replace("::", ".")
    entry = _CONVERTER_REGISTRY.get(normalized_name)
    if entry is None:
        return None
    return entry.validator


def needs_edge_program(op_name: str) -> bool:
    """Check if a converter needs access to ExportedProgram for weight extraction."""
    normalized_name = op_name.replace("::", ".")
    entry = _CONVERTER_REGISTRY.get(normalized_name)
    if entry is None:
        return False
    return entry.needs_edge_program


def supports_dynamic_shapes(op_name: str) -> bool:
    """Check if a converter supports dynamic input shapes.

    Args:
        op_name: Operation name (e.g., "aten.conv2d.default").

    Returns:
        True if the converter supports dynamic shapes, False otherwise.
        Returns False if no converter is registered for the operation.
    """
    normalized_name = op_name.replace("::", ".")
    entry = _CONVERTER_REGISTRY.get(normalized_name)
    if entry is None:
        return False
    return entry.supports_dynamic_shapes


def has_converter(op_name: str) -> bool:
    """Check if a converter is registered for the given operation."""
    normalized_name = op_name.replace("::", ".")
    return normalized_name in _CONVERTER_REGISTRY


def get_registered_ops() -> List[str]:
    """Get list of all registered operation names."""
    return list(_CONVERTER_REGISTRY.keys())


def clear_registry() -> None:
    """Clear all registered converters. Primarily useful for testing."""
    _CONVERTER_REGISTRY.clear()


def converter(
    *op_names: str,
    validator_fn: Optional[ValidatorFn] = None,
    needs_edge_program: bool = False,
    supports_dynamic_shapes: bool = False,
) -> Callable[[ConverterFn], ConverterFn]:
    """Decorator to register a converter function for one or more operations.

    Args:
        op_names: One or more operation names to register (e.g., "aten.conv2d.default").
        validator_fn: Optional validator function to check if node can be converted.
        needs_edge_program: Whether the converter needs access to ExportedProgram
                           for weight extraction (e.g., conv2d, batch_norm).
        supports_dynamic_shapes: Whether the converter supports dynamic input shapes.
                                When True, the converter can handle inputs with
                                dynamic dimensions (symbolic sizes).

    Example:
        @converter("aten.conv2d.default", needs_edge_program=True)
        def convert_conv2d(node, network, input_map, edge_program):
            ...

        @converter("aten.relu.default", supports_dynamic_shapes=True)
        def convert_relu(node, network, input_map):
            ...
    """

    def decorator(fn: ConverterFn) -> ConverterFn:
        for op_name in op_names:
            register_converter(
                op_name,
                fn,
                validator_fn,
                needs_edge_program=needs_edge_program,
                supports_dynamic_shapes=supports_dynamic_shapes,
            )
        return fn

    return decorator
