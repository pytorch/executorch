from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import tensorrt as trt
import torch
import torch_tensorrt
from torch import SymBool, SymFloat, SymInt
from torch._ops import OpOverloadPacket
from torch.fx.node import Argument, Node, Target, _get_qualified_name
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext

logger = logging.getLogger(__name__)

LegacyConverterImplSignature = Callable[
    [
        trt.INetworkDefinition,
        Target,
        Tuple[Argument, ...],
        Dict[str, Argument],
        str,
    ],
    Union[trt.ITensor, Sequence[trt.ITensor]],
]

DynamoConverterImplSignature = Callable[
    [
        ConversionContext,
        Target,
        Tuple[Argument, ...],
        Dict[str, Argument],
        str,
    ],
    Union[trt.ITensor, Sequence[trt.ITensor]],
]

ConverterImplSignature = Union[
    LegacyConverterImplSignature, DynamoConverterImplSignature
]


class ConverterPriority(Enum):
    """Enum to set a converter's priority in the registry"""

    STANDARD = auto()
    HIGH = auto()


class CallingConvention(Enum):
    """Enum representing a converter's calling convention"""

    LEGACY = auto()  # Legacy FX converters
    CTX = auto()  # New Dynamo converters


@dataclass(frozen=True)
class ConverterSupport:
    """Class representing a converter implementation and support function

    Args:
        converter_implementation: Function which converts said node to a TRT equivalent
        capability_validator: Function which takes in a Node and returns a bool indicating
            whether that node can be supported by its companion converter. Note that
            this function must not modify the node or its graph
        supports_dynamic_shapes: Boolean flag indicating if the converter has support for dynamic inputs.
        requires_output_allocator: Boolean flag indicating if the converter creates operators which require an Output Allocator to run (e.g. data dependent operators).
    """

    converter_implementation: ConverterImplSignature
    capability_validator: Callable[[Node, CompilationSettings], bool] = field(
        default=lambda node, compilation_settings: True
    )
    supports_dynamic_shapes: bool = False
    requires_output_allocator: bool = False


# Dictionary representing Dynamo aten-only converters
# Each converter maps to a sequence of at least one ConverterSupport object(s)
DYNAMO_ATEN_CONVERTERS: Dict[Target, Sequence[ConverterSupport]] = {}


def has_static_shapes(node: torch.fx.Node) -> bool:
    """Returns True if a node has static args, kwargs, or outputs"""
    return not _has_dynamic_shapes(node=node)


def node_has_dynamic_shapes(node: torch.fx.Node) -> bool:
    """Returns True if a node has dynamic args, kwargs, or outputs"""
    return _has_dynamic_shapes(node=node)


def has_dynamic_shapes_in_args(
    arg_positions_to_check: Optional[List[int]] = None,
) -> Callable[[torch.fx.Node], bool]:
    """Returns True if a node has dynamic inputs in node.args at specified positions"""
    return functools.partial(
        _has_dynamic_shapes, arg_positions_to_check=arg_positions_to_check
    )


def has_static_shapes_in_args(
    arg_positions_to_check: Optional[List[int]] = None,
) -> Callable[[torch.fx.Node, CompilationSettings], bool]:
    """Returns True if a node has static inputs in node.args at specified positions"""
    _has_static_shapes = lambda node, compilation_settings, arg_positions_to_check: not _has_dynamic_shapes(
        node, compilation_settings, arg_positions_to_check
    )
    return functools.partial(
        _has_static_shapes, arg_positions_to_check=arg_positions_to_check
    )


def _has_dynamic_shapes(
    node: torch.fx.Node,
    compilation_settings: CompilationSettings = None,
    arg_positions_to_check: Optional[List[int]] = None,
) -> bool:
    # Validate that none of the inputs to the node have Dynamic shapes
    assert isinstance(
        node, torch.fx.Node
    ), "Inputs to validator functions must be FX Nodes"

    def _is_subnode_dynamic(subnode: torch.fx.Node) -> bool:
        """Checks if a node itself has Dynamic properties"""
        _has_symbolic_sizes_strides, is_shape_dynamic = False, False
        if "val" in subnode.meta:
            _has_symbolic_sizes_strides = getattr(
                subnode.meta["val"], "_has_symbolic_sizes_strides", False
            )
            meta_val = subnode.meta["val"]
            if isinstance(meta_val, (list, tuple)):
                for val in meta_val:
                    shape = val.size()
                    if any(
                        isinstance(dim, (SymFloat, SymInt, SymBool)) for dim in shape
                    ):
                        is_shape_dynamic = True
                        break
            elif isinstance(meta_val, (SymFloat, SymInt, SymBool)):
                is_shape_dynamic = True
            else:
                shape = subnode.meta["val"].size()
                is_shape_dynamic = any(
                    isinstance(dim, (SymFloat, SymInt, SymBool)) for dim in shape
                )

        return _has_symbolic_sizes_strides or is_shape_dynamic

    # Check node value itself
    if arg_positions_to_check is None and _is_subnode_dynamic(node):
        return True

    # Check node arguments individually
    if arg_positions_to_check is None and any(
        _is_subnode_dynamic(arg) for arg in node.args if isinstance(arg, torch.fx.Node)
    ):
        return True
    # Check specific arg positions if the caller has specified positions to check
    elif arg_positions_to_check is not None and any(
        _is_subnode_dynamic(node.args[i])
        for i in arg_positions_to_check
        if isinstance(node.args[i], torch.fx.Node)
    ):
        return True

    # Check node keyword arguments individually
    if arg_positions_to_check is None and any(
        _is_subnode_dynamic(kwarg)
        for kwarg in node.kwargs.values()
        if isinstance(kwarg, torch.fx.Node)
    ):
        return True

    return False


def dynamo_tensorrt_converter(
    key: Target,
    *,
    enabled: bool = True,
    capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
) -> Callable[[ConverterImplSignature], ConverterImplSignature]:
    """Decorator for Dynamo TensorRT Converter

    Registers the decorated function in the DYNAMO_ATEN_CONVERTERS registry

    Args:
        key: Node target for which the converter is implemented for
            (for example, torch.ops.add.Tensor)
        enabled: Whether the converter should be enabled/cached or not
        capability_validator: Function which evaluates whether a node is valid for conversion
            by the decorated converter. See ConverterSupport for more details.
            Defaults to None, implying the capability_validator function is always true -
            this means all nodes of "key" kind can be supported by this converter
        priority: Converter's level of priority relative to other converters with the
            same target
        supports_dynamic_shapes: Boolean flag indicating if the converter has support for dynamic shapes.
        requires_output_allocator: Boolean flag indicating if the converter creates operators which require an Output Allocator to run (e.g. data dependent operators).
    Returns:
        The converter being decorated
    """

    def register_converter(converter: ConverterImplSignature) -> ConverterImplSignature:
        """Helper function to register the converter, then return it"""
        assert callable(converter), "Converter function must be callable"

        # If no capability_validator function is specified, use the default function - always return true
        if capability_validator is None:
            converter_support = ConverterSupport(
                converter_implementation=converter,
                supports_dynamic_shapes=supports_dynamic_shapes,
                requires_output_allocator=requires_output_allocator,
            )
        else:
            assert callable(
                capability_validator
            ), "Argument checking function must be callable"
            converter_support = ConverterSupport(
                converter_implementation=converter,
                capability_validator=capability_validator,
                supports_dynamic_shapes=supports_dynamic_shapes,
                requires_output_allocator=requires_output_allocator,
            )

        # OpOverloadPackets are only valid if they have a single overload, or
        # only the ["default", "out"] overloads, due to PyTorch conventions
        if isinstance(key, OpOverloadPacket) and (
            len(key.overloads()) >= 3
            or (len(key.overloads()) == 2 and "out" not in key.overloads())
        ):
            raise AssertionError(
                f"Detected converter for OpOverloadPacket {key}. "
                "We do not support OpOverloadPacket-keyed converters with multiple overloads. "
                "Make sure to explicitly specify each converter overload. For instance "
                "aten.mean is not a valid key, but aten.mean.default is."
            )

        # If a converter for this operator already exists, append the new converter to the list
        # Otherwise, start a new list
        if key in DYNAMO_ATEN_CONVERTERS:
            # High priority converters are inserted at the front of the list,
            # so they can be checked first by the registry
            if priority is ConverterPriority.HIGH:
                DYNAMO_ATEN_CONVERTERS[key].insert(0, converter_support)
            else:
                DYNAMO_ATEN_CONVERTERS[key].append(converter_support)
        else:
            DYNAMO_ATEN_CONVERTERS[key] = [converter_support]

        logger.debug(
            f"Converter for {key} added to Dynamo ATen Converter Registry with priority: {priority}"
        )

        return converter

    def disable_converter(converter: ConverterImplSignature) -> ConverterImplSignature:
        return converter

    # Select whether to cache/enable the converter
    if enabled:
        return register_converter
    else:
        return disable_converter


class ConverterRegistry:
    """Registry for storing multiple converter dictionaries

    Capable of storing dictionaries with the following signature:
    Dict[Target, Union[Callable, Sequence[ConverterSupport]]]

    Also able to validate converter implementations against user-provided
    argument-checking functions

    Args:
        registries: List of dictionaries representing converter registries.
            The order of the provided dictionaries is the order in which they
            will be traversed. This is only significant when using non-validated
            methods
        registry_names: Optional list of names for each registry
        registry_calling_conventions: Optional list of calling conventions
            for each registry
    """

    def __init__(
        self,
        registries: Sequence[
            Dict[Target, Union[Callable[..., Any], Sequence[ConverterSupport]]]
        ],
        registry_names: Optional[Sequence[str]] = None,
        registry_calling_conventions: Optional[Sequence[CallingConvention]] = None,
    ):
        # Copy reference to each dictionary object into attribute list
        self.registries = list(registries)

        if registry_names is not None:
            assert len(self.registries) == len(registry_names)
            self.registry_names = list(registry_names)
        else:
            self.registry_names = [
                f"Registry {i + 1}" for i in range(len(self.registries))
            ]

        if registry_calling_conventions is not None:
            assert len(self.registries) == len(registry_calling_conventions)
            self.registry_calling_conventions = list(registry_calling_conventions)
        else:
            self.registry_calling_conventions = [
                CallingConvention.CTX for _ in range(len(self.registries))
            ]

        self.compilation_settings: CompilationSettings = None
        self.disallowed_targets: Collection[Target] = set()
        self.validate_invariants()

    def set_compilation_settings(
        self, compilation_settings: CompilationSettings
    ) -> None:
        self.compilation_settings = compilation_settings
        # set torch executed ops as disallowed targets
        self.set_disallowed_targets(compilation_settings.torch_executed_ops)

    def set_disallowed_targets(self, torch_executed_ops: Collection[Target]) -> None:
        self.disallowed_targets = torch_executed_ops

    def get_disallowed_targets(self, torch_executed_ops: Collection[Target]) -> None:
        self.disallowed_targets = torch_executed_ops

    def validate_invariants(self) -> None:
        """Validates the invariants required of the dictionaries in the registries

        Raises AssertionError if any invariants have been violated
        """
        # All registries must be dictionaries
        assert all(isinstance(elt, dict) for elt in self.registries)

        # Every dictionary in the registry must have one of two signatures:
        # Dict[Target, Callable] or Dict[Target, Sequence[ConverterSupport]]
        # Where, for the latter, the sequence must be non-empty
        for registry in self.registries:
            for converters in registry.values():
                if isinstance(converters, (list, tuple)):
                    assert (
                        all(isinstance(c, ConverterSupport) for c in converters)
                        and len(converters) > 0
                    )
                else:
                    assert callable(converters), "Converter function must be callable"

    def __getitem_without_validation__(
        self, key: Target
    ) -> Tuple[
        Any, CallingConvention
    ]:  # TODO: Narrow to ConverterImplSignature this when we can remove FX converters
        """Get the first-found converter in any registry

        Searches all registries in order and returns the first converter encountered,
        along with the calling convention of the registry the converter was sourced from
        """
        if isinstance(key, Node):
            raise KeyError(
                "Unvalidated accesses to the Converter registry can only be "
                + "made with node targets. Try accessing the registry with node.target"
            )

        self.validate_invariants()

        if (
            key in self.disallowed_targets
            or self.qualified_name_or_str(key) in self.disallowed_targets
        ):
            raise KeyError(
                f"A converter exists for {key}, but it was " "explicitly disallowed"
            )

        # Iterate over all registries and return the first converter found
        for registry, calling_convention in zip(
            self.registries, self.registry_calling_conventions
        ):
            if key in registry:
                converters = registry[key]

                if isinstance(converters, (list, tuple)):
                    return converters[0].converter_implementation, calling_convention
                else:
                    return converters, calling_convention

        raise KeyError(f"None of the converter registries have an entry for {key}")

    def __getitem__(
        self, node: Node
    ) -> Tuple[
        Any, CallingConvention, Dict[str, bool]
    ]:  # TODO: Narrow to ConverterImplSignature this when we can remove FX converters
        """Get the first-found validated converter in any registry

        Searches all registries in order and returns the first converter which passes
        validation on the input node, along with the calling convention of the
        registry the converter was sourced from
        """
        if not isinstance(node, Node):
            raise KeyError(
                "Validated accesses to the Converter registry can only be "
                + "made with node inputs. Try accessing the registry with a node "
                + "or use get_unvalidated to access without node validation."
            )

        self.validate_invariants()
        key = node.target
        assume_dynamic_shape_support = False
        if self.compilation_settings:
            assume_dynamic_shape_support = (
                self.compilation_settings.assume_dynamic_shape_support
            )
        if (
            key in self.disallowed_targets
            or self.qualified_name_or_str(key) in self.disallowed_targets
        ):
            raise KeyError(
                f"A converter exists for {key}, but it was " "explicitly disallowed"
            )

        # Iterate over all registries, validating the converter on the input node
        # If no capability_validator function is found, assume full coverage
        for registry, calling_convention in zip(
            self.registries, self.registry_calling_conventions
        ):
            if key in registry:
                converters = registry[key]
                if isinstance(converters, (list, tuple)):
                    logger.debug(f"Converter options for {key}: {len(converters)}")
                    for i, candidate in enumerate(converters):
                        # We enable the converter under 4 conditions
                        # 1) capability validator is True
                        # 2) Assume dynamic_shape support is True
                        # 3) Node only has static shaped inputs
                        # 4) Node has dynamic inputs and the converter has supports_dynamic_shapes=True
                        if is_valid := candidate.capability_validator(
                            node, self.compilation_settings
                        ) and (
                            assume_dynamic_shape_support
                            or not node_has_dynamic_shapes(node)
                            or candidate.supports_dynamic_shapes
                        ):
                            logger.debug(
                                f"Selecting converter option {i} for converting {key}"
                            )
                            return (
                                candidate.converter_implementation,
                                calling_convention,
                                {
                                    "supports_dynamic_shapes": candidate.supports_dynamic_shapes,
                                    "requires_output_allocator": candidate.requires_output_allocator,
                                },
                            )
                        else:
                            logger.debug(
                                f"Skipping option {i} for {key}: (validator: {is_valid}, supports dynamic shapes: {candidate.supports_dynamic_shapes})"
                            )
                            continue
                else:
                    # Assuming FX converters don't have dynamic shapes supported
                    if not node_has_dynamic_shapes(node):
                        return (
                            converters,
                            calling_convention,
                            {
                                "supports_dynamic_shapes": False,
                                "requires_output_allocator": False,
                            },
                        )

        raise KeyError(
            f"None of the converter registries have a validated entry for {key}, with node {node}"
        )

    def keys(self) -> Set[Target]:
        """Get all unique targets across all dictionaries"""
        return self.unique_targets()

    def get_unvalidated(
        self, key: Target, value: Optional[ConverterImplSignature] = None
    ) -> Union[
        Any, Tuple[Any, CallingConvention]
    ]:  # TODO: Narrow to ConverterImplSignature this when we can remove FX converters
        """Get unvalidated converter for input target with a default return"""
        try:
            return self.__getitem_without_validation__(key)
        except KeyError:
            return value

    def get(
        self, node: Node, value: Optional[ConverterImplSignature] = None
    ) -> Union[
        Any, Tuple[Any, CallingConvention, Dict[str, bool]]
    ]:  # TODO: Narrow to ConverterImplSignature this when we can remove FX converters
        """Get validated converter for input node with a default return"""
        try:
            return self.__getitem__(node)
        except KeyError:
            return value

    def __contains__(self, key: Target | Node) -> bool:
        """Check whether a converter for an input node or target exists"""
        try:
            # Attempt to access the item in the registry
            if isinstance(key, Node):
                self.__getitem__(key)
            else:
                self.__getitem_without_validation__(key)

            return True
        except KeyError:
            return False

    def get_all_converters_with_target(
        self, key: Target, return_registry_info: bool = False
    ) -> Tuple[
        Union[List[Any], Dict[str, int], None]
    ]:  # TODO: Narrow to ConverterImplSignature this when we can remove FX converters
        """Get all converters across all registries for the target

        Returns a list of all converterts having the specified target
        """
        self.validate_invariants()
        converters_with_target = []

        # Store count of number of registered converters per registry
        if return_registry_info:
            registry_data = {name: 0 for name in self.registry_names}

        for index, registry in enumerate(self.registries):
            if key in registry:
                converters = registry[key]

                if isinstance(converters, (list, tuple)):
                    converters_with_target.extend(
                        [c.converter_implementation for c in converters]
                    )
                    # Add converter count to registry name storage
                    if return_registry_info:
                        registry_data[self.registry_names[index]] += len(converters)
                else:
                    converters_with_target.append(converters)
                    # Add converter count to registry name storage
                    if return_registry_info:
                        registry_data[self.registry_names[index]] += 1

        if return_registry_info:
            return converters_with_target, registry_data
        else:
            return converters_with_target, None

    def __setitem__(self, key: Any, value: Any) -> None:
        raise AssertionError(
            "Do not set registry members directly through the ConverterRegistry object. "
            + f"Attempted to set {key}: {value} via direct assignment to ConverterRegistry."
        )

    def __delitem__(self, key: Any) -> None:
        raise AssertionError(
            "Do not delete registry members directly through the ConverterRegistry object. "
            + f"Attempted to delete {key} via direct del on ConverterRegistry."
        )

    def __len__(self) -> int:
        """Returns the sum of lengths of all registries stored"""
        return sum(len(registry) for registry in self.registries)

    def unique_targets(self) -> Set[Target]:
        """Returns the set of unique converter targets stored across all registries"""
        return set.union(*[set(registry.keys()) for registry in self.registries])

    @staticmethod
    def qualified_name_or_str(target: Target) -> str:
        """Returns string representation of an FX Node target"""
        if isinstance(target, str):
            return target
        else:
            return cast(str, _get_qualified_name(target))

    def get_converter_support_info(self) -> Dict[str, Optional[Dict[str, int]]]:
        """Returns a dictionary of targets backed by at least one converter"""
        available_converters = {}
        for target in sorted(
            self.unique_targets(), key=lambda target: self.qualified_name_or_str(target)
        ):
            _, registry_data = self.get_all_converters_with_target(
                target, return_registry_info=True
            )
            available_converters[self.qualified_name_or_str(target)] = registry_data
        return available_converters

    def display_all_available_converters(self) -> str:
        """Returns a string with all converters and their source, separated by newlines"""
        available_converters = "Available converters in ATen registries with counts:\n"

        support_info = self.get_converter_support_info()
        for target, registry_data in support_info.items():
            available_converters += f"Node: {self.qualified_name_or_str(target)} - Registry Presence Counts: {registry_data}\n"

        return available_converters


# Initialize dynamo converter registry with the FX and Dynamo aten registries
# Note the Dynamo registry is listed first, for precedence
registries = [
    DYNAMO_ATEN_CONVERTERS,
]
registry_names = ["Dynamo ATen Converters Registry"]
registry_calling_conventions = [
    CallingConvention.CTX,
]
if torch_tensorrt.ENABLED_FEATURES.fx_frontend:
    from torch_tensorrt.fx.converter_registry import CONVERTERS as FX_CONVERTERS

    registries.append(FX_CONVERTERS)
    registry_names.append("FX Legacy ATen Converters Registry")
    registry_calling_conventions.append(CallingConvention.LEGACY)


DYNAMO_CONVERTERS: ConverterRegistry = ConverterRegistry(
    registries,
    registry_names,
    registry_calling_conventions,
)
