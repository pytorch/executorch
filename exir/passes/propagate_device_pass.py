# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Optional

import executorch.exir.schema as schema

import torch
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.tensor import TensorSpec
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger: logging.Logger = logging.getLogger(__name__)

# CompileSpec key convention for specifying the target device.
# Partitioners that target a specific device should include a CompileSpec entry
# with this key and a value encoding the device string (e.g., b"cuda:0").
TARGET_DEVICE_COMPILE_SPEC_KEY = "target_device"


def _parse_device_spec_value(value: bytes) -> tuple[schema.DeviceType, int]:
    """
    Parse a target_device CompileSpec value (e.g., b"cuda:0") into
    (DeviceType, device_index).

    The type portion is matched case-insensitively against schema.DeviceType
    member names (e.g., "cpu", "cuda").  Raises ValueError for unknown types.
    """
    device_str = value.decode("utf-8").strip().lower()
    if ":" in device_str:
        type_str, index_str = device_str.split(":", 1)
        device_index = int(index_str)
    else:
        type_str = device_str
        device_index = 0
    device_type = next(
        (dt for dt in schema.DeviceType if dt.name.lower() == type_str),
        None,
    )
    if device_type is None:
        valid = ", ".join(dt.name for dt in schema.DeviceType)
        raise ValueError(f"Unknown device type '{type_str}'. Valid types: {valid}")
    return device_type, device_index


def _get_lowered_module(
    graph_module: torch.fx.GraphModule,
    delegate_call_node: torch.fx.Node,
) -> Optional[LoweredBackendModule]:
    """
    Given an executorch_call_delegate node, retrieve the associated
    LoweredBackendModule from the graph module.
    The first argument to executorch_call_delegate is a get_attr node
    whose target names the LoweredBackendModule attribute.
    """
    if len(delegate_call_node.args) < 1:
        return None
    lowered_node = delegate_call_node.args[0]
    if not isinstance(lowered_node, torch.fx.Node) or lowered_node.op != "get_attr":
        return None
    lowered_module = getattr(graph_module, lowered_node.target, None)
    if isinstance(lowered_module, LoweredBackendModule):
        return lowered_module
    return None


def _get_target_device_from_compile_specs(
    lowered_module: LoweredBackendModule,
) -> Optional[tuple[schema.DeviceType, int]]:
    """
    Look for a CompileSpec with key TARGET_DEVICE_COMPILE_SPEC_KEY and return
    the corresponding (DeviceType, device_index), or None if not found.
    """
    for spec in lowered_module.compile_specs:
        if spec.key == TARGET_DEVICE_COMPILE_SPEC_KEY:
            return _parse_device_spec_value(spec.value)
    return None


def _set_device_on_spec(
    spec: TensorSpec,
    device_type: schema.DeviceType,
    device_index: int = 0,
) -> None:
    """Set the device attribute on a TensorSpec."""
    spec.device = device_type
    spec.device_index = device_index


def _tag_specs_with_device(
    specs: object,
    device_type: schema.DeviceType,
    device_index: int = 0,
) -> bool:
    """Apply device annotation to a TensorSpec or a collection of TensorSpecs.

    Args:
        specs: A TensorSpec, a tuple/list of TensorSpecs, or None.
        device_type: The target device type to set.
        device_index: The device index (e.g., 0 for cuda:0, 1 for cuda:1).

    Returns:
        True if any spec was modified, False otherwise.
    """
    if specs is None:
        return False
    if isinstance(specs, TensorSpec):
        _set_device_on_spec(specs, device_type, device_index)
        return True
    if isinstance(specs, (tuple, list)):
        changed = False
        for s in specs:
            if isinstance(s, TensorSpec):
                _set_device_on_spec(s, device_type, device_index)
                changed = True
        return changed
    return False


class PropagateDevicePass(PassBase):
    """
    After to_backend, walk the graph and set device metadata on TensorSpecs
    based on partitioner-assigned delegation info.

    Rules:
    1. Delegated nodes: Input and output tensors of a delegate call are marked
       with the target device derived from the delegate's CompileSpec
       (key="target_device").
    2. Non-delegated nodes: Remain on CPU (default).
    3. Getitem nodes that extract from a delegate call inherit the device from
       the delegate call's output spec at the corresponding index.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        changed = False
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target == executorch_call_delegate:
                lowered_module = _get_lowered_module(graph_module, node)
                if lowered_module is None:
                    raise RuntimeError(
                        f"executorch_call_delegate node '{node.name}' does not reference "
                        "a valid LoweredBackendModule. The first argument must be a "
                        "get_attr node pointing to a LoweredBackendModule attribute."
                    )

                result = _get_target_device_from_compile_specs(lowered_module)
                if result is None:
                    continue

                target_device_type, device_index = result

                # Tag delegate input tensors.
                # args[0] is the get_attr node for the lowered module; skip it.
                for arg in node.args[1:]:
                    if isinstance(arg, torch.fx.Node):
                        changed |= _tag_specs_with_device(
                            arg.meta.get("spec"),
                            target_device_type,
                            device_index,
                        )

                # Tag delegate output tensors.
                changed |= _tag_specs_with_device(
                    node.meta.get("spec"),
                    target_device_type,
                    device_index,
                )

                logger.debug(
                    "PropagateDevicePass: set device=%s on delegate node %s "
                    "(backend=%s)",
                    target_device_type,
                    node.name,
                    lowered_module.backend_id,
                )

        # Second pass: propagate device through getitem nodes that extract
        # individual outputs from a delegate call.
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target.__name__ == "getitem":
                source_node = node.args[0]
                if (
                    isinstance(source_node, torch.fx.Node)
                    and source_node.op == "call_function"
                    and source_node.target == executorch_call_delegate
                ):
                    spec = node.meta.get("spec")
                    source_specs = source_node.meta.get("spec")
                    idx = node.args[1]
                    if (
                        spec is not None
                        and isinstance(spec, TensorSpec)
                        and source_specs is not None
                        and isinstance(source_specs, (tuple, list))
                        and isinstance(idx, int)
                        and idx < len(source_specs)
                    ):
                        source_spec = source_specs[idx]
                        if isinstance(source_spec, TensorSpec):
                            _set_device_on_spec(
                                spec,
                                source_spec.device,
                                source_spec.device_index,
                            )
                            changed = True

        return PassResult(graph_module, changed)
