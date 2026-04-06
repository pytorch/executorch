# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
import operator
from typing import Optional

# Import to register the et_copy ops so torch.ops.et_copy is available.
import executorch.exir.passes._device_copy_ops_registry  # noqa: F401

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


def _clone_spec_with_device(
    spec: TensorSpec,
    device_type: schema.DeviceType,
    device_index: int = 0,
) -> TensorSpec:
    """Create a copy of a TensorSpec with a different device."""
    new_spec = copy.copy(spec)
    new_spec.init_mem_planning_fields()
    _set_device_on_spec(new_spec, device_type, device_index)
    return new_spec


class PropagateDevicePass(PassBase):
    """
    After to_backend, walk the graph and insert H2D/D2H copy ops at delegate
    boundaries based on partitioner-assigned device info.

    When a delegate has a target_device CompileSpec (e.g., "cuda:0"):
    - For each delegate input: insert et_copy._h2d_copy before the delegate call.
      The original input node stays CPU; the h2d_copy output is tagged as device.
    - For each delegate output: insert et_copy._d2h_copy after each getitem.
      The getitem stays device; the d2h_copy output is tagged as CPU.
    - Getitem nodes that extract from a delegate call inherit the device.

    Skip-copy optimizations:
    - skip_h2d_for_method_inputs: If the input is a graph-level placeholder
      feeding directly to a delegate, don't insert H2D — tag the placeholder
      as device instead (user provides GPU tensor at runtime).
    - skip_d2h_for_method_outputs: If the getitem feeds directly to graph
      output, don't insert D2H — the output stays on device.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def _is_placeholder(self, node: torch.fx.Node) -> bool:
        """Check if a node is a graph-level input (placeholder)."""
        return node.op == "placeholder"

    def _feeds_directly_to_output(self, node: torch.fx.Node) -> bool:
        """Check if all users of a node are output nodes."""
        return all(user.op == "output" for user in node.users)

    def _insert_h2d_copies(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        target_device_type: schema.DeviceType,
        device_index: int,
    ) -> bool:
        """Insert H2D copy nodes for each tensor input to a delegate call."""
        changed = False
        new_args = list(node.args)
        for i, arg in enumerate(node.args[1:], start=1):
            if not isinstance(arg, torch.fx.Node):
                continue
            arg_spec = arg.meta.get("spec")
            if not isinstance(arg_spec, TensorSpec):
                continue

            with graph_module.graph.inserting_before(node):
                h2d_node = graph_module.graph.call_function(
                    torch.ops.et_copy._h2d_copy.default,
                    (arg,),
                )
                h2d_spec = _clone_spec_with_device(
                    arg_spec, target_device_type, device_index
                )
                h2d_node.meta["spec"] = h2d_spec
                h2d_node.meta["val"] = arg.meta.get("val")
                if "tensor_meta" in arg.meta:
                    h2d_node.meta["tensor_meta"] = arg.meta["tensor_meta"]
                new_args[i] = h2d_node
                changed = True

        node.args = tuple(new_args)
        return changed

    def _insert_d2h_for_getitem(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
    ) -> bool:
        """If *node* is a getitem extracting from a delegate call, tag its spec
        with the delegate device and insert a D2H copy after it."""
        source_node = node.args[0]
        if not (
            isinstance(source_node, torch.fx.Node)
            and source_node.op == "call_function"
            and source_node.target == executorch_call_delegate
        ):
            return False

        spec = node.meta.get("spec")
        source_specs = source_node.meta.get("spec")
        idx = node.args[1]
        if not (
            isinstance(spec, TensorSpec)
            and isinstance(source_specs, (tuple, list))
            and isinstance(idx, int)
            and idx < len(source_specs)
        ):
            return False

        source_spec = source_specs[idx]
        if not isinstance(source_spec, TensorSpec):
            return False

        _set_device_on_spec(spec, source_spec.device, source_spec.device_index)

        with graph_module.graph.inserting_after(node):
            d2h_node = graph_module.graph.call_function(
                torch.ops.et_copy._d2h_copy.default,
                (node,),
            )
            d2h_spec = _clone_spec_with_device(spec, schema.DeviceType.CPU, 0)
            d2h_node.meta["spec"] = d2h_spec
            d2h_node.meta["val"] = node.meta.get("val")
            if "tensor_meta" in node.meta:
                d2h_node.meta["tensor_meta"] = node.meta["tensor_meta"]

            node.replace_all_uses_with(
                d2h_node,
                delete_user_cb=lambda user, _d2h=d2h_node: user != _d2h,
            )
        return True

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        changed = False

        for node in list(graph_module.graph.nodes):
            if node.op == "call_function" and node.target == executorch_call_delegate:
                lowered_module = _get_lowered_module(graph_module, node)
                if lowered_module is None:
                    continue

                result = _get_target_device_from_compile_specs(lowered_module)
                if result is None:
                    continue

                target_device_type, device_index = result

                changed |= self._insert_h2d_copies(
                    graph_module, node, target_device_type, device_index
                )

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

        # Second pass: propagate device through getitem nodes and insert D2H.
        for node in list(graph_module.graph.nodes):
            if node.op == "call_function" and node.target == operator.getitem:
                changed |= self._insert_d2h_for_getitem(graph_module, node)

        graph_module.recompile()
        return PassResult(graph_module, changed)
