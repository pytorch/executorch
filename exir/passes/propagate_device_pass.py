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
import torch.utils._pytree as pytree
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.passes.convert_constant_dim_order_pass import _should_transform
from executorch.exir.passes.insert_write_back_for_buffers_pass import (
    _alias_sets,
    _contains_node,
    _schema_paired_args,
    _schema_writes_input,
)

# Re-exported for backward compatibility; the dataclass lives in a lightweight
# module so that ExecutorchBackendConfig can reference it without importing the
# et_copy op registry above.
from executorch.exir.passes.propagate_device_config import (  # noqa: F401
    PropagateDeviceConfig,
)
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


def _copy_output_spec(
    spec: TensorSpec,
    device_type: schema.DeviceType,
    device_index: int = 0,
) -> TensorSpec:
    """Create the spec of an _h2d_copy or _d2h_copy output from the spec it copies.

    The output lives on ``device_type``. The copy writes it at run time, so it is
    never a constant, even when the copied tensor is one.
    """
    new_spec = copy.copy(spec)
    new_spec.init_mem_planning_fields()
    new_spec.const = False
    _set_device_on_spec(new_spec, device_type, device_index)
    return new_spec


def _schema_aliases_input(
    node: torch.fx.Node, schema: torch.FunctionSchema, input_node: torch.fx.Node
) -> bool:
    """Whether schema marks the result of node as an alias of input_node, as for
    a view or an in-place op."""
    returned = set().union(*(_alias_sets(ret.alias_info) for ret in schema.returns))
    return any(
        _alias_sets(schema_arg.alias_info) & returned
        for arg, schema_arg in _schema_paired_args(node, schema)
        if _contains_node(arg, input_node)
    )


def _in_place_writers(node: torch.fx.Node) -> list[torch.fx.Node]:
    """The ops that write *node* in place, directly or through a view of it,
    per the op's schema.

    A mutable op left on the host does this once to_executorch has removed
    auto_functionalized, so a copy of *node* can miss the write. If the op
    wrote a view of *node* in the model, that removal makes the view again,
    for example with as_strided. The result of an in-place op is a view of
    its input too. A view_copy is not: it still makes its own tensor here,
    and ReplaceViewCopyWithViewPass turns it into a view later only where no
    write can tell the difference. Custom op schemas are trusted, as
    functionalization trusts them. The write-back pass trusts only aten
    schemas, but here that would keep a copy for every custom op on the host,
    such as a KV cache update. Users without a schema, such as delegate calls
    and getitem, do not write their inputs.
    """
    writers: dict[torch.fx.Node, None] = {}
    aliases = {node}
    pending = [node]
    while pending:
        alias = pending.pop()
        for user in alias.users:
            schema = getattr(user.target, "_schema", None)
            if schema is None:
                continue
            if _schema_writes_input(user, schema, alias):
                writers[user] = None
            if user not in aliases and _schema_aliases_input(user, schema, alias):
                aliases.add(user)
                pending.append(user)
    return list(writers)


def _written_between(
    node: torch.fx.Node,
    after: Optional[torch.fx.Node],
    before: torch.fx.Node,
) -> bool:
    """Whether an op writes *node* in place after *after*, or anywhere when it
    is None, and before *before*, in graph order, which is the order the
    method runs in. A write outside that range cannot change what a reader at
    *before* of a copy made at *after* sees."""
    writers = _in_place_writers(node)
    if not writers:
        return False
    position = {n: i for i, n in enumerate(before.graph.nodes)}
    start = position[after] if after is not None else -1
    return any(start < position[writer] < position[before] for writer in writers)


def _reusable_copy(
    tensor: torch.fx.Node,
    delegate: torch.fx.Node,
    device_type: schema.DeviceType,
    device_index: int,
) -> Optional[torch.fx.Node]:
    """The _h2d_copy of *tensor* to this device that *delegate* can read too.

    Delegates are visited in graph order, so a copy made for an earlier
    delegate on this device already comes before *delegate*, and the last such
    copy is the newest. It is stale when an op writes *tensor* in place between
    that copy and *delegate*, so then there is none.
    """
    copy = None
    for user in tensor.users:
        if (
            user.target == torch.ops.et_copy._h2d_copy.default
            and user.meta["spec"].device == device_type
            and user.meta["spec"].device_index == device_index
        ):
            copy = user
    if copy is not None and _written_between(tensor, copy, delegate):
        return None
    return copy


def _replace_spec(
    readers: list[torch.fx.Node], old_spec: TensorSpec, new_spec: TensorSpec
) -> None:
    """Replace *old_spec* with *new_spec* in *readers*, which now read the
    tensor with *new_spec*, and in the readers of their results.

    SpecPropPass gives the output node the spec of each tensor it returns, and
    an in-place result the spec of the tensor it writes. Memory planning relies
    on both, so these specs have to follow the tensor that is read now.
    """
    pending = list(readers)
    while pending:
        node = pending.pop()
        spec = node.meta.get("spec")
        if any(value is old_spec for value in pytree.tree_leaves(spec)):
            node.meta["spec"] = pytree.tree_map(
                lambda value: new_spec if value is old_spec else value, spec
            )
            pending.extend(node.users)


class PropagateDevicePass(PassBase):
    """
    After to_backend, walk the graph and insert H2D/D2H copy ops at delegate
    boundaries based on partitioner-assigned device info.

    When a delegate has a target_device CompileSpec (e.g., "cuda:0"):
    - For each delegate input: insert et_copy._h2d_copy before the delegate call,
      or reuse the one an earlier delegate on the same device reads when no op
      writes the input in place in between. A reused copy stays in device
      memory until its last reader.
      The original input node keeps its device; the h2d_copy output is tagged as
      device.
    - For each delegate output: insert et_copy._d2h_copy after each getitem,
      or after the call itself when it returns a single tensor.
      The delegate output stays device; the d2h_copy output is tagged as CPU.
    - Getitem nodes that extract from a delegate call inherit the device.

    Skip-copy optimizations:
    - skip_h2d_for_method_inputs: If the input is a graph-level placeholder
      feeding a delegate, don't insert H2D; tag the placeholder as device
      instead (user provides GPU tensor at runtime). Every reader of the
      placeholder must then be a delegate on that device, or the pass raises.
      Constants are placeholders too, but their data is stored on the host,
      so they still get an H2D copy.
    - skip_d2h_for_method_outputs: The graph output reads a delegate output
      on device, not its D2H copy; the copy stays only for other readers. When
      one of them writes the copy in place, the graph output reads a device
      copy of the written tensor instead.
    """

    def __init__(
        self,
        skip_h2d_for_method_inputs: bool = False,
        skip_d2h_for_method_outputs: bool = False,
        enable_non_cpu_memory_planning: bool = False,
    ) -> None:
        super().__init__()
        self.skip_h2d_for_method_inputs = skip_h2d_for_method_inputs
        self.skip_d2h_for_method_outputs = skip_d2h_for_method_outputs
        self.enable_non_cpu_memory_planning = enable_non_cpu_memory_planning

        if (
            skip_h2d_for_method_inputs or skip_d2h_for_method_outputs
        ) and not enable_non_cpu_memory_planning:
            raise ValueError(
                "skip_h2d_for_method_inputs and skip_d2h_for_method_outputs are "
                "only meaningful when enable_non_cpu_memory_planning=True, since "
                "they control host/device copy insertion which only happens during "
                "device-aware memory planning. Set enable_non_cpu_memory_planning="
                "True, or leave the skip options disabled."
            )

    def _is_placeholder(self, node: torch.fx.Node) -> bool:
        """Check if a node is a graph-level input (placeholder)."""
        return node.op == "placeholder"

    def _delegate_device(
        self, graph_module: torch.fx.GraphModule, node: torch.fx.Node
    ) -> Optional[tuple[schema.DeviceType, int]]:
        """The target device of a delegate call node, or None when there is none."""
        if node.op != "call_function" or node.target != executorch_call_delegate:
            return None
        lowered_module = _get_lowered_module(graph_module, node)
        if lowered_module is None:
            return None
        return _get_target_device_from_compile_specs(lowered_module)

    def _insert_h2d_copies(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        target_device_type: schema.DeviceType,
        device_index: int,
    ) -> bool:
        """Make each tensor input of a delegate call read device memory.

        A method input that is not a constant moves to the device under
        skip_h2d_for_method_inputs.
        Any other input reads an H2D copy, reused from an earlier delegate on
        the same device when possible.
        """
        changed = False
        new_args = list(node.args)
        for i, arg in enumerate(node.args[1:], start=1):
            if not isinstance(arg, torch.fx.Node):
                continue
            arg_spec = arg.meta.get("spec")
            if not isinstance(arg_spec, TensorSpec):
                continue

            if (
                self.skip_h2d_for_method_inputs
                and self._is_placeholder(arg)
                and not arg_spec.const
            ):
                # The spec is shared by every reader, so all of them must be
                # delegates on this same device.
                other_readers = [
                    user
                    for user in arg.users
                    if self._delegate_device(graph_module, user)
                    != (target_device_type, device_index)
                ]
                if other_readers:
                    write_back = next(
                        (
                            user
                            for user in other_readers
                            if user.target == torch.ops.aten.copy_.default
                            and user.args[0] is arg
                        ),
                        None,
                    )
                    refusal = (
                        f"skip_h2d_for_method_inputs=True cannot place placeholder "
                        f"'{arg.name}' on {target_device_type.name.lower()}:"
                        f"{device_index}, because "
                        f"'{(write_back or other_readers[0]).name}'"
                    )
                    if write_back is not None:
                        raise RuntimeError(
                            f"{refusal} writes it back on the host (the method "
                            f"mutates it). Either disable "
                            f"skip_h2d_for_method_inputs, or, for a mutable buffer, "
                            f"have the partitioner put the buffer and every op that "
                            f"reads or writes it in the delegate; tag_mutated_buffer "
                            f"tags the buffer once those ops are there."
                        )
                    raise RuntimeError(
                        f"{refusal} also reads it and is not a delegate on that "
                        f"device. Either disable skip_h2d_for_method_inputs, or use "
                        f"the placeholder only from delegates on one device."
                    )
                _set_device_on_spec(arg_spec, target_device_type, device_index)
                changed = True
                continue

            val = arg.meta.get("val")
            if (
                arg_spec.const
                and isinstance(val, torch.Tensor)
                and _should_transform(val)
            ):
                raise RuntimeError(
                    f"Constant '{arg.name}' feeds a delegate on "
                    f"{target_device_type.name.lower()}:{device_index}, but it is "
                    f"neither contiguous nor channels-last. The program stores "
                    f"such a constant contiguous, so its device copy would not "
                    f"match the stored data. Make the constant contiguous where "
                    f"it is created, for example in the pass that adds it."
                )

            h2d_node = _reusable_copy(arg, node, target_device_type, device_index)
            if h2d_node is None:
                with graph_module.graph.inserting_before(node):
                    h2d_node = graph_module.graph.call_function(
                        torch.ops.et_copy._h2d_copy.default,
                        (arg,),
                    )
                    h2d_node.meta["spec"] = _copy_output_spec(
                        arg_spec, target_device_type, device_index
                    )
                    h2d_node.meta["val"] = arg.meta.get("val")
                    if "tensor_meta" in arg.meta:
                        h2d_node.meta["tensor_meta"] = arg.meta["tensor_meta"]
                if arg_spec.const:
                    logger.warning(
                        "Constant '%s' (%d bytes) feeds a delegate on %s:%d from "
                        "outside it, so it is copied to the device on every call. "
                        "Have the partitioner take it into the delegate to avoid "
                        "that copy.",
                        arg.name,
                        arg_spec.nbytes(),
                        target_device_type.name.lower(),
                        device_index,
                    )
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
        self._insert_d2h_after(graph_module, node, spec)
        return True

    def _insert_d2h_after(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        spec: TensorSpec,
    ) -> None:
        """Insert a D2H copy after *node*, a device tensor, for all of its users."""
        with graph_module.graph.inserting_after(node):
            d2h_node = graph_module.graph.call_function(
                torch.ops.et_copy._d2h_copy.default,
                (node,),
            )
            d2h_spec = _copy_output_spec(spec, schema.DeviceType.CPU, 0)
            d2h_node.meta["spec"] = d2h_spec
            d2h_node.meta["val"] = node.meta.get("val")
            if "tensor_meta" in node.meta:
                d2h_node.meta["tensor_meta"] = node.meta["tensor_meta"]

            node.replace_all_uses_with(
                d2h_node,
                delete_user_cb=lambda user, _d2h=d2h_node: user != _d2h,
            )
            _replace_spec(list(d2h_node.users), spec, d2h_spec)

    def _keep_method_outputs_on_device(
        self,
        graph_module: torch.fx.GraphModule,
    ) -> None:
        """Make the graph output read the device tensor instead of its D2H copy.

        The copy stays for its other users, such as a host op. When one of them
        writes the copy in place, the device tensor never sees that write, so the
        output reads a new device copy of the written host copy instead, and a
        warning says so.
        """
        d2h = torch.ops.et_copy._d2h_copy.default
        for node in list(graph_module.graph.nodes):
            if (
                node.op != "call_function"
                or node.target != d2h
                or not any(user.op == "output" for user in node.users)
            ):
                continue
            returned = node.args[0]
            writers = _in_place_writers(node)
            if writers:
                device_spec = returned.meta["spec"]
                output_node = graph_module.graph.output_node()
                # A later delegate on this device may already read a device copy
                # of the written host copy.
                returned = _reusable_copy(
                    node, output_node, device_spec.device, device_spec.device_index
                )
                if returned is None:
                    with graph_module.graph.inserting_before(output_node):
                        returned = graph_module.graph.call_function(
                            torch.ops.et_copy._h2d_copy.default, (node,)
                        )
                    returned.meta["spec"] = _copy_output_spec(
                        node.meta["spec"], device_spec.device, device_spec.device_index
                    )
                    returned.meta["val"] = node.meta.get("val")
                    if "tensor_meta" in node.meta:
                        returned.meta["tensor_meta"] = node.meta["tensor_meta"]
                logger.warning(
                    "A method output (%d bytes) comes from a delegate on %s:%d "
                    "and '%s' writes it in place on the host, so it is copied to "
                    "the host and back to the device on every call, even with "
                    "skip_d2h_for_method_outputs. Have the partitioner take that "
                    "op into the delegate to avoid those copies.",
                    device_spec.nbytes(),
                    device_spec.device.name.lower(),
                    device_spec.device_index,
                    writers[0].target,
                )
            _replace_spec(
                [graph_module.graph.output_node()],
                node.meta["spec"],
                returned.meta["spec"],
            )
            node.replace_all_uses_with(
                returned, delete_user_cb=lambda user: user.op == "output"
            )
            if not node.users:
                graph_module.graph.erase_node(node)

    def _elide_redundant_same_device_copies(
        self,
        graph_module: torch.fx.GraphModule,
    ) -> None:
        """Remove ``_h2d_copy(_d2h_copy(x))`` round-trips that never leave a device.

        H2D and D2H copies are inserted per delegate boundary independently (an
        H2D before every device delegate input, a D2H after every device delegate
        output), so two consecutive delegates on the *same* device leave a
        redundant ``device -> host -> device`` bounce: the first delegate's output
        is copied to host by a ``_d2h_copy`` and immediately copied back to the
        same device by a ``_h2d_copy`` feeding the second delegate.

        This walks each ``_h2d_copy`` whose input is a ``_d2h_copy`` and, when the
        tensor before the D2H already lives on the device the H2D copies back to,
        rewires the consumer straight to that device tensor and drops both copies.
        The device comparison is the correctness guard: a genuine cross-device
        move (e.g. ``cuda:0 -> host -> cuda:1``) has differing devices and is left
        intact. So is a round-trip whose host copy an op writes in place before
        the copy back, because the device tensor never sees that write. A D2H
        feeding other (host) consumers keeps those users and is removed only once
        it becomes dead.
        """
        h2d = torch.ops.et_copy._h2d_copy.default
        d2h = torch.ops.et_copy._d2h_copy.default
        changed = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target != h2d:
                continue
            src = node.args[0]
            if not (
                isinstance(src, torch.fx.Node)
                and src.op == "call_function"
                and src.target == d2h
            ):
                continue

            x = src.args[0]  # the tensor before the d2h (the device delegate output)
            x_spec = x.meta.get("spec") if isinstance(x, torch.fx.Node) else None
            h2d_spec = node.meta.get("spec")
            if not (
                isinstance(x_spec, TensorSpec) and isinstance(h2d_spec, TensorSpec)
            ):
                continue

            # Only elide a proven same-device round-trip.
            if (
                x_spec.device != h2d_spec.device
                or x_spec.device_index != h2d_spec.device_index
                or _written_between(src, None, node)
            ):
                continue

            node.replace_all_uses_with(x)
            graph_module.graph.erase_node(node)
            changed = True

        if changed:
            # Drop the d2h copies that are now dead (kept if they still feed a
            # host consumer).
            graph_module.graph.eliminate_dead_code()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:  # noqa: C901
        # Three-pass approach:
        #   Pass 1 – For each delegate with a target_device CompileSpec, insert
        #            H2D copy nodes before delegate inputs, or reuse the copy an
        #            earlier delegate on the same device reads, and tag the delegate
        #            output specs with the target device.  Delegates without a
        #            target_device are left untouched (no copies, specs stay CPU).
        #   Pass 2 – For each getitem that extracts from a device-tagged delegate
        #            (tracked in device_delegates), propagate the device onto the
        #            getitem spec and insert a D2H copy after it so downstream
        #            non-delegated ops receive CPU tensors. A delegate call that
        #            returns a single tensor gets the D2H copy itself.
        #   Pass 3 – Elide the redundant device->host->device round-trips left
        #            between consecutive same-device delegates by passes 1 and 2.
        #            With skip_d2h_for_method_outputs, then make the graph output
        #            read device tensors instead of D2H copies.
        changed = False
        device_delegates: set[torch.fx.Node] = set()

        # Pass 1: insert H2D copies and tag delegate output specs.
        for node in list(graph_module.graph.nodes):
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
                device_delegates.add(node)

                if self.enable_non_cpu_memory_planning:
                    changed |= self._insert_h2d_copies(
                        graph_module, node, target_device_type, device_index
                    )
                else:
                    for arg in node.args[1:]:
                        if isinstance(arg, torch.fx.Node):
                            changed |= _tag_specs_with_device(
                                arg.meta.get("spec"),
                                target_device_type,
                                device_index,
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

        # Second pass: propagate device through getitem nodes and insert D2H,
        # after each getitem or single-tensor call, only for delegates that have
        # a target_device.
        for node in list(graph_module.graph.nodes):
            # A lowered module that returns one tensor and is called from another
            # module has no getitem after its call, so copy its output directly.
            if (
                node in device_delegates
                and self.enable_non_cpu_memory_planning
                and isinstance(node.meta.get("spec"), TensorSpec)
            ):
                self._insert_d2h_after(graph_module, node, node.meta["spec"])
                changed = True
            elif node.op == "call_function" and node.target == operator.getitem:
                source = node.args[0]
                if isinstance(source, torch.fx.Node) and source in device_delegates:
                    if self.enable_non_cpu_memory_planning:
                        changed |= self._insert_d2h_for_getitem(graph_module, node)
                    else:
                        spec = node.meta.get("spec")
                        source_specs = source.meta.get("spec")
                        idx = node.args[1]
                        if (
                            isinstance(spec, TensorSpec)
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

        # Third pass: elide the redundant device->host->device round-trips left
        # between consecutive same-device delegates by the per-boundary H2D/D2H
        # insertion above, then, with skip_d2h_for_method_outputs, make the graph
        # output read device tensors. Only meaningful when passes 1/2 modified the
        # graph (there was a device delegate); when nothing changed there are no
        # copies to elide, so skip the scan entirely.
        if changed:
            self._elide_redundant_same_device_copies(graph_module)
            if self.skip_d2h_for_method_outputs:
                self._keep_method_outputs_on_device(graph_module)

        graph_module.recompile()
        return PassResult(graph_module, changed)
