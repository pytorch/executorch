# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast

import torch
from torch.utils import _pytree as pytree

MutableBuffers = dict[str, torch.Tensor]


def _get_attr_value(graph_module: torch.fx.GraphModule, target: str) -> Any:
    value: Any = graph_module
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


def _is_mutation_node(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and hasattr(node.target, "_schema")
        and ("copy_" in str(node.target) or "put_" in str(node.target))
        and bool(node.args)
        and isinstance(node.args[0], torch.fx.Node)
    )


def _mutable_buffer_get_attrs(
    graph_module: torch.fx.GraphModule,
) -> set[torch.fx.Node]:
    get_attrs: set[torch.fx.Node] = set()
    visited: set[torch.fx.Node] = set()
    queue = [
        cast(torch.fx.Node, node.args[0])
        for node in graph_module.graph.nodes
        if _is_mutation_node(node)
    ]

    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)

        if node.op == "get_attr" and isinstance(node.target, str):
            get_attrs.add(node)
            continue

        queue.extend(
            argument
            for argument in pytree.tree_leaves((node.args, node.kwargs))
            if isinstance(argument, torch.fx.Node)
        )

    return get_attrs


def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    if (
        left.shape != right.shape
        or left.stride() != right.stride()
        or left.storage_offset() != right.storage_offset()
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    if left.device.type == "meta":
        return left is right
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def collect_mutable_buffer_infos(
    graph_module: torch.fx.GraphModule,
) -> MutableBuffers:
    """Return mutable buffer names and their current tensor views."""
    named_buffers = dict(graph_module.named_buffers(remove_duplicate=False))
    if not named_buffers:
        return {}

    mutable_targets = {
        cast(str, node.target) for node in _mutable_buffer_get_attrs(graph_module)
    }
    return {
        target: named_buffers[target]
        for target in sorted(mutable_targets)
        if target in named_buffers
    }


def _register_mutable_buffer_targets(
    graph_module: torch.fx.GraphModule,
    mutable_buffers: MutableBuffers,
) -> None:
    registered_buffers = dict(graph_module.named_buffers(remove_duplicate=False))
    for target, value in mutable_buffers.items():
        if target in registered_buffers:
            continue
        module_path, _, buffer_name = target.rpartition(".")
        if module_path:
            try:
                owning_module = graph_module.get_submodule(module_path)
            except AttributeError:
                if not graph_module.add_submodule(module_path, torch.nn.Module()):
                    raise RuntimeError(
                        f"Could not restore mutable buffer module: {module_path}"
                    )
                owning_module = graph_module.get_submodule(module_path)
        else:
            owning_module = graph_module
        owning_module.register_buffer(buffer_name, value.detach().clone())


def _remove_unused_buffer(
    graph_module: torch.fx.GraphModule, target: str, referenced_targets: set[str]
) -> None:
    if target in referenced_targets:
        return
    module_path, _, buffer_name = target.rpartition(".")
    owning_module = (
        graph_module.get_submodule(module_path) if module_path else graph_module
    )
    if buffer_name in owning_module._buffers:
        delattr(owning_module, buffer_name)


def _merge_mutable_buffer_get_attrs(
    graph_module: torch.fx.GraphModule, mutable_targets: set[str]
) -> bool:
    modified = False
    canonical_get_attrs: dict[str, torch.fx.Node] = {}
    for node in list(graph_module.graph.nodes):
        if (
            node.op != "get_attr"
            or not isinstance(node.target, str)
            or node.target not in mutable_targets
        ):
            continue

        if node.target not in canonical_get_attrs:
            canonical_get_attrs[node.target] = node
            continue

        node.replace_all_uses_with(canonical_get_attrs[node.target])
        graph_module.graph.erase_node(node)
        modified = True

    return modified


def _mutable_buffer_target_for_node(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
    mutable_buffers: MutableBuffers,
) -> str | None:
    if node.target in mutable_buffers:
        return cast(str, node.target)

    value = _get_attr_value(graph_module, cast(str, node.target))
    if not isinstance(value, torch.Tensor):
        return None
    aliases = [
        target
        for target, mutable_buffer in mutable_buffers.items()
        if _same_tensor_view(value, mutable_buffer)
    ]
    return aliases[0] if len(aliases) == 1 else None


def _retarget_mutable_buffer_get_attrs(
    graph_module: torch.fx.GraphModule,
    mutable_buffers: MutableBuffers,
) -> tuple[set[str], set[str], bool]:
    restored_targets: set[str] = set()
    replaced_targets: set[str] = set()
    modified = False

    for node in graph_module.graph.nodes:
        if node.op != "get_attr" or not isinstance(node.target, str):
            continue

        target = _mutable_buffer_target_for_node(graph_module, node, mutable_buffers)
        if target is None:
            continue
        if target not in mutable_buffers:
            raise RuntimeError(f"Unknown mutable buffer target: {target}")
        restored_targets.add(target)
        if node.target != target:
            replaced_targets.add(node.target)
            node.target = target
            modified = True

    return restored_targets, replaced_targets, modified


def restore_mutable_buffer_targets(
    graph_module: torch.fx.GraphModule, mutable_buffers: MutableBuffers
) -> torch.fx.GraphModule:
    if not mutable_buffers:
        return graph_module

    _register_mutable_buffer_targets(graph_module, mutable_buffers)
    expected_targets = set(mutable_buffers)
    restored_targets, replaced_targets, modified = _retarget_mutable_buffer_get_attrs(
        graph_module, mutable_buffers
    )

    missing_targets = expected_targets - restored_targets
    if missing_targets:
        raise RuntimeError(
            f"Could not restore mutable buffers: {sorted(missing_targets)}"
        )

    modified = (
        _merge_mutable_buffer_get_attrs(graph_module, expected_targets) or modified
    )
    referenced_targets = {
        node.target
        for node in graph_module.graph.nodes
        if node.op == "get_attr" and isinstance(node.target, str)
    }
    for target in replaced_targets:
        _remove_unused_buffer(graph_module, target, referenced_targets)

    if modified:
        graph_module.graph.lint()
        graph_module.recompile()

    return graph_module
