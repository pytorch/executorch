# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The tensor constants of an unlifted GraphModule.

``ExportedProgram.module()`` reads each lifted input through a ``get_attr`` node:
a parameter or buffer is registered on the module, and a lifted tensor constant
is a plain tensor attribute.
"""

import operator
import re
from collections.abc import Iterable
from typing import Optional

import torch
from executorch.exir.passes.constant_prop_pass import (
    _hop_written_operands,
    _nodes_written_in_place,
)
from torch import fx
from torch._export.passes.constant_folding import ConstantFolder
from torch._guards import detect_fake_mode
from torch.export.graph_signature import InputKind
from torch.utils._python_dispatch import _disable_current_modes


def _owner(graph_module: fx.GraphModule, target: str) -> tuple[torch.nn.Module, str]:
    path, _, name = target.rpartition(".")
    return graph_module.get_submodule(path), name


def get_constant(node: fx.Node) -> Optional[torch.Tensor]:
    """The tensor a ``get_attr`` node reads, parameters unwrapped, else None."""
    graph_module = node.graph.owning_module
    if node.op != "get_attr" or graph_module is None:
        return None
    value = operator.attrgetter(str(node.target))(graph_module)
    if not isinstance(value, torch.Tensor):
        return None
    return value.data if isinstance(value, torch.nn.Parameter) else value


def get_kind(node: fx.Node) -> Optional[InputKind]:
    """Whether a ``get_attr`` node reads a parameter, a buffer or a tensor
    constant; None for any other node."""
    graph_module = node.graph.owning_module
    if graph_module is None or get_constant(node) is None:
        return None
    owner, name = _owner(graph_module, str(node.target))
    if name in owner._parameters:
        return InputKind.PARAMETER
    if name in owner._buffers:
        return InputKind.BUFFER
    return InputKind.CONSTANT_TENSOR


def add_constant(
    graph_module: fx.GraphModule, name: str, tensor: torch.Tensor, kind: InputKind
) -> fx.Node:
    """Add ``tensor`` to ``graph_module`` as a new attribute of ``kind``, and
    return a ``get_attr`` node reading it, ahead of all but the placeholders.

    As in ``ExportedProgram.module()``, a parameter or buffer is registered and a
    tensor constant is a plain attribute.
    """
    base = re.sub(r"\W", "_", name)
    attr, suffix = base, 0
    while hasattr(graph_module, attr):
        suffix += 1
        attr = f"{base}_{suffix}"
    match kind:
        case InputKind.PARAMETER:
            graph_module.register_parameter(
                attr, torch.nn.Parameter(tensor, requires_grad=False)
            )
        case InputKind.BUFFER:
            graph_module.register_buffer(attr, tensor)
        case InputKind.CONSTANT_TENSOR:
            setattr(graph_module, attr, tensor)
        case _:
            raise AssertionError(f"add_constant cannot add a {kind} attribute")

    graph = graph_module.graph
    fake_mode = detect_fake_mode(
        [
            node.meta.get("val")
            for node in graph.nodes
            if node.op in ("placeholder", "get_attr")
        ]
    )
    assert fake_mode is not None, "add_constant needs a graph with fake values"
    first = next(node for node in graph.nodes if node.op != "placeholder")
    with graph.inserting_before(first):
        node = graph.create_node("get_attr", attr)
    node.meta["val"] = fake_mode.from_tensor(tensor, static_shapes=True)
    return node


class _Folder(ConstantFolder):
    """torch's ``ConstantFolder``, with the ``unknown`` nodes unknown as well as
    the placeholders.

    Unlike torch's, it folds constructors, and it leaves impure ops (mutation,
    random draws) unfolded, as ``constant_prop_pass`` does. It does not keep the
    dequantize ops torch's keeps, which only exist after quantization and need
    torchao's ops registered to check.
    """

    def __init__(
        self, graph_module: fx.GraphModule, unknown: Iterable[fx.Node]
    ) -> None:
        super().__init__(graph_module, skip_constructors=False)
        self.graph_module = graph_module
        self.unknown = list(unknown)

    def is_impure(self, node: fx.Node) -> bool:
        return False

    def run_node(self, node: fx.Node) -> object:
        if node.op == "call_function" and node.is_impure():
            return self.unknown_value
        return super().run_node(node)

    def run(self) -> object:
        placeholders = self.graph_module.graph.find_nodes(op="placeholder")
        env = {node: self.unknown_value for node in [*placeholders, *self.unknown]}
        return fx.Interpreter.run(self, initial_env=env)


def fold(
    graph_module: fx.GraphModule, unknown: Iterable[fx.Node]
) -> dict[fx.Node, torch.Tensor]:
    """The value of each constant tensor a non-constant node reads, where a node
    is constant if it depends on neither a placeholder nor an ``unknown`` node.

    A constant read only by other constant nodes is left out, as is any
    ``get_attr`` node.
    """
    with _disable_current_modes(), torch.no_grad():
        folder = _Folder(graph_module, unknown)
        folder.run()
    return folder.node_replacements


def _erase_dead_code(graph_module: fx.GraphModule) -> None:
    """Erase the nodes nothing reads, and the tensor attributes only they read.

    As in ``constant_prop_pass``, only the top-level graph is cleaned: a higher
    order op that writes its operands has no users, yet it has to stay.
    """
    graph = graph_module.graph
    dead_targets = set()
    for node in reversed(graph.nodes):
        if (
            node.op in ("placeholder", "output")
            or node.users
            or node.is_impure()
            or _hop_written_operands(graph_module, node)
        ):
            continue
        if node.op == "get_attr":
            dead_targets.add(str(node.target))
        graph.erase_node(node)

    live_targets = {str(node.target) for node in graph.find_nodes(op="get_attr")}
    for target in dead_targets - live_targets:
        owner, name = _owner(graph_module, target)
        if isinstance(getattr(owner, name, None), torch.Tensor):
            delattr(owner, name)


def fold_constants(graph_module: fx.GraphModule) -> None:
    """Fold each subgraph computed only from constants into a new tensor
    constant, as ``constant_prop_pass`` does on an ExportedProgram, then erase
    the nodes and attributes nothing reads anymore.

    Parameters, buffers and tensor constants all count as constants, but a
    buffer the graph writes in place, such as a KV cache, does not. A value the
    graph returns is left as is.
    """
    graph = graph_module.graph
    written = _nodes_written_in_place(graph_module)
    unknown = [
        node
        for node in graph.nodes
        if node in written or (node.op == "get_attr" and get_constant(node) is None)
    ]
    output = graph.output_node()
    for node, value in fold(graph_module, unknown).items():
        if output in node.users:
            continue
        # ExecuTorch doesn't support zero strides, and contiguous() keeps one on
        # a size-1 dim.
        if 0 in value.stride():
            value = value.clone(memory_format=torch.contiguous_format)
        constant = add_constant(
            graph_module,
            "_prop_tensor_constant",
            value.detach(),
            InputKind.CONSTANT_TENSOR,
        )
        constant.meta = {**node.meta, "val": constant.meta["val"]}
        node.replace_all_uses_with(constant)

    _erase_dead_code(graph_module)
    graph_module.recompile()
