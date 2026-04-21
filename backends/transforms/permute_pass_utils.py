# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Shared utilities and base classes for permute optimization passes.

These were originally in executorch.backends.cadence.aot and are used by
both the Cadence and Arm backends.
"""

from abc import abstractmethod
from collections import deque
from typing import cast, List, Optional, Type, TypeVar, Union

import torch
import torch.fx
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import Node
from torch.fx.node import Argument

T = TypeVar("T")


def get_edge_overload_packet(edge_op: EdgeOpOverload) -> EdgeOpOverloadPacket:
    edge_op_namespace, edge_op_name = (
        edge_op.namespace,
        edge_op._schema.name.split("::")[1],
    )
    edge_op_overload_packet = getattr(
        getattr(exir_ops.edge, edge_op_namespace), edge_op_name
    )
    return edge_op_overload_packet


def get_shape(
    graph_module: torch.fx.GraphModule, node: torch.fx.Node
) -> Union[torch.Size, None]:
    """Return the shape of the tensor corresponding to node."""
    try:
        if isinstance(node, (float, int, bool)):
            return torch.Size([1])
        fake_tensor = node.meta.get("val")
        if fake_tensor is not None:
            return fake_tensor.shape
        if node.op == "get_attr":
            attr_node = getattr(graph_module, node.target)
            return attr_node.shape
        return None
    except RuntimeError:
        return None


def get_transposed_dims(
    node: torch.fx.Node, dims: Optional[List[int]] = None
) -> List[int]:
    """Applies the transposition as given by node onto the dimensions given in input."""
    assert node.target == exir_ops.edge.aten.transpose_copy.int
    assert dims is not None
    dim_len = len(dims)
    transpose_dims0 = node.args[1]
    transpose_dims1 = node.args[2]
    assert isinstance(transpose_dims0, int)
    assert isinstance(transpose_dims1, int)
    dim0 = transpose_dims0 if transpose_dims0 >= 0 else transpose_dims0 + dim_len
    dim1 = transpose_dims1 if transpose_dims1 >= 0 else transpose_dims1 + dim_len
    new_dims = list(dims)
    new_dims[dim0], new_dims[dim1] = dims[dim1], dims[dim0]
    return new_dims


def get_permuted_dims(node: torch.fx.Node, dims: List[int]) -> List[int]:
    """Applies the permutation as given by node onto the dimensions given in input."""
    assert node.target == exir_ops.edge.aten.permute_copy.default
    # pyre-fixme[6]: This combined typecheck isn't supported yet.
    permute_dims: List[int] = list(node.args[1])
    assert all(isinstance(x, int) for x in permute_dims)
    return [dims[x] for x in permute_dims]


def get_arg(
    node: torch.fx.Node,
    kwarg_name: str,
    expected_type: Type[T] = Argument,
) -> T:
    """Get the arg with kwarg_name of the node."""
    if kwarg_name in node.kwargs:
        value = node.kwargs[kwarg_name]
    else:
        normalized_args = node.normalized_arguments(
            node.graph.owning_module, normalize_to_only_use_kwargs=True
        )
        if not normalized_args:
            raise RuntimeError(
                f"get_arg: Node {node} does not support normalization of arguments"
            )
        value = normalized_args.kwargs[kwarg_name]

    if expected_type is not Argument:
        try:
            type_ok = isinstance(value, expected_type)
        except TypeError:
            # Subscripted generics (e.g. List[int]) don't support isinstance.
            # Fall through — caller is responsible for correctness.
            type_ok = True
        if not type_ok:
            raise TypeError(
                f"get_arg: expected {expected_type} for '{kwarg_name}', got {type(value)}"
            )
    return value  # type: ignore[return-value]


def set_arg(
    node: torch.fx.Node, kwarg_name: str, value: torch.fx.node.Argument
) -> None:
    """Set the node's arg with its name to the given value."""
    if kwarg_name in node.kwargs:
        node.update_kwarg(kwarg_name, value)
        return

    normalized_args = node.normalized_arguments(
        node.graph.owning_module, normalize_to_only_use_kwargs=True
    )
    if not normalized_args:
        raise RuntimeError(
            f"set_arg: Node {node} does not support normalization of arguments"
        )

    kwargs = normalized_args.kwargs
    if kwarg_name not in kwargs:
        raise ValueError(f"set_arg: invalid arg name {kwarg_name} for node {node} used")

    idx = list(kwargs.keys()).index(kwarg_name)
    if idx < len(node.args):
        node.update_arg(idx, value)
    else:
        node.update_kwarg(kwarg_name, value)


class HierarchicalInplacePassInterface(ExportPass):
    """A base class for passes that apply in-place modification to the graph module and its submodules."""

    @abstractmethod
    def _apply_flat_inplace(self, graph_module) -> bool:
        raise NotImplementedError("`_apply_flat_inplace` must be implemented")

    def _apply_hierarchical_inplace(self, graph_module: torch.fx.GraphModule) -> bool:
        modified: bool = False
        for module in filter(
            lambda m: isinstance(m, torch.fx.GraphModule), graph_module.modules()
        ):
            modified |= self._apply_flat_inplace(module)
        return modified

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = self._apply_hierarchical_inplace(graph_module)
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            return super().call(graph_module)
        return PassResult(graph_module, False)


class RemoveOrReplacePassInterface(HierarchicalInplacePassInterface):
    @property
    @abstractmethod
    def targets(self) -> list[EdgeOpOverload]:
        raise NotImplementedError("`targets` must be implemented")

    @abstractmethod
    def maybe_remove_or_replace(self, node: Node) -> bool:
        raise NotImplementedError("`maybe_remove_or_replace` must be implemented")

    def _apply_flat_inplace(self, graph_module: torch.fx.GraphModule) -> bool:
        changed = False
        for target in self.targets:
            for node in graph_module.graph.find_nodes(
                op="call_function", target=target
            ):
                if len(node.users) == 0:
                    continue
                changed |= self.maybe_remove_or_replace(node)
        return changed


class FuseOpPairsAcrossBranchesPass(ExportPass):
    """Base class for passes that fuse op pairs across branches."""

    def check_ok_to_fuse(
        self,
        producer: torch.fx.Node,
        consumers: list[torch.fx.Node],
    ) -> bool:
        return True

    def can_fuse_for_chain(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        consumer_op_packets: set[EdgeOpOverloadPacket],
    ) -> bool:
        if (
            isinstance(consumer.target, EdgeOpOverload)
            and get_edge_overload_packet(consumer.target) in consumer_op_packets
        ):
            return True
        return False

    def get_fuse_candidates(
        self,
        producer: torch.fx.Node,
        consumer_op_packets: set[EdgeOpOverloadPacket],
        bypass_ops: set[EdgeOpOverload],
    ) -> list[torch.fx.Node]:
        users = deque(producer.users.keys())
        removal_candidates = []
        while users:
            user = users.popleft()
            if user.target in bypass_ops:
                users.extend(list(user.users.keys()))
            elif self.can_fuse_for_chain(producer, user, consumer_op_packets):
                removal_candidates.append(user)
            else:
                removal_candidates.clear()
                break
        return removal_candidates

    def find_and_fuse(
        self,
        graph_module: torch.fx.GraphModule,
        producer_op_packets: set[EdgeOpOverloadPacket],
        consumer_op_packets: set[EdgeOpOverloadPacket],
        bypass_ops: set[EdgeOpOverload],
    ) -> bool:
        modified = False
        for node in graph_module.graph.nodes:
            if not (
                isinstance(node.target, EdgeOpOverload)
                and get_edge_overload_packet(node.target) in producer_op_packets
            ):
                continue
            removal_candidates = self.get_fuse_candidates(
                node, consumer_op_packets, bypass_ops
            )
            if len(removal_candidates) == 0:
                continue
            if not self.check_ok_to_fuse(node, removal_candidates):
                continue
            self.fuse(node, removal_candidates, graph_module)
            modified = True
        if modified:
            graph_module.recompile()
        return modified

    def get_fused_node(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        graph_module: torch.fx.GraphModule,
    ) -> torch.fx.Node:
        return consumer

    def fuse(
        self,
        node: torch.fx.Node,
        removal_candidates: list[torch.fx.Node],
        graph_module: torch.fx.GraphModule,
    ) -> None:
        node.replace_all_uses_with(cast(torch.fx.Node, node.args[0]))
        graph_module.graph.erase_node(node)
        for rnode in removal_candidates:
            rnode.replace_all_uses_with(self.get_fused_node(node, rnode, graph_module))
            graph_module.graph.erase_node(rnode)
