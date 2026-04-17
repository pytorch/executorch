# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Callable,
    Hashable,
    List,
    Optional,
    override,
    Set,
    Type,
    TypeVar,
    Union,
)

import torch
from beartype.door import die_if_unbearable
from executorch.backends.cadence.aot.utils import get_edge_overload_packet
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import ExportPass, PassBase, PassResult
from torch._ops import OpOverloadPacket
from torch.fx import Node
from torch.fx.node import Argument
from torch.utils import _pytree as pytree

T = TypeVar("T")


# Is an overlap in tensor lifetime and storage allowed at the current opt level?
# We allow overlap at opt level >= 2.
def allow_lifetime_and_storage_overlap(opt_level: int) -> bool:
    return opt_level >= 2


# A dataclass that bundles feature flags for edge passes.
# When adding a new flag, add a matching bool field to both this class and
# CadencePassAttribute; the pass filter will pick it up automatically.
@dataclass(frozen=True)
class EdgePassesConfig:
    use_im2row_transform: bool = False


# A dataclass that stores the attributes of an ExportPass.
@dataclass(frozen=True)
class CadencePassAttribute:
    opt_level: Optional[int] = None
    debug_pass: bool = False
    use_im2row_transform: bool = False


# A dictionary that maps an ExportPass to its attributes.
ALL_CADENCE_PASSES: dict[Type[PassBase], CadencePassAttribute] = {}


def get_cadence_pass_attribute(p: Type[PassBase]) -> Optional[CadencePassAttribute]:
    return ALL_CADENCE_PASSES.get(p, None)


# A decorator that registers a pass.
def register_cadence_pass(
    pass_attribute: CadencePassAttribute,
) -> Callable[[Type[PassBase]], Type[PassBase]]:
    def wrapper(cls: Type[PassBase]) -> Type[PassBase]:
        ALL_CADENCE_PASSES[cls] = pass_attribute
        return cls

    return wrapper


def get_all_available_cadence_passes() -> Set[Type[PassBase]]:
    return set(ALL_CADENCE_PASSES.keys())


def _check_feature_flags(
    pass_attribute: CadencePassAttribute,
    config: EdgePassesConfig,
) -> bool:
    """Check all feature flags: a pass is included only if every feature it
    requires is enabled in the config. Iterates over EdgePassesConfig fields
    so new flags are handled automatically."""
    for field in dataclasses.fields(EdgePassesConfig):
        if getattr(pass_attribute, field.name, False) and not getattr(
            config, field.name
        ):
            return False
    return True


# Create a new filter to filter out relevant passes from all passes.
def create_cadence_pass_filter(
    opt_level: int,
    debug: bool = False,
    edge_passes_config: Optional[EdgePassesConfig] = None,
) -> Callable[[Type[PassBase]], bool]:
    if edge_passes_config is None:
        edge_passes_config = EdgePassesConfig()

    def _filter(p: Type[PassBase]) -> bool:
        pass_attribute = get_cadence_pass_attribute(p)
        return (
            pass_attribute is not None
            and pass_attribute.opt_level is not None
            and pass_attribute.opt_level <= opt_level
            and (not pass_attribute.debug_pass or debug)
            and _check_feature_flags(pass_attribute, edge_passes_config)
        )

    return _filter


# Return the overload packet for the edge or torch op.
def get_overload_packet(
    op: Union[Callable[..., str], str],
) -> Union[OpOverloadPacket, EdgeOpOverloadPacket, None]:
    return (
        get_edge_overload_packet(op)
        if isinstance(op, EdgeOpOverload)
        else getattr(op, "overloadpacket", None)
    )


# Get the list of node names in a graph module (only for "call_function" ops and
# EdgeOpOverload targets). This should be used only after to_edge is called.
def get_node_names_list_from_gm(
    graph_module: torch.fx.GraphModule,
) -> list[torch.fx.Node]:
    graph_nodes = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if not isinstance(node.target, EdgeOpOverload):
            continue
        graph_nodes.append(node.name)
    return graph_nodes


def count_node(graph_module: torch.fx.GraphModule, target: torch.fx.node.Target) -> int:
    """Count the number of nodes with target `target` in the graph."""
    total = 0
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target == target:
            total += 1
    return total


def op_counts_match(
    graph_module: torch.fx.GraphModule,
    expected_op_counts: dict[EdgeOpOverload, int],
) -> bool:
    for op, count in expected_op_counts.items():
        if count_node(graph_module, op) != count:
            return False
    return True


# Testing utils
# Return the compute/function nodes in the graph
def get_compute_nodes_in_gm(graph_module: torch.fx.GraphModule) -> List[torch.fx.Node]:
    nodes = []
    for x in graph_module.graph.nodes:
        if x.op == "call_function":
            if isinstance(x.target, torch._ops.OpOverload):
                nodes.append(x.target.overloadpacket)
            elif isinstance(x.target, EdgeOpOverload):
                nodes.append(get_edge_overload_packet(x.target))
    return nodes


# Return true if there is no edge from a node with target pred_target to a
# node with target succ_target in the graph.
def nodes_not_connected_in_gm(
    graph_module: torch.fx.GraphModule,
    pred_target: torch.fx.Node,
    succ_target: torch.fx.Node,
) -> bool:
    for node in graph_module.graph.nodes:
        if node.target != pred_target:
            continue
        for user in node.users:
            if user.target == succ_target:
                return False
    return True


# Returns the position of the first entry of a node of a given kind in the graph.
def get_node_pos(
    graph_module: torch.fx.GraphModule,
    target: torch.fx.Node,
) -> int:
    pos = 0
    for node in graph_module.graph.nodes:
        if node.target == target:
            return pos
        pos += 1
    return -1


# Returns true if there is no instance of a node with target succ_target
# positioned immediately after a node with target pred_target in the graph
def nodes_not_adjacent_in_gm(
    graph_module: torch.fx.GraphModule,
    pred_target: torch.fx.Node,
    succ_target: torch.fx.Node,
) -> bool:
    for node in graph_module.graph.nodes:
        if node.target != pred_target:
            continue
        if node.next.target == succ_target:
            return False
    return True


def get_arg(
    node: torch.fx.Node,
    kwarg_name: str,
    expected_type: Type[T] = Argument,
) -> T:
    """
    Get the arg with arg_name of the node, returns default value if not set.

    Args:
        node: The FX node to extract the argument from
        kwarg_name: The name of the argument to extract
        expected_type: Optional type to validate and cast the argument to.
                      If provided, asserts the argument is an instance of this type.

    Returns:
        The argument value, optionally type-checked and cast to expected_type

    Example:
        # Get a node argument with type checking
        conv_weight_node = get_arg(node, "weight", torch.fx.Node)

        # Get a float argument with type checking
        eps = get_arg(node, "eps", float)

        # Get an argument without type checking (returns Argument)
        value = get_arg(node, "some_arg")
    """
    # Try to get the arg from kwargs first since this is faster
    if kwarg_name in node.kwargs:
        value = node.kwargs[kwarg_name]
    else:
        # If it's not found in kwargs, try to normalize the args
        normalized_args = node.normalized_arguments(
            node.graph.owning_module, normalize_to_only_use_kwargs=True
        )
        if not normalized_args:
            raise RuntimeError(
                f"get_arg: Node {node} does not support normalization of arguments"
            )
        value = normalized_args.kwargs[kwarg_name]

    # Validate type using beartype's runtime type checker when a specific
    # type is requested (not the default Argument type alias, which contains
    # recursive forward references that beartype cannot resolve).
    if expected_type is not Argument:
        die_if_unbearable(value, expected_type)
    return value  # type: ignore[return-value]


def set_arg(
    node: torch.fx.Node, kwarg_name: str, value: torch.fx.node.Argument
) -> None:
    """
    Set the node's arg with its name to the given value.
    """
    # Try to set the arg if it is present in kwargs first since this is faster
    if kwarg_name in node.kwargs:
        node.update_kwarg(kwarg_name, value)
        return

    # If it's not found in kwargs, try to normalize the args and set the arg
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


def none_throws(x: Optional[PassResult]) -> PassResult:
    assert x is not None
    return x


class HierarchicalInplacePassInterface(ExportPass):
    """A base class for passes that apply in-place modification to the graph module and its submodules.
    Also calls ExportPass.call() in case the graph module is modified to ensure all nodes have valid `meta['val']`.
    """

    @abstractmethod
    def _apply_flat_inplace(self, graph_module) -> bool:
        """Apply in-place modification to the graph module."""
        raise NotImplementedError("`_apply_flat_inplace` must be implemented")

    def _apply_hierarchical_inplace(self, graph_module: torch.fx.GraphModule) -> bool:
        """Apply in-place modification recursively to the graph module and its submodules."""

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
        """
        The list of targets to potentially remove or replace.
        """
        raise NotImplementedError("`targets` must be implemented")

    @abstractmethod
    def maybe_remove_or_replace(self, node: Node) -> bool:
        """
        If the node should be removed/replaced, removes/replaces from the graph. Returns
        True if the graph was modified, else False.
        """
        raise NotImplementedError("`maybe_remove_or_replace` must be implemented")

    @override
    def _apply_flat_inplace(self, graph_module: torch.fx.GraphModule) -> bool:
        changed = False
        for target in self.targets:
            for node in graph_module.graph.find_nodes(
                op="call_function", target=target
            ):
                if len(node.users) == 0:
                    # It is possible that maybe_remove_or_replace would have removed
                    # this target by starting from a different target. In this case,
                    # we should ignore it. If it wasn't erased, it will be handled
                    # in eliminate_dead_code.
                    continue
                changed |= self.maybe_remove_or_replace(node)
        return changed


class SwapOnCostModelPassInterface(HierarchicalInplacePassInterface):
    """
    A base class for passes that reduce op count by moving wrapper operations
    (e.g., dequant/quant, permute) from the majority side of a target op's
    inputs/outputs to the minority side.

    Given a target op with some inputs wrapped by ``input_to_swap`` and some
    outputs wrapped by ``output_to_swap``, this pass checks whether it is
    cheaper to instead wrap the *other* inputs/outputs. The swap is performed
    only when:

    1. All input wrappers share the same hash (via ``input_hash``).
    2. All output wrappers share the same hash (via ``output_hash``).
    3. The input and output hashes are compatible (via ``hashes_are_compatible``).
    4. The cost after swapping is strictly less than before (via ``subgraph_cost``).

    Subclasses must implement:
      - ``targets``: the ops to scan (e.g., cat, add, mul).
      - ``input_to_swap`` / ``output_to_swap``: the wrapper op targets.
      - ``input_hash`` / ``output_hash``: extract a hashable identity from a
        wrapper node for equality checking.
      - ``hashes_are_compatible``: define the relationship between input and
        output hashes (equality for quant/dequant, inverse for permute).
      - ``subgraph_cost``: total cost of the target node and its immediate
        wrapper neighbors, measured on the real graph.

    Example 1 — Dequant/quant around cat::

        # Before: 3 dequants on inputs, 1 fp32 input, 2 quant outputs, 1 fp32 output.
        #   dequant(A, s, zp) ──┐
        #   dequant(B, s, zp) ──┤              ┌── quant(out1, s, zp)
        #   dequant(C, s, zp) ──┼── cat(fp32) ─┼── quant(out2, s, zp)
        #               D(fp32) ┘              └── consumer(fp32)
        #   Cost: 5 wrapper ops (3 dequants + 2 quants)
        #
        # After: A, B, C feed cat directly (already quantized). D gets a quant.
        #   The cat now runs in quantized domain. The fp32 consumer gets a dequant.
        #                A ──┐
        #                B ──┤            ┌── out1 (quantized, no wrapper needed)
        #                C ──┼── cat(q) ──┼── out2 (quantized, no wrapper needed)
        #   quant(D, s, zp) ─┘            └── dequant(out3, s, zp) ── consumer
        #   Cost: 2 wrapper ops (1 quant + 1 dequant)
        #
        # Delta: 5 → 2, net savings of 3 ops.
        #
        # input_to_swap = dequantize_per_tensor
        # output_to_swap = quantize_per_tensor
        # input_hash = (scale, zero_point, quant_min, quant_max, dtype)
        # output_hash = (scale, zero_point, quant_min, quant_max, dtype)
        # hashes_are_compatible: input_hash == output_hash

    Example 2 — Permutes around a binary op::

        # Before: permute(A) and B feed add, output is inverse-permuted.
        #   permute(A, [0,3,1,2]) ──┐
        #                           ├── add ── permute(out, [0,2,3,1])
        #                      B ───┘
        #   Cost: 2 wrapper ops
        #
        # After: A and inverse-permute(B) feed add, no output permute.
        #              A ──────────────────┐
        #                                  ├── add ── out
        #   permute(B, [0,2,3,1]) ─────────┘
        #   Cost: 1 wrapper op
        #
        # input_to_swap = permute_copy
        # output_to_swap = permute_copy
        # input_hash = tuple(dims)
        # output_hash = tuple(dims)
        # hashes_are_compatible: applying input_perm then output_perm = identity

    Example 3 — Chained elimination across multiple ops::

        When targets share edges, a single pass run can cascade reductions.
        The pass visits each target in graph order; wrappers injected by
        earlier swaps become inputs/outputs for later targets.

        # Before (5 permutes total):
        #   permute(A, dims) ──┐
        #                      ├── add_1 ── permute(out1, inv_dims) ── consumer_1
        #   permute(B, dims) ──┤
        #                      └── (fp32 edge to add_2)
        #                                     │
        #   permute(C, dims) ─────────────┐   │
        #                                 ├── add_2 ── permute(out2, inv_dims)
        #                     (from add_1)┘
        #
        # After pass visits add_1 (3 wrappers → 1):
        #   Removes 2 input permutes + 1 output permute, injects 1 inverse
        #   permute on the edge flowing to add_2.
        #              A ──┐
        #                  ├── add_1 ── consumer_1
        #              B ──┤
        #                  └── permute(edge, inv_dims) ── add_2
        #   permute(C, dims) ─────────────────────────┘      │
        #                                                     └── permute(out2, inv_dims)
        #
        # After pass visits add_2 (3 wrappers → 0):
        #   add_2 now has 2 permuted inputs + 1 permuted output = 3, 0 unwrapped.
        #   All eliminated.
        #              A ──┐
        #                  ├── add_1 ── consumer_1
        #              B ──┤
        #                  └── add_2 ── out2
        #              C ──┘
        #
        # Result: all 5 original permutes eliminated in a single pass run.
    """

    @property
    @abstractmethod
    def targets(self) -> list[EdgeOpOverload]:
        """
        The list of targets that we will potentially swap inputs and outputs.
        """
        raise NotImplementedError("`targets` must be implemented")

    @property
    @abstractmethod
    def input_to_swap(self) -> EdgeOpOverload:
        """
        The wrapper op target to match on inputs (e.g., dequantize_per_tensor,
        permute_copy). Inputs to the target op whose target matches this will
        be candidates for removal during the swap.
        """
        raise NotImplementedError("You must specify the input we are trying to swap")

    @property
    @abstractmethod
    def output_to_swap(self) -> EdgeOpOverload:
        """
        The wrapper op target to match on outputs (e.g., quantize_per_tensor,
        permute_copy). Users of the target op whose target matches this will
        be candidates for removal during the swap.
        """
        raise NotImplementedError("You must specify the output we are trying to swap")

    @abstractmethod
    def input_wrapper_cost(self, data_node: Node) -> int:
        """
        Return the cost of applying the input wrapper op (``input_to_swap``)
        on ``data_node``'s tensor.

        ``data_node`` is the tensor that flows *through* the wrapper — i.e.,
        the wrapper's first positional argument.  This is the same kind of
        node regardless of whether the wrapper already exists (use
        ``wrapper.args[0]``) or is hypothetical (use the unwrapped node
        directly).
        """
        ...

    @abstractmethod
    def output_wrapper_cost(self, data_node: Node) -> int:
        """
        Return the cost of applying the output wrapper op (``output_to_swap``)
        on ``data_node``'s tensor.

        See ``input_wrapper_cost`` for the semantics of ``data_node``.
        """
        ...

    @abstractmethod
    def subgraph_cost(self, target_node: Node) -> int:
        """Return the total cost of the target node and its immediate wrapper
        neighbors.  Called on the *real* graph — both before and after a
        tentative swap — so no prediction is needed.
        """
        ...

    @abstractmethod
    def input_hash(self, node: Node) -> Hashable:
        """
        Extract a hashable identity from an input wrapper node. All input
        wrappers must produce the same hash for the swap to be valid.
        E.g., for dequant: (scale, zero_point, quant_min, quant_max, dtype).
        E.g., for permute: tuple(dims).
        """
        ...

    @abstractmethod
    def output_hash(self, node: Node) -> Hashable:
        """
        Extract a hashable identity from an output wrapper node. All output
        wrappers must produce the same hash for the swap to be valid.
        """
        ...

    @abstractmethod
    def hashes_are_compatible(
        self, input_hash: Hashable, output_hash: Hashable
    ) -> bool:
        """
        Check whether the input and output wrapper hashes are compatible,
        meaning the swap is semantically legal.
        For quant/dequant: hashes must be equal (same scale, zp, dtype).
        For permute: output dims must be the inverse permutation of input dims.
        """
        ...

    @abstractmethod
    def create_inverse_wrapper_args(
        self,
        template: Node,
    ) -> tuple[EdgeOpOverload, tuple, dict]:
        """
        Given a wrapper node from one side, return (target, args_tail, kwargs)
        for the inverse wrapper on the other side. args_tail excludes the data
        input (first arg), which is supplied by the caller.

        For quant/dequant: swap the op target, keep the same params.
        For permute: same op target, compute inverse dims.
        """
        ...

    @property
    def lossy_inverse(self) -> bool:
        """Whether the inverse wrapper introduces numerical loss.

        When True, swaps that would require injecting inverse wrappers on
        unwrapped outputs are skipped, because the inverse round-trip
        (e.g., quant → dequant) degrades precision for those consumers.
        Permute inverses are lossless, so this defaults to False.
        """
        return False

    def _partition_neighbors(
        self,
        neighbors: list[Node],
        wrapper_target: EdgeOpOverload,
        hash_fn: Callable[[Node], Hashable],
    ) -> tuple[list[Node], list[Node], Hashable | None] | None:
        """Partition neighbor nodes into (before_swap, after_swap, hash_value).

        Returns None if the partition is invalid (wrapper with multiple users
        or inconsistent hashes across wrappers).
        """
        hash_value: Hashable | None = None
        before_swap: list[Node] = []
        after_swap: list[Node] = []
        for neighbor in neighbors:
            if neighbor.target == wrapper_target:
                if len(neighbor.users) != 1:
                    return None
                h = hash_fn(neighbor)
                if hash_value is None:
                    hash_value = h
                elif hash_value != h:
                    return None
                before_swap.append(neighbor)
            else:
                after_swap.append(neighbor)
        return before_swap, after_swap, hash_value

    def _resolve_inverse_params(
        self,
        opposite_before_swap: list[Node],
        same_side_before_swap: list[Node],
    ) -> tuple[EdgeOpOverload, tuple, dict, dict]:
        """Determine (target, args_tail, kwargs, meta) for the inverse wrapper.

        Uses the opposite side's existing wrappers if available, otherwise
        falls back to create_inverse_wrapper_args on the same side.
        """
        if opposite_before_swap:
            template = opposite_before_swap[0]
            return (
                template.target,
                template.args[1:],
                dict(template.kwargs),
                template.meta.copy(),
            )
        template = same_side_before_swap[0]
        inv_target, inv_args_tail, inv_kwargs = self.create_inverse_wrapper_args(
            template
        )
        return inv_target, inv_args_tail, inv_kwargs, template.meta.copy()

    def _inject_inverse_wrappers(
        self,
        graph: torch.fx.Graph,
        unwrapped_nodes: list[Node],
        target_node: Node,
        inv_target: EdgeOpOverload,
        inv_args_tail: tuple,
        inv_kwargs: dict,
        inv_meta: dict,
        on_inputs: bool,
    ) -> list[Node]:
        """Inject inverse wrapper nodes between target_node and unwrapped_nodes.

        Returns the list of newly created wrapper nodes so they can be removed
        if the swap is rolled back.
        """
        created: list[Node] = []
        for unwrapped in list(unwrapped_nodes):
            if on_inputs:
                source, dest = unwrapped, target_node
            else:
                source, dest = target_node, unwrapped
            with graph.inserting_before(dest):
                new_wrapper = graph.call_function(
                    inv_target,
                    args=(source,) + inv_args_tail,
                    kwargs=inv_kwargs,
                )
                new_wrapper.meta = inv_meta
            dest.replace_input_with(source, new_wrapper)
            created.append(new_wrapper)
        return created

    def _perform_swap(
        self,
        graph: torch.fx.Graph,
        node: Node,
        input_before: list[Node],
        input_after: list[Node],
        output_before: list[Node],
        output_after: list[Node],
    ) -> list[Node]:
        """Execute the swap and return all injected inverse wrapper nodes."""
        injected: list[Node] = []

        # Inject inverse wrappers on unwrapped inputs
        if input_after:
            inv_target, inv_args_tail, inv_kwargs, inv_meta = (
                self._resolve_inverse_params(output_before, input_before)
            )
            injected += self._inject_inverse_wrappers(
                graph, input_after, node,
                inv_target, inv_args_tail, inv_kwargs, inv_meta,
                on_inputs=True,
            )

        # Inject inverse wrappers on unwrapped outputs
        if output_after:
            inv_target, inv_args_tail, inv_kwargs, inv_meta = (
                self._resolve_inverse_params(input_before, output_before)
            )
            injected += self._inject_inverse_wrappers(
                graph, output_after, node,
                inv_target, inv_args_tail, inv_kwargs, inv_meta,
                on_inputs=False,
            )

        # Bypass existing input wrappers (e.g., remove dequants).
        for wrapper in input_before:
            node.replace_input_with(wrapper, wrapper.args[0])

        # Bypass existing output wrappers (e.g., remove quants).
        for wrapper in output_before:
            wrapper.replace_all_uses_with(node)

        graph.eliminate_dead_code()
        return injected

    def _undo_swap(
        self,
        node: Node,
        input_before: list[Node],
        output_before: list[Node],
        injected_nodes: list[Node],
    ) -> None:
        """Reverse a swap by restoring original wiring and erasing injected nodes."""
        graph = node.graph

        # Restore output wrappers: each was bypassed via replace_all_uses_with(node).
        # Re-point the wrapper's original consumers back to the wrapper.
        for wrapper in output_before:
            node.replace_all_uses_with(wrapper)
            # The wrapper itself should consume the target node, not itself.
            # replace_all_uses_with also redirected wrapper's own input edge,
            # so fix that back.
            wrapper.replace_input_with(wrapper, node)

        # Restore input wrappers: each was bypassed via
        # node.replace_input_with(wrapper, wrapper.args[0]).
        # Re-point the target node's input back to the wrapper.
        for wrapper in input_before:
            node.replace_input_with(wrapper.args[0], wrapper)

        # Remove injected inverse wrappers (reverse order for clean removal).
        for injected in reversed(injected_nodes):
            injected.replace_all_uses_with(injected.args[0])
            graph.erase_node(injected)

    def _apply_flat_inplace(self, graph_module: torch.fx.GraphModule) -> bool:
        changed = False
        for target in self.targets:
            for node in graph_module.graph.find_nodes(
                op="call_function", target=target
            ):
                # Partition inputs into wrapped (before_swap) and unwrapped (after_swap)
                input_result = self._partition_neighbors(
                    node.all_input_nodes,
                    self.input_to_swap,
                    self.input_hash,
                )
                if input_result is None:
                    continue
                input_before, input_after, input_hash_value = input_result

                output_result = self._partition_neighbors(
                    node.users,
                    self.output_to_swap,
                    self.output_hash,
                )
                if output_result is None:
                    continue
                output_before, output_after, output_hash_value = output_result

                # We need at least one side to have wrappers to perform a swap.
                if input_hash_value is None and output_hash_value is None:
                    continue

                # Check cross-compatibility between input and output hashes
                if (
                    input_hash_value is not None
                    and output_hash_value is not None
                    and not self.hashes_are_compatible(
                        input_hash_value, output_hash_value
                    )
                ):
                    continue

                # Skip if lossy inverse would be needed on a side with mixed wrappers.
                if self.lossy_inverse and (
                    (input_before and input_after) or (output_before and output_after)
                ):
                    continue

                # Measure cost on the current (pre-swap) graph.
                before_cost = self.subgraph_cost(node)

                # Perform the swap.
                graph = graph_module.graph
                original_meta_val = node.meta["val"].clone()
                injected = self._perform_swap(
                    graph, node,
                    input_before, input_after,
                    output_before, output_after,
                )

                # Re-derive target node metadata by running the op in fake
                # tensor mode with the swapped inputs.  This correctly updates
                # dtype, shape, strides, etc. without running the real op.
                fake_args = pytree.tree_map(
                    lambda x: x.meta["val"] if isinstance(x, Node) else x,
                    node.args,
                )
                fake_kwargs = pytree.tree_map(
                    lambda x: x.meta["val"] if isinstance(x, Node) else x,
                    node.kwargs,
                )
                node.meta["val"] = node.target(*fake_args, **fake_kwargs)

                # Measure cost on the swapped graph.
                after_cost = self.subgraph_cost(node)

                # If you're wondering why we don't check lte rather than lt since we already
                # swapped anyway, it's because if we run passes like this iteratively, we want
                # to make sure that we can check for convergence.
                if after_cost < before_cost:
                    changed = True
                else:
                    # Not favorable — undo everything, including metadata.
                    node.meta["val"] = original_meta_val
                    self._undo_swap(node, input_before, output_before, injected)

        return changed
