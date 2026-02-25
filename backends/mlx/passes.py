# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph transformation passes for the MLX backend.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from executorch.backends.mlx.pattern_utils import (
    extract_lifted_tensor_constant,
    match_target,
    OpStep,
    PatternMatch,
    walk_back,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


def get_default_passes() -> List[ExportPass]:
    """
    Returns a list of passes that are enabled by default for the MLX backend.
    """
    return [
        FuseRMSNormPass(),
        CanonicalizePermutePass(),
        CollapseViewCopyPass(),
        CollapsePermutePass(),
        CollapseDtypeConversionPass(),
        RemoveNoOpsPass(),
        CSEPass(),
    ]


@dataclass
class RMSNormMatch(PatternMatch):
    """
    Matched RMSNorm pattern.

    HuggingFace Llama's RMSNorm decomposes into:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return weight * hidden_states.to(input_dtype)

    Graph pattern:
        _to_copy (to f32) [optional]
        pow(x, 2)
        mean_dim(pow_out, [-1], keepdim=True)
        add(mean_out, eps_tensor)
        rsqrt(add_out)
        mul(to_copy_out, rsqrt_out)
        _to_copy (back to original dtype) [optional]
        mul(weight, to_copy_out)
    """

    input_node: Node = None  # type: ignore[assignment]
    weight_node: Node = None  # type: ignore[assignment]
    eps: float = 0.0

    @classmethod
    def maybe_create(cls, head: Node, **context) -> Optional["RMSNormMatch"]:
        """Match RMSNorm pattern starting from final mul(weight, normalized)."""
        # Head must be mul
        if not match_target(head, torch.ops.aten.mul.Tensor):
            return None

        if len(head.args) < 2:
            return None

        # Try both orderings: mul(weight, normalized) or mul(normalized, weight)
        for weight_idx, norm_idx in [(0, 1), (1, 0)]:
            weight_node = head.args[weight_idx]
            norm_node = head.args[norm_idx]

            if not isinstance(norm_node, Node):
                continue

            # Match entire chain with single walk_back:
            #   [_to_copy] -> mul(input, rsqrt) -> rsqrt -> add -> mean -> pow -> [_to_copy]
            # The mul follows arg_index=1 to get rsqrt (not input)
            result = walk_back(
                norm_node,
                [
                    OpStep(
                        op=torch.ops.aten._to_copy.default,
                        optional=True,
                        kwargs={
                            "dtype",
                            "layout",
                            "device",
                            "pin_memory",
                            "non_blocking",
                            "memory_format",
                        },
                    ),
                    OpStep(op=torch.ops.aten.mul.Tensor, nargs=2, arg_index=1),
                    OpStep(op=torch.ops.aten.rsqrt.default),
                    OpStep(op=torch.ops.aten.add.Tensor, nargs=2),
                    OpStep(op=torch.ops.aten.mean.dim, nargs=(2, 3), kwargs={"dtype"}),
                    OpStep(op=torch.ops.aten.pow.Tensor_Scalar, nargs=2),
                    OpStep(
                        op=torch.ops.aten._to_copy.default,
                        optional=True,
                        require_single_user=False,  # _to_copy output used by both pow and mul
                        kwargs={
                            "dtype",
                            "layout",
                            "device",
                            "pin_memory",
                            "non_blocking",
                            "memory_format",
                        },
                    ),
                ],
            )
            if result is None:
                continue

            original_input, entries = result
            to_copy_out, mul, rsqrt, add, mean, pow, to_copy_in = entries

            # If input _to_copy matched, verify it has exactly 2 users: pow and mul
            if to_copy_in is not None:
                users = set(to_copy_in.users.keys())
                expected_users = {pow, mul}
                if users != expected_users:
                    continue

            # Validate pow exponent is 2
            if pow.args[1] != 2:
                continue

            # Extract epsilon from add node (it's a lifted tensor constant)
            eps_value = None
            for arg in add.args:
                eps_value = extract_lifted_tensor_constant(arg)
                if eps_value is not None:
                    break

            if eps_value is None:
                continue

            # Build body from non-None entries
            body = [n for n in entries if n is not None]

            return cls(
                head=head,
                body=body,
                input_node=original_input,
                weight_node=weight_node,
                eps=eps_value,
            )

        return None


class FuseRMSNormPass(ExportPass):
    """
    Fuses decomposed RMSNorm operations into aten.rms_norm.

    This reduces ~7 ops to 1 fused op per RMSNorm layer.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            match = RMSNormMatch.maybe_create(node)
            if match is None:
                continue

            # Get input shape for normalized_shape
            input_meta = match.input_node.meta.get("val")
            if input_meta is None:
                continue

            # Create fused rms_norm node
            with graph.inserting_before(node):
                normalized_shape = [input_meta.shape[-1]]
                rms_norm_node = graph.call_function(
                    torch.ops.aten.rms_norm.default,
                    args=(
                        match.input_node,
                        normalized_shape,
                        match.weight_node,
                        match.eps,
                    ),
                )
                rms_norm_node.meta = node.meta.copy()

            node.replace_all_uses_with(rms_norm_node)
            match.remove_body_nodes(graph)
            graph.erase_node(node)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()

        return PassResult(graph_module, modified)


class CanonicalizePermutePass(ExportPass):
    """
    Converts transpose_copy to permute_copy in the edge dialect graph.

    transpose_copy(x, dim0, dim1) is equivalent to permute_copy(x, perm)
    where perm is the identity permutation with dim0 and dim1 swapped.
    This lets the backend handle a single permute op instead of both
    transpose and permute.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.transpose_copy.int
            ):
                continue

            input_node = node.args[0]
            input_val = (
                input_node.meta.get("val") if isinstance(input_node, Node) else None
            )
            if input_val is None:
                continue

            ndim = input_val.dim()
            dim0 = node.args[1]
            dim1 = node.args[2]

            # Normalize negative dims
            if dim0 < 0:
                dim0 += ndim
            if dim1 < 0:
                dim1 += ndim

            # Build permutation: identity with dim0 and dim1 swapped
            perm = list(range(ndim))
            perm[dim0], perm[dim1] = perm[dim1], perm[dim0]

            node.target = exir_ops.edge.aten.permute_copy.default
            node.args = (input_node, perm)
            modified = True

        if modified:
            graph.lint()

        return PassResult(graph_module, modified)


class CollapseViewCopyPass(ExportPass):
    """
    Collapses consecutive view_copy nodes into a single view_copy.

    view_copy(view_copy(x, shape1), shape2) → view_copy(x, shape2)

    Only the final shape matters, so intermediate view_copys can be removed.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        view_copy_target = exir_ops.edge.aten.view_copy.default

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target != view_copy_target:
                continue

            parent = node.args[0]
            if (
                isinstance(parent, Node)
                and parent.op == "call_function"
                and parent.target == view_copy_target
                and len(parent.users) == 1
            ):
                original_input = parent.args[0]
                target_shape = node.args[1]

                # Check if final shape matches original input shape (identity).
                # Compare meta shapes (not args) so SymInt dims are handled.
                # Use try/except because shapes may contain unbacked SymInts
                # (e.g. from .item() calls) that can't be guarded on.
                original_val = (
                    original_input.meta.get("val")
                    if isinstance(original_input, Node)
                    else None
                )
                output_val = node.meta.get("val")
                is_identity = False
                if original_val is not None and output_val is not None:
                    try:
                        is_identity = original_val.shape == output_val.shape
                    except Exception:
                        is_identity = False
                if is_identity:
                    # Identity — remove both view_copys
                    node.replace_all_uses_with(original_input)
                    graph.erase_node(node)
                    graph.erase_node(parent)
                else:
                    # Collapse: view_copy(view_copy(x, s1), s2) → view_copy(x, s2)
                    node.args = (original_input, target_shape)
                    graph.erase_node(parent)
                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()

        return PassResult(graph_module, modified)


class CollapsePermutePass(ExportPass):
    """
    Collapses consecutive permute_copy nodes into a single permute_copy.

    permute(permute(x, p1), p2) → permute(x, composed)
    where composed[i] = p1[p2[i]].

    If the composed permutation is the identity, the permute is removed entirely.
    Must run after CanonicalizePermutePass so all transpose_copy nodes are permute_copy.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        permute_target = exir_ops.edge.aten.permute_copy.default

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target != permute_target:
                continue

            parent = node.args[0]
            if (
                isinstance(parent, Node)
                and parent.op == "call_function"
                and parent.target == permute_target
                and len(parent.users) == 1
            ):
                p1 = parent.args[1]
                p2 = node.args[1]
                composed = [p1[p2[i]] for i in range(len(p2))]

                if composed == list(range(len(composed))):
                    # Identity permutation — remove both permutes
                    node.replace_all_uses_with(parent.args[0])
                    graph.erase_node(node)
                    graph.erase_node(parent)
                else:
                    node.args = (parent.args[0], composed)
                    graph.erase_node(parent)

                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()

        return PassResult(graph_module, modified)


def _is_pure_dtype_cast(kwargs: dict) -> bool:
    """Check that _to_copy kwargs only specify dtype (no device/layout/memory_format)."""
    for k, v in kwargs.items():
        if k == "dtype":
            continue
        if v is not None:
            return False
    return "dtype" in kwargs


class CollapseDtypeConversionPass(ExportPass):
    """
    Collapses consecutive _to_copy (dtype conversion) nodes into a single one.

    _to_copy(dtype=bf16)(_to_copy(dtype=f32)(x)) → _to_copy(dtype=bf16)(x)

    Only the final dtype matters. Only collapses when both nodes are pure dtype
    conversions (no device/layout/memory_format changes).
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        to_copy_target = exir_ops.edge.aten._to_copy.default

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target != to_copy_target:
                continue

            parent = node.args[0]
            if not (
                isinstance(parent, Node)
                and parent.op == "call_function"
                and parent.target == to_copy_target
                and len(parent.users) == 1
            ):
                continue

            # Only collapse pure dtype conversions
            node_kw = node.kwargs
            parent_kw = parent.kwargs
            if not _is_pure_dtype_cast(node_kw) or not _is_pure_dtype_cast(parent_kw):
                continue

            # Rewrite: to_copy(to_copy(x, dtype=d1), dtype=d2) → to_copy(x, dtype=d2)
            node.args = (parent.args[0],)
            graph.erase_node(parent)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()

        return PassResult(graph_module, modified)


class RemoveNoOpsPass(ExportPass):
    """
    Removes ops that are no-ops in the MLX backend.

    - alias_copy(x): always a no-op
    - clone(x): only when memory_format is contiguous or absent
    - _to_copy(x, dtype=d): when x already has dtype d
    - view_copy(x, shape): when shape matches input shape
    - permute_copy(x, [0,1,...,n-1]): identity permutation
    - slice_copy(x, ...): when output shape matches input shape (full slice)
    """

    def call(self, graph_module: GraphModule) -> PassResult:  # noqa: C901
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue

            input_node = (
                node.args[0] if node.args and isinstance(node.args[0], Node) else None
            )
            if input_node is None:
                continue

            remove = False

            if node.target == exir_ops.edge.aten.alias_copy.default:
                remove = True

            elif node.target == exir_ops.edge.aten.clone.default:
                mem_fmt = node.kwargs.get("memory_format")
                if mem_fmt is None or mem_fmt == torch.contiguous_format:
                    remove = True

            elif node.target == exir_ops.edge.aten._to_copy.default:
                if _is_pure_dtype_cast(node.kwargs):
                    input_val = input_node.meta.get("val")
                    target_dtype = node.kwargs.get("dtype")
                    if input_val is not None and input_val.dtype == target_dtype:
                        remove = True

            elif node.target == exir_ops.edge.aten.view_copy.default:
                input_val = input_node.meta.get("val")
                output_val = node.meta.get("val")
                if input_val is not None and output_val is not None:
                    try:
                        if input_val.shape == output_val.shape:
                            remove = True
                    except Exception:
                        pass

            elif node.target == exir_ops.edge.aten.permute_copy.default:
                perm = node.args[1]
                if list(perm) == list(range(len(perm))):
                    remove = True

            elif node.target == exir_ops.edge.aten.slice_copy.Tensor:
                input_val = input_node.meta.get("val")
                output_val = node.meta.get("val")
                if input_val is not None and output_val is not None:
                    try:
                        if input_val.shape == output_val.shape:
                            remove = True
                    except Exception:
                        pass

            if remove:
                node.replace_all_uses_with(input_node)
                graph.erase_node(node)
                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()

        return PassResult(graph_module, modified)


class CSEPass(ExportPass):
    """
    Common Subexpression Elimination using structural hashing (value numbering).

    Deduplicates operations with identical computation structure, replacing
    redundant computations with references to previously computed results.

    Uses recursive structural keys: two nodes are equivalent if they have the
    same op and their inputs are structurally equivalent. This naturally handles
    chains like item(select(select(x, 0, a), 0, b)) without special cases.

    Safety is determined automatically via op schema introspection:
    - For OpOverload targets (aten ops): checks _schema for mutating arguments
    - For operator.* targets (SymInt arithmetic): always safe
    - A small denylist covers non-deterministic ops (rand, dropout, etc.)
    """

    # Ops that are pure (no mutation per schema) but non-deterministic:
    # same inputs can produce different outputs.
    UNSAFE_OPS = frozenset(
        [
            "aten::rand",
            "aten::rand_like",
            "aten::randn",
            "aten::randn_like",
            "aten::randint",
            "aten::randint_like",
            "aten::randperm",
            "aten::bernoulli",
            "aten::dropout",
            "aten::native_dropout",
            "aten::multinomial",
            "aten::normal",
            "aten::uniform",
        ]
    )

    def _is_safe_target(self, target) -> bool:
        """
        Determine if an op target is safe for CSE.

        Uses schema introspection for OpOverload targets: if no argument
        has alias_info with is_write=True, the op doesn't mutate and is
        safe (unless it's non-deterministic).

        Python operator.* targets are always safe (pure scalar arithmetic).
        """
        # Python operator module functions are always pure and deterministic
        if getattr(target, "__module__", None) == "_operator":
            return True

        # EdgeOpOverload targets (edge dialect ops)
        if isinstance(target, (torch._ops.OpOverload, EdgeOpOverload)):
            schema_name = target._schema.name

            # Only trust schema introspection for aten:: ops.
            # Custom op schemas (mlx::, torchao::, etc.) may not accurately
            # annotate mutation or side effects — default to unsafe.
            if not schema_name.startswith("aten::"):
                return False

            # Check denylist for non-deterministic/side-effecting ops
            if schema_name in self.UNSAFE_OPS:
                return False

            # Check schema for mutating arguments
            for arg in target._schema.arguments:
                if arg.alias_info is not None and arg.alias_info.is_write:
                    return False

            return True

        return False

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph

        # Discover graph output nodes — includes buffer mutation outputs
        # (e.g. index_copy for KV cache). These must never be deduplicated
        # because the graph signature references them by name for writeback.
        output_node = next(n for n in graph.nodes if n.op == "output")
        self._output_nodes: set[Node] = set()
        for arg in output_node.args[0]:
            if isinstance(arg, Node):
                self._output_nodes.add(arg)

        self._vn_cache: dict[Node, int] = {}  # Node → value number
        self._safe_cache: dict[Any, bool] = {}  # Cache for _is_safe_target
        self._sig_to_vn: dict[Any, int] = {}  # flat signature → value number
        self._vn_to_node: dict[int, Node] = {}  # value number → canonical node
        self._next_vn = 0
        modified = False

        for node in list(graph.nodes):
            vn = self._value_number(node)

            if vn in self._vn_to_node:
                canonical = self._vn_to_node[vn]
                if canonical is not node:
                    node.replace_all_uses_with(canonical)
                    graph.erase_node(node)
                    modified = True
            else:
                self._vn_to_node[vn] = node

        if modified:
            graph.eliminate_dead_code()
            graph.lint()

        return PassResult(graph_module, modified)

    def _is_safe(self, target) -> bool:
        """Cached version of _is_safe_target."""
        tid = id(target)
        if tid not in self._safe_cache:
            self._safe_cache[tid] = self._is_safe_target(target)
        return self._safe_cache[tid]

    def _new_vn(self) -> int:
        """Allocate a fresh unique value number."""
        vn = self._next_vn
        self._next_vn += 1
        return vn

    def _value_number(self, node: Node) -> int:
        """
        Assign an integer value number to a node (global value numbering).

        Two nodes with the same value number are structurally equivalent
        and can be deduplicated. All signature tuples are flat (contain
        only ints and scalars), so hashing is O(n_args) not O(graph_depth).
        """
        if node in self._vn_cache:
            return self._vn_cache[node]

        if node.op != "call_function":
            vn = self._new_vn()
        elif node in self._output_nodes:
            # Graph output node (includes buffer mutations like index_copy).
            # Must keep unique — graph signature references this node by name.
            vn = self._new_vn()
        elif not self._is_safe(node.target):
            vn = self._new_vn()
        else:
            try:
                args_sig = tuple(self._make_hashable(a) for a in node.args)
                kwargs_sig = tuple(
                    sorted((k, self._make_hashable(v)) for k, v in node.kwargs.items())
                )
                sig = (node.target, args_sig, kwargs_sig)

                if sig in self._sig_to_vn:
                    vn = self._sig_to_vn[sig]
                else:
                    vn = self._new_vn()
                    self._sig_to_vn[sig] = vn
            except TypeError:
                vn = self._new_vn()

        self._vn_cache[node] = vn
        return vn

    def _make_hashable(self, obj) -> Any:
        """Convert args/kwargs to a hashable form.

        For Node args, returns the integer value number — keeping
        all signature tuples flat and O(1) to hash.
        """
        if isinstance(obj, Node):
            return self._value_number(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return tuple(self._make_hashable(x) for x in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, torch.dtype):
            return obj
        elif isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, torch.layout):
            return str(obj)
        elif isinstance(obj, torch.memory_format):
            return str(obj)
        else:
            raise TypeError(f"Cannot make {type(obj)} hashable")
