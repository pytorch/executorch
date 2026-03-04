# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule
from torch.fx.node import Node


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
