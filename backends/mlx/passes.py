# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph transformation passes for the MLX backend.
"""

from typing import List

import torch
from executorch.backends.transforms.collapse_view_copy import CollapseViewCopyPass
from executorch.backends.transforms.fuse_gqa_with_sdpa import FuseGQAWithSDPAPass
from executorch.backends.transforms.fuse_rms_norm import FuseRMSNormPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import (
    ExportedProgramPassBase,
    ExportedProgramPassResult,
    ExportPass,
    PassResult,
)
from executorch.exir.pass_manager import PassType
from executorch.exir.passes.cse_pass import CSEPass
from torch.fx import GraphModule, Node


def get_default_passes() -> List[PassType]:
    """
    Returns a list of passes that are enabled by default for the MLX backend.
    """
    return [
        FuseRMSNormPass(fold_dtype_casts=True, allow_lossy_weight_casts=True),
        FuseGQAWithSDPAPass(),
        CanonicalizePermutePass(),
        CollapseViewCopyPass(),
        CollapsePermutePass(),
        CollapseDtypeConversionPass(),
        RemoveNoOpsPass(),
        CSEPass(),
        # Must run last: rewrites the settled/deduped graph's functional
        # elementwise chains into in-place ops so the MLX builder can donate
        # buffers (out == in).
        MLXReinplacePass(),
    ]


class MLXReinplacePass(ExportedProgramPassBase):
    """Reinplace MLX-handled elementwise ops to enable MLX buffer donation.

    Rewrites functional unary/binary chains (e.g. ``exp(log(exp(x)))``,
    ``h + ffn(h)``) into their in-place edge forms via ExecuTorch's
    ``reinplace_pass``, restricted to an explicit, MLX-owned op set. Every op in
    that set has a corresponding in-place MLX handler (see
    ``REINPLACEABLE_UNARY_BASE_NAMES`` / ``REINPLACEABLE_BINARY_BASE_OVERLOADS``
    in ops.py) that binds the output slot to the dead input slot (out == in).

    Binary ops are safe to include because ``reinplace_pass`` only reinplaces
    when the mutated argument already holds the output's shape and dtype (no
    broadcast growth / dtype change).

    We deliberately do NOT use ``DEFAULT_INPLACEABLE_OPS``: passing an explicit
    ``ops_to_inplace`` fully replaces the default, so ``index_put`` is never
    reinplaced and MLX's existing KV-cache / index_copy functional patterns are
    untouched. A fresh set is built per call so ``reinplace_pass`` cannot mutate
    shared state.

    Runs as an EP-aware pass (it needs ``graph_signature`` to protect mutable
    inputs/buffers); the pass infra hands ``ExportedProgramPassBase`` instances
    the full ExportedProgram.
    """

    def call(self, exported_program) -> ExportedProgramPassResult:
        # Imported lazily to avoid a module-load cycle with ops.py (which
        # registers handlers on import).
        from executorch.backends.mlx.ops import (
            REINPLACEABLE_BINARY_BASE_OVERLOADS,
            REINPLACEABLE_EXTRA_BASE_OVERLOADS,
            REINPLACEABLE_UNARY_BASE_NAMES,
        )
        from executorch.exir.passes.reinplace import reinplace_pass

        # Explicit, MLX-owned op set (every op has an in-place MLX handler).
        # Binary ops are safe to pass to reinplace_pass because it guards that
        # the mutated arg is full-size + dtype-matching (no broadcast growth).
        ops_to_inplace = {
            getattr(exir_ops.edge.aten, base).default
            for base in REINPLACEABLE_UNARY_BASE_NAMES
        }
        ops_to_inplace |= {
            getattr(getattr(exir_ops.edge.aten, base), overload)
            for base, overload in (
                REINPLACEABLE_BINARY_BASE_OVERLOADS + REINPLACEABLE_EXTRA_BASE_OVERLOADS
            )
        }
        if ops_to_inplace:
            reinplace_pass(exported_program, ops_to_inplace=ops_to_inplace)
        return ExportedProgramPassResult(exported_program, True)


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

    Only collapse when the intermediate cast preserves every source value.
    Narrowing or cross-kind casts may round, truncate, or overflow and must stay.
    Both nodes must be pure dtype conversions (no device/layout/memory_format changes).
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

            source = parent.args[0]
            source_val = source.meta.get("val") if isinstance(source, Node) else None
            if source_val is None:
                continue
            source_dtype = source_val.dtype
            intermediate_dtype = parent_kw["dtype"]
            if source_dtype != intermediate_dtype and (
                source_dtype,
                intermediate_dtype,
            ) not in {
                # Boolean values 0 and 1 are exact in each floating-point dtype.
                (torch.bool, torch.float16),
                (torch.bool, torch.bfloat16),
                (torch.bool, torch.float32),
                (torch.bool, torch.float64),
                (torch.float16, torch.float32),
                (torch.bfloat16, torch.float32),
                (torch.float16, torch.float64),
                (torch.bfloat16, torch.float64),
                (torch.float32, torch.float64),
            }:
                continue

            # Rewrite: to_copy(to_copy(x, dtype=d1), dtype=d2) → to_copy(x, dtype=d2)
            node.args = (source,)
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
