# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Set, Type

import torch.fx
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

logger = logging.getLogger(__name__)


def _int_arg(node: torch.fx.Node, index: int, default: int) -> int:
    """Get an integer argument from a node, with a default if missing."""
    val = node.args[index] if len(node.args) > index else default
    assert isinstance(val, int)
    return val


def _normalize_dim(dim: int, rank: int) -> int:
    """Normalize a (possibly negative) dim index to ``[0, rank)``."""
    return (dim + rank) % rank


def _slice_params(node: torch.fx.Node, dim_size: int) -> tuple[int, int, int, int]:
    """Extract (dim, start, end, step) from a slice_copy node.

    ``dim``, ``start``, and ``end`` are normalized to non-negative indices in
    ``[0, dim_size]`` (matching PyTorch slice semantics, where negative bounds
    count from the end of the dimension). ``dim_size`` is the size of the
    source tensor along the slice dimension.

    """
    rank = len(get_first_fake_tensor(node).shape)
    dim = _normalize_dim(_int_arg(node, 1, 0), rank)
    start = _int_arg(node, 2, 0)
    end = _int_arg(node, 3, dim_size)
    if start < 0:
        start += dim_size
    if end < 0:
        end += dim_size
    start = max(0, min(start, dim_size))
    end = max(0, min(end, dim_size))
    step = _int_arg(node, 4, 1)
    return dim, start, end, step


_SLICE_OP = exir_ops.edge.aten.slice_copy.Tensor


def _is_valid_slice(node: torch.fx.Node, cat_dim: int, dim_size: int) -> bool:
    """Check that node is a slice_copy on cat_dim with step=1."""
    if node.target != _SLICE_OP:
        return False
    s_dim, _, _, s_step = _slice_params(node, dim_size)
    return s_dim == cat_dim and s_step == 1


def _find_slice_replacement(
    slice_op: torch.fx.Node,
    cat_node: torch.fx.Node,
    cat_dim: int,
    s_start: int,
    s_end: int,
    offsets: list[tuple[int, int, torch.fx.Node]],
) -> torch.fx.Node | None:
    """Find a replacement for a slice that consumes a cat output.

    ``offsets`` maps each concat input to its range in the concatenated
    output: [(start, end, input_node), ...] along ``cat_dim``.

    Returns the replacement node (exact input match or adjusted sub-slice),
    or None if the slice crosses input boundaries.

    """
    for o_start, o_end, inp in offsets:
        if s_start == o_start and s_end == o_end:
            return inp
        if s_start >= o_start and s_end <= o_end:
            graph = cat_node.graph
            with graph.inserting_before(slice_op):
                new_slice = graph.call_function(
                    _SLICE_OP,
                    (inp, cat_dim, s_start - o_start, s_end - o_start),
                )
                new_slice.meta = slice_op.meta.copy()
            return new_slice
    return None


def _find_common_slice_source(
    cat_inputs: list | tuple,
    cat_dim: int,
    dim_size: int,
) -> torch.fx.Node | None:
    """Check all inputs are valid slices of the same source.

    Returns the source.

    """
    source_node = None
    for inp in cat_inputs:
        if not isinstance(inp, torch.fx.Node):
            return None
        if not _is_valid_slice(inp, cat_dim, dim_size):
            return None
        slice_source = inp.args[0]
        if source_node is None:
            source_node = slice_source
        elif slice_source is not source_node:
            return None
    assert isinstance(source_node, torch.fx.Node)
    return source_node


def _check_contiguous_slices(
    cat_inputs: list | tuple,
    source_dim_size: int,
) -> tuple[int, int] | None:
    """Check slices are contiguous.

    Returns (first_start, last_end) or None.

    """
    _, first_start, _, _ = _slice_params(cat_inputs[0], source_dim_size)
    expected_start = first_start
    for inp in cat_inputs:
        _, s_start, s_end, _ = _slice_params(inp, source_dim_size)
        if s_start != expected_start:
            return None
        expected_start = s_end

    # expected_start is now the end of the last slice
    return first_start, expected_start


class FuseConcatPass(ArmPass):
    """Eliminate redundant concat (cat) operations via graph pattern matching.

    This pass recognizes and removes concat operations that can be proven to
    produce no useful data movement. Eliminating these at the FX/TOSA level
    prevents Vela from generating MemoryCopy operations on the Ethos-U NPU.

    Five patterns are handled:

    1. Single-input concat: cat([x], dim) is a no-op; replace with x.
    2. Concat-then-slice (exact): if a consumer of cat([a, b, ...], dim) is
       a slice_copy that extracts exactly one original input, replace it
       with the corresponding concat input directly.
    3. Slice-then-concat (full): if cat([slice(x, d, s0, e0),
       slice(x, d, s1, e1), ...], dim) reconstructs x exactly (contiguous
       slices covering the full source dimension), replace with x.
    4. Concat-then-sub-slice: if a consumer of cat([a, b, ...], dim) is a
       slice_copy whose range falls entirely within one original input,
       replace it with an adjusted slice on that input directly.
    5. Slice-then-concat (partial): if contiguous slices of the same tensor
       are concatenated but cover only a sub-range of the source dimension,
       replace with a single slice on the source.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    cat_ops = {
        exir_ops.edge.aten.cat.default,
    }
    slice_op = _SLICE_OP

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in self.cat_ops:
                continue
            if node.graph is None:
                continue

            if self._eliminate_single_input_cat(node):
                modified = True
                continue

            if self._eliminate_cat_then_slice(node):
                modified = True
                continue

            if self._eliminate_slice_then_cat(node):
                modified = True
                continue

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)

    # ------------------------------------------------------------------
    # Pattern 1: single-input cat
    # ------------------------------------------------------------------
    @staticmethod
    def _eliminate_single_input_cat(cat_node: torch.fx.Node) -> bool:
        inputs = cat_node.args[0]
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 1:
            return False
        sole_input = inputs[0]
        assert isinstance(sole_input, torch.fx.Node)
        cat_node.replace_all_uses_with(sole_input)
        logger.debug("Eliminated single-input cat: %s", cat_node.name)
        return True

    # ------------------------------------------------------------------
    # Patterns 2 & 4: cat -> slice (exact input or sub-range of input)
    # ------------------------------------------------------------------
    @staticmethod
    def _eliminate_cat_then_slice(
        cat_node: torch.fx.Node,
    ) -> bool:
        cat_inputs = cat_node.args[0]
        if not isinstance(cat_inputs, (list, tuple)) or len(cat_inputs) < 2:
            return False

        # if the dim does not exist as an arg, it defaults to '0'
        output_rank = len(get_first_fake_tensor(cat_node).shape)
        cat_dim = _normalize_dim(_int_arg(cat_node, 1, 0), output_rank)

        users = list(cat_node.users.keys())
        if not users:
            return False

        # Build the offset map for each concat input along cat_dim.
        offsets = []
        offset = 0
        for inp in cat_inputs:
            assert isinstance(inp, torch.fx.Node)
            inp_shape = get_first_fake_tensor(inp).shape
            size = inp_shape[cat_dim]
            offsets.append((offset, offset + size, inp))
            offset += size

        # Every user must be a slice_copy on the same dim with step=1.
        # Collect validated (node, start, end) for replacement below.
        validated_slices: list[tuple[torch.fx.Node, int, int]] = []
        for slice_op in users:
            if not _is_valid_slice(slice_op, cat_dim, offset):
                return False
            if slice_op.args[0] is not cat_node:
                return False
            _, s_start, s_end, _ = _slice_params(slice_op, offset)
            validated_slices.append((slice_op, s_start, s_end))

        # For each user, try exact match (Pattern 2) then sub-range (Pattern 4).
        # Users that cross input boundaries are skipped.
        replacements: list[tuple[torch.fx.Node, torch.fx.Node]] = []

        for slice_op, s_start, s_end in validated_slices:
            replacement = _find_slice_replacement(
                slice_op, cat_node, cat_dim, s_start, s_end, offsets
            )
            if replacement is not None:
                replacements.append((slice_op, replacement))

        if not replacements:
            return False

        for old_node, new_node in replacements:
            old_node.replace_all_uses_with(new_node)

        logger.debug(
            "Eliminated cat-then-slice pattern: %s (%d slices redirected)",
            cat_node.name,
            len(replacements),
        )
        return True

    # ------------------------------------------------------------------
    # Patterns 3 & 5: slice -> cat (contiguous slices, full or partial)
    # ------------------------------------------------------------------
    @staticmethod
    def _eliminate_slice_then_cat(
        cat_node: torch.fx.Node,
    ) -> bool:
        cat_inputs = cat_node.args[0]
        if not isinstance(cat_inputs, (list, tuple)) or len(cat_inputs) < 2:
            return False

        output_rank = len(get_first_fake_tensor(cat_node).shape)
        cat_dim = _normalize_dim(_int_arg(cat_node, 1, 0), output_rank)

        # All inputs must be slice_copy on the same source tensor and dim,
        # with step=1.
        source_node = _find_common_slice_source(cat_inputs, cat_dim, output_rank)
        if source_node is None:
            return False

        source_shape = get_first_fake_tensor(source_node).shape
        source_dim_size = source_shape[cat_dim]

        # Verify slices are contiguous (but not necessarily starting at 0).
        bounds = _check_contiguous_slices(cat_inputs, source_dim_size)
        if bounds is None:
            return False
        first_start, last_end = bounds

        # Verify output shape matches expectations.
        cat_shape = get_first_fake_tensor(cat_node).shape

        if first_start == 0 and last_end == source_dim_size:
            # Pattern 3: full coverage — replace with source tensor.
            if list(cat_shape) != list(source_shape):
                return False
            cat_node.replace_all_uses_with(source_node)
            logger.debug(
                "Eliminated slice-then-cat (full): %s -> %s",
                cat_node.name,
                source_node.name,
            )
        else:
            # Pattern 5: partial coverage — replace with single slice.
            expected_dim_size = last_end - first_start
            if cat_shape[cat_dim] != expected_dim_size:
                return False
            for i, (cs, ss) in enumerate(zip(cat_shape, source_shape)):
                if i != cat_dim and cs != ss:  # dims must match except for cat_dim
                    return False
            graph = cat_node.graph
            with graph.inserting_before(cat_node):
                new_slice = graph.call_function(
                    _SLICE_OP,
                    (source_node, cat_dim, first_start, last_end),
                )
                new_slice.meta = cat_node.meta.copy()
            cat_node.replace_all_uses_with(new_slice)
            logger.debug(
                "Eliminated slice-then-cat (partial): %s -> slice(%s, %d, %d:%d)",
                cat_node.name,
                source_node.name,
                cat_dim,
                first_start,
                last_end,
            )
        return True
