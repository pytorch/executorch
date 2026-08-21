# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from executorch.backends.transforms.channels_last_layout import is_layout_copy
from executorch.backends.transforms.fuse_cascaded_transpose_or_permute_ops import (
    FuseCascadedTransposeOrPermuteOps,
)
from executorch.backends.transforms.fuse_cascaded_view_ops import FuseCascadedViewOps
from executorch.backends.transforms.fuse_transpose_or_permute_op_pairs_pass import (
    FuseTransposeOrPermuteOpPairsPass,
)
from executorch.backends.transforms.postpone_permute_below_squeeze_view import (
    PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView,
)
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.backends.transforms.replace_nop_transpose_or_permute_with_view import (
    ReplaceNopTransposeOrPermuteWithViewPass,
)
from executorch.backends.transforms.replace_ops_with_channels_last_variants import (
    ChannelsLastOpSpec,
    ReplaceOpsWithChannelsLastVariants,
)
from executorch.backends.transforms.replace_squeeze_unsqueeze_with_view import (
    ReplaceSqueezeAndUnsqueezeWithViewPass,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.node import Target

_BOUNDARY_TRANSPARENT_TARGETS = {
    exir_ops.edge.aten.squeeze_copy.default,
    exir_ops.edge.aten.squeeze_copy.dim,
    exir_ops.edge.aten.squeeze_copy.dims,
    exir_ops.edge.aten.unsqueeze_copy.default,
    exir_ops.edge.aten.view.default,
    exir_ops.edge.aten.view_copy.default,
}
_BOUNDARY_TRANSPARENT_TARGETS.update(
    {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
    }
)


@dataclass(frozen=True)
class ChannelsLastLayoutReport:
    candidate_anchor_count: int = 0
    converted_anchor_count: int = 0
    inserted_copy_count: int = 0
    eliminated_copy_count: int = 0
    boundary_copy_count: int = 0
    internal_copy_count: int = 0
    unknown_copy_count: int = 0
    boundary_copy_bytes: int = 0
    internal_copy_bytes: int = 0
    unknown_copy_bytes: int = 0
    copies_with_unknown_size: int = 0
    internal_copy_nodes: tuple[str, ...] = ()
    unknown_copy_nodes: tuple[str, ...] = ()


class ToContiguousChannelsLastPass(ExportPass):
    """Build and optimize explicit contiguous-NHWC regions.

    The pass replaces selected NCHW operators with channels-last dialect
    anchors surrounded by ``channels_last.permute_copy`` nodes. It then runs
    the common data-movement optimizers to a fixed point and reports only the
    surviving layout copies. Strict mode rejects structurally unsafe copies and
    supported source anchors that were not converted. It does not reject
    user-authored permutes or estimate peak arena usage after memory planning.
    ``can_propagate`` is consulted by every transform that moves a layout copy
    across a graph node.
    """

    # The matrix in test_to_contiguous_channels_last_pass reaches its fixed
    # point in at most two rounds; this only catches a pass that reports
    # progress it did not make.
    _MAX_OPTIMIZATION_ITERATIONS = 4

    def __init__(
        self,
        exported_program: ExportedProgram,
        op_map: dict[Target, ChannelsLastOpSpec] | None = None,
        can_propagate: Callable[[torch.fx.Node], bool] | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.exported_program = exported_program
        self.op_map = op_map
        self.can_propagate = can_propagate
        self.strict = strict
        self.report = ChannelsLastLayoutReport()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        existing_copy_count = len(self._layout_copy_nodes(graph_module))
        replacement_pass = ReplaceOpsWithChannelsLastVariants(
            self.exported_program,
            op_map=self.op_map,
        )
        replacement = replacement_pass.call(graph_module)
        graph_module = replacement.graph_module
        modified = replacement.modified
        preoptimization_copy_count = len(self._layout_copy_nodes(graph_module))
        inserted_copy_count = max(0, preoptimization_copy_count - existing_copy_count)

        for iteration in range(self._MAX_OPTIMIZATION_ITERATIONS):
            iteration_modified = False
            for transform in self._optimization_passes():
                result = transform.call(graph_module)
                graph_module = result.graph_module
                iteration_modified |= result.modified

            modified |= iteration_modified
            if not iteration_modified:
                break
            if iteration == self._MAX_OPTIMIZATION_ITERATIONS - 1:
                raise RuntimeError(
                    "Channels-last layout optimization did not converge after "
                    f"{self._MAX_OPTIMIZATION_ITERATIONS} iterations."
                )

        self.report = self._build_report(
            graph_module,
            replacement_pass.candidate_count,
            replacement_pass.replacement_count,
            inserted_copy_count,
            preoptimization_copy_count,
        )
        # A copy the pass cannot classify as boundary or internal means it no
        # longer knows what it did to the graph. That is the one genuinely
        # unsound outcome, so it raises unconditionally rather than behind a
        # flag: there is no configuration in which it is acceptable.
        if self.report.unknown_copy_count:
            raise RuntimeError(
                "Channels-last layout optimization produced "
                f"{self.report.unknown_copy_count} copies it cannot account for. "
                f"Unknown nodes: {self.report.unknown_copy_nodes}."
            )

        # The rest are not soundness failures. Leftover anchors and internal
        # copies are missed optimizations, and an unreadable size is a
        # reporting limit under dynamic shapes -- the graph still computes the
        # right thing in both cases. Backends wanting a guarantee opt in.
        if self.strict and (
            self.report.candidate_anchor_count != self.report.converted_anchor_count
            or self.report.internal_copy_count
            or self.report.copies_with_unknown_size
        ):
            raise RuntimeError(
                "Channels-last layout optimization left "
                f"{self.report.converted_anchor_count} converted of "
                f"{self.report.candidate_anchor_count} candidate anchors, "
                f"{self.report.internal_copy_count} internal copies, and "
                f"{self.report.copies_with_unknown_size} unknown sizes. "
                f"Internal nodes: {self.report.internal_copy_nodes}."
            )

        return PassResult(graph_module, modified)

    def _optimization_passes(self) -> tuple[ExportPass, ...]:
        return (
            ReplaceSqueezeAndUnsqueezeWithViewPass(),
            ReplaceNopTransposeOrPermuteWithViewPass(),
            PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView(
                can_propagate=self.can_propagate
            ),
            FuseCascadedViewOps(),
            FuseCascadedTransposeOrPermuteOps(can_propagate=self.can_propagate),
            RemovePermutesAroundElementwiseOps(
                exported_program=self.exported_program,
                can_propagate=self.can_propagate,
            ),
            FuseTransposeOrPermuteOpPairsPass(can_propagate=self.can_propagate),
            FuseCascadedViewOps(),
            FuseCascadedTransposeOrPermuteOps(can_propagate=self.can_propagate),
        )

    @staticmethod
    def _layout_copy_nodes(
        graph_module: torch.fx.GraphModule,
    ) -> list[torch.fx.Node]:
        return [node for node in graph_module.graph.nodes if is_layout_copy(node)]

    def _build_report(
        self,
        graph_module: torch.fx.GraphModule,
        candidate_anchor_count: int,
        converted_anchor_count: int,
        inserted_copy_count: int,
        preoptimization_copy_count: int,
    ) -> ChannelsLastLayoutReport:
        boundary_nodes: list[torch.fx.Node] = []
        internal_nodes: list[torch.fx.Node] = []
        unknown_nodes: list[torch.fx.Node] = []

        for node in self._layout_copy_nodes(graph_module):
            dims = self._normalized_dims(node)
            if dims is None:
                unknown_nodes.append(node)
            elif self._reaches_user_input(node) or self._reaches_graph_output(node):
                boundary_nodes.append(node)
            else:
                internal_nodes.append(node)

        boundary_bytes, boundary_unknown = self._copy_bytes(boundary_nodes)
        internal_bytes, internal_unknown = self._copy_bytes(internal_nodes)
        unknown_bytes, unknown_unknown = self._copy_bytes(unknown_nodes)
        surviving_count = len(boundary_nodes) + len(internal_nodes) + len(unknown_nodes)
        return ChannelsLastLayoutReport(
            candidate_anchor_count=candidate_anchor_count,
            converted_anchor_count=converted_anchor_count,
            inserted_copy_count=inserted_copy_count,
            eliminated_copy_count=max(0, preoptimization_copy_count - surviving_count),
            boundary_copy_count=len(boundary_nodes),
            internal_copy_count=len(internal_nodes),
            unknown_copy_count=len(unknown_nodes),
            boundary_copy_bytes=boundary_bytes,
            internal_copy_bytes=internal_bytes,
            unknown_copy_bytes=unknown_bytes,
            copies_with_unknown_size=(
                boundary_unknown + internal_unknown + unknown_unknown
            ),
            internal_copy_nodes=tuple(node.name for node in internal_nodes),
            unknown_copy_nodes=tuple(node.name for node in unknown_nodes),
        )

    def _reaches_user_input(self, node: torch.fx.Node) -> bool:
        current = node.args[0] if node.args else None
        visited: set[torch.fx.Node] = set()
        while isinstance(current, torch.fx.Node) and current not in visited:
            visited.add(current)
            if current.op == "placeholder":
                return current.name in self.exported_program.graph_signature.user_inputs
            if self.can_propagate is not None and not self.can_propagate(current):
                return False
            if (
                current.op != "call_function"
                or current.target not in _BOUNDARY_TRANSPARENT_TARGETS
                or not current.args
                or not isinstance(current.args[0], torch.fx.Node)
            ):
                return False
            current = current.args[0]
        return False

    def _reaches_graph_output(self, node: torch.fx.Node) -> bool:
        pending = list(node.users)
        visited: set[torch.fx.Node] = set()
        reached_output = False
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if current.op == "output":
                reached_output = True
                continue
            if self.can_propagate is not None and not self.can_propagate(current):
                return False
            if (
                current.op != "call_function"
                or current.target not in _BOUNDARY_TRANSPARENT_TARGETS
                or not current.users
            ):
                return False
            pending.extend(current.users)
        return reached_output

    @staticmethod
    def _normalized_dims(node: torch.fx.Node) -> list[int] | None:
        if len(node.args) < 2 or not isinstance(node.args[1], (list, tuple)):
            return None
        dims = list(node.args[1])
        if not all(isinstance(dim, int) for dim in dims):
            return None
        rank = len(dims)
        normalized = [dim + rank if dim < 0 else dim for dim in dims]
        if sorted(normalized) != list(range(rank)):
            return None
        return normalized

    @staticmethod
    def _copy_bytes(nodes: list[torch.fx.Node]) -> tuple[int, int]:
        known_bytes = 0
        unknown_count = 0
        for node in nodes:
            val: Any = node.meta.get("val")
            if not isinstance(val, torch.Tensor) or not all(
                isinstance(dim, int) for dim in val.shape
            ):
                unknown_count += 1
                continue
            known_bytes += val.numel() * val.element_size()
        return known_bytes, unknown_count
