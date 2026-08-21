# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass, field

import torch

from executorch.backends.transforms.channels_last_layout import is_layout_copy
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

# Elementwise per-tensor quantization is layout-agnostic, so a layout copy on
# the far side of one is still a boundary copy. Per-channel quantization is not:
# its axis is dimension-dependent.
_LAYOUT_AGNOSTIC_TARGETS = frozenset(
    {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
    }
)


def _inverse(dims: tuple[int, ...]) -> list[int]:
    inverse = [0] * len(dims)
    for position, dim in enumerate(dims):
        inverse[dim] = position
    return inverse


@dataclass(frozen=True)
class BoundaryLayoutContract:
    """Which method inputs and outputs changed layout, and to what.

    An entry ``{0: (0, 2, 3, 1)}`` in ``inputs`` means argument 0 must now be
    passed as ``argument.permute(0, 2, 3, 1)``. An entry in ``outputs`` means
    the returned tensor needs the same permutation applied to recover what the
    method used to return.
    """

    inputs: dict[int, tuple[int, ...]] = field(default_factory=dict)
    outputs: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.inputs or self.outputs)


class AbsorbBoundaryLayoutCopies(ExportPass):
    """Move layout copies that sit on the method boundary into the signature.

    A layout region formed by ``ToContiguousChannelsLastPass`` is bracketed by
    ``channels_last.permute_copy``. Those in the interior cancel against each
    other; the ones on the boundary have nothing to cancel against and survive.
    Deleting them and declaring the corresponding method input or output to be
    channels-last moves the transpose to the caller, which is free whenever the
    caller already has the data in that layout.

    Run this *after* region formation. Permuting the boundary first and hoping
    the copies cancel is measurably worse: it inserts copies into graphs that
    have no anchors at all, where nothing can cancel them.

    A copy need not touch the boundary directly. Quantized graphs put a
    per-tensor ``quantize``/``dequantize`` in between, which reorders nothing,
    so the search walks through those and relabels them on the way.

    Changing a method's layout is caller-visible, so the applied changes are
    reported in ``contract`` rather than assumed.
    """

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program
        self.contract = BoundaryLayoutContract()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        if graph_module is not self.exported_program.graph_module:
            raise RuntimeError(
                "AbsorbBoundaryLayoutCopies rewrites the ExportedProgram's graph "
                "and signature together; run it as its own transform rather than "
                "after a pass that replaces the graph module."
            )

        inputs = self._absorb_inputs(graph_module)
        outputs = self._absorb_outputs(graph_module)
        self.contract = BoundaryLayoutContract(inputs=inputs, outputs=outputs)

        modified = bool(self.contract)
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)

    def _forward_to_copies(self, node: torch.fx.Node):
        """Follow every path out of ``node`` until it reaches a layout copy.

        Returns the layout-agnostic nodes crossed, the copies terminating the
        paths, and the permutation they agree on; ``None`` if any path ends
        somewhere else or the copies disagree.
        """
        interior: list[torch.fx.Node] = []
        copies: list[torch.fx.Node] = []
        dims: tuple[int, ...] | None = None
        frontier = [node]

        while frontier:
            users = list(frontier.pop().users)
            if not users:
                return None
            for user in users:
                if is_layout_copy(user):
                    user_dims = tuple(user.args[1])
                    if dims is not None and user_dims != dims:
                        # A residual block feeds one value to several branches,
                        # each with its own copy. They collapse to a single
                        # contract entry only if they agree.
                        return None
                    dims = user_dims
                    copies.append(user)
                elif user.target in _LAYOUT_AGNOSTIC_TARGETS:
                    if user in interior:
                        continue
                    interior.append(user)
                    frontier.append(user)
                else:
                    return None

        return None if dims is None else (interior, copies, dims)

    def _backward_to_copy(self, result: torch.fx.Node):
        """Walk back from a returned value to the layout copy that produced it."""
        interior: list[torch.fx.Node] = []
        current = result

        while True:
            if is_layout_copy(current):
                if len(current.users) != 1:
                    return None
                source = current.args[0]
                if not isinstance(source, torch.fx.Node):
                    return None
                return interior, current, source, tuple(current.args[1])
            if (
                current.target not in _LAYOUT_AGNOSTIC_TARGETS
                or len(current.users) != 1
            ):
                return None
            interior.append(current)
            current = current.args[0]
            if not isinstance(current, torch.fx.Node):
                return None

    def _absorb_inputs(self, graph_module) -> dict[int, tuple[int, ...]]:
        user_inputs = list(self.exported_program.graph_signature.user_inputs)
        absorbed: dict[int, tuple[int, ...]] = {}

        for node in list(graph_module.graph.nodes):
            if node.op != "placeholder" or node.name not in user_inputs:
                continue
            found = self._forward_to_copies(node)
            if found is None:
                continue
            interior, copies, dims = found

            # .contiguous() is load-bearing: a bare permute leaves the old
            # strides, which serialize as a permuted dim order, and the kernels
            # reading these tensors require plain contiguous NHWC.
            for member in (node, *interior):
                member.meta["val"] = member.meta["val"].permute(dims).contiguous()
            for copy in copies:
                copy.replace_all_uses_with(copy.args[0])
                graph_module.graph.erase_node(copy)
            absorbed[user_inputs.index(node.name)] = dims

        return absorbed

    def _absorb_outputs(self, graph_module) -> dict[int, tuple[int, ...]]:
        output_node = graph_module.graph.output_node()
        results = list(output_node.args[0])
        specs = self.exported_program.graph_signature.output_specs
        absorbed: dict[int, tuple[int, ...]] = {}

        for index, result in enumerate(results):
            if not isinstance(result, torch.fx.Node):
                continue
            found = self._backward_to_copy(result)
            if found is None:
                continue
            interior, copy, source, dims = found

            for member in interior:
                member.meta["val"] = (
                    member.meta["val"].permute(_inverse(dims)).contiguous()
                )
            if interior:
                interior[-1].replace_input_with(copy, source)
            else:
                results[index] = source
                # The manager re-derives the signature, but the direct
                # exported_program= path does not.
                if index < len(specs) and getattr(specs[index].arg, "name", None) == (
                    result.name
                ):
                    specs[index].arg.name = source.name
            absorbed[index] = dims

        if absorbed:
            output_node.args = (results,)
        return absorbed
