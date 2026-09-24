# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import abstractmethod

from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch.export import ExportedProgram
from torch.fx import Node

# Meta key a colorer writes to claim a node for its backend. ExecuTorch's
# partitioning infrastructure uses the same key, so a claim made here is the
# same claim the partitioner will act on.
DELEGATION_TAG = "delegation_tag"


def is_colored(node: Node) -> bool:
    """Whether a backend has claimed this node.

    The generic fused_quant passes use this to leave claimed nodes alone; see
    :class:`ColorerBase` for the contract.
    """
    return DELEGATION_TAG in node.meta


class ColorerBase(ExportedProgramPassBase):
    """Tags the nodes a backend will take, by writing ``delegation_tag`` meta.

    Coloring is the step between optimization and lowering: it decides which
    fused_quant ops the backend claims, so the lowering pass knows what to rewrite
    and the partitioner knows what to delegate. A backend subclasses this and
    answers two questions -- which ops it can run (:meth:`supported_ops`) and
    whether a given instance is within its limits (:meth:`is_legal`).

    The tag is a claim of ownership, and the generic passes honour it:
    :class:`~executorch.backends.fused_quant.optimization_passes.to_channels_last.ToChannelsLast`
    and
    :class:`~executorch.backends.fused_quant.optimization_passes.to_channels_first.ToChannelsFirst`
    will not relayout a colored node, and
    :class:`~executorch.backends.fused_quant.decompose_fused_quant.DecomposeFusedQuant`
    will not decompose one. So the usual pipeline is colorer -> ToChannelsFirst
    -> DecomposeFusedQuant: the backend keeps what it claimed, and everything
    else falls back to quantize/dequantize plus ATen. That is what makes
    claiming an op sufficient to keep it intact -- a backend describes what it
    supports, not what it does not. Passes test this with :func:`is_colored`.

    Must run before any delegation has happened; a node that already carries a
    ``delegation_tag`` is an error rather than something to overwrite.
    """

    def __init__(self, delegation_tag: str) -> None:
        super().__init__()
        self.delegation_tag = delegation_tag

    @abstractmethod
    def supported_ops(self) -> set[EdgeOpOverload]:
        """Edge ops this backend can run, before per-node legality is applied."""
        ...

    def is_legal(self, exported_program: ExportedProgram, node: Node) -> bool:
        """Whether this specific node is within the backend's limits.

        Defaults to True: an op listed in :meth:`supported_ops` with no further
        constraints is always claimed. Override to reject instances whose
        qparams, attributes, or users the backend cannot handle.
        """
        return True

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        modified = False
        graph = exported_program.graph_module.graph

        for target in self.supported_ops():
            for node in graph.find_nodes(op="call_function", target=target):
                if is_colored(node):
                    raise RuntimeError(
                        f"{type(self).__name__} should only be run on non-delegated "
                        "graphs"
                    )
                if self.is_legal(exported_program, node):
                    node.meta[DELEGATION_TAG] = self.delegation_tag
                    modified = True
                else:
                    logging.info(
                        f"Could not color {node.name} to run on {self.delegation_tag}"
                    )

        return ExportedProgramPassResult(exported_program, modified)
