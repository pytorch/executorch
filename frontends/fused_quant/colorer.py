# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod

from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch.export import ExportedProgram
from torch.fx import Node

# Meta key a colorer writes to claim a node; the value names the backend. This is
# not ``delegation_tag``: partitioners write that one, one tag per partition, and
# ``to_backend`` consumes it.
CLAIM_KEY = "fused_quant_claim"


def is_colored(node: Node) -> bool:
    """Whether a backend has claimed this node.

    The generic fused_quant passes use this to leave claimed nodes alone; see
    :class:`ColorerBase` for the contract.
    """
    return CLAIM_KEY in node.meta


class ColorerBase(ExportedProgramPassBase):
    """Claims the nodes a backend will take, by writing :data:`CLAIM_KEY` meta.

    Coloring is the step between optimization and lowering: it decides which
    fused_quant ops the backend claims, so its lowering knows what to rewrite. A
    backend subclasses this and implements :meth:`can_lower`; the colorer walks
    the graph and claims every node the backend can lower.

    A claim is a statement of ownership, and the generic passes honour it:
    :class:`~executorch.backends.transforms.fused_quant.optimization_passes.to_channels_last.ToChannelsLast`
    and
    :class:`~executorch.backends.transforms.fused_quant.optimization_passes.to_channels_first.ToChannelsFirst`
    will not relayout a colored node, and
    :class:`~executorch.backends.transforms.fused_quant.decompose_fused_quant.DecomposeFusedQuant`
    will not decompose one. So the usual pipeline is colorer -> ToChannelsFirst
    -> DecomposeFusedQuant: the backend keeps what it claimed, and everything
    else falls back to quantize/dequantize plus ATen. That is what makes
    claiming an op sufficient to keep it intact -- a backend describes what it
    supports, not what it does not. Passes test this with :func:`is_colored`.

    Coloring and partitioning are meant to match: a backend lowers every node it
    claims, so no claimed node is ever left for the portable fallback. The
    intended way to guarantee that is to lower the claimed nodes to the
    backend's own ops before ``to_backend``, so its partitioner takes those ops
    and has nothing to reject.

    :class:`ReleaseClaims` is not part of that design. It exists only for
    backward compatibility and to ease the transition of backends whose
    partitioner can still reject a claimed node; new backends should not rely
    on it.

    Colorers compose by priority: run them in order and the first to claim a
    node keeps it. A node that is already claimed is skipped, never overwritten,
    so a later (lower-priority) colorer only sees what the earlier ones left.
    """

    def __init__(self, backend: str) -> None:
        super().__init__()
        self.backend = backend

    @abstractmethod
    def can_lower(self, exported_program: ExportedProgram, node: Node) -> bool:
        """Whether this backend can lower ``node``.

        Asked about every ``call_function`` node no colorer has claimed yet, so
        check the op first, then any limits on the node's qparams, attributes or
        users.
        """
        ...

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        modified = False
        for node in exported_program.graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and not is_colored(node)
                and self.can_lower(exported_program, node)
            ):
                node.meta[CLAIM_KEY] = self.backend
                modified = True
        return ExportedProgramPassResult(exported_program, modified)


class ReleaseClaims(ExportedProgramPassBase):
    """Drop every claim left on the graph. Only for backward compatibility.

    The intended flow never needs this: coloring and partitioning match, so a
    backend lowers every node it claims (see :class:`ColorerBase`). This pass
    only eases the transition of backends whose partitioner can still reject a
    claimed node, which ``to_backend`` leaves fused and still claimed, because it
    clears ``delegation_tag`` but not :data:`CLAIM_KEY`. Run it once every
    backend has run ``to_backend``, then ``ToChannelsFirst`` and
    ``DecomposeFusedQuant`` again, to fall those nodes back to the portable
    kernels, as rerunning ``DecomposeFusedQuant`` did when the claim was
    ``delegation_tag``.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        modified = False
        for node in exported_program.graph_module.graph.nodes:
            if is_colored(node):
                del node.meta[CLAIM_KEY]
                modified = True
        return ExportedProgramPassResult(exported_program, modified)
