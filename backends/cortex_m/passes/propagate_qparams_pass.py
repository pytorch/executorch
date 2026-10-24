# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from executorch.backends.transforms.channels_last_layout import PERMUTE_COPY_TARGETS
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule, Node
from torch.fx.node import Target
from torch.fx.passes.infra.pass_manager import PassResult

logger: logging.Logger = logging.getLogger(__name__)

# Ops that rearrange a tensor without changing its element type: a reshape, a
# layout copy, a transpose. Whatever dtype the producer hands them is the dtype
# they hand on.
_TRANSPARENT_TARGETS: frozenset[Target] = frozenset(
    (
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.squeeze_copy.dim,
        exir_ops.edge.aten.squeeze_copy.dims,
        exir_ops.edge.aten.unsqueeze_copy.default,
        exir_ops.edge.aten.transpose_copy.int,
        *PERMUTE_COPY_TARGETS,
    )
)


class PropagateQParamsPass(ExportPass):
    """Give the shape and layout plumbing the qparams of the tensor it moves.

    FoldAndAnnotateQParamsPass annotates the ops that compute. The ops that
    only rearrange are left with neither qparams nor a refreshed fake tensor,
    so they still carry the fp32 tensor the pre-quantization trace gave them,
    and anything asking what a plumbing node produces -- `lowered_output_dtype`
    and so every caller gating a substitution on int8, plus
    ValidateCortexMExplicitLayoutPass -- is told fp32 about a tensor that is
    int8 on device. The symptom is not a wrong answer but a missing one: a
    substitution declines, the op stays on a portable kernel, and nothing says
    so.

    Rearranging does not change an element type, so a plumbing node's qparams
    are its producer's. Annotating them here resolves the question once, in the
    graph, rather than at each of the places that ask it. Nodes are visited in
    topological order, so a chain of plumbing resolves in a single pass: each
    node reads a producer this pass has already annotated.

    Runs before ValidateCortexMExplicitLayoutPass, which is the first consumer,
    and late enough that the passes which read and write `output_qparams`
    themselves -- ActivationFusionPass, DecomposeHardswishPass -- have already
    seen the graph without these annotations.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False

        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target not in _TRANSPARENT_TARGETS:
                continue
            if node.meta.get("output_qparams"):
                continue
            if not node.args or not isinstance(producer := node.args[0], Node):
                continue
            qparams = producer.meta.get("output_qparams")
            if not qparams:
                continue
            # The same QuantArgs the producer carries: a rearrangement reads and
            # writes one tensor, so there is a single output to describe and it
            # is quantized exactly as the input was.
            node.meta["output_qparams"] = dict(qparams)
            modified = True
            logger.debug(
                "[PropagateQParams] %s inherits qparams from %s",
                node.name,
                producer.name,
            )

        return PassResult(graph_module, modified)
