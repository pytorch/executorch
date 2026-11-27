# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class CopyReturnedViewsPass(PassBase):
    """
    Copies each view that the program returns into a contiguous tensor.

    ``ReplaceViewCopyWithViewPass`` turns view copies such as ``slice_copy``
    into views, and AOTInductor returns such a view with its base's strides.
    The output's ETensor is contiguous, like the copy was, so the backend
    copies that output through host memory, with a stream sync, on every call,
    and does not capture the method in a CUDA graph. ``alias_copy`` makes the
    copy on the GPU instead. ``clone`` would too, but the Edge verifier
    rejects it in dim order mode.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        for node in graph.output_node().all_input_nodes:
            if not getattr(node.target, "is_view", False):
                continue
            # The returned node becomes the copy, so the outputs keep the names
            # the graph signature refers to.
            with graph.inserting_before(node):
                view = graph.call_function(node.target, node.args, node.kwargs)
            view.meta = dict(node.meta)
            node.target = exir_ops.edge.aten.alias_copy.default
            node.args = (view,)
            node.kwargs = {}
            modified = True
        if modified:
            graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)
