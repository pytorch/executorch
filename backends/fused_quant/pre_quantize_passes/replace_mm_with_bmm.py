# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import torch
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.fused_quant.graph_utils import compute_meta_val
from torch import fx
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class ReplaceMmWithBmm(PassBase):
    """Replace ``aten.mm(x, y)`` with a unit-batch ``aten.bmm`` between views.

    This lets the existing ``fused_quant::bmm`` quantizer and Turing TCE
    delegation handle the matmul instead of leaving it as an unfused float op
    on the DSP. Adding and dropping a leading size-1 dimension is a valid view
    for any input strides, so the reshapes cost nothing at runtime.
    """

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for mm_node in graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        ):
            self._replace(mm_node)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)

    @staticmethod
    def _replace(mm_node: fx.Node) -> None:
        graph = mm_node.graph
        inp = get_arg(mm_node, "input", fx.Node)
        mat2 = get_arg(mm_node, "mat2", fx.Node)
        inp_val = inp.meta["val"]
        mat2_val = mat2.meta["val"]

        with graph.inserting_before(mm_node):
            inp_3d = graph.call_function(
                torch.ops.aten.view.default, args=(inp, [1, *inp_val.shape])
            )
            inp_3d.meta = mm_node.meta.copy()
            inp_3d.meta["val"] = inp_val.unsqueeze(0)

            mat2_3d = graph.call_function(
                torch.ops.aten.view.default, args=(mat2, [1, *mat2_val.shape])
            )
            mat2_3d.meta = mm_node.meta.copy()
            mat2_3d.meta["val"] = mat2_val.unsqueeze(0)

            bmm = graph.call_function(
                torch.ops.aten.bmm.default, args=(inp_3d, mat2_3d)
            )
            bmm.meta = mm_node.meta.copy()
            bmm.meta["val"] = compute_meta_val(bmm)

            out_shape = list(mm_node.meta["val"].shape)
            squeezed = graph.call_function(
                torch.ops.aten.view.default, args=(bmm, out_shape)
            )
            squeezed.meta = mm_node.meta.copy()

        mm_node.replace_all_uses_with(squeezed)
        graph.erase_node(mm_node)
