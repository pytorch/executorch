# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.backends.arm._passes.convert_squeezes_to_view import (
    ConvertSqueezesToViewPass,
)
from executorch.backends.arm._passes.decompose_maxpool2d_with_dilation_pass import (
    DecomposeMaxPool2dPass,
)
from executorch.backends.arm._passes.rewrite_max_pool2d_pass import RewriteMaxPool2dPass
from executorch.backends.arm._passes.size_adjust_input_pass import SizeAdjustInputPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import Node


class NormalizeMaxPool2dInputRankPass(ArmOpTargetedPass):
    """Normalize unbatched rank-3 max_pool2d inputs to rank 4.

    Unsqueeze inputs from ``[C, H, W]`` to ``[1, C, H, W]``.

    Squeeze the leading dimension after pooling to restore the rank-3 output.

    The complete shape transformation is::

        [C, H, W]
          -> unsqueeze(0) -> [1, C, H, W]
          -> max_pool2d -> [1, C, H_out, W_out]
          -> squeeze(0) -> [C, H_out, W_out]

    """

    target_ops = (exir_ops.edge.aten.max_pool2d.default,)
    _passes_required_after: Set[Type[ExportPass]] = {
        ConvertSqueezesToViewPass,
        DecomposeMaxPool2dPass,
        RewriteMaxPool2dPass,
        SizeAdjustInputPass,
    }

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        pool_nodes = graph.find_nodes(op="call_function", target=self.target_ops[0])

        for pool_node in pool_nodes:
            input_node = pool_node.args[0]
            if not isinstance(input_node, Node):
                raise RuntimeError("Expected max_pool2d input to be a node")

            input_fake = get_first_fake_tensor(input_node)
            if input_fake.dim() != 3:
                continue

            output_fake = get_first_fake_tensor(pool_node)
            with graph.inserting_before(pool_node):
                unsqueeze = create_node(
                    graph,
                    exir_ops.edge.aten.unsqueeze_copy.default,
                    args=(input_node, 0),
                    from_node=pool_node,
                    inherit_qparams=False,
                )
                unsqueeze.meta["val"] = input_fake.unsqueeze(0)
                pool_node.replace_input_with(input_node, unsqueeze)

            pool_node.meta["val"] = output_fake.unsqueeze(0)
            original_users = list(pool_node.users)
            with graph.inserting_after(pool_node):
                squeeze = create_node(
                    graph,
                    exir_ops.edge.aten.squeeze_copy.dims,
                    args=(pool_node, [0]),
                    from_node=pool_node,
                    inherit_qparams=False,
                )
                squeeze.meta["val"] = output_fake
                for user in original_users:
                    user.replace_input_with(pool_node, squeeze)

            modified = True

        if modified:
            graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)
