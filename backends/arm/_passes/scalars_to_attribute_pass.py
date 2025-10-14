# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast, Set, Type, Union

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm._passes.match_arg_ranks_pass import MatchArgRanksPass

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix


class ScalarsToAttributePass(ExportPass):
    """
    For ops in 'targeted_ops', convert inputs that are scalar values
    to attribute Nodes that output the same value.
    """

    _passes_required_after: Set[Type[ExportPass]] = {MatchArgRanksPass}

    targeted_ops = [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub_.Tensor,
        torch.ops.aten.rsub.Scalar,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul_.Tensor,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div_.Tensor,
    ]

    def call(self, graph_module: GraphModule) -> PassResult:
        for n in graph_module.graph.nodes:
            n = cast(Node, n)
            if n.op != "call_function" or n.target not in self.targeted_ops:
                continue

            biggest_rank = 1
            for arg in n.args:
                if isinstance(arg, Node):
                    shape = get_first_fake_tensor(arg).shape
                    biggest_rank = max(biggest_rank, len(shape))

            new_args = []
            for arg in n.args:
                if isinstance(arg, Node):
                    new_args.append(arg)
                    continue
                if isinstance(arg, int) and not torch.is_floating_point(
                    get_first_fake_tensor(n)
                ):
                    new_args.append(arg)  # type: ignore[arg-type]
                    continue

                prefix = "_tensor_constant_"
                get_new_attr_name = get_new_attr_name_with_prefix(prefix)
                tensor_constant_name = get_new_attr_name(graph_module)
                float_tensor = torch.tensor(
                    float(cast(Union[int, float], arg))
                ).reshape((1,) * biggest_rank)
                graph_module.register_buffer(tensor_constant_name, float_tensor)
                fake_mode = n.meta["val"].fake_mode

                with graph_module.graph.inserting_before(n):
                    get_attr_node = graph_module.graph.create_node(
                        "get_attr", tensor_constant_name, (), {}
                    )
                    get_attr_node.meta["val"] = fake_mode.from_tensor(
                        float_tensor, static_shapes=True
                    )
                    new_args.append(get_attr_node)
            n.args = tuple(new_args)

            # Replace rsub.Scalar with sub.Tensor as retracing will fail otherwise
            if n.target == torch.ops.aten.rsub.Scalar:
                with graph_module.graph.inserting_after(n):
                    reversed_args = (n.args[1], n.args[0])
                    sub = graph_module.graph.create_node(
                        "call_function", torch.ops.aten.sub.Tensor, reversed_args, {}
                    )
                    n.replace_all_uses_with(sub)
                    sub.meta["val"] = n.meta["val"]
                graph_module.graph.erase_node(n)

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
