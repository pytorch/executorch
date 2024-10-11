# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Union

import torch
from executorch.backends.arm.tosa_mapping import extract_tensor_meta

from executorch.exir.pass_base import ExportPass, PassResult
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.fx import GraphModule, Node


class ScalarsToAttributePass(ExportPass):
    """
    For ops in 'targeted_ops', convert inputs that are scalar values
    to attribute Nodes that output the same value.
    """

    targeted_ops = [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub_.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.div.Tensor,
    ]

    def call(self, graph_module: GraphModule) -> PassResult:
        for n in graph_module.graph.nodes:
            n = cast(Node, n)
            if n.op != "call_function" or n.target not in self.targeted_ops:
                continue

            biggest_rank = 1
            for arg in n.args:
                if isinstance(arg, Node):
                    _, shape, _ = extract_tensor_meta(arg.meta)
                    biggest_rank = max(biggest_rank, len(shape))

            new_args = []
            for arg in n.args:
                if isinstance(arg, Node):
                    new_args.append(arg)
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

        graph_module.recompile()
        return PassResult(graph_module, True)
