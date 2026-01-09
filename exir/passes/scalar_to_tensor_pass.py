# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.pass_base import ExportPass, map_args, PassResult


class ScalarToTensorPass(ExportPass):
    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        modified: bool = False
        def try_coerce(value, arg):
            # Note: we want to create tensor constants instead of
            # FakeTensor or ProxyTensor. If python_dispatcher is enabled,
            # the fake_tensor_mode of inputs will be used so that we won't
            # get a constant tensor with torch.tensor() call but instead
            # a fake tensor is created.
            nonlocal modified
            with torch.utils._python_dispatch._disable_current_modes():
                if isinstance(value, (float, int, bool)) and isinstance(
                    arg.type, torch.TensorType
                ):
                    modified = True
                    return torch.tensor(value)
                return value

        args, kwargs = map_args(node.target, try_coerce, node.args, node.kwargs)
        if modified:
            node.args = args
            node.kwargs = kwargs
        return modified

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        changed = False
        for module in filter(
            lambda m: isinstance(m, torch.fx.GraphModule), graph_module.modules()
        ):
            for node in module.graph.nodes:
                if not isinstance(
                    node.target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)
                ):
                    continue
                changed |= self.maybe_remove_or_replace(node)

        if changed:
            graph_module.recompile()
            return super().call(graph_module)

        return PassResult(graph_module, False)
