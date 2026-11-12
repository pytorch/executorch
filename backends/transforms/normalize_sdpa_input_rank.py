# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult, ProxyValue


_SDPA_OPS = (
    torch.ops.aten.scaled_dot_product_attention.default,
    exir_ops.edge.aten.scaled_dot_product_attention.default,
)


class NormalizeSDPAInputRankPass(ExportPass):
    """Pad rank-2/3 SDPA inputs to rank 4 and restore the original output rank.

    Leading singleton axes preserve the head axis of [H, L, D] inputs. Rank-2
    [L, D] inputs gain both a batch and a head axis. Masks are also padded to
    rank 4 for fused-kernel compatibility. SDPA options are unchanged, and only
    the inserted output axes are squeezed.
    """

    @staticmethod
    def _needs_normalization(inputs, output):
        return (
            isinstance(output, torch.Tensor)
            and output.dim() in (2, 3)
            and all(
                isinstance(value, torch.Tensor) and value.dim() == output.dim()
                for value in inputs
            )
        )

    def call_operator(self, op, args, kwargs, meta):
        if op not in _SDPA_OPS:
            return super().call_operator(op, args, kwargs, meta)

        inputs = [
            args[i] if i < len(args) else kwargs.get(name)
            for i, name in enumerate(("query", "key", "value"))
        ]
        if not all(isinstance(x, ProxyValue) and x.is_tensor() for x in inputs):
            return super().call_operator(op, args, kwargs, meta)
        if not self._needs_normalization(
            [x.to_tensor() for x in inputs], meta.data.get("val")
        ):
            return super().call_operator(op, args, kwargs, meta)
        rank = inputs[0].to_tensor().dim()

        aten = (
            torch.ops.aten
            if op is torch.ops.aten.scaled_dot_product_attention.default
            else exir_ops.edge.aten
        )
        mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
        if isinstance(mask, ProxyValue) and mask.is_tensor():
            inputs.append(mask)
        new_args, new_kwargs = list(args), dict(kwargs)
        names = ("query", "key", "value", "attn_mask")
        for i, (name, value) in enumerate(zip(names, inputs)):
            for _ in range(4 - value.to_tensor().dim()):
                value = super().call_operator(
                    aten.unsqueeze_copy.default, (value, 0), {}, meta
                )
            if i < len(new_args):
                new_args[i] = value
            else:
                new_kwargs[name] = value
        result = super().call_operator(op, tuple(new_args), new_kwargs, meta)
        self._modified = True
        return super().call_operator(
            aten.squeeze_copy.dims, (result, list(range(4 - rank))), {}, meta
        )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self._modified = False
        # Even a no-op retrace can create unbacked symbols in the shared ShapeEnv.
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != "call_function" or node.target not in _SDPA_OPS:
                    continue
                inputs = [
                    node.args[i] if i < len(node.args) else node.kwargs.get(name)
                    for i, name in enumerate(("query", "key", "value"))
                ]
                values = [
                    arg.meta.get("val") if isinstance(arg, torch.fx.Node) else arg
                    for arg in inputs
                ]
                if self._needs_normalization(values, node.meta.get("val")):
                    result = super().call(graph_module)
                    return PassResult(result.graph_module, self._modified)
        return PassResult(graph_module, False)
