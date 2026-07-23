# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import executorch.backends.transforms.channels_last_ops  # noqa: F401

import torch

from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch.fx.passes.infra.pass_base import PassResult

_DIM_ORDER_CHANGING_OPS: frozenset = frozenset(
    {
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
    }
)

_TO_NHWC_PERMUTATION: list[int] = [0, 2, 3, 1]
_TO_NCHW_PERMUTATION: list[int] = [0, 3, 1, 2]


def _is_4d_contiguous(dim_order: Sequence[int]) -> bool:
    return list(dim_order) == [0, 1, 2, 3]


def _is_4d_channels_last(dim_order: Sequence[int]) -> bool:
    return list(dim_order) == [0, 2, 3, 1]


class ReplaceChannelsLastInputClones(ExportPass):
    """Replace `_to_dim_order_copy` and `_clone_dim_order` with an equivalent sequence in the following pattern. This
        approach allows the `channels_last.permute_copy` to be optimized out if there are subsequent channels last
        operators in the model, leaving only the `aten.permute_copy`, which is effectively a no-op. As a result, the
        input data doesn't have to be permuted in memory.
                                                                              <model input>
                                                                                    │ [N, C, H, W] shape, (0, 2, 3, 1) dim order
          <model input>                                                             │ data is stored channels last
                │ [N, C, H, W] shape, (0, 2, 3, 1) dim order              ┌─────────▼─────────┐
                │ data is stored channels last                            │ aten.permute_copy ◄──── [0, 2, 3, 1] permutation
    ┌───────────▼────────────┐                                            └─────────┬─────────┘
    │ <dim order clone/copy> │                     ────────────────►                │ [N, H, W, C] shape, (0, 1, 2, 3) dim order
    └───────────┬────────────┘                                                      │ data is stored channels last
                │ [N, C, H, W] shape, (0, 1, 2, 3) dim order         ┌──────────────▼─────────────┐
                ▼ data is stored channels first                      │ channels_last.permute_copy ◄──── [0, 3, 1, 2] permutation
                                                                     └──────────────┬─────────────┘
                                                                                    │ [N, C, H, W] shape, (0, 1, 2, 3) dim order
                                                                                    ▼ data is stored channels first
    """

    _modified: bool

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self._modified = False
        result = super().call(graph_module)
        return PassResult(result.graph_module, self._modified)

    def call_operator(self, op, args, kwargs, meta: NodeMetadata) -> ProxyValue:
        # noinspection PyProtectedMember
        if not (
            op in _DIM_ORDER_CHANGING_OPS
            and hasattr(args[0], "node")
            and isinstance(input_ := args[0].node, torch.fx.Node)
            and input_.op == "placeholder"
            and _is_4d_channels_last(input_.meta["val"].dim_order())
            and _is_4d_contiguous(kwargs.get("dim_order", []))
        ):
            return super().call_operator(op, args, kwargs, meta)

        x = super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (args[0], _TO_NHWC_PERMUTATION),
            {},
            meta,
        )
        x = super().call_operator(
            exir_ops.edge.channels_last.permute_copy.default,
            (x, _TO_NCHW_PERMUTATION),
            {},
            meta,
        )

        self._modified = True

        return x
