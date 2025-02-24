# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List, Set, Tuple, Union

import executorch.backends.vulkan.custom_ops_lib  # noqa: needed to access vk op
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue

from torch._ops import OpOverload

from torch.fx.node import Argument

OpType = Union[str, OpOverload, EdgeOpOverload]


class SqueezeUnsqueezeInputs(ExportPass):
    _squeezable_ops: Set[OpType] = {
        exir_ops.edge.et_vk.linear_weight_int4.default,
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.gelu.default,
    }

    def call_operator(
        self,
        op,  # pyre-ignore
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        def _squeezable(shape: List[int]) -> bool:
            return len(shape) > 2 and 1 in shape

        if op not in self._squeezable_ops:
            return super().call_operator(op, args, kwargs, meta)

        # pyre-ignore[16]: `None` has no attribute `node`
        input_shape = args[0].node.meta["val"].shape
        output_shape = meta["val"].shape
        if not _squeezable(input_shape):
            return super().call_operator(op, args, kwargs, meta)

        # squeeze input tensor
        squeeze_shape = list(input_shape)
        while _squeezable(squeeze_shape):
            squeeze_shape.remove(1)

        squeeze_out = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (args[0], squeeze_shape),
            kwargs,
            meta,
        )
        # call linear on squeezed output
        new_args = (squeeze_out, *args[1:])
        linear_out = super().call_operator(
            op,
            new_args,
            kwargs,
            meta,
        )
        # unsqueeze output
        unsqueeze_shape = list(output_shape)
        return super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (linear_out, unsqueeze_shape),
            kwargs,
            meta,
        )
