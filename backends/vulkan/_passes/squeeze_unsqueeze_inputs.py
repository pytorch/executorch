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
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.gelu.default,
    }

    def should_squeeze(self, op, shape: List[int]) -> bool:  # pyre-ignore
        if len(shape) == 3:
            return shape[1] == 1 and shape[0] > 1
        if len(shape) == 4:
            # No need to squeeze if all dims are 1 except the width dim
            if shape[0] == shape[1] == shape[2] == 1:
                return False
            # No need to squeeze if batch and channel dims are 1 and height and width are > 1
            if shape[0] == shape[1] == 1 and shape[2] > 1 and shape[3] > 1:
                return False
            # No need to squeeze if batch dim is 1 and channel, height and width are > 1
            if shape[0] == 1 and shape[1] > 1 and shape[2] > 1 and shape[3] > 1:
                return False
            # Otherwise, check for squeezable dim
            return 1 in shape[:-1]

        # Prefer not to introduce additional orchestration ops by default
        return False

    def call_operator(
        self,
        op,  # pyre-ignore
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in self._squeezable_ops:
            return super().call_operator(op, args, kwargs, meta)
        # pyre-ignore[16]: `None` has no attribute `node`
        input_shape = args[0].node.meta["val"].shape
        output_shape = meta["val"].shape

        if not self.should_squeeze(op, input_shape):
            return super().call_operator(op, args, kwargs, meta)

        def _squeezable(shape: List[int]) -> bool:
            return len(shape) > 2 and 1 in shape

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
