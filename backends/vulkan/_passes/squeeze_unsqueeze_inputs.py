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
from torch.fx.experimental.symbolic_shapes import (
    statically_known_false,
    statically_known_true,
    sym_and,
)

OpType = Union[str, OpOverload, EdgeOpOverload]


class SqueezeUnsqueezeInputs(ExportPass):
    _squeezable_ops: Set[OpType] = {
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.gelu.default,
    }

    @staticmethod
    def _first_static_one(shape: List[int]) -> Union[int, None]:  # pyre-ignore
        for index, dim in enumerate(shape):
            if statically_known_true(dim == 1):
                return index
        return None

    def _squeezed_shape(self, shape: List[int]) -> List[int]:  # pyre-ignore
        squeezed_shape = list(shape)
        while len(squeezed_shape) > 2:
            index = self._first_static_one(squeezed_shape)
            if index is None:
                break
            squeezed_shape.pop(index)
        return squeezed_shape

    def should_squeeze(self, op, shape: List[int]) -> bool:  # pyre-ignore
        if len(shape) == 3:
            return statically_known_true(sym_and(shape[1] == 1, shape[0] > 1))
        if len(shape) == 4:
            excluded_shapes = (
                sym_and(shape[0] == 1, shape[1] == 1, shape[2] == 1),
                sym_and(
                    shape[0] == 1,
                    shape[1] == 1,
                    shape[2] > 1,
                    shape[3] > 1,
                ),
                sym_and(
                    shape[0] == 1,
                    shape[1] > 1,
                    shape[2] > 1,
                    shape[3] > 1,
                ),
            )
            if any(
                not statically_known_false(excluded_shape)
                for excluded_shape in excluded_shapes
            ):
                return False
            return self._first_static_one(shape[:-1]) is not None

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

        # squeeze input tensor
        squeeze_shape = self._squeezed_shape(input_shape)

        squeeze_out = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (args[0], squeeze_shape),
            {},
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
            {},
            meta,
        )
