# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import NodeMetadata, ProxyValue


class FuseConsecutiveConcatShapesPass(ArmPass):
    """This pass fuses consecutive tosa.CONCAT_SHAPE operations into a single
    tosa.CONCAT_SHAPE operation with a flattened list of input shapes. E.g.
    tosa.CONCAT_SHAPE([shape1, tosa.CONCAT_SHAPE([shape2, shape3]), shape4])
    becomes tosa.CONCAT_SHAPE([shape1, shape2, shape3, shape4])

    This is necessary in order for dim-order propagation to work correctly. E.g.
    in the case of dim-order==(0, 2, 3, 1) we would need to permute input shapes
    accordingly. This is much easier if the inputs are flattened.

    """

    _passes_required_after = set()

    def _to_proxy_value(
        self, arg: ProxyValue | torch.fx.Node | Any
    ) -> ProxyValue | Any:
        if isinstance(arg, ProxyValue):
            return arg
        if isinstance(arg, torch.fx.Node):
            return ProxyValue(arg.meta["val"], self.tracer.proxy(arg))
        return arg

    def call_operator(
        self,
        op: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        meta: NodeMetadata,
        updated: bool | None = False,
    ) -> ProxyValue:
        if op != exir_ops.backend.tosa.CONCAT_SHAPE.default:
            return super().call_operator(op, args, kwargs, meta)
        arg_list = args[0]
        new_arg_list: list[Any] = []
        modified = False
        for arg in arg_list:
            if (
                hasattr(arg, "node")
                and arg.node.target == exir_ops.backend.tosa.CONCAT_SHAPE.default
            ):
                new_arg_list.extend(
                    self._to_proxy_value(nested_arg) for nested_arg in arg.node.args[0]
                )
                modified = True
            else:
                new_arg_list.append(arg)
        return super().call_operator(
            op, (new_arg_list,), kwargs, meta, updated=modified
        )
