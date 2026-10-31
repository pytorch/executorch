# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict

import torch
from executorch.exir.pass_base import ExportPass, map_args, NodeMetadata, ProxyValue
from torch import SymBool, SymFloat, SymInt
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.utils._pytree import PyTree


class RemoveMixedTypeOperators(ExportPass):
    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta: NodeMetadata):  # noqa: C901
        if op in (torch.ops.aten.cat.default, torch.ops.aten.stack.default):
            # Earlier passes edit the graph in place and can drop inputs, such as
            # empty ones, so the output metadata still has eager's promoted dtype.
            return self._cast_inputs(op, args, kwargs, meta, meta["val"].dtype)

        if len(args) <= 1:
            # Unary Operators are not mixed type
            return super().call_operator(op, args, kwargs, meta)

        promotion_type_allow_list = {
            torch.ops.aten.add.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            torch.ops.aten.mul.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            torch.ops.aten.sub.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            # The correct promotion for div depends on the mode! If there is no mode,
            # it's INT_TO_FLOAT, otherwise it's default.
            torch.ops.aten.div.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
            torch.ops.aten.div.Tensor_mode: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            torch.ops.aten.minimum.default: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        }

        if op in promotion_type_allow_list:
            promotion_kind = promotion_type_allow_list[op]
            if (
                op == torch.ops.aten.div.Tensor_mode
                and kwargs.get("rounding_mode") is None
            ):
                promotion_kind = ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        else:
            # Not in allow list, do nothing
            return super().call_operator(op, args, kwargs, meta)

        # Using tensors for type information only
        arg_tensor = []
        for arg in args:
            if isinstance(arg, ProxyValue) and arg.is_tensor():
                arg_tensor.append(arg.to_tensor())
            elif isinstance(arg, ProxyValue) and isinstance(
                arg.data,
                (
                    SymFloat,
                    SymInt,
                    SymBool,
                ),
            ):
                arg_tensor.append(torch.tensor(arg.data))
            # Note: this case can happen after scarlar_to_tensor pass
            # where we convert a scalar to a tensor.
            elif isinstance(arg, torch.Tensor):
                arg_tensor.append(arg)
            else:
                arg_tensor.append(arg.data)
        arg_tensor = tuple(arg_tensor)

        # Computes type for computation
        promote_dtype: torch.dtype = elementwise_dtypes(
            *arg_tensor,
            type_promotion_kind=promotion_kind,
        )[1]

        return self._cast_inputs(op, args, kwargs, meta, promote_dtype)

    # pyre-ignore
    def _cast_inputs(self, op, args, kwargs, meta: NodeMetadata, dtype: torch.dtype):
        # Cast a tensor that appears twice, as in cat([x, y, x]), only once. Key by
        # id() because constant tensors from ScalarToTensorPass have no node.
        casts: Dict[int, PyTree] = {}

        def coerce(value: PyTree) -> PyTree:
            if isinstance(value, ProxyValue):
                if not value.is_tensor():
                    return value
                if value.to_tensor().dtype == dtype:
                    return value

            if isinstance(value, torch.Tensor) and value.dtype == dtype:
                return value

            if id(value) not in casts:
                casts[id(value)] = self.call_operator(
                    torch.ops.aten._to_copy.default,
                    (value,),
                    {"dtype": dtype},
                    meta,
                )
            return casts[id(value)]

        def try_coerce(value: PyTree, arg: torch.Argument) -> PyTree:
            if arg.type == torch.ListType.ofTensors():
                return [coerce(v) for v in value]
            if isinstance(arg.type, torch.TensorType):
                return coerce(value)
            return value

        args, kwargs = map_args(op, try_coerce, args, kwargs)

        return super().call_operator(op, args, kwargs, meta)
