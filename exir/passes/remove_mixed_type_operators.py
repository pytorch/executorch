# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.pass_base import ExportPass, map_args, NodeMetadata, ProxyValue
from torch import SymBool, SymFloat, SymInt
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.utils._pytree import PyTree


class RemoveMixedTypeOperators(ExportPass):
    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta: NodeMetadata):  # noqa: C901
        if len(args) <= 1:
            # Unary Operators are not mixed type
            return super().call_operator(op, args, kwargs, meta)

        promotion_type_allow_list = {
            torch.ops.aten.add.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            torch.ops.aten.mul.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            torch.ops.aten.div.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            torch.ops.aten.minimum.default: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        }

        if op in promotion_type_allow_list:
            promotion_kind = promotion_type_allow_list[op]
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

        def try_coerce(value: PyTree, arg: torch.Argument) -> PyTree:
            if not isinstance(arg.type, torch.TensorType):
                return value

            if isinstance(value, ProxyValue):
                if not value.is_tensor():
                    return value
                if value.to_tensor().dtype == promote_dtype:
                    return value

            if isinstance(value, torch.Tensor) and value.dtype == promote_dtype:
                return value

            return self.call_operator(
                torch.ops.aten._to_copy.default,
                (value,),
                {"dtype": promote_dtype},
                meta,
            )

        args, kwargs = map_args(op, try_coerce, args, kwargs)

        return super().call_operator(op, args, kwargs, meta)
