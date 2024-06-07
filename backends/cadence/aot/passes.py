# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.cadence.aot.utils import get_edge_overload_packet
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, ProxyValue
from torch._subclasses import FakeTensor
from torch.utils._pytree import tree_map_only


class ReplacePT2QuantWithCadenceQuantPass(ExportPass):
    """
    Replace the pt2 quantization ops with custom cadence quantization ops.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in {exir_ops.edge.quantized_decomposed.quantize_per_tensor.default}:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.cadence.quantize_per_tensor.default,
            args,
            kwargs,
            meta,
        )


class ReplacePT2DequantWithCadenceDequantPass(ExportPass):
    """
    Replace the pt2 dequantization ops with custom cadence dequantization ops.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in {exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default}:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.cadence.dequantize_per_tensor.default,
            args,
            kwargs,
            meta,
        )


class ReplaceScalarTensorWithFullPass(ExportPass):
    """
    aten.scalar_tensor can be replaced by aten.full with a shape of [1].
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in {
            exir_ops.edge.aten.scalar_tensor.default,
            torch.ops.aten.scalar_tensor.default,
        }:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.aten.full.default,
            (
                [1],
                args[0],
            ),
            {},
            meta,
        )


class ReplaceSqueezeAndUnsqueezeWithViewPass(ExportPass):
    """
    When the shape is static, replace squeeze_copy and unsqueeze_copy ops with
    view_copy op
    """

    def call_operator(self, op, args, kwargs, meta):
        # Instead of testing EdgeOpOverload, test EdgeOpOverloadPacket,
        # which allows us to cover all overloads.
        if get_edge_overload_packet(op) not in {
            exir_ops.edge.aten.squeeze_copy,
            exir_ops.edge.aten.unsqueeze_copy,
        }:
            return super().call_operator(op, args, kwargs, meta)
        # Get the output tensor shape
        out_shape = meta["val"].shape

        # Bail out if any dim is not an int (dynamic shape)
        for dim in list(out_shape):
            if not isinstance(dim, int):
                return super().call_operator(op, args, kwargs, meta)

        # Return a view op with the new shape
        view_args = (args[0], list(out_shape))
        return super().call_operator(
            exir_ops.edge.aten.view_copy.default, view_args, kwargs, meta
        )


class RemoveZeroSizedCatArgsPass(ExportPass):
    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten.cat.default:
            return super().call_operator(op, args, kwargs, meta)

        # Remove any zero-sized tensor arg to form a new args list.
        new_args = []
        for arg in args[0]:
            arg_tensor = arg.to_tensor() if isinstance(arg, ProxyValue) else arg
            if arg_tensor.numel() > 0:
                new_args.append(arg)

        # If all the tensors were empty, we just return an empty tensor with
        # the right shape.
        if not new_args:
            args_data, kwargs_data = tree_map_only(
                ProxyValue, lambda x: x.data, (args, kwargs)
            )
            result = op(*args_data, **kwargs_data)
            # When tracing with PT2, the FakeTensor mode requires the constant
            # argument to be set to itself.
            # TODO(matthiascremon): confirm this is the best way to do this.
            if isinstance(result, FakeTensor):
                result.constant = result
            return torch.empty_like(result)

        # If there was only one tensor in the new_args list,
        # we can safely erase this cat op.
        if len(new_args) == 1:
            return new_args[0]

        # Otherwise, we replace args[0] with new_args.
        args = list(args)
        args[0] = new_args
        args = tuple(args)
        return super().call_operator(op, args, kwargs, meta)
