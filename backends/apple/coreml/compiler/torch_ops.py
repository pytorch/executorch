# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file registers torch ops that are not yet in coremltools, or are in a more recent version of
# coremltools than is used by ExecuTorch.  Each op registered here should have a link to a PR in coremltools that adds
# the op to the coremltools library.

import numpy as np
import torch as _torch
from coremltools import _logger
from coremltools.converters.mil.frontend import _utils
from coremltools.converters.mil.frontend.torch.ops import (
    _get_inputs,
    _get_kwinputs,
    NUM_TO_NUMPY_DTYPE,
    NUM_TO_TORCH_DTYPE,
    split,
    to,
    transpose,
    unbind,
)
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    register_torch_op,
)
from coremltools.converters.mil.mil import types
from executorch.exir.dim_order_utils import get_memory_format


# https://github.com/apple/coremltools/pull/2556
@register_torch_op(override=False)
def transpose_copy(context, node):
    transpose(context, node)


# https://github.com/apple/coremltools/pull/2557
@register_torch_op(override=False)
def unbind_copy(context, node):
    unbind(context, node)


# https://github.com/apple/coremltools/pull/2563
@register_torch_op(override=False)
def split_copy(context, node):
    split(context, node)


def is_fbcode():
    return not hasattr(_torch.version, "git_version")


if not is_fbcode():
    from coremltools.converters.mil.frontend.torch.dim_order_ops import (
        _empty_dim_order,
        _to_dim_order_copy,
    )

    # This is a temporary hack to register the alias "dim_order_ops._to_dim_order_copy",
    # which was missed by coremltools
    @register_torch_op(torch_alias=["dim_order_ops._to_dim_order_copy"], override=False)
    def _to_dim_order_copy_TMP_EXECUTORCH_ALIAS_HACK(context, node):
        _to_dim_order_copy(context, node)

    # This is a temporary hack to register the alias "dim_order_ops._empty_dim_order",
    # which was missed by coremltools
    @register_torch_op(torch_alias=["dim_order_ops._empty_dim_order"], override=False)
    def _empty_dim_order_TMP_EXECUTORCH_ALIAS_HACK(context, node):
        _empty_dim_order(context, node)

else:
    # TODO: remove this case when fbcode updates to coremltools 9.0
    @register_torch_op(
        torch_alias=[
            "dim_order_ops::_to_dim_order_copy",
            "dim_order_ops._to_dim_order_copy",
        ],
        override=False,
    )
    def _to_dim_order_copy(context, node):
        dim_order = _get_kwinputs(context, node, "dim_order", default=[None])[0]
        node.kwinputs.pop("dim_order")

        # In CoreML, dim_order.val will be an ndarray, so we convert it to a list
        dim_order = [int(d) for d in dim_order.val]
        memory_format = get_memory_format(dim_order)
        assert (
            memory_format == _torch.contiguous_format
        ), "Only contiguous memory format is supported in CoreML"
        to(context, node)


# https://github.com/apple/coremltools/pull/2558
@register_torch_op(
    torch_alias=["torchao::dequantize_affine", "torchao.dequantize_affine"],
    override=False,
)
def dequantize_affine(context, node):
    inputs = _get_inputs(context, node, expected=[7, 8])
    int_data = inputs[0].val
    block_size = inputs[1].val
    scale = inputs[2].val
    zero_point = (
        inputs[3].val if inputs[3] is not None and inputs[3].val is not None else None
    )
    # I do not think we need to worry about input_dtype b/c it gets cast to int4/int8
    # For now, we just check that it is int8 or int32
    input_dtype = inputs[4].val  # noqa: F841
    assert NUM_TO_TORCH_DTYPE[input_dtype] in [
        _torch.int8,
        _torch.int32,
    ], "input_dtype should be int8 or int32"

    quant_min = inputs[5].val
    quant_max = inputs[6].val

    assert len(int_data.shape) == 2, "dequantize_affine only supports rank 2 inputs"

    assert len(int_data.shape) == len(
        block_size
    ), "block_size must have the same length as int_data.shape"
    assert block_size[0] == 1, "block_size[0] must be 1"
    group_size = block_size[1]
    k = int_data.shape[1]
    assert k % group_size == 0, "k must be divisible by group_size"
    scales_per_row = k // group_size
    scale = scale.reshape(-1, scales_per_row)
    if zero_point is not None:
        zero_point = zero_point.reshape(-1, scales_per_row)

    # TODO: I don't know if CoreML can make use of this
    # We could add a cast op to the output, but I'm pretty CoreML will remove this during a later pass
    # For now, we just log a warning
    out_np_dtype = None
    if len(inputs) > 7:
        out_np_dtype = NUM_TO_NUMPY_DTYPE[inputs[7].val]
        _logger.warning(
            f"Core ML ignores output_dtype {out_np_dtype} on torchao.dequantize_affine and instead uses the native precision."
        )

    if quant_min == -8 and quant_max == 7:
        quantized_np_dtype = types.nptype_from_builtin(types.string_to_builtin("int4"))
    elif quant_min == -128 and quant_max == 127:
        quantized_np_dtype = types.nptype_from_builtin(types.string_to_builtin("int8"))
    else:
        raise ValueError(
            f"Unsupported quantization range: {quant_min} to {quant_max}.  CoreML only supports 4-bit and 8-bit quantization."
        )

    output = _utils._construct_constexpr_dequant_op(
        int_data.astype(quantized_np_dtype),
        zero_point,
        scale,
        axis=-1,
        name=node.name,
    )
    context.add(output, node.name)


@register_torch_op(
    torch_alias=["quant::dequantize_codebook", "quant.dequantize_codebook"],
    override=False,
)
def dequantize_codebook(context, node):
    inputs = _get_inputs(context, node, expected=[4, 5])
    codes = inputs[0].val
    codebook = inputs[1].val
    nbits = inputs[2].val

    # information in block_size is redundant with codebook.shape
    block_size = inputs[3].val  # noqa: F841

    assert len(codes.shape) == 2, "Only rank 2 inputs are supported"

    # Assert codebook is as expected.  codebook.dim() = codes.dim() + 2
    assert len(codebook.shape) == 4, "Only rank 4 inputs are supported for codebook"
    assert codebook.shape[0] == 1, "Only grouped_channel granularity is supported"
    n_luts = codebook.shape[1]
    assert (
        codes.shape[1] % n_luts == 0
    ), "codes.shape[1] must be divisible by codebook.shape[1]"
    assert codebook.shape[2] == 2**nbits
    assert codebook.shape[3] == 1, "Only scalar look up values are supported"

    if len(inputs) > 4:
        output_dtype = inputs[4].val
        out_np_dtype = NUM_TO_NUMPY_DTYPE[output_dtype]
        _logger.warning(
            f"Core ML ignores output_dtype {out_np_dtype} on torchao.dequantize_affine and instead uses the native precision."
        )

    output = _utils._construct_constexpr_lut_op(
        codes.astype(np.int8),
        codebook,
        name=node.name,
    )
    context.add(output, node.name)
