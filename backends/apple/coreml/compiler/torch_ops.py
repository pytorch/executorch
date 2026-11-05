# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file registers torch ops that are not yet in coremltools, or are in a more recent version of
# coremltools than is used by ExecuTorch.  Each op registered here should have a link to a PR in coremltools that adds
# the op to the coremltools library.

import copy
import functools

import numpy as np
import torch as _torch
from coremltools import _logger, target
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend import _utils
from coremltools.converters.mil.frontend.torch import ops as _ct_ops
from coremltools.converters.mil.frontend.torch.ops import (
    _get_inputs,
    _get_kwinputs,
    _make_fill_op,
    noop,
    NUM_TO_NUMPY_DTYPE,
    NUM_TO_TORCH_DTYPE,
    split,
    to,
)
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    register_torch_op,
)
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.block import (
    is_current_opset_version_compatible_with,
)
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from executorch.exir.dim_order_utils import get_memory_format


_IOS18_QUANT_HINT = (
    "ExecuTorch hint: pass `compile_specs=CoreMLBackend.generate_compile_specs("
    "minimum_deployment_target=ct.target.iOS18)` (or higher) to "
    "`CoreMLPartitioner` when lowering models that use `quantize_(...)`."
)


def _raise_with_executorch_hint(err: Exception) -> "BaseException":
    """Re-raise a coremltools quantization error with ExecuTorch-specific guidance."""
    msg = str(err)
    if "iOS18" in msg or "iOS 18" in msg:
        raise ValueError(f"{msg}\n{_IOS18_QUANT_HINT}") from err
    raise err


# https://github.com/apple/coremltools/pull/2563
@register_torch_op(override=False)
def split_copy(context, node):
    split(context, node)


# https://github.com/apple/coremltools/pull/2863
# coremltools reads approximate only from the positional inputs, but torch.export
# passes it as a keyword, so gelu(approximate="tanh") became the exact gelu.
@register_torch_op(override=True)
def gelu(context, node):
    inputs = _get_inputs(context, node, expected=[1, 2])
    if len(inputs) == 2:
        approximate = inputs[1].val
    else:
        # Read the keyword as is: _get_kwinputs would bind "tanh" to a node named tanh.
        approximate = (node.kwinputs or {}).get("approximate", ["none"])[0]
    mode = {"none": "EXACT", "tanh": "TANH_APPROXIMATION"}[approximate]
    context.add(mb.gelu(x=inputs[0], mode=mode, name=node.name))


def _torch_dtype(node):
    """The dtype torch gives the output of node, if node carries it, as a node of
    an ExportedProgram does."""
    val = (node.meta or {}).get("val")
    return val.dtype if isinstance(val, _torch.Tensor) else None


def _core_ml_type(node):
    """The numpy type Core ML uses for the dtype torch gives the output of node."""
    dtype = _torch_dtype(node)
    if dtype == _torch.bool:
        return np.bool_
    if dtype == _torch.float16:
        return np.float16
    return np.float32 if dtype.is_floating_point else np.int32


def _add_fill(context, node, shape, value):
    context.add(_make_fill_op(shape, _core_ml_type(node)(value), node.name))


def _add_fill_like(context, node, x, value):
    # A fill of a shape known at conversion becomes a constant as big as x in
    # the model, which fill_like avoids from iOS 16.
    if is_current_opset_version_compatible_with(target.iOS16):
        value = _core_ml_type(node)(value)
        context.add(mb.fill_like(ref_tensor=x, value=value, name=node.name))
    else:
        _add_fill(context, node, mb.shape(x=x), value)


# coremltools gives these fills the type of their fill value or of their input,
# or makes them float, instead of the dtype torch gives them, so an int or bool
# fill could come out as float. https://github.com/apple/coremltools/pull/2865
# fixes zeros_like; no upstream issue covers the others yet. For example,
# torch.full((1, 16), 0) is int64 in torch, and coremltools makes it fp32. A
# TorchScript node, which carries no dtype, and a fill value known only at run
# time keep the coremltools converter.
@register_torch_op(override=True)
def full(context, node):
    inputs = _get_inputs(context, node, min_expected=2)
    if _torch_dtype(node) is None or inputs[1].val is None:
        return _ct_ops.full(context, node)
    _add_fill(context, node, inputs[0], inputs[1].val)


@register_torch_op(override=True)
def new_full(context, node):
    inputs = _get_inputs(context, node, min_expected=3)
    if _torch_dtype(node) is None or inputs[2].val is None:
        return _ct_ops.new_full(context, node)
    _add_fill(context, node, inputs[1], inputs[2].val)


def _fill_like(context, node, convert):
    inputs = _get_inputs(context, node, min_expected=2)
    if _torch_dtype(node) is None or inputs[1].val is None:
        return convert(context, node)
    _add_fill_like(context, node, inputs[0], inputs[1].val)


@register_torch_op(override=True)
def full_like(context, node):
    _fill_like(context, node, _ct_ops.full_like)


@register_torch_op(override=True)
def fill(context, node):
    _fill_like(context, node, _ct_ops.fill)


# coremltools converts empty, new_empty and empty_like as zeros, new_zeros and
# zeros_like, so these overrides take their names too.
@register_torch_op(torch_alias=["empty.memory_format"], override=True)
def zeros(context, node):
    if _torch_dtype(node) is None:
        return _ct_ops.zeros(context, node)
    _add_fill(context, node, _get_inputs(context, node, min_expected=1)[0], 0)


@register_torch_op(torch_alias=["new_empty"], override=True)
def new_zeros(context, node):
    if _torch_dtype(node) is None:
        return _ct_ops.new_zeros(context, node)
    _add_fill(context, node, _get_inputs(context, node, min_expected=2)[1], 0)


@register_torch_op(torch_alias=["empty_like"], override=True)
def zeros_like(context, node):
    if _torch_dtype(node) is None:
        return _ct_ops.zeros_like(context, node)
    x = _get_inputs(context, node, min_expected=1)[0]
    _add_fill_like(context, node, x, 0)


@register_torch_op(override=True)
def ones_like(context, node):
    if _torch_dtype(node) is None:
        return _ct_ops.ones_like(context, node)
    x = _get_inputs(context, node, min_expected=1)[0]
    _add_fill_like(context, node, x, 1)


@register_torch_op(override=True)
def scalar_tensor(context, node):
    x = _get_inputs(context, node, expected=[1, 5])[0]
    if _torch_dtype(node) is None or x.val is None:
        return _ct_ops.scalar_tensor(context, node)
    context.add(mb.const(val=_core_ml_type(node)(x.val), name=node.name))


# coremltools writes y[:, i] = v to y[:, i:i + 1] without first wrapping a
# negative i, so for i = -1 it writes to an empty slice. No upstream issue
# covers this yet. For example, y[:, torch.tensor([-1])] = v fails to lower
# with "The update tensor should have shape [3, 0]. Got (3, 1)", and from
# iOS 18 Core ML skips the write when i is an input. A TorchScript node, which
# passes its indices as one list, keeps the coremltools converter.
@register_torch_op(override=True)
def index_put(context, node):
    names = node.inputs[1]
    if not isinstance(names, tuple) or None not in names:
        return _ct_ops.index_put(context, node)
    x = context[node.inputs[0]]
    wrapped = list(names)
    for dim, name in enumerate(names):
        if name is None or not types.is_int(context[name].dtype):
            continue
        index = context[name]
        size = x.shape[dim]
        if is_symbolic(size):
            size = _utils.pymil_value_at(mb.shape(x=x), dim)
        index = mb.select(
            cond=mb.less(x=index, y=0),
            a=mb.add(x=index, y=size),
            b=index,
            name=f"{node.name}_index_{dim}",
        )
        context.add(index)
        wrapped[dim] = index.name
    node = copy.copy(node)
    node.inputs = (node.inputs[0], tuple(wrapped), *node.inputs[2:])
    return _ct_ops.index_put(context, node)


def is_fbcode():
    return not hasattr(_torch.version, "git_version")


if not is_fbcode():
    from coremltools.converters.mil.frontend.torch.dim_order_ops import (
        _empty_dim_order,
        _to_dim_order_copy,
    )
    from coremltools.converters.mil.frontend.torch.ops import rms_norm as _rms_norm

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

    # coremltools 9.0 requires the eps that torch.nn.RMSNorm(eps=None) leaves out.
    # Torch then uses the float32 machine epsilon, also for float16 inputs, while
    # https://github.com/apple/coremltools/pull/2743 would use 1e-5, so keep this
    # override after that change lands.
    @register_torch_op(override=True)
    def rms_norm(context, node):
        inputs = list(node.inputs) + [None] * (4 - len(node.inputs))
        if inputs[3] is None:
            inputs[3] = float(np.finfo(np.float32).eps)
        node = copy.copy(node)
        node.inputs = inputs
        _rms_norm(context, node)

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


@register_torch_op(
    torch_alias=[
        "dim_order_ops::_clone_dim_order",
        "dim_order_ops._clone_dim_order",
    ],
    override=False,
)
def _clone_dim_order(context, node):
    dim_order = _get_kwinputs(context, node, "dim_order", default=[None])[0]
    node.kwinputs.pop("dim_order")

    # In CoreML, dim_order.val will be a ndarray, so we convert it to a list to check memory format.
    dim_order = [int(d) for d in dim_order.val]
    memory_format = get_memory_format(dim_order)
    assert (
        memory_format == _torch.contiguous_format
    ), "Only contiguous memory format is supported in CoreML"

    # Since CoreML only supports contiguous format, no dim_order preservation is needed. Treat this as a no-op clone.
    noop(context, node)


# https://github.com/apple/coremltools/pull/2558
@register_torch_op(
    torch_alias=["torchao::dequantize_affine", "torchao.dequantize_affine"],
    # coremltools did not merge the fix into 9.0 (https://github.com/apple/coremltools/pull/2589),
    # so we override here
    override=True,
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

    try:
        output = _utils._construct_constexpr_dequant_op(
            int_data.astype(quantized_np_dtype),
            zero_point,
            scale,
            name=node.name,
        )
    except ValueError as e:
        _raise_with_executorch_hint(e)
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
    assert (codebook.shape[0] == 1) or (
        codebook.shape[1] == 1
    ), "Only grouped_channel granularity is supported"
    if codebook.shape[0] == 1:
        # LUT is per column group
        n_luts = codebook.shape[1]
        assert (
            codes.shape[1] % n_luts == 0
        ), "codes.shape[1] must be divisible by codebook.shape[1]"
    else:
        # LUT is per row group
        n_luts = codebook.shape[0]
        assert (
            codes.shape[0] % n_luts == 0
        ), "codes.shape[0] must be divisible by codebook.shape[0]"

    assert codebook.shape[2] == 2**nbits
    assert codebook.shape[3] == 1, "Only scalar look up values are supported"

    if len(inputs) > 4:
        output_dtype = inputs[4].val
        out_np_dtype = NUM_TO_NUMPY_DTYPE[output_dtype]
        _logger.warning(
            f"Core ML ignores output_dtype {out_np_dtype} on torchao.dequantize_affine and instead uses the native precision."
        )

    try:
        output = _utils._construct_constexpr_lut_op(
            codes.astype(np.int8),
            codebook,
            name=node.name,
        )
    except ValueError as e:
        _raise_with_executorch_hint(e)
    context.add(output, node.name)


# Torch computes these ops in float for an int or bool input, while coremltools
# passes such an input on to Core ML ops that take only float, except for an int
# input of log and log1p. https://github.com/apple/coremltools/pull/2683 and
# https://github.com/apple/coremltools/pull/2735 cast an int input of
# reciprocal, log2 and log10 after coremltools 9.0; no upstream change covers
# the other ops or a bool input yet. For example, lowering x.long().sqrt() fails
# with "expects tensor or scalar of dtype from type domain ['fp16', 'fp32'] but
# got tensor[2,3,int32]".
def _with_float_input(convert):
    @functools.wraps(convert)
    def converter(context, node):
        x = context[node.inputs[0]]
        if types.is_int(x.dtype) or types.is_bool(x.dtype):
            context.add(mb.cast(x=x, dtype="fp32", name=f"{node.name}_input_fp32"))
            node = copy.copy(node)
            node.inputs = [f"{node.name}_input_fp32", *node.inputs[1:]]
        convert(context, node)

    return converter


for _convert in (
    _ct_ops.acos,
    _ct_ops.asin,
    _ct_ops.atan,
    _ct_ops.atanh,
    _ct_ops.cos,
    _ct_ops.cosh,
    _ct_ops.erf,
    _ct_ops.exp,
    _ct_ops.expm1,
    _ct_ops.log,
    _ct_ops.log1p,
    _ct_ops.log10,
    _ct_ops.log2,
    _ct_ops.reciprocal,
    _ct_ops.rsqrt,
    _ct_ops.sigmoid,
    _ct_ops.sin,
    _ct_ops.sinh,
    _ct_ops.sqrt,
    _ct_ops.tan,
    _ct_ops.tanh,
):
    register_torch_op(_with_float_input(_convert), override=True)


# For an int input, torch returns floor, ceil and round of it unchanged, and
# computes relu and relu6 in int, while coremltools passes the input on to Core
# ML ops that take only float. No upstream issue covers this yet. For example,
# lowering x.long().floor() fails with "expects tensor or scalar of dtype from
# type domain ['fp16', 'fp32'] but got tensor[2,3,int32]".
def _with_int_input(convert, convert_int):
    @functools.wraps(convert)
    def converter(context, node):
        x = context[node.inputs[0]]
        if types.is_int(x.dtype):
            context.add(convert_int(x=x, name=node.name))
        else:
            convert(context, node)

    return converter


for _convert, _convert_int in (
    (_ct_ops.ceil, mb.identity),
    (_ct_ops.floor, mb.identity),
    (_ct_ops.relu, lambda x, name: mb.maximum(x=x, y=0, name=name)),
    (_ct_ops.relu6, lambda x, name: mb.minimum(x=mb.maximum(x=x, y=0), y=6, name=name)),
):
    register_torch_op(_with_int_input(_convert, _convert_int), override=True)
# coremltools registers round as _round with the alias round.
register_torch_op(
    _with_int_input(_ct_ops._round, mb.identity), torch_alias=["round"], override=True
)


# Torch clips the input of hardtanh at its bounds in the dtype of the input,
# while coremltools passes the bounds as they are to clip, which takes only
# float and bounds of the dtype of the input. So nn.ReLU6 on an int tensor, and
# F.relu6(x, inplace=True), which passes int bounds, fail to lower with "the
# named input `alpha` must have the same data type as the named input `x`".
# https://github.com/apple/coremltools/pull/2894 casts the bounds to the dtype of
# a float input; no upstream change covers an int input yet. The dtype comes from
# torch, since coremltools gives some tensors another one, such as int32 to
# torch.arange(4, dtype=torch.float32). A TorchScript node, which carries no
# dtype, keeps the coremltools converter.
@register_torch_op(override=True)
def hardtanh(context, node):
    if _torch_dtype(node) is None:
        return _ct_ops.hardtanh(context, node)
    x, low, high = _get_inputs(context, node, expected=3)
    dtype = types.numpy_type_to_builtin_type(_core_ml_type(node))
    if x.dtype != dtype:
        x = mb.cast(x=x, dtype=types.builtin_to_string(dtype))
    if types.is_int(x.dtype):
        # Torch drops the fraction of each bound for an int input, as int() does.
        clipped = mb.maximum(x=x, y=int(low.val))
        context.add(mb.minimum(x=clipped, y=int(high.val), name=node.name))
    else:
        dtype = types.nptype_from_builtin(x.dtype)
        context.add(
            mb.clip(x=x, alpha=dtype(low.val), beta=dtype(high.val), name=node.name)
        )


# Torch computes clamp_min and clamp_max in the dtype it promotes the tensor and
# the bound to, while coremltools requires the two to share a dtype. So
# x.clamp_min(0) of a float x, and a float bound on an int tensor, fail to lower
# with an AssertionError. No upstream issue covers this yet. A TorchScript node,
# which carries no dtype, keeps the coremltools converter.
def _with_inputs_in_torch_dtype(op, convert):
    @functools.wraps(convert)
    def converter(context, node):
        if _torch_dtype(node) is None:
            return convert(context, node)
        dtype = types.numpy_type_to_builtin_type(_core_ml_type(node))
        x, bound = _get_inputs(context, node, expected=2)
        if x.dtype != dtype:
            x = mb.cast(x=x, dtype=types.builtin_to_string(dtype))
        if bound.dtype != dtype:
            bound = mb.cast(x=bound, dtype=types.builtin_to_string(dtype))
        context.add(op(x=x, y=bound, name=node.name))

    return converter


for _op, _convert in (
    (mb.maximum, _ct_ops.clamp_min),
    (mb.minimum, _ct_ops.clamp_max),
):
    register_torch_op(_with_inputs_in_torch_dtype(_op, _convert), override=True)


# Torch copies src into the dtype of self, while coremltools' copy keeps the
# dtype of src. Lowering turns y[:, i] = v into a copy too, so v kept its dtype
# in y. No upstream issue covers this yet. For example, lowering
# F.gelu(x.clone().copy_(x.long())) fails with "expects tensor or scalar of
# dtype from type domain ['fp16', 'fp32'] but got tensor[2,3,int32]". The dtype
# comes from torch where the node carries it, since coremltools gives some
# tensors another one, such as int32 to torch.arange(4, dtype=torch.float32).
def _copy_into_dtype_of_self(context, node):
    x, src = context[node.inputs[0]], context[node.inputs[1]]
    dtype = x.dtype
    if _torch_dtype(node) is not None:
        dtype = types.numpy_type_to_builtin_type(_core_ml_type(node))
    if src.dtype != dtype:
        name = f"{node.name}_src"
        context.add(mb.cast(x=src, dtype=types.builtin_to_string(dtype), name=name))
        node = copy.copy(node)
        node.inputs = [node.inputs[0], name, *node.inputs[2:]]
    _ct_ops.copy(context, node)


register_torch_op(
    functools.wraps(_ct_ops.copy)(_copy_into_dtype_of_self), override=True
)
