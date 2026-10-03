# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
from typing import Callable, Generic, Optional, TypeVar, Union

import torch
# Registers torch.ops.torchao.(de)quantize_affine used by QuantParamsStruct.
import torchao.quantization.quant_primitives  # noqa: F401
from executorch.backends.fused_quant.ops_utils import (
    compute_conv_out_shape,
    compute_conv_out_shape_nhwc,
)
from executorch.exir.pass_base import ProxyValue
from torch import fx
# Importing this registers the torch.ops.quantized_decomposed.* ops
# (quantize/dequantize per_tensor/per_channel) used by QuantParamsStruct.
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.library import Library, register_fake

_lib = Library("fused_quant", "DEF")

_TensorType = TypeVar(
    "_TensorType", bound=Union[torch.Tensor, "fx.Node", "fx.Proxy", ProxyValue]
)


def _get_tensor_val(
    t: Union[torch.Tensor, fx.Node, fx.Proxy, ProxyValue],
) -> torch.Tensor:
    """Extract the tensor value. For fx.Node, gets the value from meta['val']."""
    if isinstance(t, fx.Node):
        return t.meta["val"]
    if isinstance(t, torch.Tensor):
        return t
    if isinstance(t, ProxyValue):
        return t.data
    if isinstance(t, fx.Proxy):
        return t.node.meta["val"]
    raise TypeError(f"Unsupported type: {type(t)}")


@dataclasses.dataclass(frozen=True)
class QuantParamsStruct(Generic[_TensorType]):
    """Quantization parameters as a dataclass for convenient access.

    Type Parameters:
        _TensorType: The type of scale and zero_point, either torch.Tensor or fx.Node.

    Granularity is encoded entirely by the scale's shape relative to the tensor
    it quantizes -- there is no ``axis``. A scale is either:
      - a singleton (``numel() == 1``): per-tensor; broadcast over the whole
        tensor (block_size == tensor.shape); or
      - a full-rank tensor (``ndim == tensor.ndim``) whose dims encode the tiling:
        ``block_size[i] = tensor.shape[i] // scale.shape[i]``. This covers
        per-channel (exactly one non-unary dim) and per-group / blockwise (more).

    Attributes:
        scale: Scale factor (singleton or full-rank; see above).
        zero_point: Zero point; must have the same shape as scale.
        dtype: The quantized dtype for outputs / the dequantized output dtype for
               inputs.
        quant_min: Minimum quantized value.
        quant_max: Maximum quantized value.
    """

    scale: _TensorType
    zero_point: _TensorType
    dtype: torch.dtype
    quant_min: int
    quant_max: int

    @classmethod
    def maybe_from_flat_args(
        cls,
        scale: Optional[_TensorType],
        zero_point: Optional[_TensorType],
        dtype: torch.dtype,
        quant_min: int,
        quant_max: int,
    ) -> Optional["QuantParamsStruct[_TensorType]"]:
        """Construct from flat args, or return None if scale and zero_point are both None."""
        if scale is None and zero_point is None:
            return None
        if (scale is None) != (zero_point is None):
            raise ValueError(
                "scale and zero_point must both be None or both be provided, "
                f"got scale={'None' if scale is None else 'Tensor'} and "
                f"zero_point={'None' if zero_point is None else 'Tensor'}"
            )
        assert scale is not None and zero_point is not None
        return cls(scale, zero_point, dtype, quant_min, quant_max)

    @staticmethod
    def _non_unary_dims(scale: torch.Tensor) -> list[int]:
        """Indices of the scale dims whose size != 1 (the quantized/tiled dims)."""
        return [i for i, dim in enumerate(scale.shape) if dim != 1]

    def is_per_tensor(self) -> bool:
        return _get_tensor_val(self.scale).numel() == 1

    def is_per_channel(self) -> bool:
        return len(self._non_unary_dims(_get_tensor_val(self.scale))) == 1

    def is_per_group(self) -> bool:
        return len(self._non_unary_dims(_get_tensor_val(self.scale))) >= 2

    def channel_axis(self) -> Optional[int]:
        """The per-channel axis, derived from the scale shape:
        - ``0`` if the scale is a singleton (per-tensor; axis is irrelevant),
        - the single non-unary dim for per-channel,
        - ``None`` if more than one dim is non-unary (blockwise / per-group,
          which has no single channel axis).
        """
        non_unary = self._non_unary_dims(_get_tensor_val(self.scale))
        if len(non_unary) == 0:
            return 0
        if len(non_unary) == 1:
            return non_unary[0]
        return None

    def validate(self) -> None:
        """Validate that scale and zero_point have the same shape."""
        scale = _get_tensor_val(self.scale)
        zero_point = _get_tensor_val(self.zero_point)
        if scale.shape != zero_point.shape:
            raise ValueError(
                f"scale and zero_point must have the same shape, got {scale.shape} and {zero_point.shape}"
            )

    def _broadcast_scale_zp(self, ndim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (scale, zero_point) at full rank ``ndim`` for affine quant.

        A singleton (per-tensor) is reshaped to all-ones so it broadcasts over the
        whole tensor; a full-rank scale is returned as-is (its shape encodes the
        block layout). Any other rank is rejected -- non-singleton scales must be
        full-rank.
        """
        scale = _get_tensor_val(self.scale)
        zero_point = _get_tensor_val(self.zero_point)
        if scale.numel() == 1:
            return scale.reshape([1] * ndim), zero_point.reshape([1] * ndim)
        if scale.ndim == ndim:
            return scale, zero_point
        raise ValueError(
            f"scale rank {scale.ndim} must be a singleton or match the tensor rank "
            f"{ndim}; per-channel/group scales must be full-rank so their shape "
            "encodes the block layout"
        )

    @staticmethod
    def _block_size(tensor: torch.Tensor, scale: torch.Tensor) -> list[int]:
        """Derive the affine block_size: each tile is tensor.shape[i] // scale.shape[i]."""
        block_size: list[int] = []
        for tdim, sdim in zip(tensor.shape, scale.shape):
            if tdim % sdim != 0:
                raise ValueError(
                    f"tensor dim {tdim} must be divisible by scale dim {sdim}"
                )
            block_size.append(tdim // sdim)
        return block_size

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize ``tensor`` via affine (block-wise) quantization. The block
        layout is derived from the scale shape: block_size[i] = tensor.shape[i] //
        scale.shape[i] (covering per-tensor/channel/group/element uniformly)."""
        assert isinstance(self.scale, torch.Tensor)
        assert isinstance(self.zero_point, torch.Tensor)
        scale, zero_point = self._broadcast_scale_zp(tensor.ndim)
        block_size = self._block_size(tensor, scale)
        return torch.ops.torchao.quantize_affine(
            tensor,
            block_size,
            scale,
            zero_point,
            self.dtype,
            self.quant_min,
            self.quant_max,
        )

    def dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Dequantize ``tensor`` via affine (block-wise) dequantization."""
        assert isinstance(self.scale, torch.Tensor)
        assert isinstance(self.zero_point, torch.Tensor)
        scale, zero_point = self._broadcast_scale_zp(tensor.ndim)
        block_size = self._block_size(tensor, scale)
        return torch.ops.torchao.dequantize_affine(
            tensor,
            block_size,
            scale,
            zero_point,
            tensor.dtype,
            self.quant_min,
            self.quant_max,
            output_dtype=self.dtype,
        )


AnyQuantParamsStruct = Union[
    QuantParamsStruct[torch.Tensor], QuantParamsStruct[fx.Node]
]


def _validate_qparams(*qparams_list: Optional[AnyQuantParamsStruct]) -> None:
    """Validate every qparams block that is present."""
    for qp in qparams_list:
        if qp is not None:
            qp.validate()


def get_out_dtype(
    out_qparams: Optional[AnyQuantParamsStruct], default_dtype: torch.dtype
) -> torch.dtype:
    if out_qparams is None:
        return default_dtype
    return out_qparams.dtype


def _maybe_quantize(
    tensor: torch.Tensor, qparams: Optional[AnyQuantParamsStruct]
) -> torch.Tensor:
    if qparams is None:
        return tensor
    return qparams.quantize(tensor)


def _maybe_dequantize(
    tensor: torch.Tensor, qparams: Optional[AnyQuantParamsStruct]
) -> torch.Tensor:
    if qparams is None:
        return tensor
    return qparams.dequantize(tensor)


_QP = Optional[QuantParamsStruct[torch.Tensor]]


def _make_qp(
    scale: Optional[torch.Tensor],
    zero_point: Optional[torch.Tensor],
    dtype: torch.dtype,
    quant_min: int,
    quant_max: int,
) -> _QP:
    return QuantParamsStruct.maybe_from_flat_args(
        scale, zero_point, dtype, quant_min, quant_max
    )


def _permute_contiguous(tensor: torch.Tensor, dims: list[int]) -> torch.Tensor:
    return tensor.permute(*dims).clone(memory_format=torch.contiguous_format)


def _permute_qparams(
    qparams: _QP,
    dims: list[int],
) -> _QP:
    """Permute layout-sensitive qparams with their tensor."""
    if qparams is None or qparams.is_per_tensor():
        return qparams
    assert isinstance(qparams.scale, torch.Tensor)
    assert isinstance(qparams.zero_point, torch.Tensor)
    return dataclasses.replace(
        qparams,
        scale=_permute_contiguous(qparams.scale, dims),
        zero_point=_permute_contiguous(qparams.zero_point, dims),
    )


def _binary_op_impl(
    inp: torch.Tensor,
    other: torch.Tensor,
    inp_qp: _QP,
    other_qp: _QP,
    out_qp: _QP,
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    _validate_qparams(inp_qp, other_qp, out_qp)
    dq_inp = _maybe_dequantize(inp, inp_qp)
    dq_other = _maybe_dequantize(other, other_qp)
    out = fn(dq_inp, dq_other)
    return _maybe_quantize(out, out_qp)


def _binary_op_meta(
    inp: torch.Tensor,
    other: torch.Tensor,
    inp_qp: _QP,
    other_qp: _QP,
    out_qp: _QP,
) -> torch.Tensor:
    _validate_qparams(inp_qp, other_qp, out_qp)
    out_size = torch.broadcast_shapes(inp.shape, other.shape)
    return inp.new_empty(out_size, dtype=get_out_dtype(out_qp, inp.dtype))


def _unary_op_impl(
    inp: torch.Tensor,
    inp_qp: _QP,
    out_qp: _QP,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    _validate_qparams(inp_qp, out_qp)
    dq_inp = _maybe_dequantize(inp, inp_qp)
    out = fn(dq_inp)
    return _maybe_quantize(out, out_qp)


def _unary_op_meta(
    inp: torch.Tensor,
    inp_qp: _QP,
    out_qp: _QP,
) -> torch.Tensor:
    _validate_qparams(inp_qp, out_qp)
    return inp.new_empty(inp.shape, dtype=get_out_dtype(out_qp, inp.dtype))


def _binary_scalar_op_impl(
    inp: torch.Tensor,
    inp_qp: _QP,
    out_qp: _QP,
    other: float,
    fn: Callable[[torch.Tensor, float], torch.Tensor],
) -> torch.Tensor:
    """Binary op with a scalar second operand. Only the tensor input and the
    output carry quant params; the scalar is applied directly to the dequantized
    input (it has no qparams of its own)."""
    _validate_qparams(inp_qp, out_qp)
    dq_inp = _maybe_dequantize(inp, inp_qp)
    out = fn(dq_inp, other)
    return _maybe_quantize(out, out_qp)


def _binary_scalar_op_meta(
    inp: torch.Tensor,
    inp_qp: _QP,
    out_qp: _QP,
) -> torch.Tensor:
    # A scalar second operand does not broadcast, so the output keeps inp's shape.
    _validate_qparams(inp_qp, out_qp)
    return inp.new_empty(inp.shape, dtype=get_out_dtype(out_qp, inp.dtype))


# =============================================================================
# Op definitions
# =============================================================================


@torch.library.custom_op("fused_quant::requantize", mutates_args=())
def _requantize_impl(
    inp: torch.Tensor,
    inp_scale: torch.Tensor,
    inp_zero_point: torch.Tensor,
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: torch.Tensor,
    out_zero_point: torch.Tensor,
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    inp_qp = QuantParamsStruct(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = QuantParamsStruct(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, out_qp)
    return out_qp.quantize(inp_qp.dequantize(inp))


def _requantize_meta(
    inp: torch.Tensor,
    inp_scale: torch.Tensor,
    inp_zero_point: torch.Tensor,
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: torch.Tensor,
    out_zero_point: torch.Tensor,
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    inp_qp = QuantParamsStruct(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = QuantParamsStruct(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, out_qp)
    return inp.new_empty(inp.shape, dtype=out_qp.dtype)


_requantize_impl.register_fake(_requantize_meta)


@torch.library.custom_op("fused_quant::bmm", mutates_args=())
def _bmm_impl(
    inp: torch.Tensor,
    other: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    other_scale: Optional[torch.Tensor],
    other_zero_point: Optional[torch.Tensor],
    other_dtype: torch.dtype,
    other_quant_min: int,
    other_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _binary_op_impl(
        inp,
        other,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(
            other_scale,
            other_zero_point,
            other_dtype,
            other_quant_min,
            other_quant_max,
        ),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        fn=torch.bmm,
    )


def _bmm_meta(
    inp: torch.Tensor,
    other: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    other_scale: Optional[torch.Tensor],
    other_zero_point: Optional[torch.Tensor],
    other_dtype: torch.dtype,
    other_quant_min: int,
    other_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    other_qp = _make_qp(
        other_scale,
        other_zero_point,
        other_dtype,
        other_quant_min,
        other_quant_max,
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )

    _validate_qparams(inp_qp, other_qp, out_qp)

    if inp.ndim != 3 or other.ndim != 3:
        raise ValueError(
            f"Input tensors must be 3D for bmm, got {inp.ndim}D and {other.ndim}D"
        )
    if inp.shape[0] != other.shape[0]:
        raise ValueError(
            f"Input tensors must have the same batch dimension, got {inp.shape[0]} and {other.shape[0]}"
        )
    if inp.shape[2] != other.shape[1]:
        raise ValueError(
            f"Input tensors must have the same inner dimension, got {inp.shape[2]} and {other.shape[1]}"
        )

    out_shape = (inp.shape[0], inp.shape[1], other.shape[2])
    return inp.new_empty(out_shape, dtype=get_out_dtype(out_qp, inp.dtype))


_bmm_impl.register_fake(_bmm_meta)


@torch.library.custom_op("fused_quant::relu", mutates_args=())
def _relu_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _unary_op_impl(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        fn=torch.nn.functional.relu,
    )


def _relu_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _unary_op_meta(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_relu_impl.register_fake(_relu_meta)


@torch.library.custom_op("fused_quant::hardswish", mutates_args=())
def _hardswish_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _unary_op_impl(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        fn=torch.nn.functional.hardswish,
    )


def _hardswish_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _unary_op_meta(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_hardswish_impl.register_fake(_hardswish_meta)


@torch.library.custom_op("fused_quant::sigmoid", mutates_args=())
def _sigmoid_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _unary_op_impl(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        fn=torch.sigmoid,
    )


def _sigmoid_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _unary_op_meta(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_sigmoid_impl.register_fake(_sigmoid_meta)


@torch.library.custom_op("fused_quant::tanh", mutates_args=())
def _tanh_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _unary_op_impl(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        fn=torch.tanh,
    )


def _tanh_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _unary_op_meta(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_tanh_impl.register_fake(_tanh_meta)


@torch.library.custom_op("fused_quant::hard_tanh", mutates_args=())
def _hard_tanh_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    min_val: float = -1.0,
    max_val: float = 1.0,
) -> torch.Tensor:
    return _unary_op_impl(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        fn=lambda x: torch.nn.functional.hardtanh(x, min_val, max_val),
    )


def _hard_tanh_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    min_val: float = -1.0,
    max_val: float = 1.0,
) -> torch.Tensor:
    return _unary_op_meta(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_hard_tanh_impl.register_fake(_hard_tanh_meta)


@torch.library.custom_op("fused_quant::silu", mutates_args=())
def _silu_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _unary_op_impl(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        fn=torch.nn.functional.silu,
    )


def _silu_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _unary_op_meta(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_silu_impl.register_fake(_silu_meta)


@torch.library.custom_op("fused_quant::hardsigmoid", mutates_args=())
def _hardsigmoid_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _unary_op_impl(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        fn=torch.nn.functional.hardsigmoid,
    )


def _hardsigmoid_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _unary_op_meta(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_hardsigmoid_impl.register_fake(_hardsigmoid_meta)


@torch.library.custom_op("fused_quant::gelu", mutates_args=())
def _gelu_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    approximate: str,
) -> torch.Tensor:
    return _unary_op_impl(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        fn=lambda x: torch.nn.functional.gelu(x, approximate=approximate),
    )


def _gelu_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    approximate: str,
) -> torch.Tensor:
    return _unary_op_meta(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_gelu_impl.register_fake(_gelu_meta)


@torch.library.custom_op("fused_quant::add", mutates_args=())
def _add_impl(
    inp: torch.Tensor,
    other: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    other_scale: Optional[torch.Tensor],
    other_zero_point: Optional[torch.Tensor],
    other_dtype: torch.dtype,
    other_quant_min: int,
    other_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    return _binary_op_impl(
        inp,
        other,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(
            other_scale,
            other_zero_point,
            other_dtype,
            other_quant_min,
            other_quant_max,
        ),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        fn=lambda x, y: torch.add(x, y, alpha=alpha),
    )


def _add_meta(
    inp: torch.Tensor,
    other: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    other_scale: Optional[torch.Tensor],
    other_zero_point: Optional[torch.Tensor],
    other_dtype: torch.dtype,
    other_quant_min: int,
    other_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    return _binary_op_meta(
        inp,
        other,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(
            other_scale,
            other_zero_point,
            other_dtype,
            other_quant_min,
            other_quant_max,
        ),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_add_impl.register_fake(_add_meta)


@torch.library.custom_op("fused_quant::mul", mutates_args=())
def _mul_impl(
    inp: torch.Tensor,
    other: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    other_scale: Optional[torch.Tensor],
    other_zero_point: Optional[torch.Tensor],
    other_dtype: torch.dtype,
    other_quant_min: int,
    other_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _binary_op_impl(
        inp,
        other,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(
            other_scale,
            other_zero_point,
            other_dtype,
            other_quant_min,
            other_quant_max,
        ),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        fn=torch.mul,
    )


def _mul_meta(
    inp: torch.Tensor,
    other: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    other_scale: Optional[torch.Tensor],
    other_zero_point: Optional[torch.Tensor],
    other_dtype: torch.dtype,
    other_quant_min: int,
    other_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    return _binary_op_meta(
        inp,
        other,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(
            other_scale,
            other_zero_point,
            other_dtype,
            other_quant_min,
            other_quant_max,
        ),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_mul_impl.register_fake(_mul_meta)


@torch.library.custom_op("fused_quant::sub", mutates_args=())
def _sub_impl(
    inp: torch.Tensor,
    other: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    other_scale: Optional[torch.Tensor],
    other_zero_point: Optional[torch.Tensor],
    other_dtype: torch.dtype,
    other_quant_min: int,
    other_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    return _binary_op_impl(
        inp,
        other,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(
            other_scale,
            other_zero_point,
            other_dtype,
            other_quant_min,
            other_quant_max,
        ),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        fn=lambda x, y: torch.sub(x, y, alpha=alpha),
    )


def _sub_meta(
    inp: torch.Tensor,
    other: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    other_scale: Optional[torch.Tensor],
    other_zero_point: Optional[torch.Tensor],
    other_dtype: torch.dtype,
    other_quant_min: int,
    other_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    return _binary_op_meta(
        inp,
        other,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(
            other_scale,
            other_zero_point,
            other_dtype,
            other_quant_min,
            other_quant_max,
        ),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_sub_impl.register_fake(_sub_meta)


# =============================================================================
# Scalar binary ops
#
# Variants of add/mul whose second operand is a Python scalar rather than a
# tensor (e.g. RMSNorm's `x + eps`). The scalar has no quant params, so the
# schema carries qparams only for the tensor input and the output. The fusion
# pass lays args out as (tensor_input, input_qparams..., output_qparams...,
# extra_args...), so the scalar (and `alpha` for add) lands at the very end.
#
# These are registered as `.Scalar` overloads of the existing add/mul ops
# (mirroring aten's `add.Tensor`/`add.Scalar` split) rather than the
# `@custom_op` decorator, which only ever defines a `.default` overload. The
# functional op therefore goes through the lower-level Library API: an explicit
# schema, a CompositeExplicitAutograd impl, and a separately registered fake.
# =============================================================================


def _add_scalar_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    other: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    return _binary_scalar_op_impl(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        other,
        fn=lambda x, y: torch.add(x, y, alpha=alpha),
    )


def _add_scalar_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    other: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    return _binary_scalar_op_meta(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_lib.define(
    "add.Scalar(Tensor inp, "
    "Tensor? inp_scale, Tensor? inp_zero_point, ScalarType inp_dtype, "
    "SymInt inp_quant_min, SymInt inp_quant_max, "
    "Tensor? out_scale, Tensor? out_zero_point, ScalarType out_dtype, "
    "SymInt out_quant_min, SymInt out_quant_max, "
    "Scalar other, float alpha = 1.0) -> Tensor"
)
_lib.impl("add.Scalar", _add_scalar_impl, "CompositeExplicitAutograd")
register_fake("fused_quant::add.Scalar", _add_scalar_meta, lib=_lib)


def _mul_scalar_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    other: float,
) -> torch.Tensor:
    return _binary_scalar_op_impl(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        other,
        fn=torch.mul,
    )


def _mul_scalar_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    other: float,
) -> torch.Tensor:
    return _binary_scalar_op_meta(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_lib.define(
    "mul.Scalar(Tensor inp, "
    "Tensor? inp_scale, Tensor? inp_zero_point, ScalarType inp_dtype, "
    "SymInt inp_quant_min, SymInt inp_quant_max, "
    "Tensor? out_scale, Tensor? out_zero_point, ScalarType out_dtype, "
    "SymInt out_quant_min, SymInt out_quant_max, "
    "Scalar other) -> Tensor"
)
_lib.impl("mul.Scalar", _mul_scalar_impl, "CompositeExplicitAutograd")
register_fake("fused_quant::mul.Scalar", _mul_scalar_meta, lib=_lib)


@torch.library.custom_op("fused_quant::sub.Scalar", mutates_args=())
def _sub_scalar_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    other: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    return _binary_scalar_op_impl(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
        other,
        fn=lambda x, y: torch.sub(x, y, alpha=alpha),
    )


def _sub_scalar_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    other: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    return _binary_scalar_op_meta(
        inp,
        _make_qp(inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max),
        _make_qp(out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max),
    )


_sub_scalar_impl.register_fake(_sub_scalar_meta)


@torch.library.custom_op("fused_quant::linear", mutates_args=())
def _linear_impl(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    weight_scale: Optional[torch.Tensor],
    weight_zero_point: Optional[torch.Tensor],
    weight_dtype: torch.dtype,
    weight_quant_min: int,
    weight_quant_max: int,
    bias_scale: Optional[torch.Tensor],
    bias_zero_point: Optional[torch.Tensor],
    bias_dtype: torch.dtype,
    bias_quant_min: int,
    bias_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    weight_qp = _make_qp(
        weight_scale,
        weight_zero_point,
        weight_dtype,
        weight_quant_min,
        weight_quant_max,
    )
    bias_qp = _make_qp(
        bias_scale,
        bias_zero_point,
        bias_dtype,
        bias_quant_min,
        bias_quant_max,
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )

    _validate_qparams(inp_qp, weight_qp, bias_qp, out_qp)

    dq_inp = _maybe_dequantize(inp, inp_qp)
    dq_weight = _maybe_dequantize(weight, weight_qp)
    dq_bias = _maybe_dequantize(bias, bias_qp) if bias is not None else None

    dq_out = torch.nn.functional.linear(dq_inp, dq_weight, dq_bias)
    return _maybe_quantize(dq_out, out_qp)


def _linear_meta(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    weight_scale: Optional[torch.Tensor],
    weight_zero_point: Optional[torch.Tensor],
    weight_dtype: torch.dtype,
    weight_quant_min: int,
    weight_quant_max: int,
    bias_scale: Optional[torch.Tensor],
    bias_zero_point: Optional[torch.Tensor],
    bias_dtype: torch.dtype,
    bias_quant_min: int,
    bias_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    weight_qp = _make_qp(
        weight_scale,
        weight_zero_point,
        weight_dtype,
        weight_quant_min,
        weight_quant_max,
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, weight_qp, out_qp)
    output_shape = (*inp.shape[:-1], weight.shape[0])
    return inp.new_empty(output_shape, dtype=get_out_dtype(out_qp, inp.dtype))


_linear_impl.register_fake(_linear_meta)


@torch.library.custom_op("fused_quant::convolution", mutates_args=())
def _convolution_impl(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    weight_scale: Optional[torch.Tensor],
    weight_zero_point: Optional[torch.Tensor],
    weight_dtype: torch.dtype,
    weight_quant_min: int,
    weight_quant_max: int,
    bias_scale: Optional[torch.Tensor],
    bias_zero_point: Optional[torch.Tensor],
    bias_dtype: torch.dtype,
    bias_quant_min: int,
    bias_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    weight_qp = _make_qp(
        weight_scale,
        weight_zero_point,
        weight_dtype,
        weight_quant_min,
        weight_quant_max,
    )
    bias_qp = _make_qp(
        bias_scale,
        bias_zero_point,
        bias_dtype,
        bias_quant_min,
        bias_quant_max,
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )

    _validate_qparams(inp_qp, weight_qp, bias_qp, out_qp)

    num_spatial_dims = len(stride)
    assert len(padding) == num_spatial_dims
    assert len(dilation) == num_spatial_dims
    assert len(output_padding) == num_spatial_dims

    dq_inp = _maybe_dequantize(inp, inp_qp)
    dq_weight = _maybe_dequantize(weight, weight_qp)
    dq_bias = _maybe_dequantize(bias, bias_qp) if bias is not None else None

    if num_spatial_dims not in (1, 2, 3):
        raise ValueError(
            f"Unsupported number of spatial dimensions: {num_spatial_dims}. "
            "Only 1D, 2D, and 3D convolutions are supported."
        )

    dq_out = torch.ops.aten.convolution.default(
        dq_inp,
        dq_weight,
        dq_bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    return _maybe_quantize(dq_out, out_qp)


def _convolution_meta(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    weight_scale: Optional[torch.Tensor],
    weight_zero_point: Optional[torch.Tensor],
    weight_dtype: torch.dtype,
    weight_quant_min: int,
    weight_quant_max: int,
    bias_scale: Optional[torch.Tensor],
    bias_zero_point: Optional[torch.Tensor],
    bias_dtype: torch.dtype,
    bias_quant_min: int,
    bias_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    weight_qp = _make_qp(
        weight_scale,
        weight_zero_point,
        weight_dtype,
        weight_quant_min,
        weight_quant_max,
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, weight_qp, out_qp)
    output_shape = compute_conv_out_shape(
        inp,
        weight,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    return inp.new_empty(output_shape, dtype=get_out_dtype(out_qp, inp.dtype))


_convolution_impl.register_fake(_convolution_meta)


def _validate_conv_nd_args(
    name: str,
    spatial_dims: int,
    inp: torch.Tensor,
    weight: torch.Tensor,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
) -> None:
    expected_rank = spatial_dims + 2
    if inp.ndim != expected_rank or weight.ndim != expected_rank:
        raise ValueError(
            f"fused_quant::{name} expects input and weight rank {expected_rank}, "
            f"got {inp.ndim} and {weight.ndim}"
        )
    for arg_name, values in (
        ("stride", stride),
        ("padding", padding),
        ("dilation", dilation),
    ):
        if len(values) != spatial_dims:
            raise ValueError(
                f"fused_quant::{name} expects {arg_name} to contain "
                f"{spatial_dims} values, got {len(values)}"
            )


def _register_conv_nd(name: str, spatial_dims: int) -> torch.library.CustomOpDef:
    def impl(
        inp: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        inp_scale: Optional[torch.Tensor],
        inp_zero_point: Optional[torch.Tensor],
        inp_dtype: torch.dtype,
        inp_quant_min: int,
        inp_quant_max: int,
        weight_scale: Optional[torch.Tensor],
        weight_zero_point: Optional[torch.Tensor],
        weight_dtype: torch.dtype,
        weight_quant_min: int,
        weight_quant_max: int,
        bias_scale: Optional[torch.Tensor],
        bias_zero_point: Optional[torch.Tensor],
        bias_dtype: torch.dtype,
        bias_quant_min: int,
        bias_quant_max: int,
        out_scale: Optional[torch.Tensor],
        out_zero_point: Optional[torch.Tensor],
        out_dtype: torch.dtype,
        out_quant_min: int,
        out_quant_max: int,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
    ) -> torch.Tensor:
        _validate_conv_nd_args(
            name, spatial_dims, inp, weight, stride, padding, dilation
        )
        return _convolution_impl(
            inp,
            weight,
            bias,
            inp_scale,
            inp_zero_point,
            inp_dtype,
            inp_quant_min,
            inp_quant_max,
            weight_scale,
            weight_zero_point,
            weight_dtype,
            weight_quant_min,
            weight_quant_max,
            bias_scale,
            bias_zero_point,
            bias_dtype,
            bias_quant_min,
            bias_quant_max,
            out_scale,
            out_zero_point,
            out_dtype,
            out_quant_min,
            out_quant_max,
            stride,
            padding,
            dilation,
            False,
            [0] * spatial_dims,
            groups,
        )

    def meta(
        inp: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        inp_scale: Optional[torch.Tensor],
        inp_zero_point: Optional[torch.Tensor],
        inp_dtype: torch.dtype,
        inp_quant_min: int,
        inp_quant_max: int,
        weight_scale: Optional[torch.Tensor],
        weight_zero_point: Optional[torch.Tensor],
        weight_dtype: torch.dtype,
        weight_quant_min: int,
        weight_quant_max: int,
        bias_scale: Optional[torch.Tensor],
        bias_zero_point: Optional[torch.Tensor],
        bias_dtype: torch.dtype,
        bias_quant_min: int,
        bias_quant_max: int,
        out_scale: Optional[torch.Tensor],
        out_zero_point: Optional[torch.Tensor],
        out_dtype: torch.dtype,
        out_quant_min: int,
        out_quant_max: int,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
    ) -> torch.Tensor:
        _validate_conv_nd_args(
            name, spatial_dims, inp, weight, stride, padding, dilation
        )
        return _convolution_meta(
            inp,
            weight,
            bias,
            inp_scale,
            inp_zero_point,
            inp_dtype,
            inp_quant_min,
            inp_quant_max,
            weight_scale,
            weight_zero_point,
            weight_dtype,
            weight_quant_min,
            weight_quant_max,
            bias_scale,
            bias_zero_point,
            bias_dtype,
            bias_quant_min,
            bias_quant_max,
            out_scale,
            out_zero_point,
            out_dtype,
            out_quant_min,
            out_quant_max,
            stride,
            padding,
            dilation,
            False,
            [0] * spatial_dims,
            groups,
        )

    op = torch.library.custom_op(f"fused_quant::{name}", impl, mutates_args=())
    op.register_fake(meta)
    return op


_conv1d_impl: torch.library.CustomOpDef = _register_conv_nd("conv1d", 1)
_conv2d_impl: torch.library.CustomOpDef = _register_conv_nd("conv2d", 2)
_conv3d_impl: torch.library.CustomOpDef = _register_conv_nd("conv3d", 3)


def _max_pool2d_with_indices(
    inp: torch.Tensor,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    ceil_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.aten.max_pool2d_with_indices.default(
        inp,
        kernel_size,
        stride or kernel_size,
        padding,
        dilation,
        ceil_mode,
    )


@torch.library.custom_op("fused_quant::max_pool2d_with_indices", mutates_args=())
def _max_pool2d_with_indices_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    ceil_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, out_qp)
    values, indices = _max_pool2d_with_indices(
        _maybe_dequantize(inp, inp_qp),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
    return _maybe_quantize(values, out_qp), indices


def _max_pool2d_with_indices_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    ceil_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, out_qp)
    values, indices = _max_pool2d_with_indices(
        torch.ones_like(inp, dtype=torch.float32),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
    return (
        values.new_empty(values.shape, dtype=get_out_dtype(out_qp, inp.dtype)),
        indices.new_empty(indices.shape),
    )


_max_pool2d_with_indices_impl.register_fake(_max_pool2d_with_indices_meta)


@torch.library.custom_op(
    "fused_quant::max_pool2d_with_indices_channels_last", mutates_args=()
)
def _max_pool2d_with_indices_channels_last_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    ceil_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, out_qp)
    inp_nchw = _permute_contiguous(inp, [0, 3, 1, 2])
    inp_qp = _permute_qparams(inp_qp, [0, 3, 1, 2])
    values_nchw, indices_nchw = _max_pool2d_with_indices(
        _maybe_dequantize(inp_nchw, inp_qp),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
    values = _permute_contiguous(values_nchw, [0, 2, 3, 1])
    indices = _permute_contiguous(indices_nchw, [0, 2, 3, 1])
    return _maybe_quantize(values, out_qp), indices


def _max_pool2d_with_indices_channels_last_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    ceil_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, out_qp)
    values_nchw, indices_nchw = _max_pool2d_with_indices(
        inp.permute(0, 3, 1, 2),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
    values = values_nchw.permute(0, 2, 3, 1)
    indices = indices_nchw.permute(0, 2, 3, 1)
    return (
        values.new_empty(values.shape, dtype=get_out_dtype(out_qp, inp.dtype)),
        indices.new_empty(indices.shape),
    )


_max_pool2d_with_indices_channels_last_impl.register_fake(
    _max_pool2d_with_indices_channels_last_meta
)


def _avg_pool2d(
    inp: torch.Tensor,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
) -> torch.Tensor:
    return torch.ops.aten.avg_pool2d.default(
        inp,
        kernel_size,
        stride or kernel_size,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


@torch.library.custom_op("fused_quant::avg_pool2d", mutates_args=())
def _avg_pool2d_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, out_qp)
    output = _avg_pool2d(
        _maybe_dequantize(inp, inp_qp),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    return _maybe_quantize(output, out_qp)


def _avg_pool2d_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, out_qp)

    output = _avg_pool2d(
        # Doesn't matter if we are passing in float32 input, we just need
        # a valid dtype to pass in so we can get output shape, overriding
        # correct dtype later with get_out_dtype
        torch.ones_like(inp, dtype=torch.float32),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    return output.new_empty(output.shape, dtype=get_out_dtype(out_qp, inp.dtype))


_avg_pool2d_impl.register_fake(_avg_pool2d_meta)


@torch.library.custom_op("fused_quant::avg_pool2d_channels_last", mutates_args=())
def _avg_pool2d_channels_last_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, out_qp)
    inp_nchw = _permute_contiguous(inp, [0, 3, 1, 2])
    inp_qp = _permute_qparams(inp_qp, [0, 3, 1, 2])
    output_nchw = _avg_pool2d(
        _maybe_dequantize(inp_nchw, inp_qp),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    output = _permute_contiguous(output_nchw, [0, 2, 3, 1])
    return _maybe_quantize(output, out_qp)


def _avg_pool2d_channels_last_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, out_qp)
    output_nchw = _avg_pool2d(
        torch.ones_like(inp, dtype=torch.float32).permute(0, 3, 1, 2),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    output = output_nchw.permute(0, 2, 3, 1)
    return output.new_empty(output.shape, dtype=get_out_dtype(out_qp, inp.dtype))


_avg_pool2d_channels_last_impl.register_fake(_avg_pool2d_channels_last_meta)


@torch.library.custom_op("fused_quant::convolution_channels_last", mutates_args=())
def _convolution_channels_last_impl(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    weight_scale: Optional[torch.Tensor],
    weight_zero_point: Optional[torch.Tensor],
    weight_dtype: torch.dtype,
    weight_quant_min: int,
    weight_quant_max: int,
    bias_scale: Optional[torch.Tensor],
    bias_zero_point: Optional[torch.Tensor],
    bias_dtype: torch.dtype,
    bias_quant_min: int,
    bias_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    weight_qp = _make_qp(
        weight_scale,
        weight_zero_point,
        weight_dtype,
        weight_quant_min,
        weight_quant_max,
    )
    bias_qp = _make_qp(
        bias_scale,
        bias_zero_point,
        bias_dtype,
        bias_quant_min,
        bias_quant_max,
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )

    _validate_qparams(inp_qp, weight_qp, bias_qp, out_qp)

    num_spatial_dims = len(stride)
    assert len(padding) == num_spatial_dims
    assert len(dilation) == num_spatial_dims
    assert len(output_padding) == num_spatial_dims

    nhwc_to_nchw = [0, num_spatial_dims + 1] + list(range(1, num_spatial_dims + 1))
    nchw_to_nhwc = [0] + list(range(2, num_spatial_dims + 2)) + [1]
    ohwi_to_oihw = [0, num_spatial_dims + 1] + list(range(1, num_spatial_dims + 1))

    inp_nchw = _permute_contiguous(inp, nhwc_to_nchw)
    weight_oihw = _permute_contiguous(weight, ohwi_to_oihw)
    inp_qp = _permute_qparams(inp_qp, nhwc_to_nchw)
    weight_qp = _permute_qparams(weight_qp, ohwi_to_oihw)

    dq_inp = _maybe_dequantize(inp_nchw, inp_qp)
    dq_weight = _maybe_dequantize(weight_oihw, weight_qp)
    dq_bias = _maybe_dequantize(bias, bias_qp) if bias is not None else None

    if num_spatial_dims not in (1, 2, 3):
        raise ValueError(
            f"Unsupported number of spatial dimensions: {num_spatial_dims}. "
            "Only 1D, 2D, and 3D convolutions are supported."
        )

    dq_out_nchw = torch.ops.aten.convolution.default(
        dq_inp,
        dq_weight,
        dq_bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    dq_out = _permute_contiguous(dq_out_nchw, nchw_to_nhwc)
    return _maybe_quantize(dq_out, out_qp)


def _convolution_channels_last_meta(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    weight_scale: Optional[torch.Tensor],
    weight_zero_point: Optional[torch.Tensor],
    weight_dtype: torch.dtype,
    weight_quant_min: int,
    weight_quant_max: int,
    bias_scale: Optional[torch.Tensor],
    bias_zero_point: Optional[torch.Tensor],
    bias_dtype: torch.dtype,
    bias_quant_min: int,
    bias_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    weight_qp = _make_qp(
        weight_scale,
        weight_zero_point,
        weight_dtype,
        weight_quant_min,
        weight_quant_max,
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, weight_qp, out_qp)
    output_shape = compute_conv_out_shape_nhwc(
        inp,
        weight,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    return inp.new_empty(output_shape, dtype=get_out_dtype(out_qp, inp.dtype))


_convolution_channels_last_impl.register_fake(_convolution_channels_last_meta)


# =============================================================================
# Multi-output ops
# =============================================================================
#
# Multi-output ops return a tuple. Unlike the single-output ops above, only a
# subset of their outputs are quantizable, and the op schema only carries output
# qparams blocks for those outputs. ``native_layer_norm`` returns
# ``(out, mean, rstd)`` but only ``out`` (output 0) is ever quantized; ``mean``
# and ``rstd`` are always returned in float. Accordingly the schema has exactly
# one output qparams block (for output 0), and the input activation is the only
# quantizable input -- ``weight``/``bias`` pass through in float.


@torch.library.custom_op("fused_quant::native_layer_norm", mutates_args=())
def _native_layer_norm_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )

    _validate_qparams(inp_qp, out_qp)

    dq_inp = _maybe_dequantize(inp, inp_qp)
    out, mean, rstd = torch.ops.aten.native_layer_norm.default(
        dq_inp, normalized_shape, weight, bias, eps
    )
    # Only output 0 is (optionally) quantized; mean/rstd stay float.
    out = _maybe_quantize(out, out_qp)
    return out, mean, rstd


def _native_layer_norm_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, out_qp)

    # Mirror the impl exactly: the norm runs on the dequantized input, so out0
    # (when unquantized) and mean/rstd take their dtypes straight from
    # aten.native_layer_norm. ``compute_dtype`` is the dequantized input's float
    # dtype -- ``inp_qp.dtype`` (the upstream dequant's output dtype, not always
    # fp32), or the input dtype when the input isn't quantized. mean/rstd are not
    # unconditionally fp32: aten produces them in its accumulation dtype and, on
    # CPU/MTIA, casts them back to the (compute) input dtype -- so defer to aten
    # rather than assuming fp32.
    compute_dtype = inp_qp.dtype if inp_qp is not None else inp.dtype
    dq_inp = inp.new_empty(inp.shape, dtype=compute_dtype)
    out, mean, rstd = torch.ops.aten.native_layer_norm.default(
        dq_inp, normalized_shape, weight, bias, eps
    )
    # Only output 0 is (optionally) quantized; mean/rstd stay float.
    if out_qp is not None:
        out = out.new_empty(out.shape, dtype=out_qp.dtype)
    return out, mean, rstd


_native_layer_norm_impl.register_fake(_native_layer_norm_meta)


# ``rms_norm`` is single-output (unlike ``native_layer_norm``): it returns just
# the normalized tensor. Only the input activation and the output are quantized;
# ``weight`` (the learned scale) passes through in float, and ``normalized_shape``
# / ``eps`` are plain attributes. RMS norm has no bias.
@torch.library.custom_op("fused_quant::rms_norm", mutates_args=())
def _rms_norm_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )

    _validate_qparams(inp_qp, out_qp)

    dq_inp = _maybe_dequantize(inp, inp_qp)
    out = torch.ops.aten.rms_norm.default(dq_inp, normalized_shape, weight, eps)
    return _maybe_quantize(out, out_qp)


def _rms_norm_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, out_qp)

    # Output matches the input's shape. When the output is quantized its dtype is
    # the quantized out dtype. Otherwise the op returns the dequantized result,
    # whose float dtype is whatever dequantizing the input yields -- i.e.
    # ``inp_qp.dtype`` (the upstream dequant's output dtype), which is not always
    # fp32. A pipeline-interior op with stripped quantization has neither input nor
    # output quantized; then the input is already float, so use its own dtype.
    compute_dtype = inp_qp.dtype if inp_qp is not None else inp.dtype
    return inp.new_empty(inp.shape, dtype=get_out_dtype(out_qp, compute_dtype))


_rms_norm_impl.register_fake(_rms_norm_meta)


# ``_masked_softmax`` recomposes attention's ``add(mask) + softmax`` (see
# FuseAddSoftmaxIntoMaskedSoftmax). Only the input activation (the attention
# scores) and the output are quantized; the bool ``mask`` passes through
# unquantized and ``dim`` / ``mask_type`` are plain attributes. Like rms_norm it
# has no C++ kernel -- backend is expected to lower it before to_executorch.
@torch.library.custom_op("fused_quant::_masked_softmax", mutates_args=())
def _masked_softmax_impl(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    mask: torch.Tensor,
    dim: int,
    mask_type: int,
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )

    _validate_qparams(inp_qp, out_qp)

    dq_inp = _maybe_dequantize(inp, inp_qp)
    out = torch.ops.aten._masked_softmax.default(dq_inp, mask, dim, mask_type)
    return _maybe_quantize(out, out_qp)


def _masked_softmax_meta(
    inp: torch.Tensor,
    inp_scale: Optional[torch.Tensor],
    inp_zero_point: Optional[torch.Tensor],
    inp_dtype: torch.dtype,
    inp_quant_min: int,
    inp_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    mask: torch.Tensor,
    dim: int,
    mask_type: int,
) -> torch.Tensor:
    inp_qp = _make_qp(
        inp_scale, inp_zero_point, inp_dtype, inp_quant_min, inp_quant_max
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    _validate_qparams(inp_qp, out_qp)
    # Softmax preserves the input shape. When the output is quantized its dtype is
    # the quantized out dtype; otherwise it is the dequantized input's float dtype
    # (inp_qp.dtype, not always fp32), or -- for a pipeline-interior op stripped by
    # interior quantization stripping, where neither input nor output is quantized
    # -- the already-float input's own dtype.
    compute_dtype = inp_qp.dtype if inp_qp is not None else inp.dtype
    return inp.new_empty(inp.shape, dtype=get_out_dtype(out_qp, compute_dtype))


_masked_softmax_impl.register_fake(_masked_softmax_meta)


@torch.library.custom_op("fused_quant::embedding", mutates_args=())
def _embedding_impl(
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor],
    weight_zero_point: Optional[torch.Tensor],
    weight_dtype: torch.dtype,
    weight_quant_min: int,
    weight_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    indices: torch.Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    # Embedding is a compression-only op (no compute): the quantized table is
    # dequantized and the rows selected by `indices` are returned. The table
    # (the quantized "input") carries a dequant qparams block that may be
    # per-tensor, per-channel, or per-group (the group axis partitions the
    # embedding dimension). The output qparams block is optional: it is usually
    # absent (the gathered embeddings stay float, e.g. for cadence embedding_byte),
    # but the layout mirrors every other fused_quant op so the fusion and
    # optimization passes (e.g. quant absorption) can treat outputs uniformly.
    #
    # padding_idx/scale_grad_by_freq/sparse mirror aten.embedding so the generic
    # fusion can thread them through. They are training/gradient concerns with no
    # effect on the inference forward (the gather), so none of them are honored:
    # scale_grad_by_freq/sparse must be False and padding_idx must be the aten
    # default -1 (no padding token). The asserts make any model that relies on
    # padding_idx semantics fail loudly rather than silently ignoring them.
    assert padding_idx == -1, (
        "fused_quant.embedding does not honor padding_idx; expected the aten "
        f"default -1 (no padding token), got {padding_idx}"
    )
    assert not scale_grad_by_freq, (
        "fused_quant.embedding does not support scale_grad_by_freq=True (training only)"
    )
    assert not sparse, (
        "fused_quant.embedding does not support sparse=True (training only)"
    )
    weight_qp = _make_qp(
        weight_scale,
        weight_zero_point,
        weight_dtype,
        weight_quant_min,
        weight_quant_max,
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    # embedding is compression-only: an all-float form is meaningless, so unlike
    # other ops it requires the table or the output to be quantized.
    if weight_qp is None and out_qp is None:
        raise ValueError(
            "At least one of the embedding table (weight) or output must be quantized"
        )
    _validate_qparams(weight_qp, out_qp)
    if weight.ndim != 2:
        raise ValueError(
            "embedding weight (table) must be 2D [num_embeddings, embedding_dim], "
            f"got {weight.ndim}D"
        )
    dq_weight = _maybe_dequantize(weight, weight_qp)
    out = torch.nn.functional.embedding(indices, dq_weight)
    return _maybe_quantize(out, out_qp)


def _embedding_meta(
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor],
    weight_zero_point: Optional[torch.Tensor],
    weight_dtype: torch.dtype,
    weight_quant_min: int,
    weight_quant_max: int,
    out_scale: Optional[torch.Tensor],
    out_zero_point: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_quant_min: int,
    out_quant_max: int,
    indices: torch.Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    weight_qp = _make_qp(
        weight_scale,
        weight_zero_point,
        weight_dtype,
        weight_quant_min,
        weight_quant_max,
    )
    out_qp = _make_qp(
        out_scale, out_zero_point, out_dtype, out_quant_min, out_quant_max
    )
    assert padding_idx == -1, (
        "fused_quant.embedding does not honor padding_idx; expected the aten "
        f"default -1 (no padding token), got {padding_idx}"
    )
    assert not scale_grad_by_freq, (
        "fused_quant.embedding does not support scale_grad_by_freq=True (training only)"
    )
    assert not sparse, (
        "fused_quant.embedding does not support sparse=True (training only)"
    )
    if weight_qp is None and out_qp is None:
        raise ValueError(
            "At least one of the embedding table (weight) or output must be quantized"
        )
    _validate_qparams(weight_qp, out_qp)
    if weight.ndim != 2:
        raise ValueError(
            "embedding weight (table) must be 2D [num_embeddings, embedding_dim], "
            f"got {weight.ndim}D"
        )
    # Output is the gathered rows: indices.shape + [embedding_dim]. Its dtype
    # mirrors the impl: the quantized dtype when the output is quantized, else the
    # table's dequantize output dtype (weight_qp.dtype), or the table dtype when
    # the table is not dequantized -- never assumed to be float32.
    dequantized_dtype = weight_qp.dtype if weight_qp is not None else weight.dtype
    out_shape = (*indices.shape, weight.shape[1])
    return weight.new_empty(out_shape, dtype=get_out_dtype(out_qp, dequantized_dtype))


_embedding_impl.register_fake(_embedding_meta)
