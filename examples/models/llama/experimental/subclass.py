# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Define a Tensor subclass to wrap around ggml q4_0 tensor layout.
# The layout is the following:
# ┌─────────────────────┬───────────────────────────┐
# │                     │                           │
# │                     │                           │
# │  2 bytes (1xfp16)   │    16 bytes (32xint4)     │
# │  group-wise scale   │    group-wise weights     │
# │                     │                           │
# │                     │                           │
# └─────────────────────┴───────────────────────────┘
#
# Notice that the 16 bytes (32 int4) are interleved:
# [0th value, 16th value, 1st value, 17th value, ..., 15th, 31st]
#
# This layout is handled internally in the tensor subclass.
import torch
from torchao.quantization.subclass import QuantizedLinearWeightBase


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def up_size(size):
    return (*size[:-1], size[-1] * 2)


def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


def unpack_uint4(uint8_data) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    # since we are using uint8 we will decode 2 entries per byte
    # Shift elements down 4 and select out the bottom 4 bits
    shape = uint8_data.shape
    first_elements = (uint8_data & 0b1111).to(torch.uint8)
    second_elements = (uint8_data >> 4).to(torch.uint8)
    return torch.stack([first_elements, second_elements], dim=-1).view(up_size(shape))


def _pack_to_two_uint8(scale: torch.Tensor) -> torch.Tensor:
    raw_bytes = scale.numpy().tobytes()
    scale_uint8 = torch.frombuffer(raw_bytes, dtype=torch.uint8)
    scale_uint8 = scale_uint8.view(-1, 2)
    return scale_uint8


def _unpack_two_uint8(
    tensor: torch.Tensor,
) -> torch.Tensor:
    assert (
        tensor.dtype == torch.uint8
    ), f"Expecting to have a uint8 tensor but get {tensor.dtype}"
    raw_bytes = tensor.numpy().tobytes()
    scale = torch.frombuffer(raw_bytes, dtype=torch.float16)
    return scale


def _interleave(
    input: torch.Tensor,
    group_size,
) -> torch.Tensor:
    half1 = input[:, : group_size // 2]
    half2 = input[:, group_size // 2 :]
    interleaved_tensor = torch.stack((half1, half2), dim=2)
    return interleaved_tensor.view(input.size(0), -1)


def from_float(
    input: torch.Tensor,
) -> torch.Tensor:
    """
    Quantize similar to GGUF's Q4_0 quantization. Group into size of
    32 and generate a uint8 tensor. One group will result into 18 uint8s
    consisting of:
      - 1 scale (float16 represented as 2 uint8 elements)
      - 32 4-bit elements (represented as 16 uint8 elements)
    """
    group_size = 32
    zero_point = 8.5
    # pyre-fixme[16]: Callable input has no attribute dtype.
    assert input.dtype == torch.float16, f"Expecting float16 input, got {input.dtype}"
    assert (
        input.numel() % group_size
        == 0
        # pyre-fixme[16]: Callable input has no attribute numel.
    ), f"The number of input values has to be a multiple of {group_size} but got {input.numel()}"
    input = input.reshape(-1, group_size)
    abs_max_id = torch.argmax(torch.abs(input), dim=1)
    scales = input[torch.arange(input.size(0)), abs_max_id] / -8
    inv_scales = torch.div(1.0, scales.to(torch.float32))

    clamped = torch.clamp(
        input=torch.floor(inv_scales.unsqueeze(1) * input + zero_point),
        min=0,
        max=15,
    ).to(torch.uint8)
    alternate = _interleave(clamped, group_size)
    return torch.cat([_pack_to_two_uint8(scales), pack_uint4(alternate)], dim=1)


def to_float(
    input: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize GGUF's Q4_0 tensor. Expecting input to be a uint8 tensor
    with a dimension of [num_group // 2, 18], the first 2 values of each
    row represents the scale of that group.
    """
    zero_point = 8
    data_unint8 = input[:, 2:]
    data = unpack_uint4(data_unint8)
    assert data.dtype == torch.uint8
    interleave = torch.cat([data[:, ::2], data[:, 1::2]], dim=1)
    scale = _unpack_two_uint8(input[:, :2])
    a = interleave.to(torch.float16) - zero_point
    return a * scale.unsqueeze(1)


class GGMLInt4LinearWeight(QuantizedLinearWeightBase):
    """
    A Tensor subclass that when applied to a weight used in a linear op/module,
    changes that linear op to a weight-only int4 quantized linear op with groupwise
    affine quantization on the weight.
    """

    @staticmethod
    def __new__(
        cls,
        int_data,
        scales,
        shape,
        **kwargs,
    ):
        kwargs["dtype"] = kwargs.get("dtype", scales.dtype)
        return super().__new__(cls, int_data, transposed=False, shape=shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data,
        scales,
        shape,
        **kwargs,
    ):
        # the transposed flag tracks whether the tensor subclass has been transposed relative
        # to how a weight is normally stored in a linear i.e. [out_features, in_features].
        # tracking both transposed and shape is slightly redundant but corner cases like
        # square matrices can cause issues otherwise
        self.scales = scales
        self.groupsize = 32
        self.zero_point = torch.tensor(8.5, dtype=torch.float)
        super().__init__(int_data, transposed=False)

    def int_repr(self):
        return self.int_data

    def q_params(self):
        return {"q_scales": self.scales, "q_zero_points": self.zero_point}

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.int_data.to(kwargs["device"]),
            self.scales.to(kwargs["device"]),
            self.shape,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.scales),
            self.shape,
            dtype=self.dtype,
        )

    def __tensor_flatten__(self):
        return ["int_data", "scales"], (
            self.dtype,
            self.shape,
        )

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, attributes, outer_size=None, outer_stride=None
    ):
        int_data, scales = (
            tensor_data_dict["int_data"],
            tensor_data_dict["scales"],
        )
        dtype, shape = attributes
        return cls(
            int_data,
            scales,
            shape if outer_size is None else outer_size,
            dtype=dtype,
        )

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        """
        This is the quantized linear op that is used to implement the weight-only
        int4 quantized linear op.
        """
        assert isinstance(
            w_qtensor, GGMLInt4LinearWeight
        ), f"Expect {w_qtensor} to be an instance of GGMLInt4LinearWeight but got {type(w_qtensor)}"
        fp_weight = to_float(w_qtensor.int_data).view(w_qtensor.shape)
        return torch.nn.functional.linear(act_mat, fp_weight, bias)

    @classmethod
    def from_float(cls, input_float):
        """
        Method used to convert a linear weight tensor to an instance of the
        GGMLInt4LinearWeight subclass.

        Example usage::

            model.lin_mod.weight = (
                GGMLInt4LinearWeight.from_float(model.lin_mod.weight)
            )
        """
        packed = from_float(input_float)
        scale = torch.tensor(_unpack_two_uint8(packed[:, :2]), dtype=torch.float16)
        return cls(
            packed,
            scale,
            input_float.shape,
            dtype=torch.float16,
        )
