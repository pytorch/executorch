# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""MXFP Linear transform for the Arm backend.

TorchAO extension for MXFP linear. It replaces ``nn.Linear`` with a wrapper
module that stores precomputed MXFP weights and emits a backend-internal custom
op during export.

"""

import torch
import torch.nn.functional as F
from executorch.backends.arm.ao_ext.mxfp import (
    _cast_to_block_scaled_cpu_ref,
    mxfp_dtype_to_str,
    mxfp_str_to_dtype,
    MXFPDType,
    MXFPOpConfig,
)
from executorch.backends.arm.ao_ext.mxfp_tosa_lib import MXFP_TOSA_LIB
from torchao.prototype.mx_formats.mx_tensor import to_dtype, to_mx


# Define the custom TOSA operator. Note that weight_payload_dtype is needed as
# an extra argument because sub-byte dtypes (FP4 and FP6) are contained
# in uint8 tensors, meaning the weight tensor itself does not contain
# the dtype.
MXFP_TOSA_LIB.define(
    "linear(Tensor input, Tensor weight_qdata, Tensor weight_scale, "
    "Tensor? bias=None, SymInt block_size=32, str weight_payload_dtype='') -> Tensor"
)


def _get_mx_elem_dtype(
    weight_qdata: torch.Tensor,
    weight_payload_dtype: str = "",
) -> MXFPDType:
    if weight_payload_dtype:
        return mxfp_str_to_dtype(weight_payload_dtype)
    if weight_qdata.dtype == torch.uint8:
        return torch.float4_e2m1fn_x2
    return weight_qdata.dtype


def _get_num_input_features(
    weight_qdata: torch.Tensor, weight_payload_dtype: str = ""
) -> int:
    num_input_features = weight_qdata.shape[-1]
    if weight_qdata.dtype == torch.uint8 and weight_payload_dtype == mxfp_dtype_to_str(
        torch.float4_e2m1fn_x2
    ):
        # FP4 elements are packed pairwise in each byte in a uint8 tensor.
        num_input_features *= 2
    return num_input_features


@torch.library.register_fake("tosa_mxfp::linear", lib=MXFP_TOSA_LIB)  # type: ignore[misc]
def _mxfp_linear_fake(
    input: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    block_size: int = 32,
    weight_payload_dtype: str = "",
) -> torch.Tensor:
    if weight_qdata.ndim != 3:
        raise ValueError(
            f"Expected weight_qdata to be rank 3 for linear, got {weight_qdata.ndim}"
        )
    if weight_qdata.shape[0] != 1:
        raise ValueError(
            f"Expected weight_qdata batch dim to be 1, got {weight_qdata.shape[0]}"
        )
    num_input_features = _get_num_input_features(weight_qdata, weight_payload_dtype)
    if input.shape[-1] != num_input_features:
        raise ValueError(
            f"Input last dim {input.shape[-1]} must match linear in_features "
            f"{num_input_features}"
        )
    expected_scale_shape = (
        1,
        weight_qdata.shape[1],
        num_input_features // block_size,
    )
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"Expected weight_scale shape {expected_scale_shape}, got "
            f"{tuple(weight_scale.shape)}"
        )
    output_shape = (*input.shape[:-1], weight_qdata.shape[1])
    return input.new_empty(output_shape, dtype=torch.float32)


@torch.library.impl("tosa_mxfp::linear", "cpu", lib=MXFP_TOSA_LIB)
def _mxfp_linear_cpu(
    input: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    block_size: int = 32,
    weight_payload_dtype: str = "",
) -> torch.Tensor:
    """CPU reference implementation of the MXFP linear op."""

    if weight_qdata.ndim != 3 or weight_scale.ndim != 3:
        raise ValueError("Expected rank-3 weight tensors for MXFP linear")

    elem_dtype = _get_mx_elem_dtype(weight_qdata, weight_payload_dtype)

    # Cast the input to block-scaled format and back again to match the
    # expected input format of the TOSA
    dequantized_input = _cast_to_block_scaled_cpu_ref(
        input,
        elem_dtype,
        block_size,
    )
    dequantized_weight = to_dtype(
        weight_qdata,
        weight_scale,
        elem_dtype,
        block_size,
        torch.float32,
    )
    dequantized_weight = dequantized_weight.squeeze(0)
    if bias is not None:
        bias = bias.to(torch.float32)
    return F.linear(dequantized_input, dequantized_weight, bias)


class MXFPLinearOp(torch.nn.Module):
    """Linear wrapper that stores MXFP weights and emits a custom op."""

    def __init__(
        self,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        config: MXFPOpConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.weight_dtype = mxfp_dtype_to_str(config.weight_dtype)

        self.register_buffer("weight_qdata", weight_qdata, persistent=True)
        self.register_buffer("weight_scale", weight_scale, persistent=True)

        self.bias: torch.nn.Parameter | None
        bias_param = (
            torch.nn.Parameter(bias.detach(), requires_grad=False)
            if bias is not None
            else None
        )
        self.register_parameter(
            "bias",
            bias_param,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.tosa_mxfp.linear.default(
            x,
            self.weight_qdata,
            self.weight_scale,
            self.bias,
            self.config.block_size,
            self.weight_dtype,
        )


def transform_linear_to_mxfp(
    module: torch.nn.Module,
    config: MXFPOpConfig,
) -> torch.nn.Module:
    assert isinstance(module, torch.nn.Linear)

    weight = module.weight.detach().contiguous()
    if weight.shape[-1] % config.block_size != 0:
        raise ValueError(
            f"Linear in_features={weight.shape[-1]} must be divisible by "
            f"block_size={config.block_size}"
        )

    weight_scale, weight_qdata = to_mx(
        weight,
        elem_dtype=config.weight_dtype,
        block_size=config.block_size,
        scaling_mode=config.weight_scaling_mode,
    )

    # The resulting TOSA op MATMUL_T_BLOCK_SCALED only works with tensors of
    # rank 3, therefore we prepend a batch dimension of 1 to the weight tensors
    # here.
    weight_qdata = weight_qdata.unsqueeze(0)
    weight_scale = weight_scale.unsqueeze(0)

    bias = module.bias.detach().to(torch.float32) if module.bias is not None else None
    return MXFPLinearOp(weight_qdata, weight_scale, bias, config)
