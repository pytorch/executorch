# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.extension.llm.export.builder import DType


def quantize(  # noqa C901
    model: torch.nn.Module,
    qmode: str,
    computation_dtype: Optional[DType] = None,
    checkpoint_dtype: Optional[DType] = None,
    checkpoint_path: Optional[Path] = None,
    # following arguments only available when setting int4 or gptq quantization.
    group_size: Optional[int] = None,
    # following arguments are only used for GPTQ
    calibration_tasks: Optional[list] = None,
    calibration_limit: Optional[int] = None,
    calibration_seq_length: Optional[int] = None,
    pad_calibration_inputs: bool = False,
    percdamp: float = 0.01,
    blocksize: int = 128,
    tokenizer_path: Optional[Path] = None,
    verbose: bool = False,
    quantize_with_hqq: bool = True,
) -> torch.nn.Module:
    """
    Quantizes a model by converting all weights to int8.

    Args:
        model: The model to quantize.
        qmode: The quantization mode, e.g. int8, 8da4w.
        computation_dtype: The dtype that ops are performed in (the resulting dtype of dequantization).
            Also the dtype of the rest of the non-quantized compoents of the model.
        checkpoint_dtype: The dtype of the checkpoint, this arg exists since it is more accurate to
            quantize the weight in its original dtype.

    Returns:
        A quantized model.
    """
    if computation_dtype:
        computation_torch_dtype = computation_dtype.to_torch_dtype()
    else:
        computation_torch_dtype = torch.float32

    if not checkpoint_dtype:
        checkpoint_torch_dtype = computation_torch_dtype
    else:
        checkpoint_torch_dtype = checkpoint_dtype.to_torch_dtype()

    if qmode == "int8":
        # Add quantization mode options here: group size, bit width, etc.
        return WeightOnlyInt8QuantHandler(
            model, precision=checkpoint_torch_dtype
        ).quantized_model()
    elif qmode.startswith("torchao:fpa"):
        pattern = r"torchao:fpa(\d+)w"
        matches = re.findall(pattern, qmode)
        assert len(matches) == 1, f"Expected 1 match for pattern but got {len(matches)}"
        bitwidth = int(matches[0][0])
        _load_torchao_aten_lib(libname="libtorchao_ops_mps_aten")
        from torchao.experimental.quant_api import UIntxWeightOnlyLinearQuantizer

        with torch.no_grad():
            # This quantize() is currently doing a model.to(self.precision) so cannot
            # decouple computation and checkpoint dtypes.
            model = (
                UIntxWeightOnlyLinearQuantizer(
                    device="mps",
                    precision=computation_torch_dtype,
                    groupsize=group_size,
                    bitwidth=bitwidth,
                )
                .quantize(model)
                .to("cpu")
            )

        if verbose:
            print("quantized model:", model)
        return model
    elif qmode.startswith("torchao:8da"):
        # Check for required args
        if group_size is None:
            raise Exception(
                "For torchao:8daxw quantization, group size must be specified."
            )

        pattern = r"torchao:8da(\d+)w"
        matches = re.findall(pattern, qmode)
        assert len(matches) == 1, f"Expected 1 match for pattern but got {len(matches)}"
        bitwidth = int(matches[0][0])

        from torchao.quantization.granularity import PerAxis, PerGroup
        from torchao.quantization.quant_api import (
            Int8DynamicActivationIntxWeightConfig,
            quantize_,
        )
        from torchao.utils import unwrap_tensor_subclass

        with torch.no_grad():
            # Computation dtype is fixed to fp32 in the implementation of quantize_, so
            # no way to decouple checkpoint and computation dtype.
            quantize_(
                model,
                Int8DynamicActivationIntxWeightConfig(
                    weight_dtype=getattr(torch, f"int{bitwidth}"),
                    weight_granularity=(
                        PerAxis(0) if group_size == 0 else PerGroup(group_size)
                    ),
                    # pyre-ignore[6]
                    intx_packing_format="opaque_torchao_auto",
                    # pyre-ignore[6]
                    intx_choose_qparams_algorithm=(
                        "hqq_scale_only" if quantize_with_hqq else "affine"
                    ),
                ),
            )
            model = unwrap_tensor_subclass(model)
        if verbose:
            print("quantized model:", model)
        return model
    elif qmode == "8da4w":
        if group_size is None:
            # TODO: Default value for group size for 8da4w. Need this here for refactor, will clean this up.
            group_size = 128

        from torchao.quantization import (
            Int8DynamicActivationIntxWeightConfig,
            quantize_,
        )
        from torchao.quantization.granularity import PerGroup
        from torchao.utils import unwrap_tensor_subclass

        def filter_fn(m, fqn):
            # Check if it's a regular nn.Linear
            is_linear = isinstance(m, nn.Linear)

            # Check if it's a LoRALinear (which has a base weight parameter to quantize)
            is_lora_linear = False
            try:
                from executorch.examples.models.llama.lora import LoRALinear

                is_lora_linear = isinstance(m, LoRALinear)
            except ImportError:
                pass

            # Check if the weight shape is compatible with group size
            has_shape_compatible_with_group_size = False
            if is_linear or is_lora_linear:
                has_shape_compatible_with_group_size = (
                    m.weight.shape[1] % group_size == 0
                )
            return (
                is_linear or is_lora_linear
            ) and has_shape_compatible_with_group_size

        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                # pyre-ignore[16]
                weight_dtype=torch.int4,
                weight_granularity=PerGroup(group_size),
                # pyre-ignore[6]
                intx_choose_qparams_algorithm=(
                    "hqq_scale_only" if quantize_with_hqq else "affine"
                ),
            ),
            filter_fn=filter_fn,
        )
        # TODO: deal with checkpoint / computation dtype decoupling.

        if verbose:
            print("quantized model:", model)
        return model
    elif qmode == "4w":
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_
        from torchao.utils import unwrap_tensor_subclass

        q_group_size = 256 if group_size is None else group_size
        q_config = IntxWeightOnlyConfig(
            # pyre-ignore[16]
            weight_dtype=torch.int4,
            granularity=PerGroup(q_group_size),
            # pyre-ignore[6]
            intx_choose_qparams_algorithm=(
                "hqq_scale_only" if quantize_with_hqq else "affine"
            ),
        )
        quantize_(model, q_config)
        model = unwrap_tensor_subclass(model)

        return model
    else:
        raise Exception(f"Unrecognized quantize mode: {qmode}")


def dynamically_quantize_per_channel(
    x,
    quant_min,
    quant_max,
    target_dtype,
    group_size: Optional[int] = None,
    *,
    scales_dtype=torch.float16,
    enable_non_multiple_groups=True,
):
    """
    Dynamically quantize per channel.  This function is used for quantizing weights,
    for linear and embedding layers.

    Arguments:
        x: input tensor,
        quant_min: minimum value after quantization,
        quant_max: maximum value after quantization,
        target_dtype: target data type for weights after quantization,
        group_size: number of elements of the channel to quantize together

    Keyword arguments:
        scales_dtype: data type of scale,
        enable_non_multiple_groups: if True, allow the rowsize to not be a multiple of group size,
                        with a final group of a size less than group size.

    Assumptions:
        This function assumes symmetric quantization, axis ==0 and a dense memory format.
    """

    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    x_shape_1 = x.shape[1]

    if group_size is None or group_size == 0:
        items = x_shape_1
    elif ((x_shape_1 % group_size) == 0) or not enable_non_multiple_groups:
        assert group_size > 0, "group size must be positive"
        assert (
            x_shape_1 % group_size
        ) == 0, f"weights dimension 1 = {x_shape_1} must be a multiple of group size {group_size}"
        items = group_size
    else:
        assert group_size > 0, "group size must be positive"
        print(
            f"row-size of weight matrix {x_shape_1} is not divisible by group size {group_size}, using nearest neighbor rounding"
        )
        assert (
            x_shape_1 % group_size != 0
        ), f"expected x.shape[1] to not be a multiple of group size {group_size}, but got {x_shape_1}"
        padding = group_size - (x_shape_1 % group_size)
        x = F.pad(x, (0, padding))
        items = group_size

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    x = x.view(x.shape[0], x.shape[1] // items, items)
    # get min and max
    min_val, max_val = torch.aminmax(x, dim=2)
    # print(f"min_val {min_val}")
    # print(f"max_val {max_val}")

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = (
        torch.clamp(x_zp, quant_min, quant_max).to(target_dtype).view(x.shape[0], -1)
    )

    scales = scales.to(dtype=scales_dtype)
    quant = quant[:, :x_shape_1]

    return quant, scales, zero_points


#########################################################################
###                QuantHandler API definition                        ###


class QuantHandler:
    def __init__(self, mod):
        self.mod = mod

    def create_quantized_state_dict(self) -> Dict:  # "StateDict"
        pass

    def convert_for_runtime(self) -> nn.Module:
        pass

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


#########################################################################
###             Weight-only int8 per-channel quantized code           ###


def replace_linear_weight_only_int8_per_channel(module, node_type):
    for name, child in module.named_children():
        # print(f"name: {name}")
        if isinstance(child, nn.Linear):
            if (
                (node_type == "*")
                or (node_type == "output" and name == "output")
                or (node_type == "!output" and name != "output")
            ):
                # print(f"{name, child}")
                # print(f"in_features: {child.in_features}")
                # print(f"out_features: {child.out_features}")
                setattr(
                    module,
                    name,
                    WeightOnlyInt8Linear("cpu", child.in_features, child.out_features),
                )
        else:
            replace_linear_weight_only_int8_per_channel(child, node_type)


class WeightOnlyInt8QuantHandler(QuantHandler):
    def __init__(
        self,
        mod,
        device="cpu",
        *,
        node_type: str = "*",
        bitwidth: Optional[int] = None,
        group_size: Optional[int] = None,
        precision: torch.dtype = torch.float32,
    ):
        self.mod = mod
        self.group_size = group_size
        self.node_type = node_type
        if bitwidth is None:
            self.bitwidth = 8
        else:
            self.bitwidth = bitwidth
        self.precision = precision

    @torch.no_grad()
    def create_quantized_state_dict(self) -> Dict:
        cur_state_dict = self.mod.state_dict()

        if self.bitwidth == 4:
            range_min = -8
            range_max = 7
        elif self.bitwidth == 8:
            range_min = -128
            range_max = 127
        else:
            raise ValueError(f"Unsupported bitwidth {self.bitwidth}")

        for fqn, mod in self.mod.named_modules():
            # print(f"maybe? quantize {fqn}...{type(mod)}")
            if isinstance(mod, torch.nn.Linear):
                # print(f"candidate {fqn}, nodetype {self.node_type}")
                if (
                    (self.node_type == "*")
                    or (self.node_type == "output" and fqn in ["output", "final_proj"])
                    or (
                        self.node_type == "!output"
                        and fqn not in ["output", "final_proj"]
                    )
                ):
                    print(
                        f"quantize {self.node_type} {fqn, mod} with group_size {self.group_size}, bitwidth {self.bitwidth}"
                    )

                    # print(f"initial weight shape {mod.weight.shape}")
                    input_weight = mod.weight.float()

                    # print(f"expanded weight shape {input_weight.shape}")
                    weight, scales, _ = dynamically_quantize_per_channel(
                        input_weight.to(dtype=self.precision),
                        range_min,
                        range_max,
                        torch.int8,
                        self.group_size,
                        scales_dtype=mod.weight.dtype,
                    )

                    cur_state_dict[f"{fqn}.weight"] = weight
                    # squeeze makes group_size=rowsize unidimensional
                    cur_state_dict[f"{fqn}.scales"] = scales.squeeze(dim=-1)

        return cur_state_dict

    def convert_for_runtime(self) -> nn.Module:
        replace_linear_weight_only_int8_per_channel(self.mod, self.node_type)
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


class WeightOnlyInt8Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        device,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight", torch.zeros((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales
        # return F.linear(input, self.weight.to(dtype=input.dtype)) * se...


def linear_forward_8da8w(
    x,
    weight_int8,
    scales,
    zeros,
    out_features,
    precision,
):
    from torchao.quantization.utils import per_token_dynamic_quant

    x = per_token_dynamic_quant(x)
    n_bit = 8
    quant_min = -(2 ** (n_bit - 1))
    quant_max = 2 ** (n_bit - 1) - 1
    w_dq = torch.ops.quantized_decomposed.dequantize_per_channel(
        weight_int8,
        scales,
        zeros,
        0,
        quant_min,
        quant_max,
        torch.int8,
        out_dtype=precision,
    )
    c = torch.nn.functional.linear(x, w_dq)

    return c


class Int8DynActInt8WeightLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor

    """
    This module implements a dynamic quantized linear layer with int8 weight.
    Weights are per channel quantized. Parameters of importance
    precision: precision of input and output. e.g. torch.float32 means input
    activation is float32 and output is float32.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        dtype=None,
        precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.precision = precision

        if dtype is not None:
            raise ValueError("Please specify 'precision' instead of 'dtype'")

        # currently storing unpacked int8 weights
        self.register_buffer(
            "weight",
            torch.zeros((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (out_features),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(
                (out_features),
                dtype=torch.float32,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.precision)
        return linear_forward_8da8w(
            input,
            self.weight,
            self.scales,
            self.zeros,
            self.out_features,
            self.precision,
        )


#########################################################################
#####                   embedding table quantization               ######


def replace_embedding_weight_only_grouped_int8_per_channel(
    module, device, bitwidth: int = 8, group_size: Optional[int] = None, packed=False
):
    for name, child in module.named_children():
        # print(f"name: {name}")
        if isinstance(child, nn.Embedding):
            # print(f"{name, child}")
            # print(f"weights size: {child.weight.size()}")
            setattr(
                module,
                name,
                QuantizedGroupEmbedding(
                    device=device,
                    vocab_size=child.weight.shape[0],
                    embedding_dim=child.weight.shape[1],
                    group_size=group_size,
                    dtype=child.weight.dtype,
                    packed=packed,
                    bitwidth=bitwidth,
                ),
            )
        else:
            replace_embedding_weight_only_grouped_int8_per_channel(
                child, device, bitwidth, group_size, packed
            )


class EmbeddingQuantHandler(QuantHandler):
    def __init__(
        self,
        mod,
        device="cpu",
        *,
        bitwidth: int = 8,
        group_size: Optional[int] = None,
        packed=False,
        precision: Optional[torch.dtype] = None,
        quantize_with_hqq: bool = True,
    ):
        if isinstance(packed, str):
            packed = packed == "True"
        self.mod = mod
        self.device = device
        self.group_size = group_size
        self.bitwidth = bitwidth
        self.packed = packed
        # Dtype of the weights right before quantization.
        self.precision = precision
        if (bitwidth not in [2, 4]) and packed:
            raise RuntimeError("pack only works with bitsize 2, 4")
        self.quantize_with_hqq = quantize_with_hqq

    @torch.no_grad()
    def create_quantized_state_dict(self, packed=False) -> Dict:
        from torchao.quantization.granularity import PerAxis, PerGroup
        from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

        cur_state_dict = self.mod.state_dict()

        assert self.bitwidth in [2, 4, 8], f"Unsupported bitwidth {self.bitwidth}"

        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, nn.Embedding):
                # print("****")
                # print(f"Embedding identified: {fqn, mod}")
                # print(f"weights size: {mod.weight.size()}")
                # print(f"quantize {fqn}...")

                print(
                    f"quantize {fqn, mod} with group_size {self.group_size}, bitwidth {self.bitwidth}"
                )
                tmp_model = nn.Embedding(mod.weight.shape[0], mod.weight.shape[1])
                if self.precision:
                    tmp_model = tmp_model.to(dtype=self.precision)
                tmp_model.weight = nn.Parameter(mod.weight)
                config = IntxWeightOnlyConfig(
                    weight_dtype=getattr(torch, f"int{self.bitwidth}"),
                    granularity=(
                        PerAxis(0)
                        if (self.group_size is None or self.group_size == 0)
                        else PerGroup(self.group_size)
                    ),
                    # pyre-ignore[6]
                    intx_choose_qparams_algorithm=(
                        "hqq_scale_only" if self.quantize_with_hqq else "affine"
                    ),
                )
                quantize_(tmp_model, config, lambda m, fqn: isinstance(m, nn.Embedding))
                weight = tmp_model.weight.qdata  # pyre-ignore[16]
                scales = tmp_model.weight.scale  # pyre-ignore[16]

                if packed:
                    if self.bitwidth == 2:
                        if weight.shape[-1] % 4 != 0:
                            raise RuntimeError("automatic padding not implemented yet")
                        weight_range_shifted = weight.add(2).view(torch.uint8)
                        weight_view = weight_range_shifted.view(
                            weight.shape[0], weight.shape[1] // 4, 4
                        )
                        weight_0 = weight_view[:, :, 0]
                        weight_1 = weight_view[:, :, 1] << 2
                        weight_2 = weight_view[:, :, 2] << 4
                        weight_3 = weight_view[:, :, 3] << 6
                        weight_packed = weight_0 + weight_1 + weight_2 + weight_3
                        weight = weight_packed
                    elif self.bitwidth == 4:
                        if weight.shape[-1] % 2 != 0:
                            raise RuntimeError("automatic padding not implemented yet")
                        weight_range_shifted = weight.add(8).view(torch.uint8)
                        weight_view = weight_range_shifted.view(
                            weight.shape[0], weight.shape[1] // 2, 2
                        )
                        weight_even = weight_view[:, :, 0] * 16  # left shift 4
                        weight_odd = weight_view[:, :, 1]
                        weight_packed = weight_even + weight_odd
                        weight = weight_packed

                weight = weight.to(device=self.device)
                scales = scales.to(device=self.device)
                # Update state dict
                cur_state_dict[f"{fqn}.weight"] = weight
                # squeeze makes group_size=rowsize unidimensional
                cur_state_dict[f"{fqn}.scales"] = scales.squeeze(dim=-1)

        return cur_state_dict

    def convert_for_runtime(self) -> nn.Module:
        replace_embedding_weight_only_grouped_int8_per_channel(
            self.mod, self.device, self.bitwidth, self.group_size, self.packed
        )
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict(self.packed)
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict, assign=True)
        return self.mod


class QuantizedGroupEmbedding(torch.nn.Module):
    def __init__(
        self,
        device,
        vocab_size: int,
        embedding_dim: int,
        group_size: Optional[int] = None,
        dtype=torch.half,
        packed=False,
        bitwidth: int = 8,
    ) -> None:
        super().__init__()
        if group_size is None or group_size == 0:
            group_size = embedding_dim
        self.group_size = group_size
        self.dtype = dtype
        self.packed = packed
        self.bitwidth = bitwidth
        if not packed:
            self.register_buffer(
                "weight",
                torch.zeros(
                    (vocab_size, embedding_dim), dtype=torch.int8, device=device
                ),
            )
        else:  # packed
            if bitwidth == 2:
                self.register_buffer(
                    "weight",
                    torch.zeros(
                        (vocab_size, embedding_dim // 4),
                        dtype=torch.uint8,
                        device=device,
                    ),
                )
            elif bitwidth == 4:
                self.register_buffer(
                    "weight",
                    torch.zeros(
                        (vocab_size, embedding_dim // 2),
                        dtype=torch.uint8,
                        device=device,
                    ),
                )

        groups_per_row = (embedding_dim + group_size - 1) // group_size
        if groups_per_row > 1:
            self.register_buffer(
                "scales",
                torch.ones(
                    (vocab_size, groups_per_row), dtype=torch.float16, device=device
                ),
            )
        else:
            self.register_buffer(
                "scales", torch.ones((vocab_size,), dtype=torch.float16, device=device)
            )

    @torch.no_grad()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if not self.packed:  # 8bit
            return torch.ops.quantized_decomposed.embedding_byte.dtype(
                self.weight, self.scales, None, -128, 127, indices, dtype=self.dtype
            )
        else:  # packed
            if self.bitwidth == 2:
                return torch.ops.quantized_decomposed.embedding_2bit.dtype(
                    self.weight, self.scales, None, -2, 1, indices, dtype=self.dtype
                )

            # Remaining case (always return to make pyre happy)
            assert self.bitwidth == 4
            return torch.ops.quantized_decomposed.embedding_4bit.dtype(
                self.weight, self.scales, None, -8, 7, indices, dtype=self.dtype
            )


############################ Source Transform Start #######################


def get_quant_embedding_transform(
    embedding_quantize: str,
    use_shared_embedding: bool = False,
    dtype_override: Optional[DType] = None,
    quantize_with_hqq: bool = True,
):
    if embedding_quantize.startswith("torchao:"):
        from torchao.prototype.quantization.embedding.api import (
            EmbeddingQuantizer,
            TiedEmbeddingQuantizer,
        )
        from torchao.quantization.granularity import PerAxis, PerGroup
        from torchao.quantization.quant_api import MappingType

        quant_args = embedding_quantize.split(":")[1].split(",")
        if len(quant_args) == 2:
            bitwidth, group_size = quant_args
            is_asymmetric = True
        else:
            bitwidth, group_size, is_asymmetric = quant_args

        if group_size in ["none", "None", "0"]:
            group_size = 0

        group_size = int(group_size)
        bitwidth = int(bitwidth)
        is_asymmetric = bool(is_asymmetric)
        weight_dtype = getattr(torch, f"int{bitwidth}")
        granularity = PerAxis(0) if group_size == 0 else PerGroup(group_size)
        mapping_type = (
            MappingType.ASYMMETRIC if is_asymmetric else MappingType.SYMMETRIC
        )

        def _torchao_embedding_quantizer(model):
            with torch.no_grad():
                if not use_shared_embedding:
                    EmbeddingQuantizer(
                        weight_dtype=weight_dtype,
                        granularity=granularity,
                        mapping_type=mapping_type,
                        use_fallback=False,
                    ).quantize(model)
                else:
                    TiedEmbeddingQuantizer(
                        weight_dtype=weight_dtype,
                        granularity=granularity,
                        mapping_type=mapping_type,
                    ).quantize(model)
            return model

        return _torchao_embedding_quantizer

    bitwidth, group_size = embedding_quantize.split(",")
    if group_size == "none" or group_size == "None" or group_size == "0":
        group_size = None
    else:
        group_size = int(group_size)
    bitwidth = int(bitwidth)
    torch_dtype = dtype_override.to_torch_dtype() if dtype_override else None
    return lambda model: EmbeddingQuantHandler(
        model,
        bitwidth=bitwidth,
        group_size=group_size,
        packed=(bitwidth in [2, 4]),
        precision=torch_dtype,
        quantize_with_hqq=quantize_with_hqq,
    ).quantized_model()


def get_quant_weight_transform(
    quantization_mode: str,
    group_size: Optional[int] = None,
    computation_dtype: Optional[DType] = None,
    checkpoint_dtype: Optional[DType] = None,
    checkpoint_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    calibration_tasks: Optional[list] = None,
    calibration_limit: Optional[int] = None,
    calibration_seq_length: Optional[int] = None,
    quantize_with_hqq: bool = True,
):
    return partial(
        quantize,
        qmode=quantization_mode,
        computation_dtype=computation_dtype,
        checkpoint_dtype=checkpoint_dtype,
        checkpoint_path=(Path(path) if (path := checkpoint_path) is not None else None),
        group_size=group_size,
        calibration_tasks=calibration_tasks,
        calibration_limit=calibration_limit,
        calibration_seq_length=calibration_seq_length,
        tokenizer_path=(Path(path) if (path := tokenizer_path) is not None else None),
        quantize_with_hqq=quantize_with_hqq,
    )


def _load_torchao_aten_lib(libname):
    import glob
    import os

    libs = glob.glob(
        os.path.abspath(
            os.path.join(
                os.environ.get("CMAKE_INSTALL_PREFIX", ""),
                f"lib/{libname}.*",
            )
        )
    )
    assert (
        len(libs) == 1
    ), f"Expected 1 library but got {len(libs)}.  If you installed the torchao ops in a non-standard location, please set CMAKE_INSTALL_PREFIX correctly."
    logging.info(f"Loading custom ops library: {libs[0]}")
    torch.ops.load_library(libs[0])


# We want to do compute the actual ops in the computation dtype, since the precision of the
# quantized linear will initially be the dtype of the checkpoint.
def set_8da4w_computation_dtype(
    module: nn.Module, computation_dtype: torch.dtype
) -> nn.Module:
    from torchao.quantization.linear_quant_modules import Int8DynActInt4WeightLinear

    def _set_8da4w_computation_dtype(module: nn.Module, dtype: torch.dtype) -> None:
        """
        Recursively iterate through the module and set the precision attributes
        of all Int8DynActInt4WeightLinears.
        """
        for _name, child in module.named_children():
            if isinstance(child, Int8DynActInt4WeightLinear):
                child.precision = dtype
            else:
                # Recursively apply to child modules
                _set_8da4w_computation_dtype(child, dtype)

    _set_8da4w_computation_dtype(module, computation_dtype)
    return module


############################ Source Transform End #######################
