# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.extension.llm.export.builder import DType

from sentencepiece import SentencePieceProcessor

try:
    from fairseq2.nn.embedding import (
        Embedding as fsEmbedding,
        StandardEmbedding as fsStandardEmbedding,
    )

    from fairseq2.nn.projection import Linear as fsLinear

    print("Using fairseq2 modules.")
except:
    fsEmbedding = nn.Embedding
    fsStandardEmbedding = nn.Embedding
    fsLinear = nn.Linear


def quantize(
    model: torch.nn.Module,
    qmode: str,
    activation_dtype: Optional[DType],
    checkpoint_path: Optional[Path] = None,
    # following arguments only available when setting int4 or gptq quantization.
    group_size: Optional[int] = 128,
    # following arguments are only used for GPTQ
    calibration_tasks: Optional[list] = None,
    calibration_limit: Optional[int] = None,
    calibration_seq_length: Optional[int] = None,
    pad_calibration_inputs: bool = False,
    percdamp: float = 0.01,
    blocksize: int = 128,
    tokenizer_path: Optional[Path] = None,
    verbose: bool = False,
) -> torch.nn.Module:
    """
    Quantizes a model by converting all weights to int8.
    Args:
        model: A model to quantize.
        qmode: quantization mode, e.g. int8, 8da4w, 8da4w-gptq
    Returns:
        A quantized model.
    """
    if activation_dtype is not None:
        torch_dtype = activation_dtype.to_torch_dtype()
    else:
        torch_dtype = torch.float16

    assert checkpoint_path, "Need to specify a checkpoint"
    # if checkpoint_path is None:
    #     checkpoint_path = Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")

    if qmode == "int8":
        # Add quantization mode options here: group size, bit width, etc.
        return WeightOnlyInt8QuantHandler(model).quantized_model()
    elif qmode == "8da4w":
        # Check for required args
        if group_size is None:
            raise Exception("For 8da4w quantization, group size must be specified.")
        from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

        model = Int8DynActInt4WeightQuantizer(
            precision=torch_dtype, groupsize=group_size
        ).quantize(model)
        if verbose:
            print("quantized model:", model)
        return model
    elif qmode == "8da4w-gptq":
        # Check for required args
        required_args: Optional[Any] = [
            group_size,
            calibration_limit,
            calibration_seq_length,
        ]
        if any(arg is None for arg in required_args):
            raise Exception(
                "For 8da4w-gptq quantization, group size, calibration limit and calibration sequence length must be specified."
            )
        if calibration_tasks is None:
            calibration_tasks = ["wikitext"]

        try:
            # torchao 0.3+
            from torchao._eval import InputRecorder  # pyre-fixme[21]
        except ImportError:
            from torchao.quantization.GPTQ import InputRecorder  # pyre-ignore

        from torchao.quantization.quant_api import Int8DynActInt4WeightGPTQQuantizer

        if tokenizer_path is None:
            tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path
        tokenizer = SentencePieceProcessor(  # pyre-ignore[28]
            model_file=str(tokenizer_path)
        )

        inputs = (
            InputRecorder(  # pyre-fixme[16]
                tokenizer,
                calibration_seq_length,
                None,  # input_prep_func
                pad_calibration_inputs,
                model.vocab_size,
            )
            .record_inputs(
                calibration_tasks,
                calibration_limit,
            )
            .get_inputs()
        )

        gptq_quantizer = Int8DynActInt4WeightGPTQQuantizer(
            blocksize,
            percdamp,
            group_size,
        )
        model = gptq_quantizer.quantize(model, inputs)
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
    ):
        self.mod = mod
        self.group_size = group_size
        self.node_type = node_type
        if bitwidth is None:
            self.bitwidth = 8
        else:
            self.bitwidth = bitwidth

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
            if isinstance(mod, torch.nn.Linear) or isinstance(mod, fsLinear):
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
                        input_weight,
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
            "weight", torch.empty((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales
        # return F.linear(input, self.weight.to(dtype=input.dtype)) * se...


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
    ):
        if isinstance(packed, str):
            packed = packed == "True"
        self.mod = mod
        self.device = device
        self.group_size = group_size
        self.bitwidth = bitwidth
        self.packed = packed
        if (bitwidth != 4) and packed:
            raise RuntimeError("pack only works with bitsize 4")

    @torch.no_grad()
    def create_quantized_state_dict(self, packed=False) -> Dict:
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
            if isinstance(mod, nn.Embedding):
                # print("****")
                # print(f"Embedding identified: {fqn, mod}")
                # print(f"weights size: {mod.weight.size()}")
                # print(f"quantize {fqn}...")

                print(
                    f"quantize {fqn, mod} with group_size {self.group_size}, bitwidth {self.bitwidth}"
                )
                weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(),
                    range_min,
                    range_max,
                    torch.int8,
                    self.group_size,
                    scales_dtype=mod.weight.dtype,
                )

                if packed:
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
        self.mod.load_state_dict(model_updated_state_dict)
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
    ) -> None:
        super().__init__()
        if group_size is None or group_size == 0:
            group_size = embedding_dim
        self.group_size = group_size
        self.dtype = dtype
        self.packed = packed
        if not packed:
            self.register_buffer(
                "weight",
                torch.empty(
                    (vocab_size, embedding_dim), dtype=torch.int8, device=device
                ),
            )
        else:  # packed
            self.register_buffer(
                "weight",
                torch.empty(
                    (vocab_size, embedding_dim // 2), dtype=torch.uint8, device=device
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
                self.weight, self.scales, None, 0, 0, indices, dtype=self.dtype
            )
        else:  # 4bit packed
            return torch.ops.quantized_decomposed.embedding_4bit.dtype(
                self.weight, self.scales, None, 0, 0, indices, dtype=self.dtype
            )


############################ Source Transform Start #######################


def get_quant_embedding_transform(args):
    bitwidth, group_size = args.embedding_quantize.split(",")
    if group_size == "none" or group_size == "None" or group_size == "0":
        group_size = None
    else:
        group_size = int(group_size)
    bitwidth = int(bitwidth)
    return lambda model: EmbeddingQuantHandler(
        model,
        bitwidth=bitwidth,
        group_size=group_size,
        packed=(bitwidth == 4),
    ).quantized_model()


def get_quant_weight_transform(args, dtype_override, verbose):
    # If these optional args are None, don't provide them to quantize()
    quant_args_str = [
        "group_size",
        "calibration_tasks",
        "calibration_limit",
        "calibration_seq_length",
    ]
    arg_dict = vars(args)
    quant_args = {
        param: val
        for param in quant_args_str
        if (val := arg_dict.get(param)) is not None
    }

    return partial(
        quantize,
        **quant_args,
        qmode=args.quantization_mode,
        activation_dtype=dtype_override,
        checkpoint_path=(Path(path) if (path := args.checkpoint) is not None else None),
        tokenizer_path=(
            Path(path) if (path := args.tokenizer_path) is not None else None
        ),
    )


############################ Source Transform End #######################
