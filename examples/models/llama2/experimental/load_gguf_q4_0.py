# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Load llama model from a GGUF file, quantized in Q4_0 format.
# For float weights, we load them directly from the GGUF file.
# For Q4_0 weights, we load them into a Tensor subclass (GGMLInt4LinearWeight).
# This is done by replacing the linear weight with the subclass.

import logging
import os
from typing import Callable, Dict, Mapping

import torch
from executorch.examples.models.llama2.experimental.subclass import (
    _unpack_two_uint8,
    GGMLInt4LinearWeight,
    to_float,
)
from executorch.extension.gguf_util.converters.llama_converter import (
    _convert_gguf_tensor_name_to_llama_nn,
    _create_pt_model,
)
from executorch.extension.gguf_util.load_gguf import GGUFWeights, load_file
from gguf import ReaderTensor
from gguf.constants import GGMLQuantizationType
from torchao.quantization.subclass import QuantizedLinearWeightBase

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def _replace_with_custom_fn_if_matches_filter(
    # pyre-fixme[2]: Parameter must be annotated.
    model,
    replacement_fn,
    filter_fn,
    cur_fqn="",
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`
    """
    if filter_fn(model, cur_fqn[:-1]):
        model = replacement_fn(model, cur_fqn[:-1])
        return model
    else:
        for name, child in model.named_children():
            new_child = _replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, f"{cur_fqn}{name}."
            )
            if new_child is not child:
                setattr(model, name, new_child)
        return model


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _get_subclass_inserter(
    weight_map: Dict[str, ReaderTensor]
) -> Callable[[torch.nn.Module, str], torch.nn.Module]:
    def insert_subclass(lin, fqn):
        # TODO: replace weights with gguf format tensor
        # packed tensor should have size [numel / 32, 18]
        fqn = fqn + ".weight"
        assert (
            fqn in weight_map
        ), f"Expect {fqn} to be in weight map but not found. All keys are {weight_map.keys()}"
        tensor = weight_map[fqn]
        print(fqn, tensor.shape, tensor.data.shape, lin.weight.shape)
        packed = torch.from_numpy(tensor.data).reshape(-1, 18)
        scale = torch.tensor(_unpack_two_uint8(packed[:, :2]), dtype=torch.float16)
        lin.weight = torch.nn.Parameter(
            GGMLInt4LinearWeight(packed, scale, lin.weight.shape)
        )
        return lin

    return insert_subclass


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _get_filter_fn(
    weight_map: Dict[str, ReaderTensor]
) -> Callable[[torch.nn.Module, str], bool]:
    def _is_linear(mod, fqn):
        return (
            isinstance(mod, torch.nn.Linear)
            and hasattr(mod, "weight")
            and weight_map[fqn + ".weight"].tensor_type == GGMLQuantizationType.Q4_0
            and not isinstance(mod.weight, QuantizedLinearWeightBase)
        )

    return _is_linear


def change_linear_weights_to_q4_0_tensors(
    model: torch.nn.Module, gguf_weights: GGUFWeights
) -> None:
    """
    Converts all linear weight tensors to the
    `GGMLInt4LinearWeight` tensor subclass,
    effectively applying the same form of quantization
    as apply_dynamic_quant while not modifying the linear modules.
    """
    assert gguf_weights is not None, "Must provide gguf_weights"
    weight_map = {
        _convert_gguf_tensor_name_to_llama_nn(tensor.name): tensor
        for tensor in gguf_weights.tensors
    }

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(weight_map),
        _get_filter_fn(weight_map),
    )


def get_float_weights(
    pt_model: torch.nn.Module, gguf_weights: GGUFWeights
) -> Mapping[str, torch.Tensor]:
    """
    Returns a mapping from the fqn to the float weight tensor. Even though
    the model is quantized in Q4_0, these weights are still stored as float.
    Args:
        pt_model (torch.nn.Module): The model to load the weights.
        gguf_weights (GGUFWeights): The weights to extract the weights from.
    """
    state_dict = {}
    for tensor in gguf_weights.tensors:
        model_key = _convert_gguf_tensor_name_to_llama_nn(tensor.name)
        if (
            tensor.tensor_type == GGMLQuantizationType.F32
            or tensor.tensor_type == GGMLQuantizationType.F16
        ):
            print(tensor.name)
            reversed_shape = tensor.shape[::-1]
            new_tensor = tensor.data.reshape(reversed_shape)
            state_dict[model_key] = torch.from_numpy(new_tensor)
        # Load token_embd.weight which is quantized in Q4_0 and we dequantize it into float.
        elif tensor.tensor_type == GGMLQuantizationType.Q4_0:
            if tensor.name == "token_embd.weight":
                print(tensor.name)
                unpacked = to_float(torch.from_numpy(tensor.data.reshape(-1, 18)))
                state_dict[model_key] = unpacked.reshape(
                    pt_model.params.vocab_size, pt_model.params.dim
                )

    # We need to fake initialize the mask, to match with the llama_transformer.py
    for id in range(pt_model.params.n_layers):
        mask_name = f"layers.{id}.attention.mask"
        mask = torch.full(
            (1, 1, pt_model.params.max_seq_len, pt_model.params.max_seq_len),
            float("-inf"),
        )
        mask = torch.triu(mask, diagonal=1)
        state_dict[mask_name] = mask
    return state_dict


def load_gguf_q4_0(gguf_file: str) -> torch.nn.Module:
    assert os.path.isfile(gguf_file), f"Expect a valid gguf_file path, got {gguf_file}"

    logging.info(f"Loading GGUF file: {gguf_file}")
    gguf_model_args, gguf_weights = load_file(gguf_file)

    logging.info("Creating the PyTorch model")
    pt_model = _create_pt_model(
        gguf_model_args,
    )

    logging.info("Load float weights")
    state_dict = get_float_weights(pt_model, gguf_weights)
    pt_model.load_state_dict(state_dict, strict=False)

    logging.info("Change linear weights to Q4_0 tensors")
    change_linear_weights_to_q4_0_tensors(pt_model, gguf_weights)

    pt_model = pt_model.to(dtype=torch.float16)

    return pt_model
