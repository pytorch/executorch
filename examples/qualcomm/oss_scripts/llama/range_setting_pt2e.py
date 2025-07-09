# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
The goal of this is to allow range setting methods from TorchAO (formerly Quanty)
to be incorporated into the PT2E flow.

We implement the two main range setting methods:
1) MSE weight range setting (via a custom observer)
2) Activation loss weight range setting (via precomputing scales with Quanty, and loading them into a manual observer)

"""
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.qualcomm.quantizer.annotators import OP_ANNOTATOR
from executorch.backends.qualcomm.quantizer.observers.per_channel_param_observer import (
    PerChannelParamObserver,
)

from executorch.backends.qualcomm.quantizer.qconfig import (
    _derived_bias_quant_spec,
    QuantizationConfig,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

from executorch.examples.qualcomm.utils import make_quantizer

from torchao.prototype.quantization.module_swap import (
    QuantizationRecipe,
    quantize_module_swap,
    QuantizedLinear,
)
from torchao.prototype.quantization.module_swap.module_swap import (
    get_layer_parent_by_name,
)
from torchao.prototype.quantization.module_swap.quantized_modules import (
    QuantizedEmbedding,
)
from torchao.prototype.quantization.module_swap.range_setting_methods import (
    set_weight_range_activation_loss,
)

from torchao.quantization.pt2e import (
    HistogramObserver,
    MinMaxObserver,
    ObserverBase,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import QuantizationSpec


class PerChannelMSEObserver(PerChannelParamObserver):

    @torch.jit.export
    def forward(self, x_orig):
        # since params are static, one calibration is enough
        if not self.calibrated:
            x = x_orig.detach().to(self.min_val.dtype)
            self.min_val, self.max_val = self.line_search(x)
            self.calibrated = True

        return x_orig



class PerChannelFixedQParamsObserver(PerChannelMinMaxObserver):
    r"""
    Fixed scale that is set manually. Symmetric quantization, so zero point is always zero
    Used for per channel quantization
    If scale not set, defaults to minmax
    """

    def __init__(
        self,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_symmetric,
        quant_min=0,
        quant_max=255,
        is_dynamic=False,
        **kwargs,
    ):
        super().__init__(ch_axis=ch_axis, dtype=dtype, qscheme=qscheme, is_dynamic=is_dynamic, **kwargs)
        self.quant_min = quant_min
        self.quant_max = quant_max

    def set_scale(self, scale, device):
        self.scale = scale.to(device=device)
        self.zero_point = torch.zeros_like(scale).to(device=device)

    @torch.jit.export
    def calculate_qparams(self):
        if hasattr(self, "scale"):
            return self.scale, self.zero_point
        return self._calculate_qparams(self.min_val, self.max_val)


def reverse_quantize_module_swap(model: nn.Module) -> nn.Module:
    """
    Reverse `quantize_module_swap`
        QuantizedLinear --> Linear
        QuantizedEmbedding --> Embedding
    """
    model = reverse_replace_all_linear_with_quantized(model)
    model = reverse_replace_all_embedding_with_quantized(model)
    return model


def reverse_replace_all_embedding_with_quantized(
    model: nn.Module
) -> nn.Module:
    """
    Reverse `replace_all_embedding_with_quantized`
        QuantizedEmbedding --> Embedding
    """
    for name, module in model.named_modules():
        if isinstance(module, QuantizedEmbedding):
            embedding = nn.Embedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                padding_idx=module.padding_idx,
                max_norm=module.max_norm,
                norm_type=module.norm_type,
                scale_grad_by_freq=module.scale_grad_by_freq,
                sparse=module.sparse,
                _weight=module.weight,
            )
            attribute_name = name.rsplit(".", 1)[-1]
            parent_of_module = get_layer_parent_by_name(model, name)
            setattr(parent_of_module, attribute_name, embedding)

    return model


def reverse_replace_all_linear_with_quantized(
    model: nn.Module,
) -> nn.Module:
    """
    Reverse `replace_all_linear_with_quantized_linear`
        QuantizedLinear --> Linear
    """
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            linear = nn.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )
            linear.weight = module.weight
            linear.bias = module.bias

            attribute_name = name.rsplit(".", 1)[-1]
            parent_of_module = get_layer_parent_by_name(model, name)
            setattr(parent_of_module, attribute_name, linear)

    return model


def make_custom_quantizer(quant_dtype, range_setting_weight=None):
    """
    A custom quantizer which uses either the MSE or manual observer, depending
    on the weight range setting method provided.
    """
    quantizer = make_quantizer(
        quant_dtype=quant_dtype,
        per_channel_conv=True,
        per_channel_linear=True,
        act_observer=MinMaxObserver,
    )
    if range_setting_weight in ("mse", "activation_loss"):
        if range_setting_weight == "mse":
            observer = PerChannelMSEObserver.with_args(**{"steps": 200, "use_mse": True})
        else:
            observer = PerChannelFixedQParamsObserver.with_args(**{"eps": 2**-12})
        weight_dtype = (
            torch.int4
            if quant_dtype in (QuantDtype.use_16a4w, QuantDtype.use_16a4w_block)
            else torch.int8
        )
        per_channel_q_config = quantizer.default_quant_config.quant_config
        weight_qspec = QuantizationSpec(
            dtype=torch.int8 if weight_dtype == torch.int4 else weight_dtype,
            quant_min=(
                -7
                if weight_dtype == torch.int4
                else torch.iinfo(weight_dtype).min + 1
            ),
            quant_max=(
                7 if weight_dtype == torch.int4 else torch.iinfo(weight_dtype).max
            ),
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
            observer_or_fake_quant_ctr=observer,
        )
        quantizer.default_quant_config.per_channel_quant_config = (
            QuantizationConfig(
                input_activation=per_channel_q_config.input_activation,
                output_activation=per_channel_q_config.output_activation,
                weight=weight_qspec,
                bias=_derived_bias_quant_spec,
            )
        )

    return quantizer


def compute_scales(model, data, num_points=100, weight_bits=4, activation_bits=16):
    """
    Compute scales for weight quantization using activation loss range setting
    Uses function from Quanty
    1. Peform module swap
    2. Apply method from Quanty to compute optimal scales
    3. Save scales in dictionary
    4. Undo module swap
    """
    recipe = QuantizationRecipe(
        weight_bits=weight_bits,
        weight_quantization=True,
        dynamic_weights=False,
        weight_group_size="per_channel",
        activation_bits=activation_bits,
        activation_quantization=True,
        activation_group_size="per_tensor",
        input_quantization=True,
        output_quantization=True,
        dynamic_activations=False,
    )

    quantized_model = quantize_module_swap(model, recipe)

    set_weight_range_activation_loss(quantized_model, data, 1, num_points) # batch_size = 1
    scale_dict = dict()
    for name, module in quantized_model.named_modules():
        if isinstance(module, QuantizedLinear):
            scale_dict[name] = module.weight_scale.clone().detach().to(device=model.device)

    reverse_quantize_module_swap(model)

    return scale_dict


def set_scales(model, scale_dict, num_heads=32, dim=2048):
    """
    Given a prepared model with manual observers inserted after weights, set scales
    manually. This is specific to Llama architecture, prepared as in the HTP flow
    (For example, we must separate scales because of splitting attention heads)
    """
    head_dim = dim // num_heads
    for node in model.graph.nodes:
        if node.op == "get_attr":
            l = node.target.split(".")
            if len(l) > 3 and l[-3] in ("wq_sha", "wk_sha", "wv_sha"):
                shorter_name = l[-3][:2]
                key = ".".join(["model"] + l[:-3] + [shorter_name])
                observer_name = str(list(node.users.keys())[0])
                observer = getattr(model, observer_name)
                i = int(l[-2])
                observer.set_scale(scale_dict[key][head_dim*i:head_dim*(i + 1), :], device=model.device)
            elif len(l) > 1 and l[-2] in ("wo_sha", "w1_conv", "w2_conv", "w3_conv"):
                shorter_name = l[-2][:2]
                key = ".".join(["model"] + l[:-2] + [shorter_name])
                observer_name = str(list(node.users.keys())[0])
                observer = getattr(model, observer_name)
                observer.set_scale(scale_dict[key], model.device)
