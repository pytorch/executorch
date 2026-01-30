# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
The goal of this is to allow range setting methods from TorchAO (formerly Quanty)
to be incorporated into the PT2E flow.

We implement the two main range setting methods:
1) MSE weight range setting
2) Activation loss weight range setting

"""

import torch
import torch.nn as nn
from executorch.backends.qualcomm.quantizer.annotators import OP_ANNOTATOR
from executorch.backends.qualcomm.quantizer.observers.per_channel_param_observer import (
    PerChannelParamObserver,
)

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

from torchao.quantization.pt2e import MinMaxObserver, PerChannelMinMaxObserver


class WrappedLlamaModel(nn.Module):
    def __init__(
        self, model, atten_mask, use_kv_cache=False, max_seq_len=512, device="cuda"
    ):
        super(WrappedLlamaModel, self).__init__()
        self.model = model
        self.max_seq_len = max_seq_len
        self.use_kv_cache = use_kv_cache
        self.device = device
        self.atten_mask = atten_mask

    def forward(
        self,
        tokens: torch.Tensor,
        *args,
    ):
        # Pad input if necessary, since LlamaModel requires static shape
        if tokens.shape[1] != self.max_seq_len:
            tokens = torch.nn.functional.pad(
                tokens, (0, self.max_seq_len - tokens.shape[1])
            )
        return self.model.forward(tokens, self.atten_mask)


class PerChannelMSEObserver(PerChannelParamObserver):

    def forward(self, x_orig):
        # since params are static, one calibration is enough
        if not self.calibrated:
            x = x_orig.detach().to(self.min_val.dtype)
            self.min_val, self.max_val = self.line_search(x)
            self.calibrated = True

        return x_orig


class PerChannelFixedQParamsObserver(PerChannelMinMaxObserver):
    r"""
    Fixed scale that you set manually (for per channel quantization)
    Symmetric quantization, so zero point is always zero
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
        super().__init__(
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        self.quant_min = quant_min
        self.quant_max = quant_max

    def set_scale(self, scale):
        self.register_buffer("scale", scale.clone().detach())
        self.register_buffer("zero_point", torch.zeros_like(scale))

    def calculate_qparams(self):
        if hasattr(self, "scale") and hasattr(self, "zero_point"):
            print("Using precomputed scale")
            return self.scale, self.zero_point
        print("Using minmax scale")
        return self._calculate_qparams(self.min_val, self.max_val)


def reverse_quantize_module_swap(model: nn.Module) -> nn.Module:
    model = reverse_replace_all_linear_with_quantized(model)
    model = reverse_replace_all_embedding_with_quantized(
        model
    )  # if embedding_quantize was false, does nothing
    return model


def reverse_replace_all_embedding_with_quantized(model: nn.Module) -> nn.Module:
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

            # logger.info(f"replaced {name} with original embedding")
    return model


def reverse_replace_all_linear_with_quantized(
    model: nn.Module,
) -> nn.Module:
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

            # logger.info(f"replaced {name} with originallinear")
    return model


def compute_scales(model, data, weight_bits, act_bits, num_points=1600):
    recipe = QuantizationRecipe(
        weight_bits=weight_bits,  # TODO: should be based on dtype!
        weight_quantization=True,
        dynamic_weights=False,
        weight_group_size="per_channel",
        activation_bits=act_bits,  # same as above
        activation_quantization=True,
        activation_group_size="per_tensor",
        input_quantization=True,
        output_quantization=True,
        dynamic_activations=False,
    )

    quantized_model = quantize_module_swap(model, recipe)

    set_weight_range_activation_loss(
        quantized_model, data, 1, num_points
    )  # batch_size = 1 for us
    scales_state_dict = {}
    for name, module in quantized_model.named_modules():
        if isinstance(module, QuantizedLinear):
            scales_state_dict[name] = module.weight_scale.clone().detach()

    return scales_state_dict


def make_custom_quantizer(quant_dtype, custom_annotations=(), linear_only=False):
    quantizer = make_quantizer(
        quant_dtype=quant_dtype,
        per_channel_conv=True,
        per_channel_linear=True,
        act_observer=MinMaxObserver,
    )

    if linear_only:
        all_keys = set(OP_ANNOTATOR.keys())
        conv_keys = {
            op
            for op in all_keys
            if op.__name__
            in (
                "conv1d.default",
                "conv2d.default",
                "conv_transpose2d.input",
                "linear.default",
            )
        }
        quantizer.add_discard_ops(all_keys.difference(conv_keys))
    else:
        quantizer.add_custom_quant_annotations(custom_annotations)
    return quantizer


def set_scales(prepared_model, scales_state_dict, head_dim=64):
    for node in prepared_model.graph.nodes:
        if node.op == "get_attr":
            split_target = node.target.split(".")
            if len(split_target) > 3 and split_target[-3] in (
                "wq_sha",
                "wk_sha",
                "wv_sha",
            ):
                shorter = split_target[-3][:2]
                key = ".".join(["model"] + split_target[:-3] + [shorter])
                observer_name = str(list(node.users.keys())[0])
                observer = getattr(prepared_model, observer_name)
                i = int(split_target[-2])
                try:
                    observer.set_scale(
                        scales_state_dict[key][head_dim * i : head_dim * (i + 1), :]
                    )
                    print("Set scale for", key)
                except Exception:
                    print("Failed to set scale for ", key, node.target)
            elif len(split_target) > 1 and split_target[-2] in (
                "wo_sha",
                "w1_conv",
                "w2_conv",
                "w3_conv",
            ):
                shorter = split_target[-2][:2]
                key = ".".join(["model"] + split_target[:-2] + [shorter])
                observer_name = str(list(node.users.keys())[0])
                observer = getattr(prepared_model, observer_name)
                try:
                    observer.set_scale(scales_state_dict[key])
                    print("Set scale for", key)
                except Exception:
                    print("Failed to set scale for ", key, node.target)
            elif len(split_target) > 2 and split_target[-3] == "output":
                key = ".".join(["model"] + split_target[:-2])
                observer_name = str(list(node.users.keys())[0])
                observer = getattr(prepared_model, observer_name)
                try:
                    observer.set_scale(scales_state_dict[key])
                    print("Set scale for", key)
                except Exception:
                    print("Failed to set scale for ", key, node.target)
