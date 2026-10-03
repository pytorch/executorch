# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm


# ----------------------------------------------------------------------
# Replace
#     Modified from executorch.examples.models.llama.source_transformation.rms_norm.py
# ----------------------------------------------------------------------
def replace_rms_norm_with_native_rms_norm(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, LlamaRMSNorm):
            rms_norm = torch.nn.RMSNorm(
                child.weight.shape[-1], eps=child.variance_epsilon
            )
            rms_norm.weight = child.weight
            setattr(module, name, rms_norm)
        elif isinstance(child, Gemma3RMSNorm):
            rms_norm = torch.nn.RMSNorm(child.weight.shape[-1], eps=child.eps)
            rms_norm.weight = torch.nn.Parameter(child.weight + 1.0)
            setattr(module, name, rms_norm)
        else:
            replace_rms_norm_with_native_rms_norm(child)
    return module


def adding_dequant(gm, state_dict):
    dequantize_op = torch.ops.quantized_decomposed.dequantize_per_channel.default

    def convert_layer_name(name, exceptions):
        # Sort by length descending to avoid partial matches
        exceptions_sorted = sorted(exceptions, key=len, reverse=True)

        # Step 1: protect exceptions with placeholder
        protected = name
        placeholders = {}
        for i, exc in enumerate(exceptions_sorted):
            ph = f"\x00{i}\x00"
            placeholders[ph] = exc
            protected = protected.replace(exc, ph)

        # Step 2: replace remaining '_' with '.'
        replaced = protected.replace("_", ".")

        # Step 3: restore exceptions
        for ph, exc in placeholders.items():
            replaced = replaced.replace(ph, exc)

        return replaced

    EXCEPTIONS = [
        "self_attn",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "input_layernorm",
        "post_attention_layernorm",
        "embed_tokens",
        "lm_head",
    ]

    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.linear.default:
            wei_obs = node.args[1]
            name = wei_obs.args[0]  # layers_0_self_attn_q_proj_weight
            if str(name) == "lm_head_weight":
                break
            name_dict = convert_layer_name(str(name), EXCEPTIONS)
            weight = state_dict["model." + name_dict]
            weight_scale = state_dict["model." + name_dict + "_scale"]  # .squeeze()
            weight_zp = torch.zeros_like(weight_scale)

            frozen_name = str(name) + "_frozen_param_"
            scale_name = str(name) + "_scale_"
            zp_name = str(name) + "_zero_point_"
            gm.register_buffer(
                frozen_name, weight.to(torch.int8)
            )  # wei_observer.w_int.to(torch.int8))
            gm.register_buffer(scale_name, weight_scale)  # wei_observer.scale)
            gm.register_buffer(
                zp_name, weight_zp
            )  # wei_observer.offset.to(torch.int8))

            graph = gm.graph
            with graph.inserting_after(wei_obs):
                frozen_node = graph.create_node("get_attr", frozen_name, (), {})
                scale_node = graph.create_node("get_attr", scale_name, (), {})
                zp_node = graph.create_node("get_attr", zp_name, (), {})
            with graph.inserting_after(frozen_node):
                dq_node = graph.call_function(
                    dequantize_op,
                    args=(
                        frozen_node,
                        scale_node,
                        zp_node,
                        0,  # axis
                        -8,  # qmin
                        7,  # qmax
                        torch.int8,  # dtype
                    ),
                )
            wei_obs.replace_all_uses_with(dq_node)
            gm.graph.erase_node(wei_obs)

    gm.graph.lint()
    gm.recompile()
    return gm
