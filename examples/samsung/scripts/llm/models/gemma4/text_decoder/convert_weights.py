# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""
Weight conversion from a HuggingFace Gemma 4 model to the custom
ExecuTorch text_decoder format.

Maps HuggingFace model weights to the custom architecture, handling:
- Layer name mapping (self/cross decoder split)
- PLE (per-layer embedding) weight splitting
- Tied embeddings (embed_tokens -> lm_head)
- SRQ clamp semantics via ClampLinear replacement
"""

from .clamp_linear import ClampLinear
from .gemma4_config import Gemma4Config


def _get_weight_mapping(config: Gemma4Config):
    """Get mapping from HuggingFace weight names to custom weight names."""
    mapping = {}

    # Embedding mappings
    mapping["embed_tokens.weight"] = "model.self_decoder.embed_tokens.weight"
    mapping["embed_tokens_per_layer.weight"] = (
        "model.self_decoder.embed_tokens_per_layer.weight"
    )
    mapping["per_layer_model_projection.weight"] = (
        "model.self_decoder.per_layer_model_projection.weight"
    )
    mapping["per_layer_projection_norm.weight"] = (
        "model.self_decoder.per_layer_projection_norm.weight"
    )

    # Final norm
    mapping["norm.weight"] = "model.norm.weight"

    # Layer mappings
    num_self_layers = config.num_self_decoder_layers
    for layer_idx in range(config.num_hidden_layers):
        hf_prefix = f"layers.{layer_idx}"

        if layer_idx < num_self_layers:
            custom_prefix = f"model.self_decoder.layers.{layer_idx}"
        else:
            cross_idx = layer_idx - num_self_layers
            custom_prefix = f"model.cross_decoder.layers.{cross_idx}"

        # Attention
        mapping[f"{hf_prefix}.self_attn.q_proj.weight"] = (
            f"{custom_prefix}.self_attn.q_proj.weight"
        )
        mapping[f"{hf_prefix}.self_attn.k_proj.weight"] = (
            f"{custom_prefix}.self_attn.k_proj.weight"
        )
        mapping[f"{hf_prefix}.self_attn.v_proj.weight"] = (
            f"{custom_prefix}.self_attn.v_proj.weight"
        )
        mapping[f"{hf_prefix}.self_attn.o_proj.weight"] = (
            f"{custom_prefix}.self_attn.o_proj.weight"
        )
        mapping[f"{hf_prefix}.self_attn.q_norm.weight"] = (
            f"{custom_prefix}.self_attn.q_norm.weight"
        )
        mapping[f"{hf_prefix}.self_attn.k_norm.weight"] = (
            f"{custom_prefix}.self_attn.k_norm.weight"
        )

        # MLP
        mapping[f"{hf_prefix}.mlp.gate_proj.weight"] = (
            f"{custom_prefix}.mlp.gate_proj.weight"
        )
        mapping[f"{hf_prefix}.mlp.up_proj.weight"] = (
            f"{custom_prefix}.mlp.up_proj.weight"
        )
        mapping[f"{hf_prefix}.mlp.down_proj.weight"] = (
            f"{custom_prefix}.mlp.down_proj.weight"
        )

        # Layer scalar
        mapping[f"{hf_prefix}.layer_scalar"] = f"{custom_prefix}.layer_scalar"

        # Per-layer input processing
        mapping[f"{hf_prefix}.per_layer_input_gate.weight"] = (
            f"{custom_prefix}.per_layer_input_gate.weight"
        )
        mapping[f"{hf_prefix}.per_layer_projection.weight"] = (
            f"{custom_prefix}.per_layer_projection.weight"
        )

        # LayerNorms
        mapping[f"{hf_prefix}.input_layernorm.weight"] = (
            f"{custom_prefix}.input_layernorm.weight"
        )
        mapping[f"{hf_prefix}.post_attention_layernorm.weight"] = (
            f"{custom_prefix}.post_attention_layernorm.weight"
        )
        mapping[f"{hf_prefix}.pre_feedforward_layernorm.weight"] = (
            f"{custom_prefix}.pre_feedforward_layernorm.weight"
        )
        mapping[f"{hf_prefix}.post_feedforward_layernorm.weight"] = (
            f"{custom_prefix}.post_feedforward_layernorm.weight"
        )
        mapping[f"{hf_prefix}.post_per_layer_input_norm.weight"] = (
            f"{custom_prefix}.post_per_layer_input_norm.weight"
        )

    return mapping


def convert_hf_to_custom(model, config: Gemma4Config):
    hf_state_dict = model.state_dict()

    dtype = next(model.parameters()).dtype

    mapping = _get_weight_mapping(config)

    # Track which custom modules must become ClampLinear and carry the
    # SRQ scales transferred from the source ClampLinear.
    clamp_info = []  # list of (custom_module_path, in_scale, out_scale, a_bits)
    for path, module in model.named_modules():
        if isinstance(module, ClampLinear):
            w_key = f"{path}.weight"
            if w_key in mapping:
                custom_mod = mapping[w_key].rsplit(".weight", 1)[0]
                clamp_info.append(
                    (
                        custom_mod,
                        module.in_scale,
                        module.out_scale,
                        module.a_bits,
                    )
                )

    converted_state_dict = {}
    unmapped_keys = []

    for hf_key, tensor in hf_state_dict.items():
        if hf_key in mapping:
            custom_key = mapping[hf_key]
            if dtype is not None:
                tensor = tensor.to(dtype)
            converted_state_dict[custom_key] = tensor
        else:
            unmapped_keys.append(hf_key)

    # Handle tied embeddings: copy embed_tokens.weight -> lm_head.weight if not in checkpoint
    lm_head_key = "model.lm_head.weight"
    embed_tokens_key = "model.self_decoder.embed_tokens.weight"
    if (
        lm_head_key not in converted_state_dict
        and embed_tokens_key in converted_state_dict
    ):
        converted_state_dict[lm_head_key] = converted_state_dict[
            embed_tokens_key
        ].clone()

    # Per-layer embedding: the custom decoder's ModuleList expects one
    # embedding per layer; split the packed [vocab, num_layers, hidden_per_layer] weight.
    ple_key = "model.self_decoder.embed_tokens_per_layer.weight"
    if ple_key not in converted_state_dict:
        raise ValueError(
            f"'{ple_key}' is missing from the converted state dict, but the ENN "
            "text_decoder requires per-layer embeddings"
        )
    ple_weight = converted_state_dict.pop(ple_key).reshape(
        -1,
        config.num_hidden_layers,
        config.hidden_size_per_layer_input,
    )
    for idx in range(config.num_hidden_layers):
        converted_state_dict[
            f"model.self_decoder.embed_tokens_per_layer.{idx}.weight"
        ] = ple_weight[:, idx]
    return converted_state_dict, clamp_info


def replace_linear_with_clamp(root, clamp_info, device, dtype):
    # Swap the mapped nn.Linear projections in the custom decoder for
    # ClampLinear so the clamp (SRQ) semantics and scales are carried into
    # the exported text_decoder. The real weights are loaded afterwards by
    # load_state_dict; the scales are passed here directly.
    for custom_mod, in_scale, out_scale, a_bits in clamp_info:
        parent_path, leaf = custom_mod.rsplit(".", 1)
        parent = root.get_submodule(parent_path)
        lin = getattr(parent, leaf)
        new = ClampLinear(
            in_features=lin.in_features,
            out_features=lin.out_features,
            weight=lin.weight.data,
            bias=lin.bias.data if lin.bias is not None else None,
            in_scale=in_scale,
            out_scale=out_scale,
            a_bits=a_bits,
        )
        new.to(device=device, dtype=dtype)
        setattr(parent, leaf, new)
