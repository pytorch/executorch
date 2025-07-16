# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def convert_configs(config):
    # HF config keys are different from Llama configs.
    # Convert the config keys to align with Llama.
    if hasattr(config, "hidden_size"):
        config.dim = config.hidden_size
        delattr(config, "hidden_size")

    if hasattr(config, "num_attention_heads"):
        config.n_heads = config.num_attention_heads
        delattr(config, "num_attention_heads")

    if hasattr(config, "num_key_value_heads"):
        config.n_kv_heads = config.num_key_value_heads
        delattr(config, "num_key_value_heads")

    if hasattr(config, "rms_norm_eps"):
        config.norm_eps = config.rms_norm_eps
        delattr(config, "rms_norm_eps")

    if hasattr(config, "rope_theta"):
        config.rope_freq_base = config.rope_theta
        delattr(config, "rope_theta")

    if hasattr(config, "num_hidden_layers"):
        config.n_layers = config.num_hidden_layers
        delattr(config, "num_hidden_layers")

    if hasattr(config, "intermediate_size"):
        config.hidden_dim = config.intermediate_size
        delattr(config, "intermediate_size")

    if hasattr(config, "rope_scaling"):
        config.use_scaled_rope = config.rope_scaling
    # Use default value of precompute_freq_cis
    if not hasattr(config, "rope_scale_factor"):
        config.rope_scale_factor = 4

    return config
