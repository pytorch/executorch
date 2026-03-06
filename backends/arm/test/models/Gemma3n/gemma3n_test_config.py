# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""This file defines test configs used to initialize Gemma3n module tests.

Module tests in the same directory will import these configs.

"""

from transformers.models.gemma3n.configuration_gemma3n import (
    Gemma3nAudioConfig,
    Gemma3nTextConfig,
)


def get_gemma3n_text_test_config() -> Gemma3nTextConfig:
    config = Gemma3nTextConfig(
        hidden_size=2048,
        hidden_size_per_layer_input=256,
        rms_norm_eps=1e-6,
        layer_types=None,
        altup_num_inputs=4,
        altup_active_idx=0,
        altup_correct_scale=True,
    )
    # Force eager attention path to keep the module exportable in tests.
    config._attn_implementation = "eager"
    return config


def get_gemma3n_audio_test_config() -> Gemma3nAudioConfig:
    return Gemma3nAudioConfig(
        input_feat_size=128,
        hidden_size=256,
        rms_norm_eps=1e-6,
    )
