# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FLUX.2 Klein text conditioning."""

from .flux2 import _Flux2PipelineBehavior


class Flux2KleinPipelineBehavior(_Flux2PipelineBehavior):
    """Share FLUX.2 execution with Qwen conditioning and true CFG."""

    name = "flux2_klein"
    supports_cfg = True
    supports_negative_prompt = True
    default_cfg_scale = 4.0

    def tokenize_prompt(self, tokenizer, prompt, *, max_length):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return super().tokenize_prompt(tokenizer, text, max_length=max_length)
