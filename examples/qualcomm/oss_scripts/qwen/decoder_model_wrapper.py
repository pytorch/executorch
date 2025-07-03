# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from transformers import (
    PreTrainedModel,
    StaticCache,
)
from executorch.examples.models.llama.rope import hf_precompute_freqs_cis


def _get_freqs_cis(config):
    freqs_cos, freqs_sin = hf_precompute_freqs_cis(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
            config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        )

    freqs_cos = freqs_cos[:, :config.head_dim//2]
    freqs_sin = freqs_sin[:, :config.head_dim//2]
    return freqs_cos, freqs_sin

# Copy from transformers/integrations/executorch.py, transformers version: 4.47.1
class TorchExportableModuleWithStaticCache(torch.nn.Module):
    """
    A wrapper module designed to make a `PreTrainedModel` exportable with `torch.export`,
    specifically for use with static caching. This module ensures that the exported model
    is compatible with further lowering and execution in `ExecuTorch`.

    Note:
        This class is specifically designed to support export process using `torch.export`
        in a way that ensures the model can be further lowered and run efficiently in `ExecuTorch`.
    """

    def __init__(self, model: PreTrainedModel):
        """
        Initializes the wrapper module with the pretrained model.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap. The model must have caching
            enabled and use a 'static' caching implementation.

        Raises:
            AssertionError: If the pretrained model does not have caching enabled or if it does
            not use a 'static' caching implementation in `model.generation_config`.
        """
        super().__init__()

        # Sanity checks
        if model.generation_config is None:
            raise AssertionError(
                "The model must have a generation config to be exported with static caching. "
                "Please set `generation_config`."
            )

        if not model.generation_config.use_cache:
            raise AssertionError(
                "The model must have caching enabled to be exported with static caching. "
                "Please set `generation_config.use_cache=True`."
            )

        if model.generation_config.cache_implementation != "static":
            raise AssertionError(
                "The model must use a 'static' caching implementation to be exported with static caching. "
                "Please set `generation_config.cache_implementation='static'`."
            )

        self.model = model
        self.static_cache = StaticCache(
            config=self.model.config,
            batch_size=self.model.generation_config.cache_config.batch_size,
            max_cache_len=self.model.generation_config.cache_config.max_cache_len,
            dtype=self.model.dtype,
        )
        for i in range(len(self.static_cache.key_cache)):
            self.register_buffer(f"key_cache_{i}", self.static_cache.key_cache[i])
            self.register_buffer(f"value_cache_{i}", self.static_cache.value_cache[i])
        self.is_causal = any("CausalLM" in arch for arch in self.model.config.architectures)
        # ====================Qualcomm Changed=================================
        if self.is_causal:
            mask = torch.full(
                (1, 1, self.static_cache.max_cache_len, self.static_cache.max_cache_len),
                float("-65504"),
            )
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask, persistent=False)

        freqs_cos, freqs_sin = _get_freqs_cis(self.model.config)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        # =====================================================================

    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        """
        Forward pass of the module, which is compatible with the ExecuTorch runtime.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the module.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.

        This forward adapter serves two primary purposes:

        1. **Making the Model `torch.export`-Compatible**:
            The adapter hides unsupported objects, such as the `Cache`, from the graph inputs and outputs,
            enabling the model to be exportable using `torch.export` without encountering issues.

        2. **Ensuring Compatibility with `ExecuTorch` runtime**:
            The adapter matches the model's forward signature with that in `executorch/extension/llm/runner`,
            ensuring that the exported model can be executed in `ExecuTorch` out-of-the-box.
        """
        # ====================Qualcomm Changed=================================
        freqs_cos = self.freqs_cos[cache_position]
        freqs_sin = self.freqs_sin[cache_position]

        attn_mask = self.mask[:,:,cache_position, :] if self.is_causal else None
        # =====================================================================
        outs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=(freqs_cos, freqs_sin),
            cache_position=cache_position,
            past_key_values=self.static_cache,
            use_cache=True,
        )
        return outs.logits

    def get_example_inputs(self):
        example_input_ids = torch.tensor([[1]], dtype=torch.long)
        example_cache_position = torch.tensor([0], dtype=torch.long)
        return (example_input_ids, example_cache_position)

    def get_metadata(self):
        return {}