# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""
Gemma 4 Model wrapper for ExecuTorch export.

Provides interface for loading, configuring, and exporting the Gemma 4 text decoder.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .gemma4_config import Gemma4Config
from .gemma4_transformer import Gemma4ForCausalLM

logger = logging.getLogger(__name__)


class Gemma4Model:
    """ExecuTorch-compatible Gemma 4 model wrapper.

    Handles:
    - Model configuration
    - Checkpoint loading with weight conversion
    - Example input generation for export

    Args:
        config: Optional Gemma4Config. If not provided, loads from E2B config.
        checkpoint_path: Optional path to checkpoint.
        dtype: Data type for model weights.
        use_kv_cache: Whether to use KV cache for generation.
    """

    def __init__(
        self,
        config: Optional[Gemma4Config] = None,
        checkpoint_path: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        use_kv_cache: bool = True,
    ):
        if config is None:
            config = Gemma4Config.from_e2b_config()

        config.use_kv_cache = use_kv_cache

        self.config = config
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path

        self.model_ = Gemma4ForCausalLM(config)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        else:
            self._init_random_weights()

        self.model_ = self.model_.to(dtype=dtype)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint with weight conversion."""
        from .convert_weights import convert_hf_to_custom

        state_dict = convert_hf_to_custom(
            checkpoint_path, self.config, dtype=self.dtype
        )

        result = self.model_.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            logger.warning(
                f"Missing keys ({len(result.missing_keys)}): {result.missing_keys[:10]}"
            )
        if result.unexpected_keys:
            logger.warning(
                f"Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:10]}"
            )

        self._init_missing_parameters()

    def _init_missing_parameters(self) -> None:
        """Initialize any parameters that weren't loaded from checkpoint.

        Note: RMSNorm weights are initialized to zeros (offset convention)
        and loaded from checkpoint. No special initialization needed.
        """
        pass

    def _init_random_weights(self) -> None:
        """Initialize model with random weights for testing."""
        self.model_ = Gemma4ForCausalLM(self.config)

        def init_fn(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.model_.apply(init_fn)

    def get_eager_model(self) -> nn.Module:
        """Get the eager PyTorch model."""
        return self.model_

    def get_example_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Get example inputs for model tracing/export (single token decode)."""
        input_ids = torch.tensor([[1]], dtype=torch.long)
        inputs_embeds = None

        if self.config.use_kv_cache:
            input_pos = torch.tensor([0], dtype=torch.long)
            return (input_ids, input_pos, inputs_embeds)
        else:
            return (input_ids, inputs_embeds)

    def get_example_inputs_prefill(self, seq_len: int = 10) -> Tuple[torch.Tensor, ...]:
        """Get example inputs for prefill (multiple tokens)."""
        input_ids = torch.ones((1, seq_len), dtype=torch.long)
        inputs_embeds = None

        if self.config.use_kv_cache:
            input_pos = torch.arange(seq_len, dtype=torch.long)
            return (input_ids, input_pos, inputs_embeds)
        else:
            return (input_ids, inputs_embeds)

    def get_example_inputs_with_audio(
        self, seq_len: int = 200, num_audio_tokens: int = 750
    ) -> Tuple[torch.Tensor, ...]:
        """Get example inputs for prefill with audio embeddings.

        Args:
            seq_len: Total number of tokens in the sequence (text + audio).
            num_audio_tokens: Number of audio tokens (default 750 for Gemma 4).
        """
        input_ids = torch.ones((1, seq_len), dtype=torch.long)

        inputs_embeds = torch.randn(
            1, seq_len, self.config.hidden_size, dtype=self.dtype
        )

        if self.config.use_kv_cache:
            input_pos = torch.arange(seq_len, dtype=torch.long)
            return (input_ids, input_pos, inputs_embeds)
        else:
            return (input_ids, inputs_embeds)

    def get_dynamic_shapes(
        self, with_audio_embeds: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get dynamic shape specifications for export."""
        if not self.config.enable_dynamic_shape:
            return None

        from torch.export import Dim

        seq_len = Dim("seq_len", min=1, max=self.config.max_seq_len - 1)

        if self.config.use_kv_cache:
            if with_audio_embeds:
                return {
                    "input_ids": {1: seq_len},
                    "input_pos": {0: seq_len},
                    "inputs_embeds": {1: seq_len},
                }
            else:
                return {
                    "input_ids": {1: seq_len},
                    "input_pos": {0: seq_len},
                }
        else:
            if with_audio_embeds:
                return {
                    "input_ids": {1: seq_len},
                    "inputs_embeds": {1: seq_len},
                }
            else:
                return {
                    "input_ids": {1: seq_len},
                }


def create_gemma4_model(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    use_kv_cache: bool = True,
) -> Gemma4Model:
    """Factory function to create Gemma4 model."""
    if config_path is not None:
        config = Gemma4Config.from_json(config_path)
    else:
        config = None

    return Gemma4Model(
        config=config,
        checkpoint_path=checkpoint_path,
        dtype=dtype,
        use_kv_cache=use_kv_cache,
    )
