# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from transformers import StaticCache, WhisperForConditionalGeneration


class Seq2SeqLMEncoderExportableModule(torch.nn.Module):
    """
    A wrapper module designed to make a Seq2Seq LM encoder exportable with `torch.export`.
    This module ensures that the exported encoder model is compatible with ExecuTorch.
    """

    def __init__(self, encoder_model):
        super().__init__()
        self.encoder = encoder_model

    def forward(self, input_ids):
        return self.encoder(input_ids).last_hidden_state

    def get_example_inputs(self):
        return (torch.rand(1, 80, 3000),)

    def get_metadata(self):
        return {}


class Seq2SeqLMDecoderExportableModuleWithStaticCache(torch.nn.Module):
    """
    A wrapper module designed to make a Seq2Seq LM decoder exportable with `torch.export`,
    specifically for use with static caching. This module ensures the exported decoder
    is compatible with ExecuTorch.
    """

    def __init__(self, whisper_model, max_cache_length, batch_size):
        super().__init__()

        # Get the decoder component
        self.decoder = whisper_model.get_decoder()
        if isinstance(whisper_model, WhisperForConditionalGeneration):
            self.proj_out = whisper_model.proj_out
        else:
            self.proj_out = whisper_model.lm_head
        self.config = whisper_model.config
        self.batch_size = batch_size
        self.max_cache_length = max_cache_length

        # Initialize static cache
        self.static_cache = StaticCache(
            config=self.config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_length,
            device="cpu",
            dtype=torch.float32,
        )

        # Register cache buffers to make them exportable
        for i in range(len(self.static_cache.key_cache)):
            self.register_buffer(f"key_cache_{i}", self.static_cache.key_cache[i])
            self.register_buffer(f"value_cache_{i}", self.static_cache.value_cache[i])

    def forward(
        self, decoder_input_ids, attention_mask, encoder_hidden_states, cache_position
    ):
        # Get outputs from decoder
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=self.static_cache,
            use_cache=True,
            cache_position=cache_position,
        )

        # Apply linear projection (lm head) to obtain logits
        logits = self.proj_out(outputs[0])
        return logits

    def get_example_inputs(self):
        input_ids = torch.tensor([[0]], dtype=torch.long)
        encoder_hidden_states = torch.rand(1, 1500, 384)
        cache_position = torch.tensor([0], dtype=torch.long)
        atten_mask = torch.full((1, self.max_cache_length), torch.tensor(-255.0))
        atten_mask *= torch.arange(self.max_cache_length) > cache_position.reshape(
            -1, 1
        )
        atten_mask = atten_mask[None, None, :, :].expand(self.batch_size, 1, -1, -1)
        return (input_ids, atten_mask, encoder_hidden_states, cache_position)

    def get_metadata(self):
        return {
            "get_eos_id": getattr(self.config, "eos_token_id", None),
            "get_max_context_len": self.max_cache_length,
            "decoder_start_token_id": getattr(
                self.config, "decoder_start_token_id", None
            ),
        }
