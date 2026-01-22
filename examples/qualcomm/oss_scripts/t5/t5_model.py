# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5Config
from transformers.cache_utils import DynamicCache, EncoderDecoderCache, StaticCache
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Attention, T5Stack
from transformers.utils import logging

logger = logging.get_logger(__name__)


# Copy from transformers/models/t5/modeling_t5.py (transformers=4.47.1)
class CustomT5Stack(T5Stack):
    def __init__(
        self,
        config,
        max_hidden_seq_length=4096,
        max_cache_length=1024,
    ):
        super().__init__(config)

        # ====================Qualcomm Changed=================================
        # Customized position bias computation:
        # Since the calculation in `T5Attention._relative_position_bucket` is not QNN-friendly,
        # we precompute the relative position buckets as constant tensors during initialization.
        # For the encoder: use the precomputed `encoder_self_attn_position_bias`.
        # For the decoder: use the precomputed `decoder_self_attn_position_bias`.

        self.max_hidden_seq_length = max_hidden_seq_length
        self.max_cache_length = max_cache_length

        # Create relative position table for encoder
        encoder_self_attn_relative_position_bucket = (
            T5Attention._relative_position_bucket(
                torch.arange(max_hidden_seq_length)[None, :]
                - torch.arange(max_hidden_seq_length)[:, None],
                bidirectional=(not self.is_decoder),
                num_buckets=config.relative_attention_num_buckets,
                max_distance=config.relative_attention_max_distance,
            )
        )
        self.register_buffer(
            "encoder_self_attn_position_bias",
            encoder_self_attn_relative_position_bucket,
        )

        # Create relative position table for decoder
        decoder_self_attn_relative_position_bucket = (
            T5Attention._relative_position_bucket(
                torch.arange(max_cache_length)[None, :]
                - torch.arange(max_cache_length)[:, None],
                bidirectional=(not self.is_decoder),
                num_buckets=config.relative_attention_num_buckets,
                max_distance=config.relative_attention_max_distance,
            )
        )
        self.register_buffer(
            "decoder_self_attn_position_bias",
            decoder_self_attn_relative_position_bucket,
        )
        # ========================================================================

    def forward(  # noqa: C901
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError(
                    "You have to initialize the model with valid token embeddings"
                )
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(
                    f"`use_cache` can only be set to `True` if {self} is used as a decoder"
                )

        if self.is_decoder:
            if use_cache and past_key_values is None:
                if self.config.is_encoder_decoder:
                    past_key_values = EncoderDecoderCache(
                        DynamicCache(config=self.config),
                        DynamicCache(config=self.config),
                    )
                else:
                    past_key_values = DynamicCache(config=self.config)
            # ====================Qualcomm Changed=================================
            else:
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            # =====================================================================
        elif not self.is_decoder:
            # do not pass cache object down the line for encoder stack
            # it messes indexing later in decoder-stack because cache object is modified in-place
            past_key_values = None

        past_key_values_length = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_length,
                device=inputs_embeds.device,
            )

        if self.config.is_decoder:
            attention_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=(
                    past_key_values.self_attention_cache
                    if isinstance(past_key_values, EncoderDecoderCache)
                    else past_key_values
                ),
            )
        else:
            attention_mask = create_bidirectional_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

        encoder_extended_attention_mask = None
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_extended_attention_mask = create_bidirectional_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        # ====================Qualcomm Changed=================================
        # The bias is indexed by cache_position to select the correct positions for the current step.
        if self.is_decoder:
            # For decoder, use the decoder's relative position bias table.
            position_bias = (
                self.block[0]
                .layer[0]
                .SelfAttention.relative_attention_bias(
                    self.decoder_self_attn_position_bias[cache_position]
                )
                .permute([2, 0, 1])
                .unsqueeze(0)
            )
        else:
            # For encoder, use the encoder's relative position bias table.
            position_bias = (
                self.block[0]
                .layer[0]
                .SelfAttention.relative_attention_bias(
                    self.encoder_self_attn_position_bias[cache_position]
                )
                .permute([2, 0, 1])
                .unsqueeze(0)
            )
        position_bias = position_bias[:, :, -seq_length:, :]
        if self.is_decoder:
            position_bias = (
                position_bias + attention_mask[:, :, :, : self.max_cache_length]
            )
        else:
            position_bias = position_bias + attention_mask[:, :, :, :seq_length]

        # For cross-attention in decoder, precompute encoder-decoder position bias as zeros and add encoder attention mask.
        encoder_decoder_position_bias = None
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_decoder_position_bias = torch.zeros(
                (1, self.config.num_heads, seq_length, self.max_hidden_seq_length),
                dtype=encoder_extended_attention_mask.dtype,
            )
            encoder_decoder_position_bias = (
                encoder_decoder_position_bias
                + encoder_extended_attention_mask[:, :, :, : self.max_hidden_seq_length]
            )
        # =====================================================================

        hidden_states = self.dropout(inputs_embeds)

        for layer_module in self.block:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,  # as a positional argument for gradient checkpointing
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[1]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    3 if output_attentions else 2
                ]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class Seq2SeqLMEncoderExportableModule(torch.nn.Module):
    def __init__(self, model, max_hidden_seq_length):
        super().__init__()
        self.config = model.config
        self.encoder = model.get_encoder()
        self.max_hidden_seq_length = max_hidden_seq_length

    def get_example_inputs(self):
        max_hidden_seq_length = self.max_hidden_seq_length
        input_ids = torch.randint(0, max_hidden_seq_length, (1, max_hidden_seq_length))
        attn_mask = torch.randn((1, 1, 1, max_hidden_seq_length))
        return input_ids, attn_mask

    def forward(self, input_ids, attn_mask):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
        )
        return encoder_outputs.last_hidden_state


class Seq2SeqLMDecoderExportableModuleWithStaticCache(torch.nn.Module):
    def __init__(
        self,
        model,
        max_hidden_seq_length,
        max_static_cache_length,
        batch_size,
    ):
        super().__init__()

        # Get the decoder component
        self.decoder = model.get_decoder()
        self.proj_out = model.lm_head
        self.config = model.config
        self.max_hidden_seq_length = max_hidden_seq_length
        self.max_static_cache_length = max_static_cache_length

        # Initialize static cache
        self.static_cache = StaticCache(
            config=self.config,
            max_batch_size=batch_size,
            max_cache_len=max_static_cache_length,
            device="cpu",
            dtype=torch.float32,
        )
        head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        num_heads = getattr(
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        self.static_cache.early_initialization(
            batch_size, num_heads, head_dim, torch.float32, "cpu"
        )

        # Register cache buffers to make them exportable
        for i in range(len(self.static_cache.layers)):
            self.register_buffer(
                f"key_cache_{i}", self.static_cache.layers[i].keys, persistent=False
            )
            self.register_buffer(
                f"value_cache_{i}",
                self.static_cache.layers[i].values,
                persistent=False,
            )

    def get_example_inputs(self):
        max_hidden_seq_length = self.max_hidden_seq_length
        hidden_size = self.config.d_model
        decoder_input_ids = torch.tensor([[0]], dtype=torch.long)
        attn_mask = torch.full(
            (1, 1, 1, self.max_static_cache_length),
            fill_value=-255.0,
            dtype=torch.float32,
        )
        attn_mask[..., 0] = 0
        encoder_hidden_states = torch.randn(1, max_hidden_seq_length, hidden_size)
        encoder_attn_mask = torch.randn((1, 1, 1, max_hidden_seq_length))
        cache_position = torch.tensor([0], dtype=torch.long)
        return (
            decoder_input_ids,
            attn_mask,
            encoder_hidden_states,
            encoder_attn_mask,
            cache_position,
        )

    def forward(
        self,
        decoder_input_ids,
        attn_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        cache_position,
    ):
        # Get outputs from decoder
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=self.static_cache,
            use_cache=True,
            cache_position=cache_position,
        )
        sequence_output = outputs[0]
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.config.d_model**-0.5)

        # Apply linear projection (lm head) to obtain logits
        logits = self.proj_out(sequence_output)
        return logits

    def get_metadata(self):
        return {
            "get_eos_id": getattr(self.config, "eos_token_id", None),
            "get_max_context_len": self.max_static_cache_length,
            "max_hidden_seq_length": self.max_hidden_seq_length,
        }


class Seq2SeqLMExportableModulePipeline(torch.nn.Module):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: T5Config,
        max_hidden_seq_length=4096,
        max_seq_len=1024,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_len = max_seq_len

        self.max_hidden_seq_length = max_hidden_seq_length

    def __call__(
        self,
        encoder,
        decoder,
        dataset,
    ):
        self.validate(encoder, decoder, dataset, None, None)

    def validate(
        self,
        encoder,
        decoder,
        dataset,
        targets: Optional[List[torch.Tensor]] = None,
        metrics: Optional[callable] = None,
    ):
        predicted_texts = []
        target_texts = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataset)):

                token_list = self.generate(encoder, decoder, data)

                if targets is None:
                    continue

                predicted_texts.append(
                    self.tokenizer.decode(token_list[0], skip_special_tokens=True)
                )
                target_texts.append(
                    self.tokenizer.decode(targets[i], skip_special_tokens=True)
                )
                print(f"Show {i}/{len(dataset)} result:")
                print(f"\tPrediction: {predicted_texts[i]}")
                print(f"\tTarget:    {target_texts[i]}")

        if targets is None or metrics is None:
            print("No targets or metrics provided for validation.")
        else:
            results = metrics(predicted_texts, target_texts)
            print("F1 Score:", results["f1"])

    def generate(self, encoder, decoder, data):
        prompt_token_ids, encoder_attn_mask, decoder_input_ids = data

        min_dtype = torch.finfo(torch.float32).min
        attn_mask = torch.full(
            (1, 1, 1, self.max_seq_len), fill_value=min_dtype, dtype=torch.float32
        )
        attn_mask[..., 0] = 0

        with torch.no_grad():
            # Run encoder
            encoder_output = encoder(prompt_token_ids, encoder_attn_mask)
            generated_ids = [0]

            # Generate tokens one by one
            for i in range(self.max_seq_len - 1):
                # Run decoder for next token prediction
                logits = decoder(
                    decoder_input_ids,
                    attn_mask,
                    encoder_output,
                    encoder_attn_mask,
                    torch.tensor([i], dtype=torch.long),
                )

                # Get next token
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                generated_ids.append(next_token)

                # Update input for next iteration
                decoder_input_ids = torch.tensor([[next_token]], dtype=torch.long)

                # Check if EOS token
                if next_token == self.config.eos_token_id:
                    break

                # update attn_mask
                attn_mask[..., i] = 0

            return [generated_ids]

    @staticmethod
    def evaluate_with_ground_truth(
        tokenizer: AutoTokenizer,
        predicts: str,
        targets: Optional[List[torch.Tensor]],
        metrics: Optional[callable],
    ):
        predicted_texts = []
        target_texts = []
        for i, (pred, tar) in tqdm(enumerate(zip(predicts, targets))):

            predicted_texts.append(pred)
            target_texts.append(tokenizer.decode(tar, skip_special_tokens=True))
            print(f"Show {i}/{len(predicts)} result:")
            print(f"\tPrediction: {pred}")
            print(f"\tTarget:    {target_texts[i]}")
        results = metrics(predicted_texts, target_texts)
        print("F1 Score:", results["f1"])

        return results
