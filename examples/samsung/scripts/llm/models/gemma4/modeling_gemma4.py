# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.examples.samsung.scripts.llm.models.gemma4.text_decoder.clamp_linear import (
    ClampLinear,
)
from executorch.examples.samsung.scripts.llm.models.gemma4.text_decoder.convert_weights import (
    convert_hf_to_custom,
    replace_linear_with_clamp,
)
from executorch.examples.samsung.scripts.llm.models.gemma4.text_decoder.gemma4_config import (
    Gemma4Config as ExecuGemma4Config,
)
from executorch.examples.samsung.scripts.llm.util.util import free_memory
from transformers import AutoModel
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear
from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4Config as HFGemma4Config,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4CausalLMOutputWithPast,
    Gemma4MultimodalEmbedder,
    Gemma4PreTrainedModel,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


def _patch_hf_vision_for_export():
    """Monkey-patch Gemma4VisionPooler to avoid aten.div.Tensor_mode.

    Modifies the HF class globally and permanently for the lifetime of the
    process: everything in this process using Gemma4VisionPooler (including
    plain HF reference runs) sees the patched behavior. Run any HF-reference
    comparisons before calling this.
    """
    from transformers.models.gemma4 import modeling_gemma4

    pooler_cls = modeling_gemma4.Gemma4VisionPooler
    if getattr(pooler_cls, "_enn_patched", False):
        return

    def _avg_pool_by_positions(
        self, hidden_states: torch.Tensor, pixel_position_ids: torch.Tensor, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {length}: {k=}^2 times {length=} must be {input_seq_len}."
            )

        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = clamped_positions // k
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    pooler_cls._avg_pool_by_positions = _avg_pool_by_positions
    pooler_cls._enn_patched = True


class VisionEncoderWithProjection_ENN(torch.nn.Module):
    """Thin wrapper combining HF's Gemma4VisionModel + Gemma4MultimodalEmbedder.

    Calls sub-components directly instead of vision_tower.forward() because
    the top-level forward does dynamic boolean indexing (hidden_states[mask])
    to strip padding, which is incompatible with torch.export.
    """

    def __init__(self, vision_tower, embed_vision):
        super().__init__()
        self.patch_embedder = vision_tower.patch_embedder
        self.encoder = vision_tower.encoder
        self.pooler = vision_tower.pooler
        self.embed_vision = embed_vision
        self.standardize = getattr(vision_tower.config, "standardize", False)
        if self.standardize:
            self.std_bias = vision_tower.std_bias
            self.std_scale = vision_tower.std_scale
        self.pooling_kernel_size = vision_tower.config.pooling_kernel_size

    def forward(
        self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process pre-patchified image to LM-ready embeddings.

        Args:
            pixel_values: [batch, num_patches, patch_dim] (pre-patchified, [0,1])
            pixel_position_ids: [batch, num_patches, 2] (x,y coords, -1=padding)

        Returns:
            Tuple of (embeddings, output_mask):
            - embeddings: [batch, output_length, text_hidden_size]
            - output_mask: [batch, output_length] (True=valid)
        """
        pks = self.pooling_kernel_size
        output_length = pixel_values.shape[1] // (pks * pks)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)

        inputs_embeds = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )

        num_patches = padding_positions.size(1)
        attention_mask = (
            padding_positions.float().unsqueeze(1).unsqueeze(2)
        )  # [batch, 1, 1, num_patches]
        attention_mask = attention_mask.expand(-1, -1, num_patches, -1)
        attention_mask = attention_mask * -255
        encoder_output = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_position_ids=pixel_position_ids,
        )

        hidden_states, pooler_mask = self.pooler(
            hidden_states=encoder_output.last_hidden_state,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )

        if self.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale
            hidden_states = hidden_states.masked_fill(~pooler_mask.unsqueeze(-1), 0.0)
        hidden_states = hidden_states.to(pixel_values.dtype)
        embeddings = self.embed_vision(hidden_states)
        return embeddings, pooler_mask


# ---------------------------------------------------------------------------
# Audio encoder: dynamic 5D visibility mask + ENN export wrapper
# ---------------------------------------------------------------------------


def _patch_hf_audio_for_export():
    """Monkey-patch HF Gemma4 audio modules to fix torch.export blockers.

    Unwraps the decorated Gemma4AudioRelPositionalEncoding.forward (torch.export
    cannot trace through the wrapper) and converts Gemma4AudioCausalConv1d's
    cached_property `left_pad` into a plain property (cached_property caches the
    first tensor seen, which breaks re-export with different shapes).

    Modifies the HF classes globally and permanently for the lifetime of the
    process; see _patch_hf_vision_for_export for the implications.
    """
    import functools

    from transformers.models.gemma4 import modeling_gemma4

    cls = modeling_gemma4.Gemma4AudioRelPositionalEncoding
    if getattr(cls, "_enn_patched", False):
        return

    if hasattr(cls.forward, "__wrapped__"):
        cls.forward = cls.forward.__wrapped__

    if hasattr(modeling_gemma4, "Gemma4AudioCausalConv1d"):
        conv_cls = modeling_gemma4.Gemma4AudioCausalConv1d
        cached = conv_cls.__dict__.get("left_pad")
        if isinstance(cached, functools.cached_property):
            conv_cls.left_pad = property(cached.func)

    cls._enn_patched = True


def _build_dynamic_attention_mask(output_mask: torch.Tensor, config) -> torch.Tensor:
    """Build the blocked 5D visible mask dynamically from the subsampled padding mask.

    Replaces create_bidirectional_mask + _convert_4d_mask_to_blocked_5d with
    export-compatible tensor ops (no vmap, no gather).

    seq_len and num_blocks are derived from the runtime subsampled length
    (output_mask.shape[1]) instead of a baked-in constant, so the exported
    encoder accepts variable-length audio within the declared Dim range.
    """
    chunk_size = config.attention_chunk_size
    max_past = config.attention_context_left - 1
    max_future = config.attention_context_right
    context_size = chunk_size + max_past + max_future
    seq_len = output_mask.shape[1]
    num_blocks = (seq_len + chunk_size - 1) // chunk_size
    left_window = config.attention_context_left - 1
    pad_amount = num_blocks * chunk_size - seq_len

    device = output_mask.device

    padded = F.pad(output_mask, (max_past, max_future + pad_amount), value=0.0)
    pad_5d = padded.unfold(1, context_size, chunk_size)

    blocks = torch.arange(num_blocks, device=device)
    qs = torch.arange(chunk_size, device=device)
    gis = torch.arange(context_size, device=device)

    q_global = blocks[:, None] * chunk_size + qs[None, :]
    kv_global = blocks[:, None] * chunk_size + gis[None, :] - max_past
    dist = q_global[:, :, None] - kv_global[:, None, :]

    left_ok = ((dist >= 0) & (dist < left_window)).float()
    right_ok = ((dist < 0) & (-dist < max_future)).float()
    window_ok = left_ok + right_ok
    q_in_range = (q_global < seq_len).float()

    visible = (
        q_in_range[None, :, :, None] * window_ok[None, :, :, :] * pad_5d[:, :, None, :]
    )

    return visible.unsqueeze(1)


def _patch_attention_for_dynamic_mask() -> None:
    """Patch Gemma4AudioAttention.forward to use multiply-then-add masking.

    The dynamic-mask branch is taken only for encoder instances with
    `_use_dynamic_mask` set (see _patch_audio_forward); other instances fall
    through to the original HF implementation.

    Modifies the HF class globally and permanently for the lifetime of the
    process; see _patch_hf_vision_for_export for the implications.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioAttention

    if getattr(Gemma4AudioAttention, "_enn_patched", False):
        return

    _orig_attention_forward = Gemma4AudioAttention.forward

    def _patched_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if getattr(self, "_use_dynamic_mask", False):
            batch_size, seq_length, _ = hidden_states.shape
            hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)

            query_states = self.q_proj(hidden_states).float().view(hidden_shape)
            key_states = self.k_proj(hidden_states).float().view(hidden_shape)
            value_states = self.v_proj(hidden_states).float().view(hidden_shape)

            query_states = query_states * self.q_scale * F.softplus(self.per_dim_scale)
            key_states = key_states * self.k_scale

            query_states = self._convert_to_block(query_states)
            key_states = self._extract_block_context(key_states)
            value_states = self._extract_block_context(value_states)
            num_blocks = query_states.shape[1]

            relative_key_states = self.relative_k_proj(position_embeddings)
            relative_key_states = relative_key_states.view(
                -1, self.num_heads, self.head_dim
            )
            relative_key_states = relative_key_states.to(dtype=query_states.dtype)

            queries = query_states.permute(0, 3, 1, 2, 4)
            matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)

            queries_flat = queries.reshape(
                batch_size, self.num_heads, -1, self.head_dim
            )
            matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
            matrix_bd = matrix_bd.reshape(
                batch_size, self.num_heads, num_blocks, self.chunk_size, -1
            )
            matrix_bd = self._rel_shift(matrix_bd)

            attn_weights = matrix_ac + matrix_bd
            attn_weights = attn_weights / self.softcap
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.softcap

            if attention_mask is not None:
                # Masked positions only need to underflow softmax (exp(-255) ==
                # 0), and a small magnitude keeps the masked add output within
                # the int16 activation quant range instead of blowing up the
                # scale like finfo.min would.
                mask_value = -255.0
                attn_weights = (
                    attn_weights * attention_mask + (1.0 - attention_mask) * mask_value
                )

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                value_states.dtype
            )
            attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
            attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(
                batch_size, num_blocks * self.chunk_size, -1
            )
            attn_output = attn_output[:, :seq_length].contiguous()
            attn_output = self.post(attn_output.to(hidden_states.dtype))

            return attn_output, attn_weights
        else:
            return _orig_attention_forward(
                self, hidden_states, position_embeddings, attention_mask
            )

    Gemma4AudioAttention.forward = _patched_attention_forward
    Gemma4AudioAttention._enn_patched = True


def _patch_audio_forward(audio_tower) -> None:
    """Patch audio_tower.forward to build a dynamic 5D visibility mask at runtime."""
    import types

    tower = audio_tower
    num_heads = tower.config.num_attention_heads

    tower._use_dynamic_mask = True
    for layer in tower.layers:
        layer.self_attn._use_dynamic_mask = True

    def _dynamic_mask_forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        hidden_states, output_mask = self.subsample_conv_projection(
            input_features, attention_mask
        )
        position_embeddings = self.rel_pos_enc(hidden_states)

        visible_mask = _build_dynamic_attention_mask(output_mask, self.config)
        visible_mask = visible_mask.expand(-1, num_heads, -1, -1, -1)

        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=visible_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.output_proj(hidden_states)

        from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModelOutput

        return Gemma4AudioModelOutput(
            last_hidden_state=hidden_states, attention_mask=output_mask
        )

    tower.forward = types.MethodType(_dynamic_mask_forward, tower)
    _patch_attention_for_dynamic_mask()


class AudioEncoderWithDynamicMask_ENN(torch.nn.Module):
    """Audio encoder wrapper with dynamic 5D visibility mask for ENN export.

    Replaces create_bidirectional_mask (vmap) and _convert_4d_mask_to_blocked_5d
    (gather) with unfold-based tensor ops. Patches audio_tower.forward and
    Gemma4AudioAttention.forward in-place.
    """

    def __init__(self, audio_tower, embed_audio):
        super().__init__()
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio
        _patch_audio_forward(audio_tower)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask.float()
        encoder_output = self.audio_tower(features, attention_mask=mask)
        hidden = encoder_output.last_hidden_state
        output_mask = encoder_output.attention_mask
        return self.embed_audio(hidden), output_mask.bool()


def _compute_mel_frame_count(checkpoint_path: str) -> int:
    """Compute the number of mel frames for the max audio length from checkpoint config."""
    import json

    sampling_rate = 16000
    frame_length = 320
    hop_length = 160
    audio_seq_length = 750
    audio_ms_per_token = 40

    processor_path = os.path.join(checkpoint_path, "processor_config.json")
    if os.path.exists(processor_path):
        with open(processor_path, "r") as f:
            processor_config = json.load(f)
        fe = processor_config.get("feature_extractor", {})
        sampling_rate = fe.get("sampling_rate", sampling_rate)
        frame_length = fe.get("frame_length", frame_length)
        hop_length = fe.get("hop_length", hop_length)
        audio_seq_length = processor_config.get("audio_seq_length", audio_seq_length)
        audio_ms_per_token = processor_config.get(
            "audio_ms_per_token", audio_ms_per_token
        )

    max_length = int(audio_seq_length * audio_ms_per_token * sampling_rate / 1000)
    pad_left = frame_length // 2
    frame_size = frame_length + 1

    return (max_length + pad_left - frame_size) // hop_length + 1


class Gemma4Model_ENN(Gemma4PreTrainedModel):
    def __init__(self, config: HFGemma4Config):
        super().__init__(config)
        if config.vision_config is not None:
            config.vision_config._attn_implementation = "eager"
            self.vision_tower = AutoModel.from_config(config.vision_config)
        else:
            self.vision_tower = None
        self.vocab_size = config.text_config.vocab_size
        self.max_token = config.max_token_len
        self.variant = config.variant

        language_model = AutoModel.from_config(config=config.text_config)
        self.language_model = language_model
        self.vocab_size_per_layer_input = config.text_config.vocab_size_per_layer_input
        self.audio_tower = (
            AutoModel.from_config(config.audio_config)
            if config.audio_config is not None
            else None
        )
        self.embed_vision = (
            Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
            if config.vision_config is not None
            else None
        )
        self.embed_audio = (
            Gemma4MultimodalEmbedder(config.audio_config, config.text_config)
            if config.audio_config is not None
            else None
        )

        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

        self.post_init()

    def dequantize_model(self, model, dtype, padding_idx, device):
        for name, module in model.named_children():
            if isinstance(module, QuantizedLinear):
                module = module.to(torch.device("cpu"))
                # Replaces QuantizedLinear with SRQ expressed as clamp + linear + clamp.
                # Weight is dequantized to float; the learned in/out activation (SRQ)
                # scales are kept on the ClampLinear so get_text_decoder can carry the
                # same clamp ranges into the custom text_decoder.
                in_scale = module.input_activation_scale.detach().float()
                out_scale = module.output_activation_scale.detach().float()

                new_module = ClampLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    weight=module._dequantize_weights(dtype).to(dtype),
                    bias=module.bias.detach() if module.bias is not None else None,
                    in_scale=in_scale,
                    out_scale=out_scale,
                )
                new_module.to(device)
                setattr(model, name, new_module)
                del module
            elif isinstance(module, QuantizedEmbedding):
                module = module.to(torch.device("cpu"))
                new_module = Gemma4TextScaledWordEmbedding(
                    module.num_embeddings,
                    module.embedding_dim,
                    padding_idx,
                    module.scalar_embed_scale,
                )
                new_module.weight.data = module.weight.to(dtype)
                new_module.to(device)
                setattr(model, name, new_module)
                del module
            else:
                self.dequantize_model(module, dtype, padding_idx, device)

    def init_execu_model(self, dtype, device, checkpoint_path: str = ""):
        pad_token_id = self.language_model.config.pad_token_id
        self.dequantize_model(self.language_model, dtype, pad_token_id, device)
        self.text_decoder = self.get_text_decoder()
        self.language_model = None
        free_memory()

        if self.vision_tower is not None:
            self.dequantize_model(self.vision_tower, dtype, pad_token_id, device)
            self.dequantize_model(self.embed_vision, dtype, pad_token_id, device)
            self.vision_encoder = self.get_vision_encoder()
        else:
            self.vision_encoder = None
        free_memory()

        if self.audio_tower is not None:
            self.dequantize_model(self.audio_tower, dtype, pad_token_id, device)
            self.dequantize_model(self.embed_audio, dtype, pad_token_id, device)
            self.audio_encoder = self.get_audio_encoder(checkpoint_path)
        else:
            self.audio_encoder = None
        free_memory()

    def get_vision_encoder(self):
        _patch_hf_vision_for_export()
        model = VisionEncoderWithProjection_ENN(self.vision_tower, self.embed_vision)
        model.eval()
        return model

    def get_audio_encoder(self, checkpoint_path: str = ""):
        _patch_hf_audio_for_export()
        model = AudioEncoderWithDynamicMask_ENN(self.audio_tower, self.embed_audio)
        model.eval()
        return model

    def get_text_decoder(self):
        from executorch.examples.samsung.scripts.llm.models.gemma4.text_decoder.gemma4_transformer import (
            Gemma4ForCausalLM,
        )

        config = ExecuGemma4Config.from_config(self.variant)
        config.use_kv_cache = True
        config.max_seq_len = self.max_token
        config.max_context_len = self.max_token
        config.enable_dynamic_shape = True

        dtype = next(self.language_model.parameters()).dtype
        device = next(self.language_model.parameters()).device

        state_dict, clamp_info = convert_hf_to_custom(self.language_model, config)

        text_decoder = Gemma4ForCausalLM(config)
        replace_linear_with_clamp(text_decoder, clamp_info, device, dtype)

        text_decoder.load_state_dict(state_dict, strict=False)

        text_decoder.eval().to(device=device, dtype=dtype)

        del self.language_model

        self.text_config = text_decoder.config

        return text_decoder

    def get_input_embeddings(self, llm_input_ids):
        input_embeddings = (
            self.text_decoder.model.self_decoder.embed_tokens(llm_input_ids)
            * self.text_decoder.model.self_decoder.embed_scale_per_layer
        )
        return input_embeddings

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        """
        Obtains mask for multimodal placeholders (replaced by soft tokens) and hard text tokens.

        Masks are obtained from `input_ids` by comparison with
        `config.image_token_id` / `video_token_id` / `audio_token_id`.

        Args:
            input_ids: A tensor containing the hard token IDs from the text tokenizer.

        Returns:
            image_mask, video_mask, audio_mask
        """
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        special_audio_mask = input_ids == self.config.audio_token_id
        return special_image_mask, special_video_mask, special_audio_mask

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_embeddings_global: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_embeddings_local: tuple[torch.Tensor, torch.Tensor] | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        k_cache: Optional[Tuple[torch.Tensor, ...]] = None,
        v_cache: Optional[Tuple[torch.Tensor, ...]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        if per_layer_inputs is not None:
            raise ValueError(
                "You cannot specify per_layer_inputs if input_ids is provided"
            )

        image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids)
        multimodal_mask = image_mask | video_mask | audio_mask

        llm_input_ids = input_ids.clone()
        llm_input_ids = torch.where(
            multimodal_mask, self.config.text_config.pad_token_id, llm_input_ids
        )
        if inputs_embeds is None:
            # Zeros instead of real embeddings: multimodal features are
            # masked_scatter'ed on top at mm positions; text positions get their
            # embeddings from llm_input_ids inside the text_decoder.
            inputs_embeds = torch.zeros(
                1,
                input_ids.shape[1],
                self.config.text_config.hidden_size,
                device=input_ids.device,
                dtype=self.dtype,
            )

        # Merge text and images
        if pixel_values is not None:
            image_features = self.vision_encoder(pixel_values, image_position_ids)
            image_features = image_features[0][image_features[1]]
            image_features = image_features.to(input_ids.device, self.dtype)

            # Confirm the number of soft tokens from the vision tower matches the number of slots in the embeddings.
            n_image_tokens = image_mask.sum()
            image_mask = (
                image_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(image_features.device)
            )
            assert inputs_embeds[image_mask].numel() == image_features.numel(), (
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features:"
                f" {image_features.shape[0]}"
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.to(inputs_embeds.device),
                image_features.to(inputs_embeds.device),
            )

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.flatten(0, 1)
            video_position_ids = video_position_ids.flatten(0, 1)
            video_features = self.vision_encoder(
                pixel_values_videos,
                video_position_ids,
            )
            video_features = video_features[0][video_features[1]]
            video_features = video_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )

            # Confirm the number of soft tokens from the vision tower matches the number of slots in the embeddings.
            n_video_tokens = video_mask.sum()
            video_mask = (
                video_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )

            assert inputs_embeds[video_mask].numel() == video_features.numel(), (
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features:"
                f" {video_features.shape[0]}"
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.to(inputs_embeds.device),
                video_features.to(inputs_embeds.device),
            )

        # Merge text and audio
        if input_features is not None and input_features_mask is not None:
            execu_audio_result = self.audio_encoder(input_features, input_features_mask)
            audio_features = execu_audio_result[0][execu_audio_result[1]]

            n_audio_tokens = audio_mask.sum()
            audio_mask = (
                audio_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )

            assert inputs_embeds[audio_mask].numel() == audio_features.numel(), (
                f"Audio features and audio tokens do not match, tokens: {n_audio_tokens}, features:"
                f" {audio_features.shape[0] * audio_features.shape[1]}"
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                audio_mask.to(inputs_embeds.device),
                audio_features.to(inputs_embeds.device),
            )

        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            )
            position_ids = position_ids.unsqueeze(0)

        multimodal_mask = multimodal_mask.to(inputs_embeds.dtype)

        logits = self.text_decoder(
            llm_input_ids,
            inputs_embeds=inputs_embeds,
            mm_mask=multimodal_mask,
            attention_mask=attention_mask,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        return logits


class Gemma4ForConditionalGeneration_ENN(Gemma4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {
        "model.lm_head.weight": "model.language_model.embed_tokens.weight"
    }
    accepts_loss_kwargs = False
    base_model_prefix = "model"

    def __init__(self, config: HFGemma4Config):
        super().__init__(config)
        self.model = Gemma4Model_ENN(config)
        self.text_rotary_emb = Gemma4TextRotaryEmbedding(config.text_config)
        self.config = config
        self.max_token = config.max_token_len
        self.num_self_decoder_layers = (
            config.text_config.num_hidden_layers
            - config.text_config.num_kv_shared_layers
        )
        self.batch_size = 1
        self.post_init()
        self.setup_kv_cache(self.batch_size, self.max_token)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        cls._loaded_dtype = kwargs.get("torch_dtype") or kwargs.get("dtype")
        cls._loaded_device_map = kwargs.get("device_map")
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    def setup_kv_cache(
        self,
        batch_size: int = 1,
        max_token: int = 1024,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        if device is not None:
            self.target_device = device
        elif (
            hasattr(self, "_loaded_device_map") and self._loaded_device_map is not None
        ):
            self.target_device = self._loaded_device_map
        else:
            self.target_device = torch.device("cpu")

        if dtype is not None:
            pass
        elif hasattr(self, "_loaded_dtype") and self._loaded_dtype is not None:
            dtype = self._loaded_dtype
        else:
            dtype = torch.float32

        self.batch_size = batch_size
        self.max_token = max_token

        text_config = self.config.text_config
        device = self.target_device
        for layer_idx in range(self.num_self_decoder_layers):
            # Per-layer head_dim: full attention uses global_head_dim (512),
            # sliding attention uses head_dim (256).
            if text_config.layer_types[layer_idx] == "full_attention":
                head_dim = text_config.global_head_dim
            else:
                head_dim = text_config.head_dim
            cache_shape = (
                self.batch_size,
                text_config.num_key_value_heads,
                self.max_token,
                head_dim,
            )
            k = torch.zeros(cache_shape, dtype=dtype, device=device)
            v = torch.zeros(cache_shape, dtype=dtype, device=device)
            self.register_buffer(f"k_cache_{layer_idx}", k, persistent=False)
            self.register_buffer(f"v_cache_{layer_idx}", v, persistent=False)

    def generate_attention_mask(self, cache_position):
        attention_mask = {}
        attention_mask["full_attention"] = torch.tril(
            torch.ones(self.max_token, self.max_token, dtype=torch.bool)
        )
        # Create position indices
        q_indices = torch.arange(self.max_token).unsqueeze(1)  # [max_token, 1]
        kv_indices = torch.arange(self.max_token).unsqueeze(0)  # [1, max_token]

        # Causal constraint: kv_idx <= q_idx
        causal_mask = kv_indices <= q_indices

        # Sliding window constraint: kv_idx > q_idx - sliding_window_size
        sliding_window_mask = kv_indices > (
            q_indices - self.config.text_config.sliding_window
        )

        device = cache_position.device

        # Combine both constraints
        attention_mask["sliding_attention"] = causal_mask & sliding_window_mask

        attention_mask["full_attention"] = attention_mask["full_attention"][
            cache_position[0] : cache_position[-1] + 1
        ].unsqueeze(0)
        attention_mask["sliding_attention"] = attention_mask["sliding_attention"][
            cache_position[0] : cache_position[-1] + 1
        ].unsqueeze(0)

        attention_mask["full_attention"] = (
            attention_mask["full_attention"]
            .expand([self.batch_size, *attention_mask["full_attention"].shape])
            .to(device)
        )
        attention_mask["sliding_attention"] = (
            attention_mask["sliding_attention"]
            .expand([self.batch_size, *attention_mask["sliding_attention"].shape])
            .to(device)
        )

        return attention_mask

    def update_kvcache(self, cur_key_state_list, cur_value_state_list, cache_position):
        for i in range(len(cur_key_state_list)):
            torch.ops.aten.index_put_(
                getattr(self, f"k_cache_{i}"),
                [None, None, cache_position],
                cur_key_state_list[i],
            )
            torch.ops.aten.index_put_(
                getattr(self, f"v_cache_{i}"),
                [None, None, cache_position],
                cur_value_state_list[i],
            )

    def prepare_data_for_enn_kvcache(self, cache_position, attention_mask):
        # Only self-decoder layers consume cache inputs; cross-decoder layers
        # read in-graph YOCO shared KV instead.
        k_cache = tuple(
            [
                getattr(self, f"k_cache_{i}")[
                    :, :, 0 : self.max_token - len(cache_position), :
                ]
                for i in range(self.num_self_decoder_layers)
            ]
        )
        v_cache = tuple(
            [
                getattr(self, f"v_cache_{i}")[
                    :, :, 0 : self.max_token - len(cache_position), :
                ]
                for i in range(self.num_self_decoder_layers)
            ]
        )
        if len(cache_position) > 1:
            new_cache_position = torch.flip(
                (self.max_token - 1 - cache_position), dims=[0]
            )
        else:
            new_cache_position = torch.tensor([self.max_token - 1])

        first_old_idx = cache_position[0]
        first_new_idx = new_cache_position[0]

        for i, (old_idx, new_idx) in enumerate(zip(cache_position, new_cache_position)):
            attention_mask[0][:, :, i, first_new_idx : new_idx + 1] = attention_mask[0][
                :, :, i, first_old_idx : old_idx + 1
            ].clone()
            attention_mask[0][:, :, i, first_old_idx:first_new_idx] = 0

            attention_mask[1][:, :, i, first_new_idx : new_idx + 1] = attention_mask[1][
                :, :, i, first_old_idx : old_idx + 1
            ].clone()
            attention_mask[1][:, :, i, first_old_idx:first_new_idx] = 0

        attention_mask = tuple(
            (1.0 - mask.to(torch.float32)) * -255.0 for mask in attention_mask
        )

        return k_cache, v_cache, new_cache_position, attention_mask

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        attention_mask = self.generate_attention_mask(model_inputs["position_ids"][0])
        # Expanding mask size to support staticCache with a fixed size
        full_causal_mask = attention_mask["full_attention"]
        sliding_causal_mask = attention_mask["sliding_attention"]

        if full_causal_mask.shape[-1] < self.max_token:
            padding_size = self.max_token - full_causal_mask.shape[3]
            full_causal_mask = F.pad(full_causal_mask, (0, padding_size), value=0)
        elif full_causal_mask.shape[-1] > self.max_token:
            full_causal_mask = full_causal_mask[:, :, :, : self.max_token]

        if sliding_causal_mask.shape[-1] > self.max_token:
            sliding_causal_mask = sliding_causal_mask[:, :, :, : self.max_token]

        attention_mask["full_attention"] = full_causal_mask
        attention_mask["sliding_attention"] = sliding_causal_mask

        k_cache, v_cache, _, attention_mask = self.prepare_data_for_enn_kvcache(
            model_inputs["position_ids"][0], (full_causal_mask, sliding_causal_mask)
        )
        model_inputs["attention_mask"] = attention_mask
        model_inputs["k_cache"] = k_cache
        model_inputs["v_cache"] = v_cache

        # If we're in cached decoding stage, multimodal inputs are already cached and can be dropped
        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["input_features"] = input_features
            model_inputs["input_features_mask"] = input_features_mask

        # If `per_layer_inputs` was provided along with `inputs_embeds` for first forward, drop it for subsequent forwards
        if not is_first_iteration:
            _ = model_inputs.pop("per_layer_inputs", None)

        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided; see Gemma4Model_ENN.forward")
        if position_ids is None:
            raise ValueError(
                "position_ids must be provided by prepare_inputs_for_generation"
            )

        g_cos, g_sin, l_cos, l_sin = self.get_position_embeddings(position_ids)
        position_embeddings_global = (g_cos, g_sin)
        position_embeddings_local = (l_cos, l_sin)

        logits, cur_key_state_list, cur_value_state_list = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=attention_mask,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            **kwargs,
        )

        self.update_kvcache(cur_key_state_list, cur_value_state_list, position_ids[0])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits, labels, self.config.get_text_config().vocab_size, **kwargs
            )

        return Gemma4CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    def get_metadata(self, prefill_ar_len=1):
        text_config = self.config.text_config
        return {
            "get_bos_id": text_config.bos_token_id,
            "get_eos_ids": self.config.eos_token_id,
            "get_boa_token_id": self.config.boa_token_id,
            "get_boi_token_id": self.config.boi_token_id,
            "get_eoa_token_id": self.config.eoa_token_id,
            "get_eoa_token_index": self.config.eoa_token_index,
            "get_eoi_token_id": self.config.eoi_token_id,
            "get_image_token_id": self.config.image_token_id,
            "get_audio_token_id": self.config.audio_token_id,
            "get_max_seq_len": self.max_token,
            "get_max_context_len": self.max_token,
            "get_n_layers": text_config.num_hidden_layers,
            "get_n_kv_shared_layers": text_config.num_kv_shared_layers,
            "get_vocab_size": text_config.vocab_size,
            "use_kv_cache": True,
            "use_sdpa_with_kv_cache": False,
            "enable_dynamic_shape": False,
            "get_prefill_ar_len": prefill_ar_len,
            "get_sliding_window": text_config.sliding_window,
            "get_global_head_dim": text_config.global_head_dim,
            "get_head_dim": text_config.head_dim,
            "get_n_heads": text_config.num_attention_heads,
            "get_n_kv_heads": text_config.num_key_value_heads,
            "get_hidden_dim": text_config.hidden_size,
        }

    def get_position_embeddings(
        self, position_ids: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the RoPE tables the graphs take as inputs.

        Returns (global cos, global sin, local cos, local sin), the order the
        runtime expects them laid out in pos_embedding.bin. With
        ``position_ids`` None, positions [0, max_token) are used so the
        runtime can slice per-step rows out of the full table.
        """
        if position_ids is None:
            position_ids = torch.arange(
                self.max_token, device=self.target_device
            ).unsqueeze(0)
        hidden_states_dummy = torch.zeros(
            (
                self.batch_size,
                position_ids.shape[1],
                self.config.text_config.hidden_size,
            )
        ).to(self.target_device)
        g_cos, g_sin = self.text_rotary_emb(
            hidden_states_dummy, position_ids, layer_type="full_attention"
        )
        l_cos, l_sin = self.text_rotary_emb(
            hidden_states_dummy, position_ids, layer_type="sliding_attention"
        )
        return (g_cos, g_sin, l_cos, l_sin)

    def _prepare_vision_input(self, device, dtype, max_patches=1260):
        pooling_kernel_size = self.config.vision_config.pooling_kernel_size
        _num_groups = torch.export.Dim(
            "_num_groups", min=1, max=max_patches // (pooling_kernel_size**2)
        )
        num_patches_dim = (pooling_kernel_size**2) * _num_groups

        example_pixels = torch.randn(
            1, max_patches, 3 * 16 * 16, device=device, dtype=dtype
        )
        example_positions = torch.zeros(
            1, max_patches, 2, dtype=torch.long, device=device
        )

        dynamic_shapes = {
            "pixel_values": {1: num_patches_dim},
            "pixel_position_ids": {1: num_patches_dim},
        }

        return (example_pixels, example_positions), dynamic_shapes

    def _prepare_audio_input(self, device, dtype):
        # Dynamic shapes: num_frames = 48*k - 25
        _num_frames = torch.export.Dim("_num_frames", min=2, max=63)
        num_frames_dim = 48 * _num_frames - 25
        valid_frames = 48 * 63 - 25  # 2999
        example_features = torch.randn(1, valid_frames, 128, device=device, dtype=dtype)
        example_mask = torch.ones(1, valid_frames, dtype=torch.bool, device=device)

        dynamic_shapes = {
            "features": {1: num_frames_dim},
            "mask": {1: num_frames_dim},
        }
        return (example_features, example_mask), dynamic_shapes

    def _prepare_text_input(self, device, dtype, num_tokens=1):
        input_ids = torch.ones((1, num_tokens), dtype=torch.long).to(device)

        inputs_embeds = torch.randn(
            1, num_tokens, self.config.text_config.hidden_size, dtype=dtype
        ).to(device)
        input_pos = torch.arange(num_tokens, dtype=torch.long).to(device)

        multimodal_mask = torch.zeros(1, num_tokens, device=device, dtype=dtype)

        attention_mask = (
            torch.randn(
                1, 1, num_tokens, self.max_token, device=device, dtype=dtype
            ),  # 'full_attention'
            torch.randn(
                1, 1, num_tokens, self.max_token, device=device, dtype=dtype
            ),  # 'sliding_attention'
        )
        k_cache = []
        v_cache = []
        for i in range(self.num_self_decoder_layers):
            k_cache.append(
                getattr(self, f"k_cache_{i}")[
                    :, :, 0 : self.max_token - num_tokens, :
                ].to(device)
            )
            v_cache.append(
                getattr(self, f"v_cache_{i}")[
                    :, :, 0 : self.max_token - num_tokens, :
                ].to(device)
            )
        k_cache = tuple(k_cache)
        v_cache = tuple(v_cache)

        token_dim = torch.export.Dim.AUTO(min=1, max=self.max_token)
        cache_dim = torch.export.Dim.AUTO(min=1, max=self.max_token)
        max_token_dim = torch.export.Dim.AUTO(min=1, max=self.max_token)

        kv_cache_dyn_list = []
        for _ in range(self.num_self_decoder_layers):
            kv_cache_dyn_list.append({2: cache_dim})
        kv_cache_dyn = tuple(kv_cache_dyn_list)

        dynamic_shapes = {
            "input_ids": {1: token_dim},
            "inputs_embeds": {1: token_dim},
            "mm_mask": {1: token_dim},
            "attention_mask": (
                {2: token_dim, 3: max_token_dim},  # 'full_attention'
                {2: token_dim, 3: max_token_dim},  # 'sliding_attention'
            ),
            "position_embeddings_global": (
                {1: token_dim},
                {1: token_dim},
            ),
            "position_embeddings_local": (
                {1: token_dim},
                {1: token_dim},
            ),
            "k_cache": kv_cache_dyn,
            "v_cache": kv_cache_dyn,
        }

        input_pos = input_pos.unsqueeze(0)
        self.text_rotary_emb = self.text_rotary_emb.to(device)
        g_cos, g_sin, l_cos, l_sin = self.get_position_embeddings(input_pos)
        position_embeddings_global = (g_cos, g_sin)
        position_embeddings_local = (l_cos, l_sin)

        example_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "mm_mask": multimodal_mask,
            "attention_mask": attention_mask,
            "position_embeddings_global": position_embeddings_global,
            "position_embeddings_local": position_embeddings_local,
            "k_cache": k_cache,
            "v_cache": v_cache,
        }

        return example_inputs, dynamic_shapes
