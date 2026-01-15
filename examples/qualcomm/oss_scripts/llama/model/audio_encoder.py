# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from executorch.examples.qualcomm.utils import replace_module_with_custom_class
from torch import nn

from transformers.models.blip_2.modeling_blip_2 import (
    Blip2QFormerConfig,
    Blip2QFormerSelfOutput,
)

from transformers.models.granite_speech.modeling_granite_speech import (
    GraniteSpeechConfig,
    GraniteSpeechConformerAttention,
    GraniteSpeechCTCEncoder,
    GraniteSpeechEncoderConfig,
    GraniteSpeechEncoderProjector,
)


# A `GraniteSpeechConformerAttention` implementation based on Transformers v5.0.0rc1.
#
# Reshapes the query_states to avoid 6D tensors Matmul, which are not supported by HTP.
class CustomGraniteSpeechConformerAttention(GraniteSpeechConformerAttention):
    def __init__(self, config: GraniteSpeechEncoderConfig):
        super().__init__(config)

    def forward(
        self, hidden_states: torch.Tensor, attention_dists: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states)
        bsz, num_features, _ = hidden_states.shape

        num_blocks = math.ceil(num_features / self.context_size)
        remainder = num_features % self.context_size
        if remainder > 0:
            # right padding to reach block size
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, self.context_size - remainder)
            )

        query_states = self.to_q(hidden_states)
        key_states, value_states = self.to_kv(hidden_states).chunk(2, dim=-1)

        query_states = query_states.reshape(
            bsz, num_blocks, self.context_size, self.num_heads, -1
        ).transpose(2, 3)
        key_states = key_states.reshape(
            bsz, num_blocks, self.context_size, self.num_heads, -1
        ).transpose(2, 3)
        value_states = value_states.reshape(
            bsz, num_blocks, self.context_size, self.num_heads, -1
        ).transpose(2, 3)

        # ========================== Qualcomm Changed: Pre-merge dimensions to avoid 6D tensor matmul ==========================
        rel_pos_emb = self.rel_pos_emb(attention_dists)
        b, m, h, c, d = query_states.shape
        c, r, d = rel_pos_emb.shape
        rel = rel_pos_emb.transpose(-1, -2)  # [c, d, r]
        q = query_states.reshape(-1, c, d)  # [b*m*h, c, d]
        out = (
            torch.einsum("b c d, c d r -> b c r", q, rel) * self.scale
        )  # [b*m*h, c, r]
        pos_attn = out.view(b, m, h, c, r)
        # ======================================================================================================================

        if remainder > 0:
            # masked attention in the extended block
            mask = torch.ones(
                self.context_size,
                self.context_size,
                dtype=bool,
                device=hidden_states.device,
            )
            mask[:remainder, :remainder] = 0
            mask_value = -torch.finfo(pos_attn.dtype).max
            pos_attn[:, -1, :].masked_fill_(mask, mask_value)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=pos_attn,
                scale=self.scale,
            )
        out = out.transpose(2, 3).reshape(bsz, hidden_states.shape[1], -1)
        out = self.to_out(out[:, :num_features, :])
        return self.dropout(out)


# Custom implementation based on `transformers.models.blip_2.modeling_blip_2.Blip2QFormerSelfOutput` (Transformers v5.0.0rc1).
#
# Workaround:
# Adds an identity matrix computation before LayerNorm as a workaround for an
# HTP preparation failure.
class CustomBlip2QFormerSelfOutput(Blip2QFormerSelfOutput):
    def __init__(self, config: Blip2QFormerConfig):
        super().__init__(config=config)
        self.identity = nn.Linear(config.hidden_size, config.hidden_size)
        self.identity.weight.data.copy_(torch.eye(config.hidden_size))
        self.identity.bias.data.zero_()

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        # Workaround for HTP preparation failure: insert an identity matrix
        # to break pattern match.
        hidden_states = self.identity(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class GraniteSpeechCTCEncoderWrapper(nn.Module):
    def __init__(self, config: GraniteSpeechConfig):
        super().__init__()
        self.encoder = GraniteSpeechCTCEncoder(config.encoder_config)
        self.projector = GraniteSpeechEncoderProjector(config)

        replace_module_with_custom_class(
            self.encoder,
            target_class=GraniteSpeechConformerAttention,
            custom_class=CustomGraniteSpeechConformerAttention,
            strict=True,
            extra_custom_kwargs={"config": config.encoder_config},
        )

        replace_module_with_custom_class(
            self.projector,
            target_class=Blip2QFormerSelfOutput,
            custom_class=CustomBlip2QFormerSelfOutput,
            strict=False,  # Set to False because the custom class adds an 'identity' matrix not present in the original QFormer.
            extra_custom_kwargs={"config": config.projector_config},
        )

        self.config = config

    def get_example_inputs(self):
        return (torch.randn((1, 844, 160), dtype=torch.float32),)

    def forward(self, hidden_states: torch.Tensor):
        encoder_embeds = self.encoder(hidden_states)
        projected_embeds = self.projector(encoder_embeds)
        return projected_embeds
