# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch

from transformers.models.mobilebert.modeling_mobilebert import (
    MobileBertLayer as MobileBertLayerBase,
    MobileBertModel as MobileBertModelBase,
)


class MobileBertLayer(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        assert isinstance(base, MobileBertLayerBase)
        self.base = base

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        if self.base.use_bottleneck:
            query_tensor, key_tensor, value_tensor, layer_input = self.base.bottleneck(
                hidden_states
            )
        else:
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4

        self_attention_outputs = self.base.attention(
            query_tensor,
            key_tensor,
            value_tensor,
            layer_input,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        s = (attention_output,)
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        if self.base.num_feedforward_networks != 1:
            for i, ffn_module in enumerate(self.base.ffn):
                attention_output = ffn_module(attention_output)
                s += (attention_output,)

        intermediate_output = self.base.intermediate(attention_output)
        layer_output = self.base.output(
            intermediate_output, attention_output, hidden_states
        )
        outputs = (
            (layer_output,)
            + outputs
            + (
                torch.scalar_tensor(1000, dtype=torch.int64),
                query_tensor,
                key_tensor,
                value_tensor,
                layer_input,
                attention_output,
                intermediate_output,
            )
            + s
        )
        return outputs


class MobileBertModel(MobileBertModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layer = self.encoder.layer
        assert isinstance(layer, torch.nn.ModuleList)
        self.encoder.layer = torch.nn.ModuleList([MobileBertLayer(x) for x in layer])
