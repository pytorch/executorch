# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from executorch.examples.qualcomm.oss_scripts.llama.inference.decoder import (
    DecoderInference,
)
from executorch.examples.qualcomm.oss_scripts.llama.inference.encoder import (
    EncoderInference,
)
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import AttentionMask


class ModelInference:
    """
    LLM:  ModelInference(decoder=DecoderInference(...))
    MLLM: ModelInference(decoder=DecoderInference(...), encoder=EncoderInference())
    """

    def __init__(
        self,
        decoder: DecoderInference,
        encoder: Optional[EncoderInference] = None,
    ):
        self.decoder = decoder
        self.encoder = encoder

    def predict_step(
        self,
        decoder_module: torch.nn.Module,
        input_ids: torch.Tensor,
        attn_mask: Optional[AttentionMask] = None,
        hidden_states: Tuple[torch.Tensor, ...] = (),
        tok_embedding: Optional[torch.nn.Module] = None,
        encoder_module: Optional[torch.nn.Module] = None,
        encoder_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # If pre-computed hidden_states are not provided but an encoder is available, run it.
        if (
            not hidden_states
            and self.encoder is not None
            and encoder_module is not None
        ):
            outputs = []
            for data in encoder_inputs:
                outputs.append(self.encoder.predict_step(encoder_module, data))
            hidden_states = tuple(outputs)
        return self.decoder.predict_step(
            decoder_module,
            input_ids,
            attn_mask=attn_mask,
            hidden_states=hidden_states,
            tok_embedding=tok_embedding,
        )
