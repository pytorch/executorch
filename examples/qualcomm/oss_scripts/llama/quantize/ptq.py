# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
from executorch.backends.qualcomm._passes import SeqMSE
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    TEXT_DECODER,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.inference import ModelInference
from executorch.examples.qualcomm.oss_scripts.llama.quantize.strategy import (
    QuantizationStrategy,
)
from executorch.examples.qualcomm.oss_scripts.llama.utils import safe_dataloader_iter
from torch.utils.data import DataLoader
from tqdm import tqdm


class PTQStrategy(QuantizationStrategy):
    """Post-training quantization strategy: calibrate only."""

    def __init__(
        self,
        inference: ModelInference,
        module: torch.fx.GraphModule,
        seq_mse_candidates: int = 0,
        tok_embedding: Optional[torch.fx.GraphModule] = None,
    ):
        super().__init__(
            inference=inference, module=module, tok_embedding=tok_embedding
        )
        self._seq_mse_candidates = seq_mse_candidates

    def _calibrate(self, calib_loader: Dict[str, DataLoader]) -> None:
        audio_dataloader = calib_loader.get(AUDIO_ENCODER)
        vision_dataloader = calib_loader.get(VISION_ENCODER)
        text_dataloader = calib_loader[TEXT_DECODER]

        for batch_idx, (audio_batch, vision_batch, text_batch) in tqdm(
            enumerate(
                zip(
                    safe_dataloader_iter(audio_dataloader),
                    safe_dataloader_iter(vision_dataloader),
                    text_dataloader,
                )
            ),
            total=len(text_dataloader),
        ):
            input_ids = text_batch["input_ids"]
            attn_mask = text_batch["attention_mask"]
            hidden_states = (*audio_batch, *vision_batch)
            self._inference.predict_step(
                self._module,
                input_ids=input_ids,
                attn_mask=attn_mask,
                hidden_states=hidden_states,
                tok_embedding=self._tok_embedding,
            )
            if batch_idx == 0 and self._seq_mse_candidates:
                with SeqMSE(self._module, self._seq_mse_candidates):
                    self._inference.predict_step(
                        self._module,
                        input_ids=input_ids,
                        attn_mask=attn_mask,
                        hidden_states=hidden_states,
                        tok_embedding=self._tok_embedding,
                    )

    def quantize(self, calib_loader, **kwargs):
        self._calibrate(calib_loader)
        return self._module
