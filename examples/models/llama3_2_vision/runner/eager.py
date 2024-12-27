# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Optional

import torch

from executorch.examples.models.llama.export_llama_lib import _prepare_for_llama_export
from executorch.examples.models.llama.runner.eager import execute_runner
from executorch.examples.models.llama3_2_vision.runner.generation import (
    TorchTuneLlamaRunner,
)
from executorch.extension.llm.export import LLMEdgeManager


class EagerLlamaRunner(TorchTuneLlamaRunner):
    """
    Runs llama in eager mode with provided checkpoint file.
    """

    def __init__(self, args):
        with open(args.params, "r") as f:
            params = json.loads(f.read())
        super().__init__(
            tokenizer_path=args.tokenizer_path,
            max_seq_len=args.max_seq_length,
            max_batch_size=1,
            use_kv_cache=args.use_kv_cache,
            vocab_size=params["vocab_size"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.encoder_input = torch.load(
            "/home/jackzhxng/torchrepos3/executorch/vision_encoder_output.pt"
        )[0].to(
            torch.float32
        )  # Only support one image at the moment.
        print(self.encoder_input.shape)

        encoder_sequence_length = self.encoder_input.shape[1]
        self.encoder_mask = torch.ones(
            1, args.max_seq_length, 8192, dtype=torch.bool
        ) # 8192 for encoder max seq len, not seq len.
        self.encoder_mask = self.encoder_mask[:, :7]
        print(f"encoder_mask: {self.encoder_mask.shape}")
        # TODO: make demo_config.json contain the encoder params.
        self.prefill = False
        
        manager: LLMEdgeManager = _prepare_for_llama_export(args)
        self.model = manager.model.eval().to(device=self.device)

    def forward(
        self,
        tokens: Optional[torch.LongTensor] = None,
        input_pos: Optional[torch.LongTensor] = None,
        mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        ret = self.model.forward(tokens=tokens, input_pos=input_pos, mask=mask, encoder_input=self.encoder_input, encoder_mask=self.encoder_mask)
        if not self.prefill:
            self.prefill = True
            self.encoder_input = torch.full_like(self.encoder_input, torch.nan)
            self.encoder_mask = self.encoder_mask[:,-1:]
        return ret
        # return self.model.forward(tokens=tokens, input_pos=input_pos, mask=mask)


def main() -> None:
    execute_runner(EagerLlamaRunner)


if __name__ == "__main__":
    main()  # pragma: no cover
