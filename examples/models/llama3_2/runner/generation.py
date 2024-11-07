# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Optional, TypedDict

import torch

from executorch.extension.llm.tokenizer.utils import get_tokenizer
from executorch.examples.models.llama.runner.generation import LlamaRunner, next_token, sample_top_p


class TorchTuneLlamaRunner(LlamaRunner):
    def __init__(
        self,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        use_kv_cache: bool,
        vocab_size: int,
        device: str = "cpu",
    ):
        super().__init__(
            tokenizer_path,
            max_seq_len,
            max_batch_size,
            use_kv_cache,
            vocab_size,
            device,
        )

        self.causal_mask = torch.tril(
            torch.ones(
                size=(self.max_seq_len, self.max_seq_len),
                dtype=torch.bool,
            )
        )
        self.input_pos = torch.arange(self.max_seq_len)

    def generate(  # noqa: C901
        self,
        prompt_tokens: List[int],
        temperature: float = 0.8,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> List[int]:
        # Prefill
        seq_len = len(prompt_tokens)
        input_pos = self.input_pos[None, :seq_len]
        mask = self.causal_mask[None, :seq_len]
        if self.use_kv_cache:
            logits = self.forward(
                tokens=torch.tensor([prompt_tokens], dtype=torch.long, device=self.device),
                input_pos=input_pos,
                mask=mask,
            )
        else:
            logits = self.forward(
                tokens=torch.tensor([prompt_tokens], dtype=torch.long, device=self.device),
            )

        # Only need the last logit.
        current_token = next_token(logits[:, -1, :], temperature, top_p)
        tokens = prompt_tokens + [current_token]

        i = 0
        while len(tokens) < self.max_seq_len:
            print(f"{i} out of {self.max_seq_len} max tokens generated")
            mask = self.causal_mask[None, seq_len, None, :]
            input_pos = self.input_pos[None, seq_len, None]
            if self.use_kv_cache:
                logits = self.forward(
                    tokens=torch.tensor(
                        [[current_token]], dtype=torch.long, device=self.device
                    ),
                    input_pos=input_pos,
                    mask=mask,
                )
            else:
                logits = self.forward(
                    tokens=torch.tensor([tokens], dtype=torch.long, device=self.device),
                )

            # Only need the last logit.
            current_token = next_token(logits[:, -1, :], temperature, top_p)

            if current_token == self.tokenizer.eos_id or (
                hasattr(self.tokenizer, "stop_tokens")
                and current_token in self.tokenizer.stop_tokens
            ):
                break

            tokens.append(current_token)
            i += 1
            seq_len += 1

        return tokens if echo else tokens[len(prompt_tokens) :]

