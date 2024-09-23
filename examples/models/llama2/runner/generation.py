# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Optional, TypedDict

import torch

from executorch.examples.models.llama2.llama_transformer import ModelArgs
from executorch.examples.models.llama2.tokenizer.tiktoken import Tokenizer


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is re-normalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature > 0:
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        return sample_top_p(probs, top_p).item()
    return torch.argmax(logits[:, -1], dim=-1).item()


class LlamaRunner(ABC):
    def __init__(self, tokenizer_path: str, model_args: ModelArgs):
        self.params = model_args
        self.tokenizer = Tokenizer(tokenizer_path)
        assert model_args.vocab_size == self.tokenizer.n_words

    @abstractmethod
    def forward(
        self,
        tokens: Optional[torch.LongTensor] = None,
        input_pos: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        pass

    def generate(  # noqa: C901
        self,
        prompt_tokens: List[int],
        temperature: float = 0.8,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> List[int]:
        # prefill
        logits = self.forward(
            tokens=torch.tensor([prompt_tokens], dtype=torch.long),
            input_pos=(
                torch.tensor([0], dtype=torch.long)
                if self.params.use_kv_cache
                else None
            ),
        )

        current_token = next_token(logits, temperature, top_p)
        tokens = prompt_tokens + [current_token]

        while len(tokens) < self.params.max_seq_len:
            if self.params.use_kv_cache:
                logits = self.forward(
                    tokens=torch.tensor([[current_token]], dtype=torch.long),
                    input_pos=torch.tensor([len(tokens) - 1], dtype=torch.long),
                )
            else:
                logits = self.forward(tokens=torch.tensor([tokens], dtype=torch.long))
            current_token = next_token(logits, temperature, top_p)
            if current_token in self.tokenizer.stop_tokens:
                break
            tokens.append(current_token)

        return tokens if echo else tokens[len(prompt_tokens) :]

    def text_completion(
        self,
        prompt: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> CompletionPrediction:
        """
        Perform text completion for a prompt using the language model.

        Args:
            prompt (str): Text prompt for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            CompletionPrediction: Completion prediction, which contains the generated text completion.

        Note:
            This method generates text completion for the provided prompt, employing nucleus sampling to introduce controlled randomness.
        """
        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        generation_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
        )
        return {
            "generation": self.tokenizer.decode(generation_tokens),
            "tokens": generation_tokens,
        }
