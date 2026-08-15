# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import requests
import torch
from datasets import load_dataset as hf_load_dataset
from executorch.examples.qualcomm.oss_scripts.llama.dataset.schema import MessageSample
from huggingface_hub import hf_hub_download
from pytorch_tokenizers.hf_tokenizer import HuggingFaceTokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer
from pytorch_tokenizers.tiktoken import TiktokenTokenizer


def load_audio_file(path: str, repo_id: str) -> torch.Tensor:
    """Returns float waveform [1, T] from a URL, local file, or HF Hub path."""
    try:
        import soundfile
    except ImportError:
        raise ImportError(
            "Please install the `soundfile` package via `pip install soundfile`"
        )
    if path.startswith(("http://", "https://")):
        resp = requests.get(path, timeout=60)
        resp.raise_for_status()
        wav, _ = soundfile.read(io.BytesIO(resp.content), always_2d=False)
    else:
        if not os.path.exists(path):
            try:
                path = hf_hub_download(repo_id=repo_id, filename=path)
            except Exception:
                raise FileNotFoundError(
                    f"Audio file {path} not found locally or in HuggingFace repository {repo_id}."
                )
        wav, _ = soundfile.read(path, always_2d=False)
    return torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]


def load_conversation_samples(samples_paths: List[Path]) -> List[MessageSample]:
    """Load and merge MessageSamples from one or more flat JSON files."""
    samples = []
    for p in samples_paths:
        with open(p) as f:
            samples.extend(json.load(f))
    return [MessageSample(**d) for d in samples]


def collect_lm_eval_tokens(
    tokenizer: Union[SentencePieceTokenizer, TiktokenTokenizer, HuggingFaceTokenizer],
    max_context_length: int,
    vocab_size: int,
    tasks: Union[str, List[str]] = "wikitext",
    tasks_limit: int = 1,
    num_fewshot: Optional[int] = None,
) -> List[List[int]]:
    """
    Collect max_context_length-length token sequences from any lm_eval task(s).

    Delegates request-building entirely to lm_eval: simple_evaluate handles
    rolling windows, few-shot context, and multiple-choice continuations.
    Returns raw token sequences; callers are responsible for wrapping in a DataLoader.
    """
    try:
        from executorch.examples.models.llama.evaluate.eager_eval import (
            EagerEvalWrapper,
        )
        from lm_eval.evaluator import simple_evaluate
    except ImportError as e:
        raise ImportError(
            "Please install lm_eval and datasets: "
            "examples/models/llama/install_requirements.sh"
        ) from e

    class _TokenCollector(EagerEvalWrapper):
        def __init__(self):
            super().__init__(
                model=None, tokenizer=tokenizer, max_seq_length=max_context_length
            )
            self.sequences: List[List[int]] = []

        def _model_call(self, inps):
            self.sequences.extend(inps.tolist())
            return torch.zeros(inps.shape[0], inps.shape[1], vocab_size)

    collector = _TokenCollector()
    with torch.no_grad():
        simple_evaluate(
            model=collector, tasks=tasks, num_fewshot=num_fewshot, limit=tasks_limit
        )

    logging.info(
        "Calibration: collected %d sequences (max_context_length=%d, limit=%d)",
        len(collector.sequences),
        max_context_length,
        tasks_limit,
    )
    return collector.sequences


def _assistant_mask_from_boundaries(
    messages: List[dict],
    full_tokens: List[int],
    tokenizer_wrapper,
) -> List[int]:
    """Token-level assistant mask computed by re-tokenizing message prefixes.

    Fallback for chat templates that lack the `{% generation %}` /
    `{% endgeneration %}` markers HuggingFace's `return_assistant_tokens_mask`
    relies on. Instead of trusting the template annotation, we recover each turn's
    token span from the length of tokenize(messages[:i+1]) and mark assistant spans.
    The prefix tokenization must match the full tokenization (same template,
    add_generation_prompt=False) for boundaries to line up.
    """
    mask = [0] * len(full_tokens)
    boundaries = [0]
    for i in range(len(messages)):
        prefix = tokenizer_wrapper.chat_template(
            messages[: i + 1],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        boundaries.append(len(prefix))

    for msg, start, end in zip(messages, boundaries, boundaries[1:]):
        if msg.get("role") == "assistant":
            for j in range(start, min(end, len(full_tokens))):
                mask[j] = 1
    return mask


def load_hf_chat_dataset(
    dataset_name: str,
    tokenizer_wrapper,
    max_context_len: int,
    num_samples: int = 1000,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Tokenize a HuggingFace chat dataset (requires a 'messages' field).

    Returns (token_sequences, assistant_masks_list).
    assistant_masks_list[i] is a token-level 0/1 mask: 1 = assistant turn.
    Samples that fail tokenization or exceed max_context_len are dropped.
    """

    ds = hf_load_dataset(dataset_name, split="train")
    ds = ds.select(range(min(num_samples, len(ds))))

    input_sequences: List[List[int]] = []
    assistant_masks_list: List[List[int]] = []

    for sample in ds:
        messages = sample.get("messages", [])
        full_tokens = []
        if not messages:
            continue
        try:
            result = tokenizer_wrapper.chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                return_dict=True,
            )
            full_tokens = result["input_ids"]
            assistant_masks = result.get("assistant_masks", [0] * len(full_tokens))
            if not any(assistant_masks):
                assistant_masks = _assistant_mask_from_boundaries(
                    messages, full_tokens, tokenizer_wrapper
                )
        except Exception:
            continue
        if not full_tokens or len(full_tokens) > max_context_len:
            continue
        input_sequences.append(full_tokens)
        assistant_masks_list.append(assistant_masks)

    logging.info(
        "%s: loaded %d sequences (requested %d)",
        dataset_name,
        len(input_sequences),
        num_samples,
    )
    return input_sequences, assistant_masks_list
