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
from typing import List, Union

import requests
import torch
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
        simple_evaluate(model=collector, tasks=tasks, limit=tasks_limit)

    logging.info(
        "Calibration: collected %d sequences (max_context_length=%d, limit=%d)",
        len(collector.sequences),
        max_context_length,
        tasks_limit,
    )
    return collector.sequences
