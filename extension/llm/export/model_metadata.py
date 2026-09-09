#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-constant model-metadata writers for exported LLM programs.

Exports publish their metadata as PTE constant methods; the shared typed C++
readers in ``extension/llm/runner/model_metadata.h`` consume them. The method
names below are the single Python-side source of truth and must stay in sync
with the constants in ``extension/llm/runner/constants.h``.

Each ``write_*`` returns the ``{name: value}`` for one constant. Producers
compose the set they publish (the MLX export's ``model_constant_methods`` builds
the full set); the backend-neutral runner test composes them too. This module
deliberately depends only on ``torch`` (and a lazy ``ScalarType`` import), so
those consumers pull in no backend.
"""

from typing import Optional

import torch

# Constant-method names. Keep in sync with extension/llm/runner/constants.h.
MAX_CONTEXT_LEN_METHOD = "get_max_context_len"
MAX_SEQ_LEN_METHOD = "get_max_seq_len"
VOCAB_SIZE_METHOD = "get_vocab_size"
ACTIVATION_DTYPE_METHOD = "get_activation_dtype"
LOGITS_TO_KEEP_MODE_METHOD = "get_logits_to_keep_mode"

# Serialized logits-to-keep modes. Keep in sync with LogitsToKeepMode in
# extension/llm/runner/model_metadata.h.
LOGITS_TO_KEEP_MODES = {"full": 0, "last": 1, "selected": 2}


def _require_positive(name: str, value: int) -> dict[str, int]:
    if value <= 0:
        raise ValueError(f"Invalid value for {name}: {value}")
    return {name: value}


def write_max_context_len(max_context_len: int) -> dict[str, int]:
    """max_context_len -> get_max_context_len (the KV-cache capacity)."""
    return _require_positive(MAX_CONTEXT_LEN_METHOD, max_context_len)


def write_vocab_size(vocab_size: int) -> dict[str, int]:
    """vocab_size -> get_vocab_size."""
    return _require_positive(VOCAB_SIZE_METHOD, vocab_size)


def write_max_seq_len(max_seq_len: Optional[int]) -> dict[str, int]:
    """max_seq_len -> get_max_seq_len (largest single forward step); optional."""
    if max_seq_len is None:
        return {}
    return _require_positive(MAX_SEQ_LEN_METHOD, max_seq_len)


def write_activation_dtype(activation_dtype: str) -> dict[str, int]:
    """activation_dtype name -> get_activation_dtype (ExecuTorch ScalarType)."""
    from executorch.exir.scalar_type import ScalarType

    table = {
        "fp16": ScalarType.HALF,
        "fp32": ScalarType.FLOAT,
        "bf16": ScalarType.BFLOAT16,
    }
    try:
        return {ACTIVATION_DTYPE_METHOD: int(table[activation_dtype])}
    except KeyError as error:
        raise ValueError(f"Unsupported activation dtype: {activation_dtype}") from error


def write_logits_to_keep_mode(logits_to_keep: str) -> dict[str, int]:
    """logits_to_keep name -> get_logits_to_keep_mode."""
    try:
        return {LOGITS_TO_KEEP_MODE_METHOD: LOGITS_TO_KEEP_MODES[logits_to_keep]}
    except KeyError as error:
        raise ValueError(
            f"Unsupported logits-to-keep mode: {logits_to_keep}"
        ) from error


def model_vocab_size(model: torch.nn.Module) -> int:
    """Return the model's actual output vocabulary width."""
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("Model has no output embedding weight")
    vocab_size = int(output_embeddings.weight.shape[0])
    if vocab_size <= 0:
        raise ValueError(f"Invalid vocabulary size: {vocab_size}")
    return vocab_size
