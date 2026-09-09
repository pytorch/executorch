#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Canonical model-metadata writer for exported LLM programs.

Exports publish their metadata as PTE constant methods; the shared typed C++
reader in ``extension/llm/runner/model_metadata.h`` consumes them. The method
names below are the single Python-side source of truth and must stay in sync
with the constants in ``extension/llm/runner/constants.h``.

This module deliberately depends only on ``torch`` (and a lazy ``ScalarType``
import), so backend-neutral consumers -- including the generic runner
round-trip test -- can produce metadata fixtures without pulling in any
backend.
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


def model_constant_methods(
    *,
    max_context_len: int,
    logits_to_keep: str,
    activation_dtype: str,
    vocab_size: int,
    max_seq_len: Optional[int] = None,
) -> dict[str, int]:
    """Build constant methods shared by exported LLM programs.

    ``max_seq_len`` is the largest single forward step this export traces; it is
    serialized as ``get_max_seq_len``. ``max_context_len`` is the KV-cache
    capacity, serialized as ``get_max_context_len``.
    """
    if max_context_len <= 0:
        raise ValueError(f"Invalid maximum context length: {max_context_len}")
    if max_seq_len is not None and max_seq_len <= 0:
        raise ValueError(f"Invalid max sequence length: {max_seq_len}")
    try:
        logits_to_keep_mode = LOGITS_TO_KEEP_MODES[logits_to_keep]
    except KeyError as error:
        raise ValueError(
            f"Unsupported logits-to-keep mode: {logits_to_keep}"
        ) from error
    from executorch.exir.scalar_type import ScalarType

    try:
        activation_dtype_value = int(
            {
                "fp16": ScalarType.HALF,
                "fp32": ScalarType.FLOAT,
                "bf16": ScalarType.BFLOAT16,
            }[activation_dtype]
        )
    except KeyError as error:
        raise ValueError(f"Unsupported activation dtype: {activation_dtype}") from error
    if vocab_size <= 0:
        raise ValueError(f"Invalid vocabulary size: {vocab_size}")
    methods = {
        MAX_CONTEXT_LEN_METHOD: max_context_len,
        VOCAB_SIZE_METHOD: vocab_size,
        LOGITS_TO_KEEP_MODE_METHOD: logits_to_keep_mode,
        ACTIVATION_DTYPE_METHOD: activation_dtype_value,
    }
    if max_seq_len is not None:
        methods[MAX_SEQ_LEN_METHOD] = max_seq_len
    return methods


def model_vocab_size(model: torch.nn.Module) -> int:
    """Return the model's actual output vocabulary width."""
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("Model has no output embedding weight")
    vocab_size = int(output_embeddings.weight.shape[0])
    if vocab_size <= 0:
        raise ValueError(f"Invalid vocabulary size: {vocab_size}")
    return vocab_size
