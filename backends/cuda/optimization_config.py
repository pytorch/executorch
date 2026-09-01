# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scoped CUDA optimization choices used while AOTInductor traces kernels."""

import contextlib
import contextvars
from typing import Iterator


_Q4K_FP8_PREFILL_ENABLED = contextvars.ContextVar(
    "q4k_fp8_prefill_enabled", default=False
)
_TMA_CAUSAL_PREFILL_ENABLED = contextvars.ContextVar(
    "tma_causal_prefill_enabled", default=False
)


def q4k_fp8_prefill_enabled() -> bool:
    return _Q4K_FP8_PREFILL_ENABLED.get()


def tma_causal_prefill_enabled() -> bool:
    return _TMA_CAUSAL_PREFILL_ENABLED.get()


@contextlib.contextmanager
def cuda_optimization_context(
    *, q4k_fp8_prefill: bool, tma_causal_prefill: bool = False
) -> Iterator[None]:
    q4k_token = _Q4K_FP8_PREFILL_ENABLED.set(q4k_fp8_prefill)
    tma_token = _TMA_CAUSAL_PREFILL_ENABLED.set(tma_causal_prefill)
    try:
        yield
    finally:
        _TMA_CAUSAL_PREFILL_ENABLED.reset(tma_token)
        _Q4K_FP8_PREFILL_ENABLED.reset(q4k_token)
