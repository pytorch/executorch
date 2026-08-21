# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export-time CUDA shared-memory targeting for cross-architecture artifacts.

This experiment-only helper is intentionally inactive unless
``ET_CUDA_TARGET_SMEM_BYTES`` is set to a positive integer.
"""

import os
from contextlib import contextmanager
from typing import Iterator


_TARGET_SMEM_ENV = "ET_CUDA_TARGET_SMEM_BYTES"


def _target_smem_bytes() -> int | None:
    raw = os.environ.get(_TARGET_SMEM_ENV)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_TARGET_SMEM_ENV} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{_TARGET_SMEM_ENV} must be positive, got {value}")
    return value


@contextmanager
def target_smem_context() -> Iterator[int | None]:
    """Constrain generated CUDA kernels to a target dynamic-smem budget.

    Triton kernels are checked after each candidate is compiled, using the
    exact ``metadata.shared`` value while retaining the local GPU limit as an
    upper bound.
    """

    target = _target_smem_bytes()
    if target is None:
        yield None
        return

    from triton.compiler import compiler as triton_compiler

    original_max_shared_mem = triton_compiler.max_shared_mem

    def target_max_shared_mem(device):
        return min(int(original_max_shared_mem(device)), target)

    triton_compiler.max_shared_mem = target_max_shared_mem
    print(
        f"CUDA export target shared-memory budget: {target} bytes "
        "(exact Triton metadata filter)"
    )
    try:
        yield target
    finally:
        triton_compiler.max_shared_mem = original_max_shared_mem
