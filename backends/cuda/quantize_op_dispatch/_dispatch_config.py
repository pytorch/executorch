# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_plain_mm_max_m = ContextVar("cuda_plain_mm_max_m", default=4)


def get_plain_mm_max_m() -> int:
    return _plain_mm_max_m.get()


@contextmanager
def plain_mm_max_m(value: int) -> Iterator[None]:
    if not 1 <= value <= 16:
        raise ValueError(f"plain_mm_max_m must be in [1, 16], got {value}")
    token = _plain_mm_max_m.set(value)
    try:
        yield
    finally:
        _plain_mm_max_m.reset(token)
