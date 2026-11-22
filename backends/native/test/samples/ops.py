# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recorded ops, keyed by the name their packages are written under."""

from collections.abc import Callable
from dataclasses import dataclass

from executorch.backends.native.test.samples import (
    core_ops,
    elu,
    embedding,
    gelu,
    linear,
    pad_pool,
)
from executorch.backends.native.test.samples.cases import Case
from executorch.backends.native.test.samples.record import DTYPES


@dataclass(frozen=True)
class Op:
    cases: Callable[[], list[Case]]
    dtypes: tuple[str, ...] = tuple(DTYPES)


def ops() -> dict[str, Op]:
    return {
        "elu": Op(elu.cases),
        "embedding": Op(embedding.cases),
        "linear": Op(linear.cases),
        "pad": Op(pad_pool.pad_cases),
        "max_pool2d": Op(pad_pool.max_pool2d_cases),
        "gelu": Op(gelu.cases),
        "ne": Op(core_ops.ne_cases),
        "split": Op(core_ops.split_cases),
        "topk": Op(core_ops.topk_cases),
        "sum": Op(core_ops.sum_cases, ("f32",)),
        "mean": Op(core_ops.mean_cases, ("f32",)),
        "amax": Op(core_ops.amax_cases, ("f32",)),
        "softmax": Op(core_ops.softmax_cases, ("f32",)),
        "repeat_interleave": Op(core_ops.repeat_interleave_cases),
    }
