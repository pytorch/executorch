# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recorded ops, keyed by the name their packages are written under."""

from collections.abc import Callable
from dataclasses import dataclass

from executorch.backends.native.test.samples import linear
from executorch.backends.native.test.samples.cases import Case
from executorch.backends.native.test.samples.record import DTYPES


@dataclass(frozen=True)
class Op:
    cases: Callable[[], list[Case]]
    dtypes: tuple[str, ...] = tuple(DTYPES)


def ops() -> dict[str, Op]:
    return {"linear": Op(linear.cases)}
