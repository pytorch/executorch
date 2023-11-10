# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.exir.serialize._program import (
    deserialize_from_flatbuffer,
    serialize_to_flatbuffer,
)

__all__ = [
    "deserialize_from_flatbuffer",
    "serialize_to_flatbuffer",
]
