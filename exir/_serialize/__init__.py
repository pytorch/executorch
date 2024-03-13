# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.exir._serialize._program import (
    deserialize_pte_binary as _deserialize_pte_binary,
    serialize_pte_binary as _serialize_pte_binary,
)

# Internal APIs that should not be used outside of exir.
__all__ = [
    "_deserialize_pte_binary",
    "_serialize_pte_binary",
]
