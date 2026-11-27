# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.exir._serialize._flatbuffer_program import (
    deserialize_program_method_io,
    ProgramMethodIO,
)
from executorch.exir._serialize._program import (
    deserialize_pte_binary as _deserialize_pte_binary,
    PTEFile as _PTEFile,
    serialize_pte_binary as _serialize_pte_binary,
)

# Underscore-prefixed APIs are internal to exir.
__all__ = [
    "deserialize_program_method_io",
    "ProgramMethodIO",
    "_deserialize_pte_binary",
    "_serialize_pte_binary",
    "_PTEFile",
]
