# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.apple.coreai.passes.decompose_add_sub_alpha import (
    DecomposeAddSubAlphaPass,
)
from executorch.backends.apple.coreai.passes.narrow_dtypes import (
    NarrowToCoreAIDtypesPass,
)
from executorch.backends.apple.coreai.passes.normalize_op_args import (
    NormalizeCoreAIOpArgsPass,
)

__all__ = [
    "DecomposeAddSubAlphaPass",
    "NarrowToCoreAIDtypesPass",
    "NormalizeCoreAIOpArgsPass",
]
