# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.backends.vulkan._passes.fold_qdq import FoldQDQPass
from executorch.backends.vulkan._passes.fuse_patterns import FusePatternsPass
from executorch.backends.vulkan._passes.fuse_quantized_ops import (
    FuseQuantizedOpsTransform,
)
from executorch.backends.vulkan._passes.insert_prepack_nodes import insert_prepack_nodes
from executorch.backends.vulkan._passes.remove_asserts import (
    remove_asserts,
    RemoveAssertsTransform,
)
from executorch.backends.vulkan._passes.remove_redundant_ops import (
    RemoveRedundantOpsTransform,
)
from executorch.backends.vulkan._passes.squeeze_unsqueeze_inputs import (
    SqueezeUnsqueezeInputs,
)
from executorch.backends.vulkan._passes.tag_memory_meta_pass import TagMemoryMetaPass

__all__ = [
    "FoldQDQPass",
    "FusePatternsPass",
    "FuseQuantizedOpsTransform",
    "insert_prepack_nodes",
    "remove_asserts",
    "RemoveAssertsTransform",
    "RemoveRedundantOpsTransform",
    "SqueezeUnsqueezeInputs",
    "TagMemoryMetaPass",
]
