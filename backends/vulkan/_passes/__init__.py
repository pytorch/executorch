# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.backends.vulkan._passes.insert_prepack_nodes import insert_prepack_nodes
from executorch.backends.vulkan._passes.int4_weight_only_quantizer import (
    VkInt4WeightOnlyQuantizer,
)
from executorch.backends.vulkan._passes.remove_asserts import (
    remove_asserts,
    RemoveAssertsTransform,
)
from executorch.backends.vulkan._passes.remove_local_scalar_dense_ops import (
    RemoveLocalScalarDenseOpsTransform,
)
from executorch.backends.vulkan._passes.remove_redundant_ops import (
    RemoveRedundantOpsTransform,
)
from executorch.backends.vulkan._passes.squeeze_int4_linear_inputs import (
    SqueezeInt4LinearInputs,
)
from executorch.backends.vulkan._passes.tag_memory_meta_pass import TagMemoryMetaPass

__all__ = [
    "insert_prepack_nodes",
    "VkInt4WeightOnlyQuantizer",
    "remove_asserts",
    "RemoveAssertsTransform",
    "RemoveLocalScalarDenseOpsTransform",
    "RemoveRedundantOpsTransform",
    "SqueezeInt4LinearInputs",
    "TagMemoryMetaPass",
]
