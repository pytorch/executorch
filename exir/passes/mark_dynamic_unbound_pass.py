# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from executorch.exir.pass_base import ExportPass


class MarkDynamicUnboundPass(ExportPass):
    """
    Marks matching placeholder nodes with ``et_dynamic_unbound`` metadata.

    After ``SpecPropPass`` creates ``TensorSpec`` for each placeholder,
    ``update_placeholder_tensor_specs`` reads this flag and sets the spec's
    ``shape_dynamism`` to ``DYNAMIC_UNBOUND``.  The memory planner then skips
    those tensors, and the runtime allocates their memory lazily via
    ``DynamicAllocator``.

    Typical usage: mark KV cache buffers so they start unallocated and grow
    on demand, avoiding the full upfront memory cost of max_context_length.
    """

    def __init__(
        self,
        name_patterns: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.name_patterns = name_patterns or ["k_cache", "v_cache"]

    def placeholder(self, name: str, arg, meta):
        if any(pattern in name for pattern in self.name_patterns):
            meta["et_dynamic_unbound"] = True
        return super().placeholder(name, arg, meta)
