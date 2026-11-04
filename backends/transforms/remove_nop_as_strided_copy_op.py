# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast, Optional

import torch
from executorch.backends.transforms.permute_pass_utils import (
    get_arg,
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch.fx.node import Node


class RemoveNopAsStridedCopyOpPass(RemoveOrReplacePassInterface):
    """Remove as_strided_copy ops that preserve every logical element."""

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.as_strided_copy.default]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        input_node = node.args[0]
        assert isinstance(input_node, Node)
        input_value = cast(torch.Tensor, input_node.meta["val"])
        size = get_arg(node, "size", list[int])
        stride = get_arg(node, "stride", list[int])
        storage_offset = get_arg(node, "storage_offset", Optional[int])

        if tuple(size) != tuple(input_value.shape) or len(stride) != len(size):
            return False
        if (
            storage_offset is not None
            and storage_offset != input_value.storage_offset()
        ):
            return False
        if not all(
            dim_size == 1 or output_stride == input_stride
            for dim_size, output_stride, input_stride in zip(
                size, stride, input_value.stride()
            )
        ):
            return False

        node.replace_all_uses_with(input_node)
        return True
