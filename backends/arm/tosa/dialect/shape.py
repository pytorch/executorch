# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm.tosa.mapping import TosaSpecialDtype


def is_shape_op_node(node: torch.fx.Node) -> bool:
    """Check if a node is a shape operation node based on TosaSpecialDtype
    metadata.
    """
    return meta_has_shape_mark(node.meta)


def meta_has_shape_mark(meta: dict) -> bool:
    """Check if NodeMetadata has TosaSpecialDtype set to SHAPE."""
    tosa_special_dtype_val = meta.get(TosaSpecialDtype.meta_key())
    return tosa_special_dtype_val == TosaSpecialDtype.SHAPE
