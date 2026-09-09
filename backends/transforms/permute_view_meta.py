# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from collections.abc import Sequence
from typing import cast

import torch
import torch.fx
from executorch.backends.transforms.dim_maps import (
    _Dim,
    _normalize_dims,
    normalize_view_shape,
)
from executorch.exir.dialects._ops import ops as exir_ops


def refresh_permute_view_meta(node: torch.fx.Node) -> None:
    """Compute new meta-vals, specifically preserving SymInts for view/permute
    nodes.
    """
    input_node = node.all_input_nodes[0]
    input_val = input_node.meta.get("val")
    if input_val is None or node.target not in {
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.permute_copy.default,
    }:
        return

    if not isinstance(input_val, torch.Tensor):
        node.meta["val"] = node.target(input_val, *node.args[1:])  # type: ignore[operator]
        return

    # Compute new meta shapes to preserve SymInts.
    match node.target:
        case exir_ops.edge.aten.view_copy.default:
            node.meta["val"] = input_val.new_empty(
                tuple(
                    normalize_view_shape(
                        input_val.shape, cast(Sequence[_Dim], node.args[1])
                    )
                )
            )
        case exir_ops.edge.aten.permute_copy.default:
            dims = _normalize_dims(
                cast(Sequence[int], node.args[1]), len(input_val.shape)
            )
            node.meta["val"] = input_val.new_empty(
                tuple(input_val.shape[dim] for dim in dims)
            )
        case _:
            node.meta["val"] = node.target(input_val, *node.args[1:])  # type: ignore[operator]
