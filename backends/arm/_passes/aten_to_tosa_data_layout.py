# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import cast

from executorch.backends.transforms.aten_to_dialect_pass import (
    AtenToDialectPass,
    DialectNodeSpec,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node


def _get_arg(node: Node, index: int, name: str, default=None):
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name, default)


def _normalize_dim(dim: int, rank: int) -> int:
    return (dim + rank) % rank


def _input_rank(node: Node) -> int:
    input_node = cast(Node, node.args[0])
    return len(input_node.meta["val"].shape)


def _rewrite_cat(node: Node, pass_: AtenToDialectPass) -> DialectNodeSpec:
    tensors = cast(Sequence[Node], node.args[0])
    dim = _get_arg(node, 1, "dim", 0)
    first_tensor = tensors[0]
    axis = _normalize_dim(cast(int, dim), len(first_tensor.meta["val"].shape))
    return DialectNodeSpec(
        exir_ops.backend.tosa.CONCAT.default,
        (tensors,),
        {"axis": axis},
    )


def _rewrite_view_copy(node: Node, pass_: AtenToDialectPass) -> DialectNodeSpec:
    return DialectNodeSpec(
        exir_ops.backend.tosa.RESHAPE.default,
        node.args,
        dict(node.kwargs),
    )


def _rewrite_repeat(node: Node, pass_: AtenToDialectPass) -> DialectNodeSpec:
    return DialectNodeSpec(
        exir_ops.backend.tosa.TILE.default,
        node.args,
        dict(node.kwargs),
    )


def _rewrite_permute_copy(node: Node, pass_: AtenToDialectPass) -> DialectNodeSpec:
    permutation = list(cast(Sequence[int], _get_arg(node, 1, "dims")))
    rank = _input_rank(node)
    permutation = [_normalize_dim(dim, rank) for dim in permutation]
    return DialectNodeSpec(
        exir_ops.backend.tosa.TRANSPOSE.default,
        (node.args[0], permutation),
        {},
    )


def _rewrite_flip(node: Node, pass_: AtenToDialectPass) -> DialectNodeSpec | None:
    dims = list(cast(Sequence[int], _get_arg(node, 1, "dims")))
    if len(dims) != 1:
        return None

    return DialectNodeSpec(
        exir_ops.backend.tosa.REVERSE.default,
        (node.args[0],),
        {"axis": _normalize_dim(dims[0], _input_rank(node))},
    )


def rewrite_data_layout_operator(
    node: Node, pass_: AtenToDialectPass
) -> DialectNodeSpec | None:
    match node.target:
        case exir_ops.edge.aten.cat.default:
            return _rewrite_cat(node, pass_)
        case exir_ops.edge.aten.view_copy.default:
            return _rewrite_view_copy(node, pass_)
        case exir_ops.edge.aten.repeat.default:
            return _rewrite_repeat(node, pass_)
        case exir_ops.edge.aten.permute_copy.default:
            return _rewrite_permute_copy(node, pass_)
        case exir_ops.edge.aten.flip.default:
            return _rewrite_flip(node, pass_)
        case _:
            return None
