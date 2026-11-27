# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import cast, Union

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


def _normalize_dim(dim: int, node: Node) -> int | None:
    input_node = cast(Node, node.args[0])
    rank = len(input_node.meta["val"].shape)
    if rank == 0:
        return None
    return (dim + rank) % rank


def _single_dim(dim) -> int | None:
    if isinstance(dim, Sequence) and not isinstance(dim, str):
        if len(dim) != 1:
            return None
        return cast(int, dim[0])
    return cast(int, dim)


def _rewrite_reduction(
    node: Node,
    target,
    *,
    dim,
    keepdim: bool,
    nan_mode: str | None = None,
) -> DialectNodeSpec | None:
    dim = _single_dim(dim)
    if dim is None or not keepdim:
        return None

    axis = _normalize_dim(dim, node)
    if axis is None:
        return None

    kwargs: dict[str, Union[int, str]] = {"axis": axis}
    if nan_mode is not None:
        kwargs["nan_mode"] = nan_mode
    return DialectNodeSpec(target, (node.args[0],), kwargs)


def rewrite_reduction_operator(
    node: Node, pass_: AtenToDialectPass
) -> DialectNodeSpec | None:
    match node.target:
        case exir_ops.edge.aten.amax.default:
            return _rewrite_reduction(
                node,
                exir_ops.backend.tosa.REDUCE_MAX.default,
                dim=_get_arg(node, 1, "dim", []),
                keepdim=cast(bool, _get_arg(node, 2, "keepdim", False)),
                nan_mode="PROPAGATE",
            )
        case exir_ops.edge.aten.amin.default:
            return _rewrite_reduction(
                node,
                exir_ops.backend.tosa.REDUCE_MIN.default,
                dim=_get_arg(node, 1, "dim", []),
                keepdim=cast(bool, _get_arg(node, 2, "keepdim", False)),
                nan_mode="PROPAGATE",
            )
        case exir_ops.edge.aten.any.dim:
            return _rewrite_reduction(
                node,
                exir_ops.backend.tosa.REDUCE_ANY.default,
                dim=_get_arg(node, 1, "dim"),
                keepdim=cast(bool, _get_arg(node, 2, "keepdim", False)),
            )
        case exir_ops.edge.aten.any.dims:
            return _rewrite_reduction(
                node,
                exir_ops.backend.tosa.REDUCE_ANY.default,
                dim=_get_arg(node, 1, "dim"),
                keepdim=cast(bool, _get_arg(node, 2, "keepdim", False)),
            )
        case exir_ops.edge.aten.prod.dim_int:
            return _rewrite_reduction(
                node,
                exir_ops.backend.tosa.REDUCE_PRODUCT.default,
                dim=_get_arg(node, 1, "dim"),
                keepdim=cast(bool, _get_arg(node, 2, "keepdim", False)),
            )
        case exir_ops.edge.aten.sum.dim_IntList:
            return _rewrite_reduction(
                node,
                exir_ops.backend.tosa.REDUCE_SUM.default,
                dim=_get_arg(node, 1, "dim"),
                keepdim=cast(bool, _get_arg(node, 2, "keepdim", False)),
            )
        case _:
            return None
