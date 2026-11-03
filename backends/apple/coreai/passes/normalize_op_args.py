# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Spell out op arguments that coreai-torch misreads when left implicit."""

from typing import Callable, Dict

import torch
from executorch.backends.apple.coreai.passes.replace_copy_ops import functional_aten_op
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import PassResult
from torch.fx import GraphModule, Node


def coreai_aten_op(target):
    """The ATen overload coreai-torch lowers ``target`` as (edge and ``*_copy``
    variants unwrapped)."""
    if not isinstance(target, EdgeOpOverload):
        return target
    return functional_aten_op(target) or target._op


class NormalizeCoreAIOpArgsPass:
    """Rewrite op arguments into the explicit form coreai-torch expects.

    coreai-torch reads some arguments positionally and doesn't apply PyTorch's
    defaults, so these lower wrongly or raise unless rewritten:

    - ``mean.dim`` / ``amax`` / ``amin`` with ``dim`` omitted, ``None`` or
      ``[]`` (all meaning "every dim"): spell out every dim.
    - ``avg_pool2d`` / ``avg_pool3d`` with ``ceil_mode`` but no
      ``count_include_pad``, or an empty ``stride``: fill in the defaults.
      ``ceil_mode`` is cleared when it doesn't change the output shape.
    - ``squeeze.dims`` naming a dim whose size isn't 1 (a no-op in PyTorch):
      drop that dim, or replace the squeeze with ``clone`` if none remain.

    Runs on ATen or edge graphs.
    """

    def __call__(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            normalize = _NORMALIZERS.get(coreai_aten_op(node.target))
            if normalize is not None and normalize(node):
                modified = True
        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)


def _explicit_reduce_dims(node: Node) -> bool:
    if len(node.args) > 1 and node.args[1]:
        return False
    all_dims = list(range(node.args[0].meta["val"].dim()))
    node.args = (node.args[0], all_dims, *node.args[2:])
    return True


def _explicit_avg_pool_args(node: Node) -> bool:
    args = list(node.args)
    kernel_size = args[1]
    defaults = [kernel_size, [0] * len(kernel_size), False, True]
    new_args = args[:2] + [
        args[i] if i < len(args) else defaults[i - 2] for i in range(2, 6)
    ]
    new_args += args[6:]
    if not new_args[2]:
        new_args[2] = kernel_size
    # The converter only implements floor mode, which is exact whenever
    # ceil_mode doesn't change the output shape.
    if new_args[4] and _floor_mode_shape_matches(node, *new_args[1:4]):
        new_args[4] = False
    if new_args == args:
        return False
    node.args = tuple(new_args)
    return True


def _floor_mode_shape_matches(node: Node, kernel_size, stride, padding) -> bool:
    n = len(kernel_size)
    in_sizes = node.args[0].meta["val"].shape[-n:]
    out_sizes = node.meta["val"].shape[-n:]
    for size, out, k, s, p in zip(
        in_sizes,
        out_sizes,
        kernel_size,
        _expand(stride, n),
        _expand(padding, n),
    ):
        if not isinstance(size, int) or (size + 2 * p - k) // s + 1 != out:
            return False
    return True


def _expand(values, n):
    return list(values) * n if len(values) == 1 else list(values)


def _drop_non_singleton_squeeze_dims(node: Node) -> bool:
    shape = node.args[0].meta["val"].shape
    dims = node.args[1]
    kept = [d for d in dims if not _is_static_non_singleton(shape[d])]
    if len(kept) == len(dims):
        return False
    if kept:
        node.args = (node.args[0], kept, *node.args[2:])
    else:
        # The converter aborts on an empty squeeze, so make it an explicit copy.
        node.target = (
            exir_ops.edge.aten.clone.default
            if isinstance(node.target, EdgeOpOverload)
            else torch.ops.aten.clone.default
        )
        node.args = (node.args[0],)
    return True


def _is_static_non_singleton(size) -> bool:
    return isinstance(size, int) and size != 1


_NORMALIZERS: Dict[object, Callable[[Node], bool]] = {
    torch.ops.aten.mean.dim: _explicit_reduce_dims,
    torch.ops.aten.amax.default: _explicit_reduce_dims,
    torch.ops.aten.amin.default: _explicit_reduce_dims,
    torch.ops.aten.avg_pool2d.default: _explicit_avg_pool_args,
    torch.ops.aten.avg_pool3d.default: _explicit_avg_pool_args,
    torch.ops.aten.squeeze.dims: _drop_non_singleton_squeeze_dims,
}
