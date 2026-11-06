# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph Transformation Pass for Complex Numbers.

Inductor generates no code for complex numbers. It calls an ATen kernel for
every op that reads or writes a complex tensor, and this backend ships none of
those kernels. So this pass keeps each complex tensor as a real tensor with a
last dim of size 2 that holds the real and imaginary parts, the layout of
``torch.view_as_real``, and rewrites the ops on it as ops on real numbers:

    view_as_complex(x)  ->  x
    view_as_real(z)     ->  z
    polar(r, theta)     ->  stack([r * cos(theta), r * sin(theta)], -1)
    z * w               ->  stack([zr * wr - zi * wi, zr * wi + zi * wr], -1)

Views, copies, indexing, repeats, splits, concatenation and stacking act on
both parts alike, so they keep their op, with the dims they name counted in
the complex tensor. A cast of a real tensor to complex gets zero imaginary
parts. A complex constant, parameter or buffer is stored in the real layout
too. A decomposition table cannot do this, since a decomposition returns the
dtype of the op it replaces.

The pass rewrites a program only when it can rewrite every op that reads or
writes a complex tensor in it. Otherwise it leaves the program as it is and
logs why, and the backend reports the ops it has no kernels for. It never
rewrites control flow branches, such as those of torch.cond, or complex32
values.
"""

import logging
import operator
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.utils._pytree as pytree
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.fx import Graph, Node

logger = logging.getLogger(__name__)

_aten = exir_ops.edge.aten


def _complex_vals(node: Node) -> List[torch.Tensor]:
    val = node.meta.get("val")
    vals = val if isinstance(val, (list, tuple)) else (val,)
    return [v for v in vals if isinstance(v, torch.Tensor) and v.is_complex()]


# The pair's extra dim comes last, so a dim of the complex tensor names the
# same dim of the pair once it is counted from the front.
def _dim(dim: int, ndim: int) -> int:
    return dim + ndim if dim < 0 else dim


def _dims(dims: List[int], ndim: int) -> List[int]:
    return [_dim(d, ndim) for d in dims]


def _new_dim(dim: int, ndim: int) -> int:
    return _dim(dim, ndim + 1)


def _permutation(dims: List[int], ndim: int) -> List[int]:
    return _dims(dims, ndim) + [ndim]


def _repeats(repeats: List[Any], ndim: int) -> List[Any]:
    return [*repeats, 1]


def _size(size: List[Any], ndim: int) -> List[Any]:
    return [*size, 2]


# Ops that act on the real and imaginary parts alike. For each argument that
# names dims or sizes of the complex tensor, the entry gives the argument's
# position, and how to name the same dims of the pair. These arguments'
# default values never need changing.
_SAME_ON_BOTH_PARTS: Dict[Any, Tuple[Tuple[int, Callable[..., Any]], ...]] = {
    # Torch-TensorRT turns every view into _reshape_copy.
    _aten._reshape_copy.default: ((1, _size),),
    _aten.alias_copy.default: (),
    _aten.cat.default: ((1, _dim),),
    _aten.clone.default: (),
    _aten.expand_copy.default: ((1, _size),),
    _aten.index.Tensor: (),
    _aten.permute_copy.default: ((1, _permutation),),
    _aten.repeat.default: ((1, _repeats),),
    _aten.select.int: ((1, _dim),),
    _aten.select_copy.int: ((1, _dim),),
    _aten.slice.Tensor: ((1, _dim),),
    _aten.slice_copy.Tensor: ((1, _dim),),
    _aten.split.Tensor: ((2, _dim),),
    _aten.split_copy.Tensor: ((2, _dim),),
    _aten.split_with_sizes_copy.default: ((2, _dim),),
    _aten.squeeze_copy.default: (),
    _aten.squeeze_copy.dim: ((1, _dim),),
    _aten.squeeze_copy.dims: ((1, _dims),),
    _aten.stack.default: ((1, _new_dim),),
    _aten.transpose_copy.int: ((1, _dim), (2, _dim)),
    _aten.unbind_copy.int: ((1, _dim),),
    _aten.unsqueeze_copy.default: ((1, _new_dim),),
    _aten.view_copy.default: ((1, _size),),
}

_LIST_INPUTS = (_aten.cat.default, _aten.stack.default)

# Copies that may change the dtype. The pair gets the precision of the
# complex dtype, and the complex tensor's dim order with its extra dim last.
_COPIES = (
    _aten._to_copy.default,
    exir_ops.edge.dim_order_ops._clone_dim_order.default,
    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
)
_DENSE_FORMATS = (None, torch.preserve_format, torch.contiguous_format)
_PROVENANCE = ("from_node", "nn_module_stack", "source_fn_stack", "stack_trace")

# The fake kernels of some view copies, such as permute_copy, make dynamic
# sizes static or fail on them, so a new view copy gets its value from its
# view, laid out as the copy's would be. view_copy keeps its own kernel, which
# keeps dynamic sizes, because view fails on some strides that view_copy
# accepts.
_VIEWS: Dict[Any, Any] = {
    _aten.alias_copy.default: torch.ops.aten.alias.default,
    _aten.expand_copy.default: torch.ops.aten.expand.default,
    _aten.permute_copy.default: torch.ops.aten.permute.default,
    _aten.select_copy.int: torch.ops.aten.select.int,
    _aten.slice_copy.Tensor: torch.ops.aten.slice.Tensor,
    _aten.split_copy.Tensor: torch.ops.aten.split.Tensor,
    _aten.split_with_sizes_copy.default: torch.ops.aten.split_with_sizes.default,
    _aten.squeeze_copy.default: torch.ops.aten.squeeze.default,
    _aten.squeeze_copy.dim: torch.ops.aten.squeeze.dim,
    _aten.squeeze_copy.dims: torch.ops.aten.squeeze.dims,
    _aten.transpose_copy.int: torch.ops.aten.transpose.int,
    _aten.unbind_copy.int: torch.ops.aten.unbind.int,
    _aten.unsqueeze_copy.default: torch.ops.aten.unsqueeze.default,
}


class _Rewriter:
    """Builds, in graph order, the real replacement of every node that reads
    or writes a complex tensor."""

    def __init__(self, graph: Graph, complex_nodes: Set[Node]) -> None:
        self.graph = graph
        self.complex_nodes = complex_nodes
        self.ndims = {
            node: node.meta["val"].dim()
            for node in complex_nodes
            if isinstance(node.meta["val"], torch.Tensor)
        }
        self.replacements: Dict[Node, Node] = {}
        self.source: Optional[Node] = None

    def get(self, value: Any) -> Any:
        """``value`` with each complex node in it replaced by its pair."""
        return torch.fx.map_arg(
            value,
            lambda n: self.replacements[n] if n in self.complex_nodes else n,
        )

    def call(self, target: Any, *args: Any, **kwargs: Any) -> Node:
        node = self.graph.call_function(target, args, kwargs)
        # Errors and profiles then point at the op that this one replaces.
        node.meta.update(
            {k: v for k, v in self.source.meta.items() if k in _PROVENANCE}
        )
        fake_args, fake_kwargs = torch.fx.map_arg(
            (args, kwargs), lambda n: n.meta["val"]
        )
        value = _VIEWS.get(target, target)(*fake_args, **fake_kwargs)
        if target in _VIEWS:
            value = pytree.tree_map_only(
                torch.Tensor,
                lambda t: t.clone(memory_format=torch.contiguous_format),
                value,
            )
        node.meta["val"] = value
        return node

    def parts(self, value: Node) -> List[Node]:
        pair = self.get(value)
        return [self.call(_aten.select.int, pair, -1, i) for i in (0, 1)]


def _view_as_complex(rw: _Rewriter, node: Node) -> Node:
    return node.args[0]


def _view_as_real(rw: _Rewriter, node: Node) -> Node:
    return rw.get(node.args[0])


def _polar(rw: _Rewriter, node: Node) -> Node:
    magnitude, angle = node.args
    parts = [
        rw.call(_aten.mul.Tensor, magnitude, rw.call(trig, angle))
        for trig in (_aten.cos.default, _aten.sin.default)
    ]
    return rw.call(_aten.stack.default, parts, -1)


def _mul(rw: _Rewriter, node: Node) -> Node:
    (ar, ai), (br, bi) = (rw.parts(arg) for arg in node.args)
    mul = _aten.mul.Tensor
    real = rw.call(_aten.sub.Tensor, rw.call(mul, ar, br), rw.call(mul, ai, bi))
    imag = rw.call(_aten.add.Tensor, rw.call(mul, ar, bi), rw.call(mul, ai, br))
    return rw.call(_aten.stack.default, [real, imag], -1)


def _copy(rw: _Rewriter, node: Node) -> Node:
    source, kwargs = node.args[0], dict(node.kwargs)
    if kwargs.get("dtype") is not None:
        kwargs["dtype"] = kwargs["dtype"].to_real()
    if source not in rw.complex_nodes:
        # Edge programs cast a real operand to complex before complex math
        # with it. The cast gives zero imaginary parts.
        real = rw.call(node.target, source, **kwargs)
        imag = rw.call(_aten.zeros_like.default, real)
        return rw.call(_aten.stack.default, [real, imag], -1)
    if kwargs.get("dim_order") is not None:
        kwargs["dim_order"] = [*kwargs["dim_order"], len(kwargs["dim_order"])]
    return rw.call(node.target, rw.get(source), **kwargs)


def _same_on_both_parts(rw: _Rewriter, node: Node) -> Node:
    args = list(node.args)
    tensor = args[0][0] if node.target in _LIST_INPUTS else args[0]
    for index, pair_value in _SAME_ON_BOTH_PARTS[node.target]:
        if index < len(args):
            args[index] = pair_value(args[index], rw.ndims[tensor])
    return rw.call(node.target, *rw.get(args), **rw.get(dict(node.kwargs)))


def _getitem(rw: _Rewriter, node: Node) -> Node:
    return rw.call(operator.getitem, rw.get(node.args[0]), node.args[1])


_REWRITES: Dict[Any, Callable[[_Rewriter, Node], Node]] = {
    _aten.view_as_complex_copy.default: _view_as_complex,
    _aten.view_as_real_copy.default: _view_as_real,
    _aten.polar.default: _polar,
    _aten.mul.Tensor: _mul,
    operator.getitem: _getitem,
    **{op: _copy for op in _COPIES},
    **{op: _same_on_both_parts for op in _SAME_ON_BOTH_PARTS},
}


def _keeps_dims_in_order(node: Node) -> bool:
    # With the pair's dim after them, the dims of a channels-last layout are
    # in an order that no memory format describes, and the dim order ops
    # accept only those that one does.
    dim_order = node.kwargs.get("dim_order")
    return node.kwargs.get("memory_format") in _DENSE_FORMATS and (
        dim_order is None or list(dim_order) == list(range(len(dim_order)))
    )


def _can_rewrite(node: Node, complex_nodes: Set[Node]) -> bool:
    if node.op != "call_function" or node.target not in _REWRITES:
        return False
    # The rewrites read by position the arguments that exported programs
    # pass by position.
    schema = getattr(node.target, "_schema", None)
    if schema is not None and any(
        not arg.kwarg_only for arg in schema.arguments if arg.name in node.kwargs
    ):
        return False
    if node.target in _COPIES:
        # A cast from complex to real has no rewrite here.
        return node in complex_nodes and _keeps_dims_in_order(node)
    if node.target is _aten.clone.default:
        return _keeps_dims_in_order(node)
    if node.target is _aten.mul.Tensor:
        # Edge programs cast a real factor to complex first.
        return all(arg in complex_nodes for arg in node.args)
    if node.target in _LIST_INPUTS:
        return all(tensor in complex_nodes for tensor in node.args[0])
    return True


def _constant_name(exported_program: ExportedProgram, node: Node) -> Optional[str]:
    """The name of the constant, parameter or buffer that ``node`` reads,
    unless it is a buffer that the graph writes."""
    signature = exported_program.graph_signature
    for inputs in (
        signature.inputs_to_lifted_tensor_constants,
        signature.inputs_to_parameters,
    ):
        if node.name in inputs:
            return inputs[node.name]
    buffer = signature.inputs_to_buffers.get(node.name)
    if buffer in signature.buffers_to_mutate.values():
        return None
    return buffer


def _store_as_real(exported_program: ExportedProgram, node: Node) -> None:
    name = _constant_name(exported_program, node)
    tensors = (
        exported_program._constants
        if name in exported_program.constants
        else exported_program._state_dict
    )
    tensor = tensors[name]
    # Partitioning can give one tensor several placeholders, and the first of
    # them stores it.
    if tensor.is_complex():
        pair = torch.view_as_real(tensor.detach().resolve_conj()).clone()
        if isinstance(tensor, torch.nn.Parameter):
            pair = torch.nn.Parameter(pair, requires_grad=False)
        tensors[name] = pair
    node.meta["val"] = torch.view_as_real(node.meta["val"].resolve_conj())


def _reason_to_keep(
    exported_program: ExportedProgram, complex_nodes: Set[Node], affected: Set[Node]
) -> Optional[str]:
    """Why the pass must leave the program as it is, if it must."""
    graph_module = exported_program.graph_module
    for module in graph_module.modules():
        if module is not graph_module and isinstance(module, torch.fx.GraphModule):
            for node in module.graph.nodes:
                if _complex_vals(node):
                    return f"it does not rewrite control flow branches: {node.format_node()}"
    for node in graph_module.graph.nodes:
        if node not in affected:
            continue
        if node.op == "placeholder":
            supported = _constant_name(exported_program, node) is not None
        else:
            supported = _can_rewrite(node, complex_nodes)
        if not supported:
            return f"it cannot rewrite {node.format_node()}"
        # PyTorch computes complex32 math in float32, and its pair would
        # compute in float16.
        if any(val.dtype == torch.complex32 for val in _complex_vals(node)):
            return f"it does not rewrite complex32: {node.format_node()}"
    return None


class ReplaceComplexWithRealPass:
    """
    Replace complex tensors with real tensors that hold their real and
    imaginary parts in a last dim of size 2, and the ops on them with ops on
    real numbers, so that Inductor can generate code for them. See the module
    docstring for the ops it handles.
    """

    requires_exported_program = True

    def __call__(self, exported_program: ExportedProgram) -> None:
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        complex_nodes = {node for node in graph.nodes if _complex_vals(node)}
        affected = complex_nodes | {
            user for node in complex_nodes for user in node.users
        }
        reason = _reason_to_keep(exported_program, complex_nodes, affected)
        if reason is not None:
            logger.warning(
                "The CUDA backend cannot lower the complex numbers in this "
                "program, since %s",
                reason,
            )
            return
        if not complex_nodes:
            return

        rw = _Rewriter(graph, complex_nodes)
        order = [node for node in graph.nodes if node in affected]
        for node in order:
            if node.op == "placeholder":
                _store_as_real(exported_program, node)
                rw.replacements[node] = node
                continue
            rw.source = node
            with graph.inserting_before(node):
                rw.replacements[node] = _REWRITES[node.target](rw, node)
        signature = exported_program.graph_signature
        for node in reversed(order):
            replacement = rw.replacements[node]
            if replacement is not node:
                node.replace_all_uses_with(replacement)
                # The signature names the program's outputs by node, and a
                # view_as_real can be one of them.
                signature.replace_all_uses(node.name, replacement.name)
                graph.erase_node(node)
        graph.lint()
        graph_module.recompile()
