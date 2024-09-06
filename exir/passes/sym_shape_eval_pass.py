# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Callable, List, Optional

import torch
import torch.utils._pytree as pytree

from executorch.exir._warnings import deprecated
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassBase, PassResult
from executorch.exir.sym_util import eval_expr, eval_shape, eval_upper_bound
from executorch.exir.tensor import TensorSpec
from torch.fx import GraphModule

upper_bound_shape_inference_table = {}


def register_upper_bound_inference(fn):
    def inference_deco(f: Callable):
        upper_bound_shape_inference_table[fn] = f
        return f

    return inference_deco


@register_upper_bound_inference(exir_ops.edge.aten.nonzero.default)
@register_upper_bound_inference(torch.ops.aten.nonzero.default)
def nonzero(args, kwargs) -> List[Optional[int]]:
    return [eval_expr(args[0].numel()), len(args[0].shape)]


@register_upper_bound_inference(exir_ops.edge.aten.index.Tensor)
@register_upper_bound_inference(torch.ops.aten.index.Tensor)
def index_Tensor(args, kwargs) -> List[Optional[int]]:  # noqa: C901
    tensor = args[0]
    indices = args[1]

    # Compute numbers of contiguous blocks of non-null indices.
    # For example, if A, B, C, D, E are non-null tensors, then
    # [None, None, A, B, None, C, D, E, None] has 2 blocks.
    index_blocks = 0
    in_block = False
    for index in indices:
        if index is not None:
            if not in_block:
                in_block = True
                index_blocks += 1
        else:
            in_block = False

    if index_blocks == 0:
        # If no dimensions are actually being indexed, either because the indices list is empty
        # or all indices are null, then the result is just the same as the input tensor.
        return tensor.shape

    adjacent = index_blocks == 1

    # Number of leading null indices in the indices list.
    num_leading_null_indices = 0
    for index in indices:
        if index is None:
            num_leading_null_indices += 1
        else:
            break

    # Number of null indices in total in the indices list.
    num_null_indices = sum([ix is None for ix in indices])

    # Number of dimensions being indexed (bool/byte tensors are treated as masks, and index as
    # many input dimensions as their number of dimensions.
    num_indexed_dims = 0
    mask_indices = []
    int_indices = []
    for index in indices:
        if index is not None:
            if index.dtype in [torch.bool, torch.uint8]:
                num_indexed_dims += index.dim()
                mask_indices.append(index)
            else:
                num_indexed_dims += 1
                int_indices.append(index)

    broadcast_sizes = []
    if len(int_indices) > 0:
        # All of the integer index tensors (non-mask & non-null index tensors) need to broadcast.
        # We need to compute the resulting shape.
        curr_ndim = 0
        rev_shape = []
        for index in int_indices:
            for j in range(index.dim()):
                rev_j_size = eval_expr(index.size(index.dim() - j - 1))
                if j >= curr_ndim:
                    curr_ndim += 1
                    rev_shape.append(rev_j_size)
                elif rev_shape[j] == 1:
                    rev_shape[j] = rev_j_size
        broadcast_sizes = list(reversed(rev_shape))

    # The number of True elements in the mask indices must broadcast (i.e some might be 1
    # but the others must all be equal). They also need to broadcast with broadcast_sizes[0]
    # Therefore, if broadcast_sizes[0] != 1, we don't need to worry about the mask indices,
    # since we are assuming that the inputs are valid. However, if broadcast_sizes[0] = 1,
    # we do need to consider them. We can't know how many True elements there are in each mask,
    # but we know that the broadcasted size, can't be larger than the minimum number of True
    # elements across all mask indices with a number of elements other than 1.
    if len(mask_indices) > 0 and (len(broadcast_sizes) == 0 or broadcast_sizes[0] == 1):
        upper_bound_broadcast_size = 1
        intialized = False
        for mask in mask_indices:
            mask_numel = eval_expr(mask.numel())
            if mask_numel != 1:
                if intialized:
                    assert isinstance(
                        mask_numel, int
                    ), "Expect mask_numel to be a concrete int"
                    assert isinstance(
                        upper_bound_broadcast_size, int
                    ), "Expect upper_bound_broadcast_size to be a concrete int"
                    if upper_bound_broadcast_size > mask_numel:
                        upper_bound_broadcast_size = mask_numel
                else:
                    upper_bound_broadcast_size = mask_numel
                    intialized = True
        if len(broadcast_sizes) == 0:
            broadcast_sizes.append(upper_bound_broadcast_size)
        else:
            broadcast_sizes[0] = upper_bound_broadcast_size

    broadcast_ndim = len(broadcast_sizes)

    out_ndim = tensor.dim() + broadcast_ndim - num_indexed_dims
    out_sizes: List[Optional[int]] = [0 for _ in range(out_ndim)]

    if adjacent:
        for i in range(num_leading_null_indices):
            out_sizes[i] = eval_expr(tensor.size(i))
        for i in range(broadcast_ndim):
            out_sizes[i + num_leading_null_indices] = broadcast_sizes[i]
        for i in range(num_indexed_dims + num_leading_null_indices, tensor.dim()):
            out_sizes[i + broadcast_ndim - num_indexed_dims] = eval_expr(tensor.size(i))
    else:
        for i in range(broadcast_ndim):
            out_sizes[i] = broadcast_sizes[i]
        in_i = 0
        out_i = broadcast_ndim
        for index in indices:
            if index is None:
                out_sizes[out_i] = eval_expr(tensor.size(in_i))
                out_i += 1
                in_i += 1
            else:
                if index.dtype in [torch.bool, torch.uint8]:
                    in_i += index.dim()
                else:
                    in_i += 1

        for i in range(num_indexed_dims + num_null_indices, tensor.dim()):
            out_sizes[i + broadcast_ndim - num_indexed_dims] = eval_expr(tensor.size(i))

    return out_sizes


@deprecated(
    "`HintBasedSymShapeEvalPass` is deprecated "
    "and will be removed in a future version of ExecuTorch. "
    "Please use `ConstraintBasedSymShapeEvalPass` instead.",
    category=FutureWarning,
)
class HintBasedSymShapeEvalPass(PassBase):
    """

    .. warning::

        'HintBasedSymShapeEvalPass` is deprecated
        and will be removed in a future version of ExecuTorch.
        Please use `ConstraintBasedSymShapeEvalPass` instead.

    If we enable dynamic shape tracing, a tensor's shape may become a symbolic
    formula. We should convert those symbolic formula to concrete value for
    static/upperbound tensors so we can properly do memory planning for them.

    HintBasedSymShapeEvalPass evalutes the symbolic expression of shapes based
    on its hint, which is a concrete integer that backs the sym expression. The original
    hint comes from the sizes of the inputs that user uses for tracing and hints of
    symbolic expressions are propagated via meta tensor computation.
    For example, when export f(x), we use x = torch.ones(3, 4) as an exmaple input to f and
    suppose we constrain both dimensions of x as dynamic. We'll have two symbols s0, s1 created
    and they are backed up with hints 3 and 4 respectively. If there is a y = x[0] operation in f,
    the shape of y is inferred to be s1, which is backed up with hint 4.

    Warning: if you're using torch.export with constrain API, this method doesn't respect the input constraints.

    Not inherited from ExportPass since we simply need a way to iterate thru
    every node's output. PassBase is easier for that purpose.
    """

    def call(self, graph_module: GraphModule):
        for subgm in graph_module.modules():
            if not isinstance(subgm, GraphModule):
                continue
            for node in subgm.graph.nodes:
                for spec in pytree.tree_flatten(node.meta.get("spec", []))[0]:
                    # Node for function like aten.sym_size does not have spec
                    if isinstance(spec, TensorSpec):
                        concrete_shape = eval_shape(spec.shape)
                        concrete_spec = eval_shape(spec.stride)
                        if any(s is None for s in concrete_shape) or any(
                            s is None for s in concrete_spec
                        ):
                            # None indicates unbacked symints, see: https://fburl.com/code/v7hj5zv6
                            # Use value range to get the upper bounds of unbacked symints.
                            from torch._guards import detect_fake_mode

                            fake_mode = detect_fake_mode(node.meta.get("val"))
                            if fake_mode is not None:
                                from torch.utils._sympy.numbers import int_oo

                                shape_env = fake_mode.shape_env
                                for i, v in enumerate(spec.shape):
                                    if concrete_shape[i] is None:
                                        # get updated shape from var_to_range
                                        _value_range = shape_env.var_to_range[
                                            v._sympy_()  # pyre-fixme[16] Undefined attribute: `int` has no attribute `_sympy_`.
                                        ]
                                        # cannot handle unbounded, unbacked symints; add a range to bound it.
                                        assert _value_range.upper is not int_oo
                                        concrete_shape[i] = int(_value_range.upper)
                                for i, v in enumerate(spec.stride):
                                    if concrete_spec[i] is None:
                                        _expr = (
                                            v.node.expr  # pyre-fixme[16] Undefined attribute: `int` has no attribute `node`.
                                        )
                                        _value_range = v.node.shape_env.var_to_range
                                        from torch.utils._sympy.value_ranges import (
                                            bound_sympy,
                                        )

                                        _bound_sympy = bound_sympy(_expr, _value_range)
                                        # cannot handle unbounded, unbacked symints; add a range to bound it.
                                        assert _bound_sympy.upper is not int_oo
                                        concrete_spec[i] = int(_bound_sympy.upper)

                        assert all(isinstance(s, int) for s in concrete_shape) and all(
                            isinstance(s, int) for s in concrete_spec
                        )
                        spec.shape = concrete_shape
                        spec.stride = concrete_spec
        return PassResult(graph_module, True)


class ConstraintBasedSymShapeEvalPass(PassBase):
    """
    If we enable dynamic shape tracing, a tensor's shape may become a symbolic
    formula. We should convert those symbolic formula to concrete value for
    static/upperbound tensors so we can properly do memory planning for them.

    Not inherited from ExportPass since we simply need a way to iterate through
    every node's output. PassBase is easier for that purpose.
    """

    def call(self, graph_module: GraphModule):
        for subgm in graph_module.modules():
            if not isinstance(subgm, GraphModule):
                continue
            for node in subgm.graph.nodes:
                for spec in pytree.tree_flatten(node.meta.get("spec", []))[0]:
                    # Node for function like aten.sym_size does not have spec
                    if isinstance(spec, TensorSpec):
                        concrete_shape = [eval_upper_bound(s) for s in spec.shape]
                        concrete_stride = [eval_upper_bound(s) for s in spec.stride]
                        if any(not isinstance(s, int) for s in concrete_shape) or any(
                            not isinstance(s, int) for s in concrete_stride
                        ):
                            raise RuntimeError(
                                f"Cannot evalute the shape upper bound of a dynamic-shaped tensor to a concrete bounded integer. Got tensor spec: {spec}."
                                f"The upper bound shape we get {concrete_shape}, the upper bound stride we get {concrete_stride}"
                                "This tensor could either be from 1. a data-dependent operation such as nonzero. Or 2. an input, whose don't have a constraint for the upper bound."
                                "Please use export's constrain_as_size() or constrain_as_value() apis and set a concrete upper bound to resolve this."
                            )

                        spec.shape = concrete_shape
                        spec.stride = concrete_stride  # pyre-ignore[8]: Attribute `stride` declared in class `TensorSpec` has type `Tuple[int]` but is used as type `List[int]`
        return PassResult(graph_module, True)
