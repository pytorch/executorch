# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional

import torch
import torch.utils._pytree as pytree
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassBase, PassResult
from executorch.exir.sym_util import eval_expr, eval_shape, eval_upper_bound
from executorch.exir.tensor import TensorSpec
from sympy import Integer
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


class HintBasedSymShapeEvalPass(PassBase):
    """
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

    Not inherit from ExportPass since we simply need a way to iterate thru
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

                            def get_val(arg):
                                assert "val" in arg.meta and isinstance(
                                    arg.meta["val"], torch.Tensor
                                )
                                return arg.meta["val"]

                            # TODO (yidi): Replace with range based shape inference using var_to_range.
                            concrete_shape = upper_bound_shape_inference_table[
                                node.target
                            ](*pytree.tree_map(get_val, (node.args, node.kwargs)))

                            for sym_int, i in zip(spec.shape, concrete_shape):
                                if isinstance(sym_int, torch.SymInt):
                                    # We cache the symbolic ints' value as the concrete interger upper bounds.
                                    # So that future use of the unbacked symbols will get a concrete value.
                                    sym_int.node.shape_env.var_to_val[
                                        sym_int.node._expr
                                    ] = Integer(i)

                            # spec.stride is guaranteed to use a subset of symbols in spec.shape, since
                            # we cached the map between symbols and the concrete upper bounds. Can directly eval here.
                            concrete_spec = eval_shape(spec.stride)

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

    Not inherit from ExportPass since we simply need a way to iterate thru
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

                        spec.shape = concrete_shape  # pyre-ignore[8]: Attribute `stride` declared in class `TensorSpec` has type `Tuple[int]` but is used as type `List[Optional[int]]`
                        spec.stride = concrete_stride  # pyre-ignore[8]: Attribute `stride` declared in class `TensorSpec` has type `Tuple[int]` but is used as type `List[Optional[int]]`
        return PassResult(graph_module, True)
