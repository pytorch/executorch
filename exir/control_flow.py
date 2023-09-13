# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This module contains tools for rewriting a dynamic PyTorch program such
that the dynamic part (e.g. control flow) can be properly captured by
DispatchTracer.
The core idea is annotating all branches in the graph with unique keys,
and using a dictionary of supplemental inputs as arguments to these
local branches so that every path gets a canonical input during tracing.

For example, consider the following usage of Python if statement:

.. code-block:: python

    if pred:
        ...
        ret = a
    else:
        ...
        ret = b

To rewrite the code to be tracable, users may use tracing_key decorator
and cond operator:

.. code-block:: python

    @control_flow.tracing_context(inputs)
    def branch_true(args):
        ...
        return a

    @control_flow.tracing_context(inputs)
    def branch_false(args):
        ...
        return b

    ret = control_flow.cond(pred, branch_true, branch_false, args)

and we can use the usual exir.capture() function.

.. code-block:: python

    exir.capture(module, args)

"""

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree
from executorch.exir.error import ExportError, ExportErrorType, internal_assert
from executorch.exir.tracer import (
    DispatchTracer,
    flattened_dispatch_trace,
    PythonTensor,
    tree_return,
    unwrap_functional,
    unwrap_proxy,
    using_tracer,
    Value,
)
from executorch.exir.wrap import update_with_proxy


def shape(x: torch.Tensor) -> Union[torch._C.Size, torch.Tensor]:
    """
    A helper function for capturing the shape as a tensor from a tensor
    value.
    """
    tracer = DispatchTracer.get()
    if tracer is None:
        return x.shape
    x = unwrap_functional(x)
    if not isinstance(x, PythonTensor):
        raise ExportError(
            ExportErrorType.INVALID_INPUT_TYPE,
            f"exir custom shape function only takes EXIR dispatch tensor, but got: {type(x)}",
        )
    # TODO _shape_as_tensor should work with functional tensor but currently not.
    # TODO torch.tensor() should succeed under functionalization but currently not.
    #      see: https://github.com/pytorch/pytorch/pull/76319
    tmp = torch.empty(len(x.shape), dtype=torch.int64)
    for i, s in enumerate(x.shape):
        tmp[i] = s
    proxy = torch.ops.aten._shape_as_tensor.default(x.proxy)
    return PythonTensor(unwrap_functional(tmp), proxy)


def _make_submodule(
    fn: Callable[..., Union[torch.Tensor, Tuple[torch.Tensor]]],
    example_returns: Optional[List[torch.Tensor]] = None,
    single_return: bool = False,
) -> torch.fx.GraphModule:
    if not hasattr(fn, "__tracing_inputs__"):
        raise ExportError(
            ExportErrorType.MISSING_PROPERTY,
            f"Expect function '{fn.__name__}' to be decorated with tracing_context.",
        )
    # pyre-ignore
    args = fn.__tracing_inputs__
    # TODO(yidi): we don't want to enable here because we are not gonna use this code path in the future anyways
    gm, _ = flattened_dispatch_trace(fn, args, set(), enable_functionalization=False)
    output = next(iter(reversed(gm.graph.nodes)))
    if example_returns:
        internal_assert(
            len(example_returns) == len(output.args[0]),
            f"Eager mode of this {gm} returns {len(example_returns)} elements, but this graph returns {len(output.args[0])} elements",
        )

    if single_return:
        # Force number of returned value to be 1.
        internal_assert(
            len(output.args[0]) == 1,
            f"Graph {gm} should return just one element, but got {len(output.args[0])}",
        )
        output.args = tuple(output.args[0])
        gm.recompile()
    gm.__tracing_inputs__ = args
    return gm


def while_loop(
    cond_fn: Callable[..., torch.Tensor],
    body_fn: Callable[..., Tuple[torch.Tensor]],
    init_val: pytree.PyTree,
) -> Union[Tuple[torch.Tensor], Value]:
    """
    A higher order function returning the result based on executing body_fn
    until cond_fn returns False.
    """
    flattened_inputs, _ = pytree.tree_flatten(init_val)
    if not all(isinstance(i, torch.Tensor) for i in flattened_inputs):
        raise ExportError(
            ExportErrorType.INVALID_INPUT_TYPE,
            f"control_flow.while_loop() expects all inputs values to be tensors, actual inputs: {init_val}",
        )

    with using_tracer(None):
        val = init_val
        while cond_fn(*val):
            val = body_fn(*val)

    flattened_outputs, _ = pytree.tree_flatten(val)
    if not all(isinstance(o, torch.Tensor) for o in flattened_outputs):
        raise ExportError(
            ExportErrorType.INVALID_OUTPUT_TYPE,
            f"control_flow.while_loop() expects all returned values to be tensors, actual outputs: {val}",
        )

    tracer = DispatchTracer.get()

    if tracer is None:
        return val

    gm_cond = _make_submodule(cond_fn, single_return=True)
    gm_body = _make_submodule(body_fn)

    proxies = tuple([unwrap_proxy(v) for v in flattened_inputs])

    proxy = tracer.create_proxy(
        "call_function",
        while_loop,
        (gm_cond, gm_body, proxies),
        {},
    )

    return tree_return(val, proxy, update_with_proxy)


def tracing_context(
    inputs: Tuple[torch.Tensor, ...],
) -> Callable[..., Callable[..., Union[torch.Tensor, Tuple[torch.Tensor]]]]:
    """
    A decorator function to annotate code path that we conditionally
    run during tracing. We need to annotate these paths for now because
    during exir.capture(), the tracer does not know what's the proper
    local inputs to be passed to the untaken path.
    """

    def decorator(
        f: Callable[..., Tuple[torch.Tensor]]
    ) -> Callable[..., Union[torch.Tensor, Tuple[torch.Tensor]]]:
        def wrapper(
            *args: torch.Tensor, **kwargs: Tuple[torch.Tensor]
        ) -> Tuple[torch.Tensor]:
            if kwargs:
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    "kwargs are not supported for @tracing_context decorated functions.",
                )

            return f(*args)

        wrapper.__tracing_inputs__ = inputs  # pyre-ignore
        return wrapper

    return decorator
