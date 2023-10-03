# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    get_proxy_slot,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import _get_current_dispatch_mode
from torch.utils._pytree import tree_flatten


executorch_call_delegate = HigherOrderOperator(
    "executorch_call_delegate", _deprecated_global_ns=True
)
# pyre-ignore
executorch_call_delegate.fallthrough(torch._C.DispatchKey.PythonDispatcher)
# pyre-ignore
executorch_call_delegate.fallthrough(torch._C.DispatchKey.PythonTLSSnapshot)
executorch_call_delegate.fallthrough(torch._C.DispatchKey.ADInplaceOrView)
executorch_call_delegate.fallthrough(torch._C.DispatchKey.BackendSelect)
# pyre-ignore
executorch_call_delegate.fallthrough(torch._C.DispatchKey.AutocastCPU)

LOWERED_BACKEND_MODULE_TYPE = "LoweredBackendModule"

# pyre-ignore
def trace_call_delegate(proxy_mode, func_overload, lowered_module, *args):
    # pyre-ignore
    def _unwrap_proxy(e):
        if not isinstance(e, (torch.Tensor, torch.SymInt, torch.SymFloat)):
            return e
        return get_proxy_slot(e, proxy_mode.tracer, e, lambda e: e.proxy)

    if not is_lowered_module(lowered_module):
        raise ValueError(
            "executorch_call_delegate()'s first argument must be a LoweredBackendModule"
        )

    with disable_proxy_modes_tracing():
        out = lowered_module.original_module(*args)

    lowered_name = get_lowered_module_name(proxy_mode.tracer.root, lowered_module)
    proxy_mode.tracer.root.register_module(lowered_name, lowered_module)

    node_args = (lowered_module, *args)
    proxy_args = pytree.tree_map(_unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="executorch_call_delegate"
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@executorch_call_delegate.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
# pyre-ignore
def call_delegate_cpu(lowered_module, *args):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU key"
    return lowered_module.original_module(*args)


@executorch_call_delegate.py_impl(torch._C.DispatchKey.Autograd)
# pyre-ignore
def call_delegate_autograd(lowered_module, *args):
    # TODO: support autograd
    flat_operands, _ = tree_flatten([lowered_module, *args])
    requires_grad = any(
        f.requires_grad for f in flat_operands if isinstance(f, torch.Tensor)
    )

    with torch._C._ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
    ):
        res = executorch_call_delegate(lowered_module, *args)

        if requires_grad:
            # Create aliases of the output that has requires_grad=True. We need
            # at least one of the inputs to err_fn to require grad so that the
            # output will have a grad_fn.

            # pyre-ignore
            def fake_requires_grad(var):
                if var is not None:
                    var = var.detach()
                    if torch.is_floating_point(var) or torch.is_complex(var):
                        var.requires_grad = True
                return var

            return pytree.tree_map(fake_requires_grad, res)

        return res


@executorch_call_delegate.py_impl(ProxyTorchDispatchMode)
# pyre-ignore
def call_delegate_proxy_torch_dispatch_mode(mode, lowered_module, *args):
    res = trace_call_delegate(mode, executorch_call_delegate, lowered_module, *args)
    return res


@executorch_call_delegate.py_impl(FakeTensorMode)
# pyre-ignore
def call_delegate_fake_tensor_mode(mode, lowered_module, *args):
    with mode:
        return lowered_module.original_module(*args)


@executorch_call_delegate.py_functionalize_impl
# pyre-ignore
def call_delegate_functionalize(ctx, lowered_module, *args):
    unwrapped_args = tuple(ctx.unwrap_tensors(arg) for arg in args)
    with ctx.redispatch_to_next():
        res = executorch_call_delegate(lowered_module, *unwrapped_args)
        return ctx.wrap_tensors(res)


# pyre-ignore: Missing parameter annotation [2]: Parameter `obj` must have a type other than `Any`.Pyre
def is_lowered_module(obj: Any) -> bool:
    """
    This function is added to avoid using isinstance(obj, LoweredBackendModule) as it will import LoweredBackendModule, which may cause a circular import.
    """
    return type(obj).__name__ == LOWERED_BACKEND_MODULE_TYPE


def get_lowered_module_name(
    root: torch.nn.Module,
    # pyre-ignore: Undefined or invalid type [11]: Annotation `LoweredBackendModule` is not defined as a type.
    lowered_module: LOWERED_BACKEND_MODULE_TYPE,  # noqa
) -> str:
    """
    Adds the given lowered_module into the given root module and returns the
    name of the module added.
    """
    # Find a qualifying name for the lowered submodule
    qualname = None
    i = 0
    while True:
        qualname = f"lowered_module_{i}"
        if not hasattr(root, qualname):
            break
        i += 1
    assert qualname is not None

    root.add_module(qualname, lowered_module)
    return qualname
