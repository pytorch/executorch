import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Set, Tuple

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._C import _disabled_torch_function_impl
from torch.fx import GraphModule, Tracer
from torch.fx.experimental.normalize import NormalizeArgs
from torch.fx.passes.shape_prop import _extract_tensor_metadata

DEFAULT_LEAF_MODULE_LIST = {}


@contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def unwrap_proxy(e):
    return e.proxy if isinstance(e, DispatchTensor) else e


def build_outputs(func, func_overload, args, kwargs, proxy_out, call_module=False):
    # Kind of a hacky way to test if an op is in-place or not
    if func.__name__[-1] == "_" and func.__name__[0] != "_":
        args[0].proxy = proxy_out

    with no_dispatch():
        real_out = func_overload(*args, **kwargs)

    def wrap_with_proxy(e, proxy):
        if e is None:
            e = torch.empty(())
        if type(e) == torch.Tensor:
            return DispatchTensor(e, proxy)
        # if module output is dispatchTensor, then all op inside it are in-place
        elif type(e) == DispatchTensor and call_module:
            e.proxy = proxy_out
        else:
            return e

    if isinstance(real_out, tuple):
        return tuple(
            [wrap_with_proxy(e, proxy_out[idx]) for idx, e in enumerate(real_out)]
        )
    elif isinstance(real_out, list):
        return [wrap_with_proxy(e, proxy_out[idx]) for idx, e in enumerate(real_out)]
    elif type(real_out) == torch.Tensor:
        return wrap_with_proxy(real_out, proxy_out)
    else:
        return real_out


class DispatchTensor(torch.Tensor):
    """
    Copied from the python key tensor in functorch
    https://github.com/pytorch/functorch/blob/b83273b25213f556f05a065163163ba531e24750/functorch/_src/python_key.py.
    and tracer tensor in subclass_zoo
    https://github.com/albanD/subclass_zoo/blob/main/tracer_tensor.py

    The differences are
        1. when creating the tensor we always set require_grad to false as we are only using
           it here for inference purposes.
    """

    @staticmethod
    def __new__(cls, elem, proxy):
        return torch.Tensor._make_subclass(cls, elem, require_grad=False)

    def __init__(self, elem, proxy):
        self.proxy = proxy
        proxy.node.meta["tensor_meta"] = _extract_tensor_metadata(self)

    def __repr__(self):
        return f"DispatchTensor({torch.Tensor._make_subclass(torch.Tensor, self)})"

    __torch_function__ = _disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
        func = func_overload.overloadpacket
        proxy_args = pytree.tree_map(unwrap_proxy, args)
        proxy_kwargs = pytree.tree_map(unwrap_proxy, kwargs)
        proxy_out = func(*proxy_args, **proxy_kwargs)
        return build_outputs(func, func_overload, args, kwargs, proxy_out)


class DispatchTracer(Tracer):
    """
    Copied from the python key tracer in functorch
    https://github.com/pytorch/functorch/blob/b83273b25213f556f05a065163163ba531e24750/functorch/_src/python_key.py.

    The differences are
        1. this tracer allows specifying leaf module and will preserve it as a call module node
           in the graph.
    """

    def __init__(self, leaf_module_list: Optional[Set[str]] = None):
        super().__init__()
        self.leaf_module_list = (leaf_module_list or set()).union(
            DEFAULT_LEAF_MODULE_LIST
        )

    # User can use leaf_module_list but it won't work combine with functionalize
    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if self.is_leaf_module(m):
            i = 0
            while True:
                qualname = f"{type(m).__name__}_{i}"
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, m)
            proxy_args = pytree.tree_map(unwrap_proxy, args)
            proxy_kwargs = pytree.tree_map(unwrap_proxy, kwargs)
            proxy_out = self.create_proxy(
                "call_module", qualname, proxy_args, proxy_kwargs
            )

            return build_outputs(
                forward, forward, args, kwargs, proxy_out, call_module=True
            )
        return forward(*args, **kwargs)

    def is_leaf_module(self, m) -> bool:
        return torch.typename(m) in self.leaf_module_list

    def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
        if isinstance(attr_val, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        proxy = self.create_proxy("get_attr", n, (), {})
                        parameter_proxy_cache[n] = DispatchTensor(attr_val, proxy)
                    return parameter_proxy_cache[n]
            return attr_val
        return attr_val

    def create_arg(self, a: Any):
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node("get_attr", n, (), {})
            qualname: Optional[str] = None

            i = 0
            while True:
                qualname = f"_param_constant{i}"
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})
        return super().create_arg(a)


def dispatch_trace(
    root: torch.nn.Module,
    leaf_module_list: Optional[Set[str]] = None,
    concrete_args=None,
) -> GraphModule:
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    tracer = DispatchTracer(leaf_module_list)
    graph = tracer.trace(root, concrete_args=concrete_args)
    gm = GraphModule(tracer.root, graph, name)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def wrap_key(f, inps):
    flat_inps, inp_spec = pytree.tree_flatten(inps)

    @functools.wraps(f)
    def wrapped(*args):
        flat_args, args_spec = pytree.tree_flatten(args)
        assert len(flat_args) == len(flat_inps)
        for idx, arg in enumerate(flat_args):
            if isinstance(flat_inps[idx], torch.Tensor):
                flat_args[idx] = DispatchTensor(flat_inps[idx], arg)
            else:
                flat_args[idx] = flat_inps[idx]
        tree_args = pytree.tree_unflatten(flat_args, args_spec)
        out = f(*tree_args)
        flat_outs, out_spec = pytree.tree_flatten(out)
        for idx in range(len(flat_outs)):
            if isinstance(flat_outs[idx], torch.Tensor) and isinstance(
                flat_outs[idx], DispatchTensor
            ):
                flat_outs[idx] = flat_outs[idx].proxy
        return pytree.tree_unflatten(flat_outs, out_spec)

    return wrapped


def make_fx(f, leaf_module_list: Optional[Set[str]] = None):
    @functools.wraps(f)
    def wrapped(*args):
        phs = pytree.tree_map(lambda x: fx.PH, args)
        t = dispatch_trace(
            wrap_key(f, args),
            concrete_args=tuple(phs),
            leaf_module_list=leaf_module_list,
        )
        return t

    return wrapped
