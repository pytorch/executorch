# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import json
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import executorch.extension.pytree as ex_pytree
import torch
import torch._dynamo as torchdynamo
import torch.fx as fx

import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from executorch.exir.common import (
    extract_out_arguments,
    format_schema_name,
    no_dispatch,
    setting_python_recursive_limit,
)
from executorch.exir.error import ExportError, ExportErrorType, InternalError
from executorch.exir.graph_module import LeafValue
from executorch.exir.operator.convert import is_out_variant
from executorch.exir.types import ValueSpec

from torch._C import _EnableTorchFunction, DisableTorchFunctionSubclass  # @manual
from torch._decomp import core_aten_decompositions, get_decompositions
from torch._dynamo.guards import Guard
from torch._functorch.eager_transforms import _maybe_unwrap_functional_tensor
from torch.func import functionalize
from torch.fx.operator_schemas import normalize_function
from torch.utils._pytree import TreeSpec

from typing_extensions import TypeAlias


Value: TypeAlias = Union[
    LeafValue,
    Tuple["Value", ...],
    List["Value"],
    Dict[str, "Value"],
]

torchdynamo_enabled = False


def get_stacktrace() -> List[Dict[str, str]]:
    """
    Get the current stacktrace (between trace() and __torch_dispatch__())
    Include the filename, function name, line number, and source code from the
    start of the function to the given instruction.

    Return:
        A list of stacktraces for each instruction along with the source code
        context surrounding each instruction
    """

    stacktrace = traceback.extract_stack()

    # The stacktrace typically looks like this:
    #
    #   1. I stack frames from the top level runner (e.g., the
    #      test suite runner)
    #   2. J frames in executorch/exir/tracer.py setting up tracing
    #      (call this INIT_EXIR)
    #   3. K frames in user model code (this is what we want to save!)
    #   4. 1 frame in executorch/exir/tracer.py __torch_function__
    #      returning to tracer (call this TRACE_EXIR)
    #   5. H frames in executorch/exir/tracer.py AND torch/_tensor.py
    #      doing all of the internal tracer handling
    #
    # The PyE tests assert that executorch/exir/tracer.py never shows
    # up in the user provided stack traces, so we must oblige them.
    #
    # Assumptions:
    #   - Reentrant tracing is not a thing.  Thus, the first time
    #     executorch/exir/tracer.py shows up in the trace, we know
    #     THAT is the point at which we start tracing.  (An alternative
    #     is that the tracer entry point could record the stack trace
    #     at this time, but I didn't do this.)
    #
    # Our plan is to do a miniature stack machine traversing these
    # stack machines.

    # Remove parts before the trace function and parts after entering
    # __torch_dispatch__.  Defaults to returning the entire stack trace.
    init_exir_end = 0
    trace_exir_start = None
    # A miniature state machine, referring to the frame segments described
    # above.  The locations are closed-open interval.
    FIND_INIT_EXIR_START, FIND_INIT_EXIR_END, FIND_TRACE_EXIR_START = range(3)
    state = FIND_INIT_EXIR_START
    for i, frame in enumerate(stacktrace):
        if state == FIND_INIT_EXIR_START:
            if "executorch/exir/tracer.py" in frame.filename:
                state = FIND_INIT_EXIR_END
        elif state == FIND_INIT_EXIR_END:
            if "executorch/exir/tracer.py" not in frame.filename:
                init_exir_end = i
                state = FIND_TRACE_EXIR_START
        elif state == FIND_TRACE_EXIR_START:
            if "executorch/exir/tracer.py" in frame.filename:
                trace_exir_start = i
                break

    stacktrace = stacktrace[init_exir_end:trace_exir_start]

    # Get the source code from the errored line to it
    contexts: List[str] = []
    for s in stacktrace:
        try:
            with open(s.filename) as file:
                # pyre-fixme[6]: For 1st param expected `Union[SupportsTrunc, bytes,
                #  str, SupportsInt, SupportsIndex]` but got `Optional[int]`.
                lineno = int(s.lineno)
                # Get the source code 5 lines above/below the current instruction
                file_contents = [
                    str(index + 1) + line for index, line in enumerate(file.readlines())
                ]
                file_contents_above = "".join(
                    file_contents[max(lineno - 5, 0) : lineno]
                )
                file_contents_below = "".join(
                    file_contents[lineno : min(lineno + 5, len(file_contents))]
                )
                context = (
                    file_contents_above
                    + "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
                    + file_contents_below
                )
                contexts.append(context)
        except FileNotFoundError:
            contexts.append("<unknown file: unknown line>")

    # torch.fx stack preservation logic expects strings to
    # be passed around. Working with dictionary is lot easier
    # to convert to string and vice versa.
    frames: List[Dict[str, str]] = []
    for i, frame in enumerate(stacktrace):
        frames.append(
            {
                "filename": str(frame.filename),
                "lineno": str(frame.lineno),
                "name": str(frame.name),
                "line": str(frame.line),
                "context": contexts[i],
            }
        )

    return frames


def unwrap_functional(t: torch.Tensor) -> torch.Tensor:
    assert isinstance(t, torch.Tensor)
    return _maybe_unwrap_functional_tensor(t, reapply_views=False)


def unwrap_proxy(t: LeafValue) -> Union[LeafValue, torch.fx.Proxy]:
    if not isinstance(t, torch.Tensor):
        return t
    t = unwrap_functional(t)
    return t.proxy if isinstance(t, PythonTensor) else t


def single_return(
    output: LeafValue,
    proxy: torch.fx.Proxy,
    wrapper: Callable[..., LeafValue],
) -> LeafValue:
    if isinstance(output, torch.Tensor):
        return wrapper(output, proxy)

    return output


def tree_return(
    outputs: Value,
    proxy: torch.fx.Proxy,
    wrapper: Callable[..., LeafValue],
    meta_type: Callable[..., Iterable[ValueSpec]] = tuple,
) -> Value:
    i: int = 0

    def wrap(o: LeafValue) -> LeafValue:
        nonlocal i
        ret = single_return(o, proxy[i], wrapper)
        i += 1
        return ret

    return pytree.tree_map(wrap, outputs)


class DummyProxy:
    def __init__(self) -> None:
        class DummyNode:
            def __init__(self):
                self.meta = {}

        self.node = DummyNode()

    def __getitem__(self, key: str) -> "DummyProxy":
        return DummyProxy()


class PythonTensor(torch.Tensor):
    """
    A wrapper tensor subclass used in the DispatchTracer to keep track of
    proxies to construct the FX graph.

    Wrapping something in PythonTensor implicitly detaches gradients.  If
    something required grad, we will collect it as if it were a leaf.  A
    consequence of detaching in this way is you need to maintain a parameter
    cache when translating tensors into PythonTensor, so you don't create
    multiple copies of a gradient (they are aliased, but they would count as
    independent leaves).  An alternate strategy would be to avoid implicitly
    detaching and instead "catch" gradients as they exit the PythonTensor
    boundary.
    """

    __slots__ = ["proxy", "is_immutable"]

    @staticmethod
    def __new__(
        cls, elem: torch.Tensor, proxy: torch.fx.Proxy, is_immutable: bool = False
    ) -> torch.Tensor:
        # assert not elem.requires_grad or not torch.is_grad_enabled()

        r = torch.Tensor._make_subclass(cls, elem, elem.requires_grad)
        assert isinstance(r, PythonTensor)
        r.is_immutable: bool = is_immutable
        r.update_proxy(proxy)
        return r

    def update_proxy(self, proxy: torch.fx.Proxy) -> None:
        self.proxy = proxy

    def __repr__(self, *, tensor_contents: None = None) -> str:
        with no_dispatch():
            return f"PythonTensor({self.as_subclass(torch.Tensor)})"

    @classmethod
    def __torch_function__(
        cls,
        # pyre-ignore: Missing parameter annotation [2]
        func,
        # pyre-ignore: Missing parameter annotation [2]
        types,
        args: Tuple[Value, ...] = (),
        kwargs: Optional[Dict[str, Value]] = None,
    ) -> Value:
        if kwargs is None:
            kwargs = {}
        if torch.is_inference_mode_enabled():
            if func is torch.nn.functional.layer_norm:
                args, kwargs = normalize_function(func, args, kwargs)  # pyre-fixme[23]
                input, normalized_shape = args
                normalized_shape = list(normalized_shape)
                return cls.__torch_dispatch__(
                    torch.ops.aten.layer_norm.default,
                    types,
                    (input, normalized_shape),
                    kwargs,
                )
            elif func is torch.nn.functional.linear:
                return cls.__torch_dispatch__(
                    torch.ops.aten.linear.default, types, args, kwargs
                )
        with DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(  # noqa: C901
        cls,
        func_overload: torch._ops.OpOverload,
        # pyre-ignore: Missing parameter annotation [2]
        types,
        args: Tuple[Value, ...] = (),
        kwargs: Optional[Dict[str, Value]] = None,
    ) -> Value:
        """
        This function is invoked every time an aten operation is called.

        Args:
            func_overload: The function that was called that invoked this
                torch_dispatch call
            types:
            args: Arguments that were passed into the function. Each argument
                has type PythonTensor.
            kwargs: Keyword arguments that were passed into the function. Each
                argument has type PythonTensor.
        """
        func = func_overload.overloadpacket

        kwargs = kwargs or {}
        if is_out_variant(func._qualified_op_name, func_overload._overloadname):
            out_args = extract_out_arguments(func_overload._schema, kwargs)
            out_args_iter = [out_args] if not isinstance(out_args, list) else out_args
            for out_arg_name, out_arg_val in out_args_iter:
                if isinstance(out_arg_val, PythonTensor) and out_arg_val.is_immutable:
                    raise RuntimeError(
                        "Immutable tensor `{}` is potentially getting modified by {}".format(
                            out_arg_name, format_schema_name(func_overload._schema)
                        )
                    )

        # pyre-fixme[16]: Module `pytree` has no attribute `tree_map`.
        proxy_args = ex_pytree.tree_map(unwrap_proxy, args)
        # pyre-fixme[16]: Module `pytree` has no attribute `tree_map`.
        proxy_kwargs = ex_pytree.tree_map(unwrap_proxy, kwargs)

        # Get the output of the function
        g = _EnableTorchFunction()
        try:
            proxy_out = (
                func_overload(*proxy_args, **proxy_kwargs)
                if DispatchTracer.get() or torchdynamo_enabled
                # Disable node creation when no tracer is active.
                else DummyProxy()
            )
        finally:
            del g

        with no_dispatch():
            real_out = func_overload(*args, **kwargs)

        # Kind of a hacky way to test if an op is in-place or not
        if func.__name__[-1] == "_" and func.__name__[0] != "_":
            if isinstance(args[0], PythonTensor):
                args[0].proxy = proxy_out

        if not torch.fx.traceback.has_preserved_node_meta():
            proxy_out.node.meta["stack_trace"] = json.dumps(get_stacktrace())

        # Wrap the output tensors with the PythonTensor subclass to propagate to
        # future tracing
        def wrap_with_proxy(e: LeafValue, proxy: torch.fx.Proxy) -> LeafValue:
            # Some ops (like native_batch_norm_backward) return undefined tensors that get
            # converted into None in python.
            # As the function signature expects tensors, if we directly return these None
            # tensors back to C++, we'll error.
            if e is None:
                e = torch.empty(())

            if isinstance(e, torch.Tensor):
                return PythonTensor(e, proxy)

            # Inplace and out-variant ops may return one of their arguments, which is already
            # a PythonTensor. In this case, we need to update the PythonTensor's associated
            # proxy to the newly created proxy.
            if isinstance(e, PythonTensor):
                e.update_proxy(proxy)
                return e

            return e

        retval = None
        if not isinstance(real_out, (list, tuple)):
            retval = single_return(real_out, proxy_out, wrap_with_proxy)
        else:
            retval = tree_return(real_out, proxy_out, wrap_with_proxy, type(real_out))
        return retval


@contextmanager
def using_tracer(tracer: Optional["DispatchTracer"]) -> Generator[None, None, None]:
    """
    Set the "current" global tracer within the scope of using_tracer
    context manager.

    Since various things we want to capture today with torch_dispatch
    does not "trap" into dispatcher really (for example, cond() and
    shape()), we need a separate singleton tracer exposed to user space
    in addition to Dispatcher to trigger graph capturing.
    """
    global TRACER
    TRACER, prev = tracer, TRACER
    try:
        yield
    finally:
        TRACER = prev


class DispatchTracer(fx.Tracer):
    def __init__(self) -> None:
        super().__init__()
        self.root: torch.nn.Module = torch.nn.Module()
        self.tensor_attrs: Dict[torch.Tensor, str] = {}
        self.submodules: Dict[fx.GraphModule, str] = {}

    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Value],
        args: Tuple[Value, ...],
        kwargs: Dict[str, Value],
    ) -> Value:
        return forward(*args, **kwargs)

    def _module_getattr(
        self, attr: str, attr_val: Value, parameter_proxy_cache: Dict[str, torch.Tensor]
    ) -> Value:
        if isinstance(attr_val, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        proxy = self.create_proxy("get_attr", n, (), {})
                        parameter_proxy_cache[n] = PythonTensor(attr_val, proxy)
                    return parameter_proxy_cache[n]
            return attr_val
        return attr_val

    def create_arg(self, a: Value) -> torch.fx.Node:  # noqa: C901
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node("get_attr", n, (), {})
            qualname: Optional[str] = None

            if not qualname:
                i = 0
                while True:
                    qualname = f"_param_constant{i}"
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})

        if isinstance(a, torch.Tensor):
            qualname: Optional[str] = self.tensor_attrs.get(a)

            if not qualname:
                i = 0
                while True:
                    qualname = f"_tensor_constant{i}"
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                self.tensor_attrs[a] = qualname
                self.root.register_buffer(qualname, a)

            return self.create_node("get_attr", qualname, (), {})

        # higher-order operator
        if isinstance(a, fx.GraphModule):
            if a not in self.submodules:
                name_submodule = f"submodule_{len(self.submodules)}"
                self.root.add_module(name_submodule, a)
                self.submodules[a] = name_submodule
            return self.create_node("get_attr", self.submodules[a], (), {})

        return super().create_arg(a)  # pyre-fixme[7]

    @staticmethod
    def get() -> "DispatchTracer":
        return TRACER

    def trace(  # pyre-fixme[14,15]
        self,
        root: Callable[..., Value],
        concrete_args: Tuple[Value, ...] = (),
        in_spec: Optional[TreeSpec] = None,
    ) -> Value:
        """
        Traces the given graph module.
        """
        with using_tracer(self):
            return self._trace(root, concrete_args=concrete_args, in_spec=in_spec)

    def _trace(
        self,
        root: Callable[..., Value],
        concrete_args: Tuple[Value, ...],
        in_spec: Optional[TreeSpec],
    ) -> Value:
        self.root = torch.nn.Module()
        root_fn = root

        tracer_cls = getattr(self, "__class__", None)
        self.graph = fx.Graph(tracer_cls=tracer_cls)
        # Don't support module, so tensor_attrs is always empty
        self.tensor_attrs = {}

        # Wrap all inputs as a PythonTensor subclass and insert them into the FX
        # graph as placeholder nodes
        def wrap(arg: Value, i: int) -> Value:
            placeholder = self.create_proxy("placeholder", f"ph_{i}", (), {})
            if isinstance(arg, torch.Tensor):
                return PythonTensor(arg, placeholder, is_immutable=True)
            else:
                # torch._assert(
                #     placeholder == arg,
                #     f"ph_{i} has been specialized to have value {arg}",
                # )
                return arg

        tree_args = [wrap(arg, i) for i, arg in enumerate(concrete_args)]
        if in_spec:
            tree_args = pytree.tree_unflatten(tree_args, in_spec)

        tree_out = root_fn(*tree_args)

        out_args, _ = pytree.tree_flatten(tree_out)

        def unwrap(out: LeafValue) -> Union[LeafValue, torch.fx.Proxy]:
            # it's legit for a model to return a list of items some of which
            # are None
            if out is None:
                return None
            if not isinstance(out, torch.Tensor):
                raise TypeError(
                    f"Expect model to return torch.Tensor, got type: '{type(out)}' (value: {out})."
                )
            return unwrap_proxy(out)

        returns = [unwrap(out) for out in out_args]

        return_annotation = None
        # some ops like torch.sub doesn't have annotations
        if hasattr(root_fn, "__annotations__"):
            return_annotation = root_fn.__annotations__.get("return", None)

        self.create_proxy(
            "output",
            "output",
            (returns,),
            {},
            type_expr=return_annotation,
        )

        self.submodule_paths = None

        return tree_out


TRACER: Optional[DispatchTracer] = None
TORCHDYNAMO_ENABLED: bool = False


@contextmanager
def using_dynamo(val: bool) -> Generator[None, None, None]:
    global TORCHDYNAMO_ENABLED
    TORCHDYNAMO_ENABLED, prev = val, TORCHDYNAMO_ENABLED
    try:
        yield
    finally:
        TORCHDYNAMO_ENABLED = prev


def flattened_dispatch_trace(
    f: Callable[..., Value],
    args: Tuple[LeafValue, ...],
    guards: Set[Guard],
    in_spec: Optional[TreeSpec] = None,
    enable_functionalization: bool = True,
) -> Tuple[torch.fx.GraphModule, Value]:
    if not isinstance(args, tuple):
        raise TypeError(f"Expecting 'args' to be a tuple, got: {type(args)}")

    tracer = DispatchTracer()

    if enable_functionalization:
        f = functionalize(f, remove="mutations_and_views")
    tree_out = tracer.trace(f, concrete_args=args, in_spec=in_spec)

    name = type(f).__name__ if isinstance(f, torch.nn.Module) else f.__name__
    gm = torch.fx.GraphModule(tracer.root, tracer.graph, name)

    return (gm, tree_out)


@dataclass
class ExirDynamoConfig:
    """
    Manage Exir-specific configurations of Dynamo.
    """

    allow_rnn: bool = True
    verbose: bool = True
    assume_static_by_default: bool = False


def flatten_output(gm: torch.fx.GraphModule) -> None:
    """
    Modifies the output nodes in the submodules to return the result
    as a flattened list. This keeps it consistent with the result of
    EXIR's tracer
    """
    for node in reversed(gm.graph.nodes):
        if node.op == "output":
            assert len(node.args) == 1
            outputs = node.args[0]
            returns, _ = pytree.tree_flatten(outputs)
            node.args = (returns,)
            return
    raise RuntimeError(f"Could not find an output node in {gm.graph}")


def _default_decomposition_table(
    _use_old_decomp_table=False,
) -> Dict[torch._ops.OpOverload, Callable[..., Value]]:
    if _use_old_decomp_table:
        decomp_opset = [
            torch.ops.aten.log_sigmoid_forward,
            torch.ops.aten.ones,
            torch.ops.aten.arange.default,
            torch.ops.aten.arange.start,
            torch.ops.aten.transpose,
        ]
        # pyre-fixme[7]: Expected `Dict[OpOverload, typing.Callable[..., executorch.e...
        return get_decompositions(decomp_opset)
    # pyre-fixme[7]: Expected `Dict[OpOverload, typing.Callable[..., executorch.exir....
    return core_aten_decompositions()


def dynamo_trace(
    f: Callable[..., Value],
    # pyre-ignore
    args: Tuple[Any, ...],
    aten_graph: bool,
    tracing_mode: str = "real",
    dynamo_config: Optional[ExirDynamoConfig] = None,
    # pyre-ignore
    dynamic_shapes: Optional[List[Any]] = None,
    _use_old_decomp_table: bool = False,
) -> Tuple[torch.fx.GraphModule, Set[Guard]]:
    """
    TODO: Once we fully migrate to torchdynamo frontend, we will remove
    this config option alltogether.  For now, it helps with quick
    experiments with playing around with TorchDynamo
    """
    if dynamo_config is None:
        dynamo_config = ExirDynamoConfig()

    with torchdynamo.config.patch(
        asdict(dynamo_config)
    ), setting_python_recursive_limit(2000):
        torchdynamo.reset()
        try:
            # TODO merge executorch functionalization with official
            # functionalization
            # pyre-fixme[7]: Expected `Tuple[GraphModule, Set[Guard]]` but got
            #  `ExportResult`.
            return torchdynamo.export(
                f,
                aten_graph=aten_graph,
                tracing_mode=tracing_mode,
                assume_static_by_default=dynamo_config.assume_static_by_default,
                decomposition_table=(
                    _default_decomposition_table(_use_old_decomp_table)
                    if aten_graph
                    else None
                ),
                dynamic_shapes=dynamic_shapes,
            )(
                *copy.deepcopy(args),
            )
        except torchdynamo.exc.Unsupported as exc:
            raise ExportError(
                ExportErrorType.NOT_SUPPORTED,
                "The user code is using a feature we don't support. "
                "Please try torchdynamo.explain() to get possible the reasons",
            ) from exc
        except Exception as exc:
            raise InternalError(
                "torchdynamo internal error occured. Please see above stacktrace"
            ) from exc


def dispatch_trace(
    f: Callable[..., Value],
    args: Tuple[Value, ...],
) -> torch.fx.GraphModule:
    """
    Executes a given callable `f` with a given tuple of arguments. During
    execution, Tensor operations are recorded in a fx.GraphModule, which is then
    returned.

    Args:
        f: A `nn.Module` or a Python function that implements an ML program.
        args: A tuple of arguments of any type to be used as inputs for the tracing run.

    Returns:
        EXIR contained in a fx.GraphModule
    """
    trace_func = f
    guards = set()
    if TORCHDYNAMO_ENABLED:
        # Copying args is safer in case downstream implementations of trace_func mutate them
        trace_func, guards = dynamo_trace(trace_func, args, False)

    # Copying args is safer in case downstream implementations of trace_func mutate them
    trace_args, in_spec = pytree.tree_flatten(args)

    in_args = copy.deepcopy(tuple(trace_args))
    gm, tree_out = flattened_dispatch_trace(
        trace_func,
        in_args,
        guards,
        in_spec,
        enable_functionalization=False,
    )

    _, out_spec = pytree.tree_flatten(tree_out)

    gm.in_spec = in_spec
    gm.out_spec = out_spec

    # TODO (tmanlaibaatar) This is bit clowny, but our
    # dispatch_trace sometimes creates unused node that
    # breaks functionalization. it seems too much trouble
    # to fix it properly since dispatch_trace will be deprecated soon.
    # Basically dispatch_trace struggles on:
    # def f(x: torch.Tensor) -> torch.Tensor:
    #    return torch.ones(6, dtype=x.dtype)
    changed = gm.graph.eliminate_dead_code()
    if changed:
        gm.recompile()

    in_args = copy.deepcopy(tuple(trace_args))
    assert callable(gm)

    # This wrapper is used for preserving the stacktrace
    # during second round of tracing.
    # pyre-ignore
    def graph_with_interpreter(*args):
        try:
            args = fx_pytree.tree_flatten_spec(args, gm.in_spec)  # type: ignore[assignment]
        except Exception:
            _, received_spec = pytree.tree_flatten(args)
            raise RuntimeError(
                "Trying to flatten user inputs with exported input tree spec: \n"
                f"{gm.in_spec}\n"
                "but actually got inputs with tree spec of: \n"
                f"{received_spec}"
            )
        with torch.fx.traceback.preserve_node_meta():
            res = gm(*args)

        if gm.out_spec is not None:
            try:
                res = pytree.tree_unflatten(res, gm.out_spec)
            except Exception:
                _, received_spec = pytree.tree_flatten(res)
                raise RuntimeError(
                    "Trying to flatten user outputs with exported output tree spec: \n"
                    f"{gm.out_spec}\n"
                    "but actually got outputs with tree spec of: \n"
                    f"{received_spec}"
                )
        return res

    gm, tree_out = flattened_dispatch_trace(
        graph_with_interpreter,
        in_args,
        guards,
        in_spec,
        enable_functionalization=True,
    )

    gm.in_spec = in_spec
    gm.out_spec = out_spec

    return gm
