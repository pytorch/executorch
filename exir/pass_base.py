# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import operator
import traceback
from contextlib import nullcontext
from typing import (
    Any,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
    Protocol,
    runtime_checkable,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import torch
from executorch.exir import memory

from executorch.exir.delegate import executorch_call_delegate, is_lowered_module

from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.error import ExportError, ExportErrorType
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses import FakeTensorMode, UnsupportedFakeTensorException
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch.fx import traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import PythonKeyTracer
from torch.fx.graph import CodeGen
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import PyTree

Fn = Callable[..., Any]  # pyre-ignore
Argument = Any  # pyre-ignore
Value = Any  # pyre-ignore
NodeMetadataValue = Any  # pyre-ignore
K = TypeVar("K")
PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]


_TORCH_SYM_OPS: Set[Any] = {  # pyre-ignore
    torch.sym_int,
    torch.sym_float,
    torch.sym_ite,
    torch.sym_max,
    torch.sym_min,
    torch.sym_not,
    torch.sym_sqrt,
}


PROTECTED_KEYS: Set[str] = {
    "val",
    "stack_trace",
    "nn_module_stack",
    "debug_handle",
    "tensor_meta",
}


def _unstack_pytree(xs) -> List[PyTree]:  # pyre-ignore
    flat_xs, inspec = pytree.tree_flatten(xs)
    if not all(isinstance(xs, torch.Tensor) for xs in flat_xs):
        raise RuntimeError(f"Leaves of xs must be Tensor {flat_xs}")

    if not all(xs.shape[0] == flat_xs[0].shape[0] for xs in flat_xs):
        raise RuntimeError(
            f"Leaves of xs must have same leading dimension size {[xs.shape for xs in flat_xs]}"
        )

    ctx = (
        FunctionalTensorMode
        if any(isinstance(x, FunctionalTensor) for x in flat_xs)
        else nullcontext
    )
    with ctx():
        a = zip(*flat_xs)

    pytrees = []
    for tuple in a:
        pytrees.append(pytree.tree_unflatten(tuple, inspec))
    return pytrees


class NodeMetadata:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.data: Dict[str, Any] = data.copy()

    def __getitem__(self, key: str) -> NodeMetadataValue:
        return self.data[key]

    def __setitem__(self, key: str, value: NodeMetadataValue) -> NodeMetadataValue:
        if key in PROTECTED_KEYS:
            raise RuntimeError(f"Could not override node key: {key}")
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def copy(self) -> "NodeMetadata":
        return NodeMetadata(self.data.copy())


class ProxyValue:
    # pyre-ignore
    def __init__(self, data, proxy: Union[torch.fx.Proxy, torch.fx.Node]):
        # pyre-ignore
        self.data = data
        self.proxy_or_node = proxy

    @property
    def node(self) -> torch.fx.Node:
        if isinstance(self.proxy_or_node, torch.fx.Node):
            return self.proxy_or_node
        assert isinstance(self.proxy_or_node, torch.fx.Proxy)
        return self.proxy_or_node.node

    @property
    def proxy(self) -> torch.fx.Proxy:
        if not isinstance(self.proxy_or_node, torch.fx.Proxy):
            raise RuntimeError(
                f"ProxyValue doesn't have attached Proxy object. Node: {self.proxy_or_node.format_node()}"
            )
        return self.proxy_or_node

    def to_tensor(self) -> torch.Tensor:
        assert isinstance(self.data, torch.Tensor)
        return self.data

    def is_tensor(self) -> bool:
        return isinstance(self.data, torch.Tensor)

    # pyre-ignore
    def __iter__(self):
        yield from self.data

    def __bool__(self) -> bool:
        return bool(self.data)


class ExportPassBaseError(RuntimeError):
    pass


class _ExportPassBase(PassBase):
    """
    Interpreter-based pass class to help users maintain the IR spec while writing
    transformations.
    """

    @staticmethod
    def _create_dummy_node_metadata() -> NodeMetadata:
        return NodeMetadata({"stack_trace": "".join(traceback.format_stack(limit=1))})

    class ExportTracer(PythonKeyTracer):
        def __init__(self, callback: "_ExportPassBase", codegen: CodeGen) -> None:
            super().__init__()
            self.callback = callback
            self.root = torch.nn.Module()
            self.graph = torch.fx.Graph()
            self.graph.set_codegen(codegen)
            self.tensor_attrs: Dict[str, torch.Tensor] = {}  # type: ignore[assignment]
            self.fake_tensor_mode: Optional[FakeTensorMode] = None
            self.submodules: Dict[torch.nn.Module, str] = {}

        def trace(self) -> None:  # pyre-fixme[14,15]
            raise ExportPassBaseError("ExportTracer doesn't support trace().")

        def create_arg(self, a: Argument) -> torch.fx.Node:
            if isinstance(a, torch.nn.Module):
                if a not in self.submodules:
                    name_submodule = f"submodule_{len(self.submodules)}"
                    self.root.add_module(name_submodule, a)
                    self.submodules[a] = name_submodule
            elif isinstance(a, FakeTensor):
                if not hasattr(a, "constant") or a.constant is None:
                    raise ExportPassBaseError(f"Cannot add {a} to graph.")
                a = a.constant
            node = super().create_arg(a)
            if (
                isinstance(a, torch.Tensor)
                and isinstance(node, torch.fx.Node)
                and node.op == "get_attr"
            ):
                self.set_metadata(node, a)
                self.callback.on_attr(ProxyValue(a, node))
            return node

        def set_metadata(  # noqa: C901
            self,
            node: torch.fx.Node,
            value: Argument,
        ) -> None:
            # propagate the fake tensor or sym nodes
            def make_val(
                x: Argument,
            ) -> Union[
                FakeTensor,
                torch.SymInt,
                torch.SymFloat,
                torch.SymBool,
                int,
                float,
                bool,
                str,
                None,
            ]:
                if isinstance(x, FakeTensor):
                    return x
                elif isinstance(x, torch.Tensor):
                    if x.is_quantized:
                        # TODO (tmanlaibaatar) properly support Quantized FakeTensor
                        x = torch.dequantize(x)

                    try:
                        assert self.fake_tensor_mode is not None
                        # TODO we should allocate static shapes
                        # for param/buffer values
                        if isinstance(x, torch.nn.Parameter):
                            fake_tensor = self.fake_tensor_mode.from_tensor(
                                x, static_shapes=True
                            )
                        else:
                            fake_tensor = self.fake_tensor_mode.from_tensor(x)
                    except UnsupportedFakeTensorException:
                        # TODO: This is just a workaround to get over the
                        # x.as_subclass error
                        print(
                            "Fakeifying a Tensor subclass is not supported \
                            right now. Instead a TensorMetadata is used."
                        )
                        fake_tensor = None
                    return fake_tensor
                elif isinstance(
                    x,
                    (
                        torch.SymInt,
                        torch.SymFloat,
                        torch.SymBool,
                        int,
                        float,
                        bool,
                        str,
                    ),
                ):
                    return x
                else:
                    return None

            node.meta["val"] = pytree.tree_map(make_val, value)

            # Set the tensor_metadata for values that do not have a corresponding FakeTensor
            def make_tensor_meta(x: Argument) -> Optional[TensorMetadata]:
                if not isinstance(x, FakeTensor) and isinstance(x, torch.Tensor):
                    if x.is_quantized:
                        # TODO (tmanlaibaatar) properly support Quantized FakeTensor
                        x = torch.dequantize(x)

                    try:
                        assert self.fake_tensor_mode is not None
                        _ = self.fake_tensor_mode.from_tensor(x)
                        tensor_meta = None
                    except UnsupportedFakeTensorException:
                        # TODO: This is just a workaround to get over the
                        # x.as_subclass error
                        tensor_meta = _extract_tensor_metadata(x)
                    return tensor_meta
                else:
                    return None

            node.meta["tensor_meta"] = pytree.tree_map(make_tensor_meta, value)

    class ExportInterpreter(fx.Interpreter):
        def __init__(self, callback: "_ExportPassBase", gm: fx.GraphModule) -> None:
            super().__init__(gm)
            self.callback = callback
            self.node: torch.fx.Node = next(iter(gm.graph.nodes))

        def placeholder(  # pyre-fixme[14]
            self,
            target: str,
            args: Tuple[Argument, ...],
            kwargs: Dict[str, Argument],
        ) -> ProxyValue:
            arg = super().placeholder(target, args, kwargs)
            return self.callback.placeholder(target, arg, NodeMetadata(self.node.meta))

        def output(
            self,
            target: torch.fx.node.Target,
            args: Tuple[Argument, ...],
            kwargs: Dict[str, Argument],
        ) -> ProxyValue:
            return self.callback.output(args[0], NodeMetadata(self.node.meta)).data

        def call_function(
            self,
            target: torch.fx.node.Target,
            args: Tuple[Argument, ...],
            kwargs: Dict[str, Argument],
        ) -> ProxyValue:
            meta = NodeMetadata(self.node.meta)

            if target == operator.getitem:
                value, key = args
                return self.callback.call_getitem(value, key, meta)
            elif getattr(target, "__module__", None) in {"_operator", "math"}:
                assert callable(target)
                return self.callback.call_sym(target, args, meta)
            elif target in _TORCH_SYM_OPS:
                assert callable(target)
                return self.callback.call_sym(target, args, meta)
            elif isinstance(
                target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)
            ):
                return self.callback.call_operator(
                    target,
                    args,
                    kwargs,
                    meta,
                )
            elif target == torch.ops.higher_order.cond:
                pred, true_fn, false_fn, inputs = args
                return self.callback.call_cond(pred, true_fn, false_fn, inputs, meta)
            elif target == torch.ops.higher_order.map_impl:
                f, mapped_args, operands = args  # type: ignore[assignment]
                return self.callback.call_map(f, mapped_args, operands, meta)
            # For other unregistered HigherOrderOps, just interpret them blindly
            elif isinstance(target, torch._ops.HigherOrderOperator):
                return self.callback._fx(
                    "call_function",
                    target,
                    args,
                    kwargs,
                    meta,
                )
            else:
                raise ExportPassBaseError(f"Unsupported target type: {target}")

        def get_attr(  # pyre-fixme[14]
            self, target: str, args: Tuple[Argument, ...], kwargs: Dict[str, Argument]
        ) -> Argument:
            return super().get_attr(target, args, kwargs)

        def call_module(
            self,
            target: torch.fx.node.Target,
            args: Tuple[Argument, ...],
            kwargs: Dict[str, Argument],
        ) -> None:
            raise ExportPassBaseError("call_module is not supported.")

        def call_method(  # pyre-fixme[14]
            self, target: str, args: Tuple[Argument, ...], kwargs: Dict[str, Argument]
        ) -> None:
            raise ExportPassBaseError("call_method is not supported.")

        def run_node(self, n: torch.fx.Node) -> Argument:
            self.node = n
            self.callback.node_debug_str = n.format_node()
            return super().run_node(n)

    def __init__(self) -> None:
        self.interpreter = torch.fx.Interpreter(
            torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        )
        self.tracer = self.ExportTracer(self, CodeGen())  # pyre-ignore
        self.fake_tensor_mode: Optional[FakeTensorMode] = None
        self._initialized = True
        self.node_debug_str: Optional[str] = None

    def _fx(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        res_data = getattr(self.interpreter, kind)(target, args_data, kwargs_data)
        args_proxy, kwargs_proxy = pytree.tree_map_only(
            ProxyValue, lambda x: x.proxy, (args, kwargs)
        )

        name = None
        if isinstance(target, torch._ops.OpOverload):
            name = self.tracer.graph._target_to_str(target.overloadpacket.__name__)

        res_proxy = self.tracer.create_proxy(
            kind, target, args_proxy, kwargs_proxy, name=name
        )
        res_proxy.node.meta.update(meta.data)
        self.tracer.set_metadata(res_proxy.node, res_data)
        return ProxyValue(res_data, res_proxy)

    def inputs(self, graph_module: torch.fx.GraphModule) -> List[Argument]:
        # TODO(angelayi): Update this with what we decide to do for metadata in
        # the exported graph module
        if (args := graph_module.meta.get("args", None)) is not None:
            return list(args)

        def extract_input(node: torch.fx.Node) -> Optional[FakeTensor]:
            if "val" in node.meta:
                fake = node.meta["val"]
                if hasattr(fake, "constant") and fake.constant is not None:
                    return fake.constant
                return fake
            elif tensor_meta := node.meta.get("tensor_meta"):
                assert self.fake_tensor_mode is not None
                return FakeTensor(
                    self.fake_tensor_mode,
                    torch.empty(
                        tensor_meta.shape,
                        dtype=tensor_meta.dtype,
                        device="meta",
                        requires_grad=tensor_meta.requires_grad,
                        memory_format=tensor_meta.memory_format,
                    ),
                    torch.device("cpu"),
                )
            elif len(node.users) == 0:
                return None
            raise ExportPassBaseError(
                f"Cannot construct an input for graph module: {graph_module}.",
            )

        return [
            extract_input(node)
            for node in graph_module.graph.nodes
            if node.op == "placeholder"
        ]

    def on_attr(self, attr: ProxyValue) -> None:
        pass

    def placeholder(self, name: str, arg: Argument, meta: NodeMetadata) -> ProxyValue:
        arg_proxy = self.tracer.create_proxy("placeholder", name, (), {})
        arg_proxy.node.meta = meta.data
        self.tracer.set_metadata(arg_proxy.node, arg)
        return ProxyValue(arg, arg_proxy)

    def call_operator(
        self,
        op,  # pyre-ignore
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        return self._fx("call_function", op, args, kwargs, meta)

    def call_sym(
        self,
        target: Fn,
        args: Tuple[Argument, ...],
        meta: NodeMetadata,
    ) -> ProxyValue:
        return self._fx("call_function", target, args, {}, meta)

    def call_cond(
        self,
        pred: ProxyValue,
        true_fn: torch.fx.GraphModule,
        false_fn: torch.fx.GraphModule,
        inputs: List[Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        true_branch = self.call_submodule(true_fn, tuple(inputs))
        false_branch = self.call_submodule(false_fn, tuple(inputs))
        assert true_branch is not None
        assert false_branch is not None
        return self._fx(
            "call_function",
            torch.ops.higher_order.cond,
            (pred, true_branch.graph_module, false_branch.graph_module, list(inputs)),
            {},
            meta,
        )

    def call_map(
        self,
        f: torch.fx.GraphModule,
        mapped_args: List[ProxyValue],
        operands: List[ProxyValue],
        meta: NodeMetadata,
    ) -> ProxyValue:
        xs = _unstack_pytree([arg.data for arg in mapped_args])[0]
        f_branch = self.call_submodule(f, tuple(xs + [arg.data for arg in operands]))
        assert f_branch is not None
        return self._fx(
            "call_function",
            torch.ops.higher_order.map_impl,
            (f_branch.graph_module, mapped_args, operands),
            {},
            meta,
        )

    def call_getitem(
        self, value: ProxyValue, key: int, meta: NodeMetadata
    ) -> ProxyValue:
        return self._fx("call_function", operator.getitem, (value, key), {}, meta)

    def output(self, results: List[Argument], meta: NodeMetadata) -> ProxyValue:
        return self._fx("output", "output", (results,), {}, meta)

    def call_submodule(
        self, graph_module: fx.GraphModule, inputs: Tuple[Argument, ...]
    ) -> PassResult:
        prev_tracer, self.tracer = self.tracer, self.ExportTracer(
            self, graph_module.graph._codegen
        )
        self.tracer.fake_tensor_mode = prev_tracer.fake_tensor_mode
        interpreter = self.ExportInterpreter(self, graph_module)
        prev_interpreter, self.interpreter = self.interpreter, torch.fx.Interpreter(
            torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        )
        inputs_data = pytree.tree_map_only(ProxyValue, lambda x: x.data, inputs)
        with fx_traceback.preserve_node_meta():
            interpreter.run(*inputs_data)

        new_graph_module = torch.fx.GraphModule(self.tracer.root, self.tracer.graph)

        self.tracer = prev_tracer
        self.interpreter = prev_interpreter
        return PassResult(
            new_graph_module,
            True,
        )

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        if not getattr(self, "_initialized", False):
            raise ExportPassBaseError(
                "ExportPass is not initialized with __init__().",
            )

        inputs = self.inputs(graph_module)

        fake_tensor_mode = None
        for i in inputs:
            if isinstance(i, FakeTensor):
                assert (
                    fake_tensor_mode is None or fake_tensor_mode is i.fake_mode
                ), "Multiple fake tensor mode detected."
                fake_tensor_mode = i.fake_mode
        if fake_tensor_mode is None:
            self.tracer.fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True)
            fake_tensor_mode = nullcontext()  # type: ignore[assignment]
            dispatcher_mode = nullcontext()  # type: ignore[assignment]
        else:
            fake_tensor_mode.allow_non_fake_inputs = True
            self.tracer.fake_tensor_mode = fake_tensor_mode
            dispatcher_mode = enable_python_dispatcher()  # type: ignore[assignment]
        self.fake_tensor_mode = self.tracer.fake_tensor_mode

        with fake_tensor_mode, dispatcher_mode:  # type: ignore[assignment, union-attr]
            result = self.call_submodule(graph_module, tuple(inputs))

        return result


class ExportPass(_ExportPassBase):
    class ExportTracer(_ExportPassBase.ExportTracer):
        def create_arg(self, a: Argument) -> torch.fx.Node:
            if isinstance(a, torch.nn.Module):
                if a not in self.submodules:
                    prefix = "lowered_module" if is_lowered_module(a) else "submodule"
                    name_submodule = f"{prefix}_{len(self.submodules)}"
                    self.root.add_module(name_submodule, a)
                    self.submodules[a] = name_submodule
            return super().create_arg(a)

    class ExportInterpreter(_ExportPassBase.ExportInterpreter):
        """
        Interpreter to callback on any ExportPassBase functions
        """

        def __init__(self, callback: "ExportPass", gm: fx.GraphModule) -> None:
            super().__init__(callback, gm)

        def call_function(
            self,
            target: torch.fx.node.Target,
            args: Tuple[Argument, ...],
            kwargs: Dict[str, Argument],
        ) -> ProxyValue:
            meta = NodeMetadata(self.node.meta)
            if target == operator.getitem:
                value, key = args
                return self.callback.call_getitem(value, key, meta)
            elif isinstance(target, EdgeOpOverload):
                return self.callback.call_operator(
                    target,
                    args,
                    kwargs,
                    meta,
                )

            # TODO according to zhengxu ExportPassBase should not be aware of
            # memory.alloc. Check this comment:
            # https://www.internalfb.com/diff/D42758019?dst_version_fbid=5906016402813292&transaction_fbid=1104713900200176
            elif target == memory.alloc:
                return self.callback._fx(
                    "call_function",
                    target,
                    args,
                    kwargs,
                    meta,
                )

            elif target == executorch_call_delegate:
                lowered_module = args[0]
                args = args[1:]
                return self.callback.call_delegate(  # pyre-ignore
                    lowered_module,
                    args,
                    kwargs,
                    NodeMetadata(self.node.meta),
                )

            return super().call_function(target, args, kwargs)

    def call_delegate(
        self,
        # pyre-ignore: Undefined or invalid type [11]: Annotation `LoweredBackendModule` is not defined as a type.
        lowered_module: "LoweredBackendModule",  # noqa
        args: Tuple[ProxyValue, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        args = (lowered_module,) + args
        return self._fx(
            "call_function",
            executorch_call_delegate,
            args,
            kwargs,
            meta,
        )

    def call_submodule(
        self, graph_module: fx.GraphModule, inputs: Tuple[Argument, ...]
    ) -> PassResult:
        res = super().call_submodule(graph_module, inputs)

        def preserve_original_ph_meta_val(
            gm: torch.fx.GraphModule, new_gm: torch.fx.GraphModule
        ) -> None:
            def get_phs(gm: torch.fx.GraphModule) -> List[torch.fx.Node]:
                return [node for node in gm.graph.nodes if node.op == "placeholder"]

            def migrate_meta_val(
                orig_phs: List[torch.fx.Node], new_phs: List[torch.fx.Node]
            ) -> None:
                if len(orig_phs) != len(new_phs):
                    raise ExportError(
                        ExportErrorType.NOT_SUPPORTED,
                        "ExportPassBase doesn't support changing the placeholders",
                    )
                for ph, new_ph in zip(orig_phs, new_phs):
                    if isinstance(new_ph.meta["val"], torch.Tensor):
                        if (
                            not isinstance(ph.meta["val"], torch.Tensor)
                            or new_ph.meta["val"].size() != ph.meta["val"].size()
                        ):
                            raise ExportError(
                                ExportErrorType.NOT_SUPPORTED,
                                "ExportPassBase doesn't support changing the placeholders",
                            )
                    new_ph.meta["val"] = ph.meta["val"]

            migrate_meta_val(get_phs(gm), get_phs(new_gm))

        # After one pass, new_graph_module's placeholders will always hold fake tensors in
        # meta['val'] but sometimes we want to preserve the original meta['val'] of placeholders
        #
        # For example, custom flows and certain passes assume no fake_tensor_mode is activated
        # and it doesn't quite work with fake_tensor_mode. but we don't bother to fix them.
        # So we'll just reset the meta of placeholders to its original value. It's safe because that
        # 1. For models captured with pt2_mode, the meta['val'] of placeholders are fake_tensors already, so
        # preserving it to the new graph module won't hurt.
        # 2. For models captured with dispatch_trace, the meta['val'] field
        # Note that it's only safe when passes don't modify the inputs.
        preserve_original_ph_meta_val(graph_module, res.graph_module)

        return res


@runtime_checkable
class ArgSchema(Protocol):
    name: str
    kwarg_only: bool
    type: Any  # pyre-ignore


def map_args(
    op: torch._ops.OpOverload,
    fn: Fn,
    args: Argument,
    kwargs: Dict[str, Argument],
) -> Tuple[Argument, Dict[str, Argument]]:
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    args = list(args)
    kwargs = kwargs.copy()

    def update(key: K, args: MutableMapping[K, PyTree], schema: ArgSchema) -> None:
        args[key] = fn(args[key], schema)

    for i, schema in enumerate(op._schema.arguments):
        assert isinstance(schema, ArgSchema)
        if schema.name in kwargs:
            update(schema.name, kwargs, schema)
        elif not schema.kwarg_only and i < len(args):
            update(i, args, schema)  # pyre-ignore

    return tuple(args), kwargs
