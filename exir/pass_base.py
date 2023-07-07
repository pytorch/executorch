# pyre-strict

import logging
import operator
import typing
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
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from executorch.exir import delegate, error, memory
from executorch.exir.delegate import LoweredBackendModule

from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.error import ExportError, ExportErrorType
from executorch.exir.graph_module import get_exir_meta, make_export_graph_module
from executorch.exir.pass_infra.node_metadata import NodeMetadata
from executorch.exir.pass_infra.proxy_value import ProxyValue
from functorch.experimental import control_flow
from functorch.experimental._map import _unstack_pytree
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses import FakeTensor, UnsupportedFakeTensorException
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import PythonKeyTracer
from torch.fx.graph import CodeGen
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import PyTree

Argument = Any  # pyre-ignore
Value = Any  # pyre-ignore
Fn = Callable[..., Any]  # pyre-ignore
K = TypeVar("K")


def make_inline_interpreter(
    parent: Type[torch.fx.Interpreter],
) -> Type[torch.fx.Interpreter]:
    class InlineInterpreter(parent):
        def call_function(self, target, args, kwargs):
            if target == torch.ops.higher_order.cond:
                pred, true, false, params = args
                return InlineInterpreter(true).run(*params)
            elif target == torch.ops.map:
                f, xs, *params = args
                sample_out = InlineInterpreter(f).run(xs[0], *params)
                return sample_out.new_empty([xs.shape[0], *sample_out.shape])
            else:
                return super().call_function(target, args, kwargs)

    return typing.cast(Type[torch.fx.Interpreter], InlineInterpreter)


class ExportTracer(PythonKeyTracer):
    def __init__(self, callback: "ExportPassBase", codegen: CodeGen) -> None:
        super().__init__()
        self.callback = callback
        self.root = torch.nn.Module()
        self.graph = torch.fx.Graph()
        self.graph.set_codegen(codegen)
        self.tensor_attrs: Dict[str, torch.Tensor] = {}
        self.fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True)
        self.submodules: Dict[torch.nn.Module, str] = {}

    def trace(self) -> None:
        raise error.InternalError("ExportTracer doesn't support trace().")

    def create_arg(self, a: Argument) -> torch.fx.Node:
        if isinstance(a, torch.nn.Module):
            if a not in self.submodules:
                prefix = (
                    "lowered_module"
                    if isinstance(a, LoweredBackendModule)
                    else "submodule"
                )
                name_submodule = f"{prefix}_{len(self.submodules)}"
                self.root.add_module(name_submodule, a)
                self.submodules[a] = name_submodule
        elif isinstance(a, FakeTensor):
            if not hasattr(a, "constant") or a.constant is None:
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED, f"Cannot add {a} to graph."
                )
            a = a.constant
        node = super().create_arg(a)
        if (
            isinstance(a, torch.Tensor)
            and isinstance(node, torch.fx.Node)
            and node.op == "get_attr"
        ):
            self.set_metadata(node, a, ())
            self.callback.on_attr(ProxyValue(a, node))
        return node

    def set_metadata(
        self, node: torch.fx.Node, value: Argument, args: pytree.PyTree
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
            None,
        ]:
            if isinstance(x, FakeTensor):
                return x
            elif isinstance(x, torch.Tensor):
                if x.is_quantized:
                    # TODO (tmanlaibaatar) properly support Quantized FakeTensor
                    x = torch.dequantize(x)

                try:
                    fake_tensor = self.fake_tensor_mode.from_tensor(x)
                except UnsupportedFakeTensorException:
                    # TODO: This is just a workaround to get over the
                    # x.as_subclass error
                    logging.error(
                        "Fakeifying a Tensor subclass is not supported \
                        right now. Instead a TensorMetadata is used."
                    )
                    fake_tensor = None
                return fake_tensor
            elif isinstance(x, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                return x
            elif isinstance(x, (int, float, bool)):
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


class ExportPassBase(PassBase):
    class ExportInterpreter(fx.Interpreter):
        def __init__(self, callback: "ExportPassBase", gm: fx.GraphModule) -> None:
            super().__init__(gm)
            self.callback = callback
            self.node: torch.fx.Node = next(iter(gm.graph.nodes))

        def placeholder(
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
            elif getattr(target, "__module__", None) == "_operator":
                assert callable(target)
                return self.callback.call_sym(target, args, meta)
            elif isinstance(target, (torch._ops.OpOverload, EdgeOpOverload)):
                return self.callback.call_operator(
                    target,
                    args,
                    kwargs,
                    meta,
                )
            elif target == control_flow.cond:
                pred, true_fn, false_fn, inputs = args
                return self.callback.call_cond(pred, true_fn, false_fn, inputs, meta)
            elif target == torch.ops.map_impl:
                f, num_mapped, *args = args  # pyre-ignore
                return self.callback.call_map(f, num_mapped, args, meta)

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
                    self.callback.interpreter,
                )

            elif target == delegate.executorch_call_delegate:
                lowered_module = args[0]
                args = args[1:]
                return self.callback.call_delegate(
                    lowered_module,
                    args,
                    kwargs,
                    NodeMetadata(self.node.meta),
                )
            else:
                raise error.InternalError(
                    f"Unsupported target type: {target}, {type(target)}"
                )

        def get_attr(
            self, target: str, args: Tuple[Argument, ...], kwargs: Dict[str, Argument]
        ) -> Argument:
            return super().get_attr(target, args, kwargs)

        def call_module(
            self,
            target: torch.fx.node.Target,
            args: Tuple[Argument, ...],
            kwargs: Dict[str, Argument],
        ) -> None:
            raise error.InternalError("call_module is not supported.")

        def call_method(
            self, target: str, args: Tuple[Argument, ...], kwargs: Dict[str, Argument]
        ) -> ProxyValue:
            raise error.InternalError("call_method is not supported.")

        def run_node(self, n: torch.fx.Node) -> Argument:
            self.node = n
            self.callback.node_debug_str = n.format_node()
            return super().run_node(n)

    def __init__(self) -> None:
        self.interpreter = torch.fx.Interpreter(
            torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        )
        self.tracer = ExportTracer(self, CodeGen())
        self.fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True)
        self._initialized = True
        self.node_debug_str: typing.Optional[str] = None

    def _create_result_node(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
        res_data: Value,
    ) -> ProxyValue:
        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        args_proxy, kwargs_proxy = pytree.tree_map_only(
            ProxyValue, lambda x: x.proxy, (args, kwargs)
        )
        res_proxy = self.tracer.create_proxy(kind, target, args_proxy, kwargs_proxy)
        res_proxy.node.meta.update(meta.data)
        self.tracer.set_metadata(res_proxy.node, res_data, (args_data, kwargs_data))
        return ProxyValue(res_data, res_proxy)

    def _fx(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
        interpreter: torch.fx.Interpreter,
    ) -> ProxyValue:
        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        res_data = getattr(interpreter, kind)(target, args_data, kwargs_data)
        return self._create_result_node(kind, target, args, kwargs, meta, res_data)

    def inputs(self, graph_module: torch.fx.GraphModule) -> List[Argument]:
        def extract_input(node: torch.fx.Node) -> Optional[FakeTensor]:
            if "val" in node.meta:
                return node.meta["val"]
            elif len(node.users) == 0:
                return None
            raise ExportError(
                ExportErrorType.VIOLATION_OF_SPEC,
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
        self.tracer.set_metadata(arg_proxy.node, arg, ())
        return ProxyValue(arg, arg_proxy)

    def call_operator(
        self,
        op: Fn,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        return self._fx("call_function", op, args, kwargs, meta, self.interpreter)

    def call_sym(
        self,
        target: Fn,
        args: Tuple[Argument, ...],
        meta: NodeMetadata,
    ) -> ProxyValue:
        return self._fx("call_function", target, args, {}, meta, self.interpreter)

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
            control_flow.cond,
            (pred, true_branch.graph_module, false_branch.graph_module, inputs),
            {},
            meta,
            self.interpreter,
        )

    def call_map(
        self,
        f: torch.fx.GraphModule,
        num_mapped: int,
        args: Tuple[ProxyValue, ...],
        meta: NodeMetadata,
    ) -> ProxyValue:
        xs = _unstack_pytree([arg.data for arg in args[:num_mapped]])[0]
        pos_args = args[num_mapped:]
        f_branch = self.call_submodule(f, tuple(xs + [arg.data for arg in pos_args]))
        assert f_branch is not None
        return self._fx(
            "call_function",
            torch.ops.map_impl,
            (f_branch.graph_module, num_mapped, *args),
            {},
            meta,
            self.interpreter,
        )

    def call_getitem(
        self, value: ProxyValue, key: int, meta: NodeMetadata
    ) -> ProxyValue:
        return self._fx(
            "call_function", operator.getitem, (value, key), {}, meta, self.interpreter
        )

    def output(self, results: List[Argument], meta: NodeMetadata) -> ProxyValue:
        return self._fx("output", "output", (results,), {}, meta, self.interpreter)

    def call_delegate(
        self,
        lowered_module: LoweredBackendModule,
        args: Tuple[ProxyValue, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        # pyre-ignore
        args = (lowered_module,) + args
        return self._fx(
            "call_function",
            delegate.executorch_call_delegate,
            args,
            kwargs,
            meta,
            self.interpreter,
        )

    def call_submodule(
        self, graph_module: fx.GraphModule, inputs: Tuple[Argument, ...]
    ) -> PassResult:
        prev_tracer, self.tracer = self.tracer, ExportTracer(
            self, graph_module.graph._codegen
        )
        self.tracer.fake_tensor_mode = prev_tracer.fake_tensor_mode
        interpreter = ExportPass.ExportInterpreter(self, graph_module)
        prev_interpreter, self.interpreter = self.interpreter, typing.cast(
            torch.fx.Interpreter, super(type(interpreter), interpreter)
        )
        inputs_data = pytree.tree_map_only(ProxyValue, lambda x: x.data, inputs)
        with fx_traceback.preserve_node_meta():
            interpreter.run(*inputs_data)

        new_graph_module = make_export_graph_module(self.tracer.root, self.tracer.graph)

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
        # For example, Jarvis's quantization flow and certain passes assume no fake_tensor_mode is activated
        # and it doesn't quite work with fake_tensor_mode. but we don't bother to fix them.
        # So we'll just reset the meta of placeholders to its original value. It's safe because that
        # 1. For models captured with pt2_mode, the meta['val'] of placeholders are fake_tensors already, so
        # preserving it to the new graph module won't hurt.
        # 2. For models captured with dispatch_trace, the meta['val'] field
        # Note that it's only safe when passes don't modify the inputs.
        preserve_original_ph_meta_val(graph_module, new_graph_module)

        self.tracer = prev_tracer
        self.interpreter = prev_interpreter
        return PassResult(
            new_graph_module,
            True,
        )

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        if not getattr(self, "_initialized", False):
            raise ExportError(
                ExportErrorType.UNINITIALIZED,
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
            fake_tensor_mode = nullcontext()
            dispatcher_mode = nullcontext()
        else:
            self.tracer.fake_tensor_mode = fake_tensor_mode
            dispatcher_mode = enable_python_dispatcher()
        with fake_tensor_mode, dispatcher_mode:
            # pyre-ignore
            result = self.call_submodule(graph_module, inputs)

        new_graph_module = result.graph_module
        new_meta = get_exir_meta(new_graph_module)
        old_meta = get_exir_meta(graph_module)
        new_meta.in_spec = old_meta.in_spec
        new_meta.out_spec = old_meta.out_spec
        return result


class ExportPass(ExportPassBase):
    ...


class BackendPass(ExportPassBase):
    ...


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
