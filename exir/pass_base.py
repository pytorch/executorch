# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import operator
import traceback
from contextlib import contextmanager, nullcontext
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


# Namespaces of ops that are safe to cache in the FakeTensor dispatch cache.
# By default, FakeTensorMode only caches ops in {"aten", "prim", "prims"}.
# ExecuTorch passes commonly use quantization and TOSA ops that are
# deterministic and shape-preserving, so we extend caching to cover them
# during pass execution to avoid redundant FakeTensor dispatches.
_EXTRA_CACHEABLE_NAMESPACES: frozenset[str] = frozenset(
    {
        "quantized_decomposed",
        "tosa",
        "dim_order_ops",
        "cortex_m",
    }
)


@contextmanager
# pyre-ignore[3]
def _extend_faketensor_cache_builtins():  # noqa: C901
    """Temporarily extend FakeTensor dispatch cache to cover ExecuTorch ops.

    The FakeTensor dispatch cache (``FakeTensorMode``) only caches "builtin"
    ops whose namespace is in ``{"aten", "prim", "prims"}``.  ExecuTorch
    passes operate on graphs that contain ``quantized_decomposed``, ``tosa``,
    and other non-builtin ops that are nonetheless safe to cache -- they are
    deterministic and their output metadata depends only on input metadata.

    Without caching these ops, every pass re-dispatches them through the full
    PyTorch stack (~0.5 ms each), leading to tens of seconds of overhead
    across 50+ passes on a ~1200-node graph.

    This context manager monkey-patches ``torch._library.utils.is_builtin``
    so that the cache also covers the extra namespaces, then restores the
    original function on exit.
    """
    import torch._library.utils as _library_utils

    _original_is_builtin = _library_utils.is_builtin

    def _extended_is_builtin(op: torch._ops.OpOverload) -> bool:
        if not isinstance(op, torch._ops.OpOverload):
            raise AssertionError(f"op must be OpOverload, got {type(op)}")
        return op.namespace in {"aten", "prim", "prims"} or (
            op.namespace in _EXTRA_CACHEABLE_NAMESPACES
        )

    _library_utils.is_builtin = _extended_is_builtin  # pyre-ignore[8]

    # Evict negative cache entries ("non-builtin" bypass entries) that were
    # stored before the extension was active.  FakeTensorMode stores
    # _DispatchCacheBypassEntry objects as negative cache hits — once stored,
    # _validate_cache_key is never re-evaluated for that key.  We must evict
    # these so the first dispatch under the extension re-evaluates is_builtin
    # and creates a proper positive cache entry instead.
    #
    # There are TWO caches that can hold negative entries:
    #   1. FakeTensorMode.cache -- the global (class-level) cache, used when
    #      the dispatch has no SymInt inputs.
    #   2. shape_env.fake_tensor_cache -- per-ShapeEnv cache, used when the
    #      dispatch involves SymInt/SymFloat inputs (cache_on_shape_env=True).
    # We must evict from both.
    try:
        from torch._subclasses.fake_tensor import (
            _DispatchCacheBypassEntry,
            FakeTensorMode,
        )

        def _is_nonbuiltin_bypass(v: object) -> bool:
            return (
                isinstance(v, _DispatchCacheBypassEntry) and v.reason == "non-builtin"
            )

        # 1. Evict from the global class-level cache.
        FakeTensorMode.cache = {
            k: v
            for k, v in FakeTensorMode.cache.items()
            if not _is_nonbuiltin_bypass(v)
        }

        # 2. Evict from the per-ShapeEnv cache of the currently active
        #    FakeTensorMode (if any).  When ExportPass enters _fx(), the
        #    FakeTensorMode is already on the dispatch stack before this CM
        #    is entered, so we can reach its shape_env cache.
        try:
            from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

            for mode in _get_current_dispatch_mode_stack():
                if isinstance(mode, FakeTensorMode):
                    se = getattr(mode, "shape_env", None)
                    if se is not None:
                        se_cache = getattr(se, "fake_tensor_cache", None)
                        if se_cache:
                            se.fake_tensor_cache = {
                                k: v
                                for k, v in se_cache.items()
                                if not _is_nonbuiltin_bypass(v)
                            }
        except (ImportError, AttributeError):
            pass
    except (ImportError, AttributeError):
        pass  # Graceful degradation if internals change

    try:
        yield
    finally:
        _library_utils.is_builtin = _original_is_builtin  # pyre-ignore[8]


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
            elif isinstance(a, torch.SymInt):
                if a.node.constant is not None:
                    return a.node.constant
                else:
                    return a
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

    # Types whose nodes are eligible for the fast-copy optimisation in
    # ``run_node``.  Subclass interpreters (e.g. ``ExportPass``) extend
    # this tuple to include dialect-specific overload types such as
    # ``EdgeOpOverload``.
    _OPERATOR_TARGET_TYPES: Tuple[type, ...] = (
        torch._ops.OpOverload,
        torch._ops.OpOverloadPacket,
    )

    class ExportInterpreter(fx.Interpreter):
        def __init__(self, callback: "_ExportPassBase", gm: fx.GraphModule) -> None:
            super().__init__(gm)
            self.callback = callback
            self.node: torch.fx.Node = next(iter(gm.graph.nodes))

            # --- fast-copy bookkeeping ---------------------------------
            # When the owning pass declares ``targeted_ops``, cold nodes
            # (those whose target is *not* in the set) can be copied into
            # the new graph without an expensive FakeTensor dispatch.
            targeted: Optional[Set[Any]] = getattr(callback, "targeted_ops", None)
            self._targeted_ops: Optional[Set[Any]] = targeted if targeted else None

            # Fast-copy relies on the existing ``n.meta["val"]`` being
            # correct for cold nodes.  If the pass overrides ``call()``
            # it may modify the graph (e.g. insert nodes with metadata
            # copied from unrelated ops) before calling ``super().call()``,
            # which would make cold-node metadata unreliable.  Disable the
            # optimisation in that case.
            call_overridden = type(callback).call is not _ExportPassBase.call
            self._fast_copy_enabled: bool = (
                self._targeted_ops is not None and not call_overridden
            )

            # Maps old-graph nodes to their new-graph equivalents so that
            # ``_fast_copy_node`` can remap arguments (including get_attr
            # nodes that are stored in ``self.env`` as raw tensors rather
            # than ProxyValues).
            self._node_remap: Dict[torch.fx.Node, torch.fx.Node] = {}

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
            elif getattr(target, "__module__", None) in {
                "_operator",
                "builtins",
                "math",
            }:
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
            elif target == torch.ops.higher_order.while_loop:
                cond, body, carried_inputs, additional_inputs = args
                return self.callback.call_while(
                    cond, body, carried_inputs, additional_inputs, meta
                )
            elif target == torch.ops.higher_order.map_impl:
                f, mapped_args, operands = args  # type: ignore[assignment]
                return self.callback.call_map(f, mapped_args, operands, meta)
            elif target == torch.ops.higher_order.scan:
                combine_fn, init, xs, additional_inputs = args  # type: ignore[assignment]
                return self.callback.call_scan(
                    combine_fn, init, xs, additional_inputs, meta
                )
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

        # -- fast-copy helpers ------------------------------------------

        def _fast_copy_node(self, n: torch.fx.Node) -> "ProxyValue":
            """Copy *n* into the new graph without FakeTensor dispatch.

            This is the fast path for "cold" nodes — nodes whose target is
            not in the pass's ``targeted_ops``.  Instead of running the
            full ``_fx`` pipeline (unwrap → dispatch → create_proxy →
            set_metadata), we use ``graph.node_copy`` to clone the node
            directly and reuse the original ``val`` metadata.

            Typical savings: ~0.4 ms → ~0.02 ms per node.
            """

            tracer = self.callback.tracer

            def _arg_transform(old_node: torch.fx.Node) -> torch.fx.Node:
                # 1. Check the remap dict (populated for processed nodes
                #    whose result is a ProxyValue).
                new_node = self._node_remap.get(old_node)
                if new_node is not None:
                    return new_node
                # 2. Fallback: extract from ProxyValue in env.
                pv = self.env.get(old_node)
                if pv is not None and hasattr(pv, "proxy"):
                    mapped = pv.proxy.node
                    self._node_remap[old_node] = mapped
                    return mapped
                # 3. For get_attr / placeholder nodes that were processed
                #    via the normal path but returned raw tensors (not
                #    ProxyValue), they won't be in _node_remap.  Copy
                #    them into the new graph on demand.
                if old_node.op in ("get_attr", "placeholder"):
                    copied = tracer.graph.node_copy(
                        old_node, lambda x: self._node_remap.get(x, x)
                    )
                    self._node_remap[old_node] = copied
                    # For get_attr, also register the attribute on the
                    # new module so GraphModule.__init__ can find it.
                    if old_node.op == "get_attr":
                        val = self.fetch_attr(old_node.target)
                        target_atoms = old_node.target.split(".")
                        root = tracer.root
                        for atom in target_atoms[:-1]:
                            if not hasattr(root, atom):
                                setattr(root, atom, torch.nn.Module())
                            root = getattr(root, atom)
                        setattr(root, target_atoms[-1], val)
                    return copied
                return old_node

            new_node = tracer.graph.node_copy(n, _arg_transform)
            # node_copy already does copy.copy(node.meta)

            val = n.meta.get("val")
            proxy = torch.fx.Proxy(new_node, tracer)
            result = ProxyValue(val, proxy)
            self._node_remap[n] = new_node
            return result

        def run_node(self, n: torch.fx.Node) -> Argument:
            self.node = n
            self.callback.node_debug_str = n.format_node()

            # Fast-copy path: skip the full interpreter dispatch for cold
            # call_function nodes whose operator is not targeted by this
            # pass.  This avoids the expensive FakeTensor re-dispatch and
            # proxy reconstruction for nodes the pass will not modify.
            if (
                self._fast_copy_enabled
                and n.op == "call_function"
                and isinstance(n.target, self.callback._OPERATOR_TARGET_TYPES)
                and n.target not in self._targeted_ops  # type: ignore[operator]
                and n.meta.get("val") is not None
            ):
                return self._fast_copy_node(n)

            result = super().run_node(n)

            # Record old→new node mapping for fast-copy arg remapping.
            if self._fast_copy_enabled and isinstance(result, ProxyValue):
                self._node_remap[n] = result.proxy.node

            # After a hot node runs through full dispatch, verify that
            # it did not change output shapes.  If it did, downstream
            # cold nodes' original ``val`` metadata would be stale, so
            # we disable the fast-copy optimisation for the remainder
            # of this interpreter walk.
            if (
                self._fast_copy_enabled
                and n.op == "call_function"
                and self._targeted_ops is not None
                and n.target in self._targeted_ops
                and isinstance(result, ProxyValue)
            ):
                original_val = n.meta.get("val")
                new_val = result.data
                if isinstance(original_val, torch.Tensor) and isinstance(
                    new_val, torch.Tensor
                ):
                    if (
                        original_val.shape != new_val.shape
                        or original_val.dtype != new_val.dtype
                    ):
                        self._fast_copy_enabled = False

            return result

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
        arg_proxy.node.meta["val"] = arg
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

    def call_while(
        self,
        cond_fn: torch.fx.GraphModule,
        body_fn: torch.fx.GraphModule,
        carried_inputs: List[Argument],
        additional_inputs: List[Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        cond_fn = self.call_submodule(cond_fn, (*carried_inputs, *additional_inputs))
        body_fn = self.call_submodule(body_fn, (*carried_inputs, *additional_inputs))
        assert cond_fn is not None
        assert body_fn is not None
        return self._fx(
            "call_function",
            torch.ops.higher_order.while_loop,
            (
                cond_fn.graph_module,
                body_fn.graph_module,
                carried_inputs,
                additional_inputs,
            ),
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

    def call_scan(
        self,
        combine_fn: torch.fx.GraphModule,
        init: List[ProxyValue],
        xs: List[Argument],
        additional_inputs: List[ProxyValue],
        meta: NodeMetadata,
    ) -> ProxyValue:
        # Get the expected x element shapes from the combine_fn's placeholders
        # The combine_fn expects: (carry..., x_element..., additional_inputs...)
        combine_fn_placeholders = [
            n for n in combine_fn.graph.nodes if n.op == "placeholder"
        ]
        num_init = len(init)
        # The x_element placeholders are at indices [num_init : num_init + num_xs]
        xs_element_data = []
        for i in range(0, len(xs)):
            ph = combine_fn_placeholders[num_init + i]
            # Use the placeholder's val which has the correct shape
            xs_element_data.append(ph.meta["val"])

        combine_fn_result = self.call_submodule(
            combine_fn, (*init, *xs_element_data, *additional_inputs)
        )
        assert combine_fn_result is not None

        return self._fx(
            "call_function",
            torch.ops.higher_order.scan,
            (combine_fn_result.graph_module, init, xs, additional_inputs),
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
        prev_tracer, self.tracer = (
            self.tracer,
            self.ExportTracer(self, graph_module.graph._codegen),
        )
        self.tracer.fake_tensor_mode = prev_tracer.fake_tensor_mode
        interpreter = self.ExportInterpreter(self, graph_module)
        prev_interpreter, self.interpreter = (
            self.interpreter,
            torch.fx.Interpreter(
                torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
            ),
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

    def should_run(self, graph_module: fx.GraphModule) -> bool:
        """Override to declare when this pass can be skipped entirely.

        When this method returns False, the expensive FakeTensor graph
        re-interpretation is bypassed and the original graph module is returned
        unchanged.  Subclasses should override this to inspect the graph cheaply
        (e.g. checking whether any node targets an op this pass cares about).

        The default implementation returns True so existing passes are
        unaffected.
        """
        return True

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        if not getattr(self, "_initialized", False):
            raise ExportPassBaseError(
                "ExportPass is not initialized with __init__().",
            )

        if not getattr(self, "_skip_should_run", False) and not self.should_run(
            graph_module
        ):
            return PassResult(graph_module, False)

        prev_skip = getattr(self, "_skip_should_run", False)
        self._skip_should_run = True

        inputs = self.inputs(graph_module)

        fake_tensor_mode = None
        for i in inputs:
            if isinstance(i, FakeTensor):
                assert (
                    fake_tensor_mode is None or fake_tensor_mode is i.fake_mode
                ), "Multiple fake tensor mode detected."
                fake_tensor_mode = i.fake_mode
        if fake_tensor_mode is None:
            fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True)
            dispatcher_mode = nullcontext()  # type: ignore[assignment]
        else:
            fake_tensor_mode.allow_non_fake_inputs = True
            dispatcher_mode = enable_python_dispatcher()  # type: ignore[assignment]
        self.tracer.fake_tensor_mode = fake_tensor_mode
        self.fake_tensor_mode = fake_tensor_mode

        with fake_tensor_mode, dispatcher_mode:  # type: ignore[assignment, union-attr]
            with _extend_faketensor_cache_builtins():
                result = self.call_submodule(graph_module, tuple(inputs))

        self._skip_should_run = prev_skip
        return result


class ExportPass(_ExportPassBase):
    # Extend operator target types to include the Edge dialect overloads so
    # that the fast-copy optimisation in ``run_node`` also covers Edge ops.
    _OPERATOR_TARGET_TYPES: Tuple[type, ...] = (
        torch._ops.OpOverload,
        torch._ops.OpOverloadPacket,
        EdgeOpOverload,
    )

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
        if schema.name in kwargs:
            update(schema.name, kwargs, schema)
        elif not schema.kwarg_only and i < len(args):
            update(i, args, schema)  # pyre-ignore

    return tuple(args), kwargs
