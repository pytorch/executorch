# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import operator
import traceback
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
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
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.error import ExportError, ExportErrorType
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses import FakeTensorMode, UnsupportedFakeTensorException
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch.export import ExportedProgram
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


@dataclass(frozen=True, slots=True)
class _SymbolicTensorSnapshot:
    shape: Tuple[Optional[str], ...]


@dataclass(frozen=True, slots=True)
class _TensorMetadataSnapshot:
    shape: Tuple[Any, ...]
    dtype: torch.dtype
    layout: torch.layout
    device: torch.device
    requires_grad: bool
    stride: Optional[Tuple[Any, ...]]


def _symbolic_scalar_snapshot(
    value: Argument,
) -> Optional[Tuple[str, str]]:
    if isinstance(value, torch.SymInt):
        return ("SymInt", str(value))
    if isinstance(value, torch.SymFloat):
        return ("SymFloat", str(value))
    if isinstance(value, torch.SymBool):
        return ("SymBool", str(value))
    return None


def _leaf_symbolic_snapshot(value: Argument) -> Any:
    scalar_snapshot = _symbolic_scalar_snapshot(value)
    if scalar_snapshot is not None:
        return scalar_snapshot

    if isinstance(value, FakeTensor):
        if value.constant is not None:
            return None
        dims = []
        has_symbolic_dim = False
        for dim in value.shape:
            dim_snapshot = _symbolic_scalar_snapshot(dim)
            if dim_snapshot is None:
                dims.append(None)
            else:
                has_symbolic_dim = True
                dims.append(dim_snapshot[1])
        if has_symbolic_dim:
            return _SymbolicTensorSnapshot(tuple(dims))

    return None


def _extract_symbolic_snapshot(value: Argument) -> Any:
    snapshot = pytree.tree_map(_leaf_symbolic_snapshot, value)
    leaves = pytree.tree_leaves(snapshot)
    if any(leaf is not None for leaf in leaves):
        return snapshot
    return None


def _target_matches_by_identity(target: Any, targets: Tuple[Any, ...]) -> bool:
    return any(target is candidate for candidate in targets)


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
        if isinstance(self.data, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            raise ExportPassBaseError(
                "ProxyValue with symbolic data cannot be used in boolean context."
            )
        return bool(self.data)

    def __int__(self):
        if isinstance(self.data, torch.SymInt):
            raise ExportPassBaseError(
                "ProxyValue with SymInt data cannot be converted to int."
            )
        return int(self.data)

    def __float__(self):
        if isinstance(self.data, torch.SymFloat):
            raise ExportPassBaseError(
                "ProxyValue with SymFloat data cannot be converted to float."
            )
        return float(self.data)

    def __index__(self):
        if isinstance(self.data, torch.SymInt):
            raise ExportPassBaseError(
                "ProxyValue with SymInt data cannot be used in index context."
            )
        return self.__int__()


class ExportPassBaseError(RuntimeError):
    pass


class _FastCopyFallback(Exception):
    pass


def _metadata_dimension_snapshot(value: Any) -> Any:
    symbolic = _symbolic_scalar_snapshot(value)
    return symbolic if symbolic is not None else value


def _tensor_metadata_snapshot(value: torch.Tensor) -> _TensorMetadataSnapshot:
    stride = None
    if value.layout == torch.strided:
        stride = tuple(_metadata_dimension_snapshot(dim) for dim in value.stride())
    return _TensorMetadataSnapshot(
        shape=tuple(_metadata_dimension_snapshot(dim) for dim in value.shape),
        dtype=value.dtype,
        layout=value.layout,
        device=value.device,
        requires_grad=value.requires_grad,
        stride=stride,
    )


def _metadata_leaf_snapshot(value: Any) -> Tuple[type, Any]:
    symbolic = _symbolic_scalar_snapshot(value)
    if symbolic is not None:
        return (type(value), symbolic)
    if isinstance(
        value,
        (
            type(None),
            bool,
            int,
            float,
            complex,
            str,
            bytes,
            torch.dtype,
            torch.layout,
            torch.device,
            torch.memory_format,
        ),
    ):
        return (type(value), value)
    # Avoid arbitrary equality/repr work. Distinct unknown objects may be
    # equivalent, but treating them as drift is the conservative choice.
    return (type(value), id(value))


def _tensor_metadata_changed(original: Argument, new: Argument) -> bool:
    if isinstance(original, ProxyValue):
        original = original.data
    if isinstance(new, ProxyValue):
        new = new.data
    original_leaves, original_spec = pytree.tree_flatten(original)
    new_leaves, new_spec = pytree.tree_flatten(new)
    if original_spec != new_spec:
        return True

    for original_leaf, new_leaf in zip(original_leaves, new_leaves):
        if isinstance(original_leaf, ProxyValue):
            original_leaf = original_leaf.data
        if isinstance(new_leaf, ProxyValue):
            new_leaf = new_leaf.data

        original_is_tensor = isinstance(original_leaf, torch.Tensor)
        new_is_tensor = isinstance(new_leaf, torch.Tensor)
        if original_is_tensor != new_is_tensor:
            return True
        if original_is_tensor:
            if _tensor_metadata_snapshot(original_leaf) != _tensor_metadata_snapshot(
                new_leaf
            ):
                return True
        elif _metadata_leaf_snapshot(original_leaf) != _metadata_leaf_snapshot(new_leaf):
            return True

    return False


@dataclass(frozen=True)
class ExportedProgramPassResult:
    exported_program: ExportedProgram
    modified: bool


class ExportedProgramPassBase(ABC):
    """
    Base interface for implementing passes that operate on ExportedProgram.
    """

    def __call__(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        """
        Runs the precondition check, the pass itself, and the postcondition check.
        """

        self.requires(exported_program)
        res = self.call(exported_program)
        self.ensures(exported_program)
        return res

    @abstractmethod
    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        """
        The pass that is run through the given exported program. To implement a
        pass, it is required to implement this function.

        Args:
            exported_program: The exported program we will run a pass on
        """

    def requires(self, exported_program: ExportedProgram) -> None:  # noqa: B027
        """
        This function will be called before the pass is run and will check that
        the given exported program contains the preconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            exported_program: The exported program we will run checks on
        """

    def ensures(self, exported_program: ExportedProgram) -> None:  # noqa: B027
        """
        This function will be called after the pass is run and will check that
        the given exported program contains the postconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            exported_program: The exported program we will run checks on
        """


# Replaying convolution and linear operators can refresh layout-sensitive
# metadata that downstream passes rely on. Keep every ATen convolution spelling
# that can survive export, plus the Edge aliases exposed here, on the replay path.
_FAST_COPY_UNSAFE_TARGETS: Tuple[Any, ...] = (
    torch.ops.aten.convolution,
    torch.ops.aten.convolution.default,
    torch.ops.aten.conv1d,
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv1d.padding,
    torch.ops.aten.conv2d,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.conv2d.padding,
    torch.ops.aten.conv3d,
    torch.ops.aten.conv3d.default,
    torch.ops.aten.conv3d.padding,
    torch.ops.aten.conv_transpose1d,
    torch.ops.aten.conv_transpose1d.default,
    torch.ops.aten.conv_transpose2d,
    torch.ops.aten.conv_transpose2d.input,
    torch.ops.aten.conv_transpose3d,
    torch.ops.aten.conv_transpose3d.input,
    torch.ops.aten.linear,
    torch.ops.aten.linear.default,
    exir_ops.edge.aten.convolution.default,
    exir_ops.edge.aten.conv2d.default,
    exir_ops.edge.aten.conv2d.padding,
    exir_ops.edge.aten.conv3d.default,
    exir_ops.edge.aten.conv3d.padding,
    exir_ops.edge.aten.linear.default,
)
_FAST_COPY_UNSAFE_TARGET_IDS = frozenset(
    id(target) for target in _FAST_COPY_UNSAFE_TARGETS
)


def _is_fast_copy_unsafe_target(target: Any) -> bool:
    return id(target) in _FAST_COPY_UNSAFE_TARGET_IDS


class _ExportPassBase(PassBase):
    """
    Interpreter-based pass class to help users maintain the IR spec while writing
    transformations.
    """

    enable_fast_copy = False

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
            self.tensor_attrs: Dict[torch.Tensor, str] = {}
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
            # (those whose target is not one of the exact targets) can be copied into
            # the new graph without an expensive FakeTensor dispatch.
            self._targeted_ops = callback.get_fast_copy_target_ops()

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

        @staticmethod
        def _proxy_value_node(
            value: Any,
            tracer: "_ExportPassBase.ExportTracer",
        ) -> Optional[torch.fx.Node]:
            if not isinstance(value, ProxyValue):
                return None
            proxy_or_node = value.proxy_or_node
            node = (
                proxy_or_node.node
                if isinstance(proxy_or_node, torch.fx.Proxy)
                else proxy_or_node
            )
            if not isinstance(node, torch.fx.Node) or node.graph is not tracer.graph:
                raise _FastCopyFallback
            return node

        @staticmethod
        def _source_attr_registration(
            parent: torch.nn.Module,
            name: str,
        ) -> Tuple[str, bool]:
            if name in parent._parameters:
                return ("parameter", True)
            if name in parent._buffers:
                return ("buffer", name not in parent._non_persistent_buffers_set)
            if name in parent._modules:
                return ("module", True)
            return ("attribute", True)

        def _fetch_attr_for_fast_copy(
            self, target: str
        ) -> Tuple[Any, Tuple[str, ...], str, bool]:
            target_atoms = tuple(target.split("."))
            if not target_atoms or any(not atom for atom in target_atoms):
                raise _FastCopyFallback

            parent = self.module
            for atom in target_atoms[:-1]:
                try:
                    parent = getattr(parent, atom)
                except AttributeError as exc:
                    raise _FastCopyFallback from exc
                if not isinstance(parent, torch.nn.Module):
                    raise _FastCopyFallback

            try:
                value = getattr(parent, target_atoms[-1])
            except AttributeError as exc:
                raise _FastCopyFallback from exc
            registration, persistent = self._source_attr_registration(
                parent, target_atoms[-1]
            )
            return value, target_atoms, registration, persistent

        @staticmethod
        def _destination_attr_matches(
            parent: torch.nn.Module,
            name: str,
            value: Any,
            registration: str,
            persistent: bool,
        ) -> bool:
            if not hasattr(parent, name) or getattr(parent, name) is not value:
                return False
            if registration == "parameter":
                return name in parent._parameters
            if registration == "buffer":
                return (
                    name in parent._buffers
                    and (name not in parent._non_persistent_buffers_set) == persistent
                )
            if registration == "module":
                return name in parent._modules
            return (
                name not in parent._parameters
                and name not in parent._buffers
                and name not in parent._modules
            )

        def _preflight_get_attr_destinations(
            self,
            tracer: "_ExportPassBase.ExportTracer",
            get_attr_values: Dict[
                torch.fx.Node, Tuple[Any, Tuple[str, ...], str, bool]
            ],
        ) -> None:
            planned: Dict[Tuple[str, ...], Tuple[Any, str, bool]] = {}
            for value, target_atoms, registration, persistent in get_attr_values.values():
                previous = planned.get(target_atoms)
                if previous is not None and (
                    previous[0] is not value
                    or previous[1] != registration
                    or previous[2] != persistent
                ):
                    raise _FastCopyFallback
                planned[target_atoms] = (value, registration, persistent)

            for path in planned:
                for index in range(1, len(path)):
                    if path[:index] in planned:
                        raise _FastCopyFallback

            for path, (value, registration, persistent) in planned.items():
                parent = tracer.root
                for atom in path[:-1]:
                    if not hasattr(parent, atom):
                        break
                    child = getattr(parent, atom)
                    if not isinstance(child, torch.nn.Module):
                        raise _FastCopyFallback
                    parent = child
                else:
                    leaf_name = path[-1]
                    if hasattr(parent, leaf_name) and not self._destination_attr_matches(
                        parent,
                        leaf_name,
                        value,
                        registration,
                        persistent,
                    ):
                        raise _FastCopyFallback

        def _preflight_fast_copy_inputs(
            self,
            n: torch.fx.Node,
            tracer: "_ExportPassBase.ExportTracer",
        ) -> Tuple[
            Dict[torch.fx.Node, torch.fx.Node],
            Dict[torch.fx.Node, Tuple[Any, Tuple[str, ...], str, bool]],
        ]:
            resolved_nodes: Dict[torch.fx.Node, torch.fx.Node] = {}
            get_attr_values: Dict[
                torch.fx.Node, Tuple[Any, Tuple[str, ...], str, bool]
            ] = {}
            for old_node in n.all_input_nodes:
                new_node = self._node_remap.get(old_node)
                if new_node is not None:
                    if new_node.graph is not tracer.graph:
                        raise _FastCopyFallback
                    resolved_nodes[old_node] = new_node
                    continue

                env_node = self._proxy_value_node(self.env.get(old_node), tracer)
                if env_node is not None:
                    resolved_nodes[old_node] = env_node
                    continue
                if old_node.op != "get_attr" or not isinstance(old_node.target, str):
                    raise _FastCopyFallback

                get_attr_values[old_node] = self._fetch_attr_for_fast_copy(
                    old_node.target
                )

            self._preflight_get_attr_destinations(tracer, get_attr_values)
            return resolved_nodes, get_attr_values

        @staticmethod
        def _get_attr_parent(
            root: torch.nn.Module,
            target_atoms: Tuple[str, ...],
        ) -> torch.nn.Module:
            parent = root
            for atom in target_atoms[:-1]:
                if not hasattr(parent, atom):
                    parent.add_module(atom, torch.nn.Module())
                child = getattr(parent, atom)
                assert isinstance(child, torch.nn.Module)
                parent = child
            return parent

        @staticmethod
        def _install_attr(
            parent: torch.nn.Module,
            name: str,
            value: Any,
            registration: str,
            persistent: bool,
        ) -> None:
            if hasattr(parent, name):
                return
            if registration == "parameter":
                parent.register_parameter(name, value)
            elif registration == "buffer":
                parent.register_buffer(name, value, persistent=persistent)
            elif registration == "module":
                parent.add_module(name, value)
            else:
                setattr(parent, name, value)

        def _commit_fast_copy_get_attrs(
            self,
            tracer: "_ExportPassBase.ExportTracer",
            resolved_nodes: Dict[torch.fx.Node, torch.fx.Node],
            get_attr_values: Dict[
                torch.fx.Node, Tuple[Any, Tuple[str, ...], str, bool]
            ],
        ) -> None:
            for old_node, (
                value,
                target_atoms,
                registration,
                persistent,
            ) in get_attr_values.items():
                parent = self._get_attr_parent(tracer.root, target_atoms)
                self._install_attr(
                    parent,
                    target_atoms[-1],
                    value,
                    registration,
                    persistent,
                )
                copied = tracer.graph.node_copy(old_node, lambda node: resolved_nodes[node])
                proxy_value = ProxyValue(value, torch.fx.Proxy(copied, tracer))
                if isinstance(value, torch.Tensor):
                    tracer.tensor_attrs[value] = str(old_node.target)
                    tracer.set_metadata(copied, value)
                    tracer.callback.on_attr(proxy_value)
                resolved_nodes[old_node] = copied
                self._node_remap[old_node] = copied

        def _fast_copy_node(self, n: torch.fx.Node) -> "ProxyValue":
            tracer = self.callback.tracer
            resolved_nodes, get_attr_values = self._preflight_fast_copy_inputs(
                n, tracer
            )
            self._commit_fast_copy_get_attrs(
                tracer, resolved_nodes, get_attr_values
            )

            new_node = tracer.graph.node_copy(n, lambda old_node: resolved_nodes[old_node])
            val = n.meta.get("val")
            result = ProxyValue(val, torch.fx.Proxy(new_node, tracer))
            self._node_remap[n] = new_node
            return result

        def _record_slow_path_result(
            self, n: torch.fx.Node, result: Argument
        ) -> None:
            for leaf in pytree.tree_leaves(result):
                try:
                    self._proxy_value_node(leaf, self.callback.tracer)
                except _FastCopyFallback:
                    self._fast_copy_enabled = False
                    break
            else:
                if isinstance(result, ProxyValue):
                    mapped = self._proxy_value_node(result, self.callback.tracer)
                    if mapped is not None:
                        self._node_remap[n] = mapped

            if "val" in n.meta and _tensor_metadata_changed(n.meta["val"], result):
                self._fast_copy_enabled = False

        def run_node(self, n: torch.fx.Node) -> Argument:
            self.node = n
            fast_copied = False

            # Fast-copy path: skip the full interpreter dispatch for cold
            # call_function nodes whose operator is not targeted by this
            # pass.  This avoids the expensive FakeTensor re-dispatch and
            # proxy reconstruction for nodes the pass will not modify.
            if (
                self._fast_copy_enabled
                and n.op == "call_function"
                and isinstance(n.target, self.callback._OPERATOR_TARGET_TYPES)
                and self._targeted_ops is not None
                and not _target_matches_by_identity(n.target, self._targeted_ops)
                and self.callback.should_fast_copy_node(n.target)
                and n.meta.get("val") is not None
                and "tensor_meta" in n.meta
            ):
                try:
                    result = self._fast_copy_node(n)
                    fast_copied = True
                except _FastCopyFallback:
                    result = super().run_node(n)
            else:
                result = super().run_node(n)

            if not fast_copied and self._fast_copy_enabled:
                self._record_slow_path_result(n, result)

            return result

    def __init__(self) -> None:
        self.interpreter = torch.fx.Interpreter(
            torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        )
        self.tracer = self.ExportTracer(self, CodeGen())  # pyre-ignore
        self.fake_tensor_mode: Optional[FakeTensorMode] = None
        self._initialized = True
        self.node_debug_str: Optional[str] = None

    def should_preserve_symbolic_input_metadata(self) -> bool:
        """Returns whether replay should validate symbolic input preservation.

        Override to ``False`` for passes that intentionally change symbolic
        input metadata during replay.
        """
        return True

    def get_fast_copy_target_ops(self) -> Optional[Tuple[Any, ...]]:
        """Return exact targets only when this pass explicitly enables fast-copy."""
        if not self.enable_fast_copy:
            return None
        targeted_ops = getattr(self, "targeted_ops", None)
        if targeted_ops is None:
            return None
        try:
            return tuple(targeted_ops)
        except TypeError:
            return None

    def should_fast_copy_node(self, target: torch.fx.node.Target) -> bool:
        """Return whether a cold call_function node can bypass replay.

        Passes with subclass-wide ``call_operator`` behavior can override this
        to keep selected non-targeted operators on the normal replay path.
        """
        return not _is_fast_copy_unsafe_target(target)

    def _capture_symbolic_input_snapshots(
        self, graph_module: fx.GraphModule
    ) -> List[Any]:
        return [
            _extract_symbolic_snapshot(node.meta.get("val"))
            for node in graph_module.graph.nodes
            if node.op == "placeholder"
        ]

    def _validate_symbolic_input_snapshots(
        self,
        graph_module: fx.GraphModule,
        new_graph_module: fx.GraphModule,
    ) -> None:
        if not self.should_preserve_symbolic_input_metadata():
            return

        symbolic_inputs = self._capture_symbolic_input_snapshots(graph_module)
        if all(snapshot is None for snapshot in symbolic_inputs):
            return

        new_symbolic_inputs = self._capture_symbolic_input_snapshots(new_graph_module)
        for input_index, snapshot in enumerate(symbolic_inputs):
            if snapshot is None:
                continue
            if input_index >= len(new_symbolic_inputs):
                raise ExportPassBaseError(
                    f"Input at position {input_index} did not preserve symbolic metadata across pass replay."
                )

            current_snapshot = new_symbolic_inputs[input_index]
            if current_snapshot != snapshot:
                raise ExportPassBaseError(
                    f"Input at position {input_index} did not preserve symbolic metadata across pass replay."
                )

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

    def call_size_operator(
        self,
        tensor_proxy: ProxyValue,
        dim: int,
        meta: NodeMetadata,
        *,
        edge_dialect: bool = False,
    ) -> Union[ProxyValue, int]:
        """Read ``tensor_proxy.size(dim)`` as a value usable in graph args.
        Returns a plain ``int`` for static dims; emits and returns a
        ``sym_size.int`` ``ProxyValue`` for SymInt dims (since
        ``Graph.create_node`` rejects raw SymInts in call_function args).

        When ``edge_dialect`` is True, emits ``exir_ops.edge.aten.sym_size.int``
        so the node fits an edge-lowered graph; otherwise emits the raw
        ``torch.ops.aten.sym_size.int``.
        """
        size = tensor_proxy.data.shape[dim]
        if isinstance(size, torch.SymInt):
            sym_size_op = (
                exir_ops.edge.aten.sym_size.int
                if edge_dialect
                else torch.ops.aten.sym_size.int
            )
            new_proxy = self.call_operator(sym_size_op, (tensor_proxy, dim), {}, meta)
            # Mirror source's "example_value" if present, so the new node
            # matches the surrounding graph's meta-key convention. "val"
            # is already set by call_operator → _fx → set_metadata.
            if "example_value" in tensor_proxy.node.meta:
                new_proxy.node.meta["example_value"] = new_proxy.node.meta["val"]
            return new_proxy
        return int(size)

    def call_size_operator_all(
        self,
        tensor_proxy: ProxyValue,
        meta: NodeMetadata,
        *,
        edge_dialect: bool = False,
    ) -> list[Union[ProxyValue, int]]:
        """Return all dims of ``tensor_proxy.shape`` as a list of values
        usable in graph args. Each entry is an ``int`` (static dim) or a
        ``sym_size.int`` ``ProxyValue`` (dynamic dim) — see
        ``call_size_operator``.

        ``edge_dialect`` selects the edge vs raw ATen ``sym_size.int`` op.
        """
        return [
            self.call_size_operator(tensor_proxy, d, meta, edge_dialect=edge_dialect)
            for d in range(len(tensor_proxy.data.shape))
        ]

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
        self._validate_symbolic_input_snapshots(graph_module, new_graph_module)

        # Preserve GraphModule-level metadata from the input module.
        new_graph_module.meta = graph_module.meta.copy()

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
            fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True)
            dispatcher_mode = nullcontext()  # type: ignore[assignment]
        else:
            fake_tensor_mode.allow_non_fake_inputs = True
            dispatcher_mode = enable_python_dispatcher()  # type: ignore[assignment]
        self.tracer.fake_tensor_mode = fake_tensor_mode
        self.fake_tensor_mode = fake_tensor_mode

        with fake_tensor_mode, dispatcher_mode:  # type: ignore[assignment, union-attr]
            result = self.call_submodule(graph_module, tuple(inputs))

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
