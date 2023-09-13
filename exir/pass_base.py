# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import operator
from typing import (
    Any,
    Callable,
    Dict,
    List,
    MutableMapping,
    Protocol,
    runtime_checkable,
    Tuple,
    TypeVar,
)

import torch
from executorch.exir import memory

from executorch.exir.delegate import executorch_call_delegate, is_lowered_module

from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.error import ExportError, ExportErrorType
from torch import fx
from torch._export.pass_base import (  # noqa
    _ExportPassBase,
    Argument,
    NodeMetadata,
    PassBase,
    PassResult,
    ProxyValue,
)
from torch.utils._pytree import PyTree

Fn = Callable[..., Any]  # pyre-ignore
K = TypeVar("K")


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
