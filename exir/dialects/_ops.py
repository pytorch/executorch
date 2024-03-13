# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
from typing import Callable

import torch
from executorch.exir.dialects.backend._ops import (
    _BACKEND_OP_LIB,
    BackendOpOverloadPacket,
)
from executorch.exir.dialects.edge._ops import EdgeOpOverloadPacket
from torch._C import DispatchKey  # @manual
from torch.library import Library
from torchgen.model import FunctionSchema

_OPOVERLOAD_PACKET_CLS_MAPPING = {
    "edge": EdgeOpOverloadPacket,
    "backend": BackendOpOverloadPacket,
}


def bind_pattern_to_op(library: Library, schema_or_name: str):
    """Bind a pattern of ops to a backend op. A backend op should only appear when a user wants to replace a pattern of nodes to a custom op.
    On this front, the kernel being registered to it determines the decomposing behavior.

    *   If the backend op is registered with an CompositeExplicitAutograd (or Meta) kernel, once the graph is lowered (meaning the pass
        of replacing a pattern to an op is executed) it will stick in the graph and we won't get the original graph even retrace.
    *   Otherwise, the backend op should be able to support retracing and be able to "promote" back to the original graph through retracing.

    This macro is aiming to handle this complexity for users and they just need to use this macro on the pattern and we can make a decision for them.

    Args:
        library (Library): torch library
        schema_or_name (str): schema string, e.g., "add.int(SymInt a, SymInt b) -> SymInt", or a qualified op name
    """

    def wrapper(f: Callable):
        if library.ns not in _BACKEND_OP_LIB:
            _BACKEND_OP_LIB.append(library.ns)
        no_namespace = schema_or_name.split("::")[-1]
        try:
            # can parse it into a FunctionSchema
            func = FunctionSchema.parse(no_namespace)
            name, overload_name = func.name.name.base, func.name.overload_name
            library.define(no_namespace)
        except AssertionError:
            if "." in no_namespace:
                name, overload_name = no_namespace.split(".")
            else:
                name, overload_name = no_namespace, None
        opname = name + ("." + overload_name if overload_name else "")
        overload_name = overload_name if overload_name else "default"
        torch_op = getattr(getattr(getattr(torch.ops, library.ns), name), overload_name)
        # we can't have both CompositeExplicitAutograd and CompositeImplicitAutograd kernel,
        # we can't have both Meta and CompositeImplicitAutograd kernel either.
        keys = [
            DispatchKey.CompositeExplicitAutograd,
            DispatchKey.CompositeImplicitAutograd,
            DispatchKey.Meta,
        ]
        if not any(torch_op.has_kernel_for_dispatch_key(k) for k in keys):
            library.impl(opname, f, "CompositeImplicitAutograd")
        op = getattr(getattr(getattr(ops.backend, library.ns), name), overload_name)
        op._equivalent_callable = f
        return f

    return wrapper


class _OpNamespace(types.ModuleType):
    """
    EXIR Dialect op namespace object. Contains ops and overloads registered into PyTorch dispatcher.
    """

    def __init__(self, dialect, name):
        super().__init__(f"exir.ops.{dialect}.{name}")
        self._dialect = dialect
        if dialect == "backend" and name not in _BACKEND_OP_LIB:
            raise RuntimeError(f"{name} op library does not belong to backend ops.")
        self._name = name
        self._dir = []
        self._op_namespace = getattr(torch.ops, name)

    def __iter__(self):
        return iter(self._dir)

    def __getattr__(self, op_name):
        # It is not a valid op_name when __file__ is passed in
        if op_name == "__file__":
            return "exir.ops"

        if op_name in self.__dict__:
            return getattr(self, op_name)

        try:
            parent_packet = getattr(self._op_namespace, op_name)
        except AttributeError as e:
            # Turn this into AttributeError so getattr(obj, key, default)
            # works (this is called by TorchScript with __origin__)
            raise AttributeError(
                f"'_OpNamespace' '{self._dialect}.{self._name}' object has no attribute '{op_name}'"
            ) from e
        qualified_op_name = f"{self._name}::{op_name}"
        opoverload_packet_cls = _OPOVERLOAD_PACKET_CLS_MAPPING[self._dialect]
        opoverloadpacket = opoverload_packet_cls(
            qualified_op_name,
            op_name,
            parent_overload_packet=parent_packet,
        )
        opoverloadpacket.__module__ = self.__module__ + "." + self._name
        # cache the opoverloadpacket to ensure that each op corresponds to
        # a unique OpOverloadPacket object
        setattr(self, op_name, opoverloadpacket)
        self._dir.append(op_name)
        return opoverloadpacket


class _DialectNamespace(types.ModuleType):
    """
    Dialect namespace. Currently the dialects are:
    - ATen Dialect: core ATen ops and overloads, see torch._ops._OpNamespace
    - Edge Dialect: ATen ops with explicit Tensor dtype
    - Backend Dialect: backend ops only meaningful to the backend we are lowering into
    - Execution Dialect: memory planning ready, all out-variants
    """

    def __init__(self, dialect_name):
        super().__init__("exir.ops" + "." + dialect_name)
        self._dialect_name = dialect_name
        self._dir = []

    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self, name)
        # Here we are creating `exir.ops.<dialect_ns>.<my_namespace>`
        namespace = _OpNamespace(self._dialect_name, name)
        setattr(self, name, namespace)
        self._dir.append(name)
        return namespace


class _Ops(types.ModuleType):
    __file__ = "_ops.py"

    def __init__(self):
        super().__init__("exir.ops")
        self._dir = []

    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self, name)
        dialect = _DialectNamespace(name)
        setattr(self, name, dialect)
        self._dir.append(name)
        return dialect

    def __iter__(self):
        return iter(self._dir)


# The ops "namespace"
ops = _Ops()
