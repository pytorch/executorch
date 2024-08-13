# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from torch._C import DispatchKey  # @manual

_BACKEND_OP_LIB = [
    "executorch_prim",
    "quantized_decomposed",
    "DO_NOT_USE_TEST_ONLY",
]


class BackendOpOverload(EdgeOpOverload):
    """OpOverload for backend ops.
    A Backend operator is a custom op that doesn't show up in ATen dialect.
    Therefore it must be replacing an existing node or a pattern of nodes in Edge dialect.
    This data structure makes sure after lower (part of) Edge dialect to backend ops, the whole graph can still be captured properly.

    Difference to delegate:
    1. delegate result is still a module (a target of call_module, at least for now) while backend op is an operator (a target of call_function).
    2. backend op is stateless while delegation doesn't have to
    3. backend op stays in executor standard runtime but delegation doesn't have to

    Examples for backend ops including fused ops for a specific backend, ExecuTorch prim ops to handle symbolic shape.

    Note that the assumption here is that the backend op and the original callable / equivalent callable is 1 - 1 mapping.

    BackendOpOverload makes sure:
    1. The backend op contains either a CompositeExplicitAutograd or a meta kernel.
    2. It also holds a reference to the original node/pattern it replaces.
    Example:

    add -> relu
        |
        v
    add_relu(only works on dsp): hold reference to add -> relu pattern, for re-capturing purpose.

    Retrace example:

    A very common practice in delegate, is that the module needs to be lowered to a backend, then the lowered module needs to be composed with original nn.Module and retrace.

    LoweredModule l_of_m = to_backend(g_of_m.to_edge(), ...)
    Module main(l_of_m)
    export(main, inputs)

    """

    def __init__(
        self,
        op_overload: EdgeOpOverload,
    ):
        super(self.__class__, self).__init__(
            op_overload._op,
            op_overload._schema,
        )
        self._equivalent_callable = None
        self._has_meta_kernel = self._op.has_kernel_for_dispatch_key(DispatchKey.Meta)
        self._has_composite_explicit_autograd_kernel = (
            self._op.has_kernel_for_dispatch_key(DispatchKey.CompositeExplicitAutograd)
        )
        self._has_composite_implicit_autograd_kernel = (
            self._op.has_kernel_for_dispatch_key(DispatchKey.CompositeImplicitAutograd)
        )
        assert (
            self._has_meta_kernel
            or self._has_composite_explicit_autograd_kernel
            or self._has_composite_implicit_autograd_kernel
        ), "A backend op must contain either CompositeExplicitAutograd or Meta or CompositeImplicitAutograd kernel."


class BackendOpOverloadPacket(EdgeOpOverloadPacket):
    """OpOverloadPacket for backend ops.
    Wraps EdgeOpOverloadPacket and overrides __getattr__ to return OpOverload
    for backend ops.
    """

    def __init__(
        self,
        qualified_op_name: str,
        op_name: str,
        parent_overload_packet: torch._ops.OpOverloadPacket,
    ):
        super(self.__class__, self).__init__(
            qualified_op_name, op_name, parent_overload_packet
        )

    def __repr__(self):
        return "<BackendOpOverloadPacket(op='{}', parent_op='{}')>".format(
            self._qualified_op_name.replace("::", "."),
            self._parent_qualified_op_name.replace("::", "."),
        )

    def __hash__(self):
        return hash(self._op)

    def __str__(self):
        return "{}".format(self._qualified_op_name.replace("::", "."))

    @property
    def op(self):
        return self._op

    def __getattr__(self, key):
        try:
            # get edge op, set it as attribute. Note that this way we don't have `_original_pattern`.
            result = super().__getattr__(key)
            if isinstance(result, EdgeOpOverload):
                backend_op = BackendOpOverload(result)
                setattr(self, key, backend_op)
                return backend_op
            else:
                return result
        except AttributeError as e:
            raise AttributeError(
                "The underlying op of '{}' has no overload name '{}'. Original error message: \n {}".format(
                    self, key, e
                )
            ) from e
