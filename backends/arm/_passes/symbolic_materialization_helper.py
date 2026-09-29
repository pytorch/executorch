# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Iterable, List, Tuple

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm.tosa.dialect.shape import meta_has_shape_mark
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import NodeMetadata, ProxyValue


ShapeList = List | Tuple


class SymbolMaterializationHelpers:
    """Build canonical TOSA shape operands for symbolic shape lowering."""

    def __init__(self, owning_pass: ArmPass):
        self._shape_to_proxyval: dict[str, ProxyValue] = {}
        self.builder = owning_pass
        self.materialized_shape_ops = 0

    def _ensure_value(
        self,
        value: ProxyValue | int,
        meta: NodeMetadata,
    ) -> ProxyValue:
        if isinstance(value, ProxyValue):
            if not meta_has_shape_mark(value.node.meta) and isinstance(value.data, int):
                return self._materialize_int(value.data, meta)
            return value
        elif isinstance(value, int):
            return self._materialize_int(value, meta)
        else:
            raise TypeError(
                f"Unsupported value type {type(value)} for symbolic materialization"
            )

    def materialize_arglist(
        self, shape_arg: ShapeList, meta: NodeMetadata
    ) -> ProxyValue:
        elements = list(self._iter_materialized_shape_elements(shape_arg, meta))
        if len(elements) == 1:
            return elements[0]
        self.materialized_shape_ops += 1
        return self.builder.call_shape_operator(
            exir_ops.backend.tosa.CONCAT_SHAPE.default,
            (elements,),
            {},
            meta,
            True,
        )

    def _iter_materialized_shape_elements(
        self,
        shape_arg: ShapeList,
        meta: NodeMetadata,
    ) -> Iterable[ProxyValue]:
        for element in shape_arg:
            if isinstance(element, (list, tuple)):
                yield from self._iter_materialized_shape_elements(element, meta)
            else:
                yield self._ensure_value(element, meta)

    def _register_proxyval(self, key: str, proxyval: ProxyValue) -> None:
        self._shape_to_proxyval[key] = proxyval

    def _materialize_int(self, value: int, meta: NodeMetadata) -> ProxyValue:
        maybe_proxy = self._shape_to_proxyval.get(str(value), None)
        if maybe_proxy is not None:
            return maybe_proxy
        self.materialized_shape_ops += 1
        proxy_value = self.builder.call_shape_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default,
            ([value],),
            {},
            meta,
            True,
        )
        self._register_proxyval(str(value), proxy_value)
        return proxy_value

    def materialize_shape_op(self, target, args: Tuple, kwargs, meta) -> ProxyValue:
        output_shape = meta.data["val"]
        maybe_output_proxy = self._shape_to_proxyval.get(str(output_shape), None)
        if maybe_output_proxy is not None:
            return maybe_output_proxy
        if target == exir_ops.backend.tosa.DIM.default:
            args = (args[0],)
        else:
            args = tuple([self.materialize_arglist([arg], meta) for arg in args])

        shape_meta = copy.copy(meta)
        shape_meta.data = dict(meta.data)
        if not isinstance(output_shape, (list, tuple)):
            shape_meta.data["val"] = [output_shape]
        self.materialized_shape_ops += 1
        proxy = self.builder.call_shape_operator(
            target,
            args,
            kwargs,
            shape_meta,
            True,
        )
        self._register_proxyval(str(output_shape), proxy)
        return proxy
