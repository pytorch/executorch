# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Iterable, List, Tuple

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm.tosa.dialect.shape import meta_has_shape_mark
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import NodeMetadata, ProxyValue


ShapeList = List | Tuple

logger = logging.getLogger(__name__)


class SymbolMaterializationHelpers:
    """Build canonical TOSA shape operands for symbolic shape lowering."""

    def __init__(self, owning_pass: ArmPass):
        self._shape_to_proxyval: dict[str, ProxyValue] = {}
        self.builder = owning_pass

    def _ensure_value(
        self,
        value: ProxyValue | int,
        meta: NodeMetadata,
    ) -> ProxyValue:
        if isinstance(value, ProxyValue):
            if not meta_has_shape_mark(value.node.meta) and isinstance(value.data, int):
                logger.debug(
                    "Materializing scalar ProxyValue node=%s data=%s",
                    value.node.name,
                    value.data,
                )
                return self._materialize_int(value.data, meta)
            logger.debug(
                "Using existing ProxyValue node=%s data=%s",
                value.node.name,
                value.data,
            )
            return value
        elif isinstance(value, int):
            logger.debug("Materializing integer shape value=%s", value)
            return self._materialize_int(value, meta)
        else:
            logger.debug("Unsupported symbolic materialization value=%r", value)
            raise TypeError(
                f"Unsupported value type {type(value)} for symbolic materialization"
            )

    def materialize_arglist(
        self, shape_arg: ShapeList, meta: NodeMetadata
    ) -> ProxyValue:
        logger.debug("Materializing shape arglist=%s", shape_arg)
        elements = list(self._iter_materialized_shape_elements(shape_arg, meta))
        logger.debug(
            "Materialized arglist elements=%s",
            [(element.node.name, element.data) for element in elements],
        )
        if len(elements) == 1:
            logger.debug(
                "Arglist has one element; reusing node=%s", elements[0].node.name
            )
            return elements[0]
        logger.debug("Creating CONCAT_SHAPE for %d elements", len(elements))
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
        logger.debug(
            "Registering shape proxy key=%s node=%s data=%s",
            key,
            proxyval.node.name,
            proxyval.data,
        )
        self._shape_to_proxyval[key] = proxyval

    def _materialize_int(self, value: int, meta: NodeMetadata) -> ProxyValue:
        maybe_proxy = self._shape_to_proxyval.get(str(value), None)
        if maybe_proxy is not None:
            logger.debug("Reusing CONST_SHAPE for integer value=%s", value)
            return maybe_proxy
        logger.debug("Creating CONST_SHAPE for integer value=%s", value)
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
        logger.debug(
            "Materializing shape op target=%s output_shape=%s args=%s kwargs=%s",
            target,
            output_shape,
            args,
            kwargs,
        )
        maybe_output_proxy = self._shape_to_proxyval.get(str(output_shape), None)
        if maybe_output_proxy is not None:
            logger.debug(
                "Reusing cached shape op target=%s output_shape=%s node=%s",
                target,
                output_shape,
                maybe_output_proxy.node.name,
            )
            return maybe_output_proxy
        if target == exir_ops.backend.tosa.DIM.default:
            args = (args[0],)
        else:
            args = tuple([self.materialize_arglist([arg], meta) for arg in args])

        logger.debug(
            "Calling shape op target=%s with materialized args=%s", target, args
        )
        shape_meta = copy.copy(meta)
        shape_meta.data = dict(meta.data)
        if not isinstance(output_shape, (list, tuple)):
            shape_meta.data["val"] = [output_shape]
        proxy = self.builder.call_shape_operator(
            target,
            args,
            kwargs,
            shape_meta,
            True,
        )
        self._register_proxyval(str(output_shape), proxy)
        return proxy
