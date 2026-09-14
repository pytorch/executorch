# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator

from typing import Any, cast, Optional

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass

from executorch.backends.arm._passes.symbolic_materialization_helper import (
    SymbolMaterializationHelpers,
)
from executorch.backends.arm._passes.symbolic_shape_utils import materialize_symints
from executorch.backends.arm.tosa.dialect.shape import meta_has_shape_mark
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import NodeMetadata, ProxyValue

logger = logging.getLogger(__name__)


_SYMBOLIC_SHAPE_OPS: dict[Any, Any] = {
    operator.add: exir_ops.backend.tosa.ADD_SHAPE.default,
    operator.sub: exir_ops.backend.tosa.SUB_SHAPE.default,
    operator.mul: exir_ops.backend.tosa.MUL_SHAPE.default,
    operator.mod: exir_ops.backend.tosa.MOD_SHAPE.default,
    operator.floordiv: exir_ops.backend.tosa.DIV_FLOOR_SHAPE.default,
}


class SymbolicToTosaShapesPass(ArmPass):
    """Lower PyTorch symbolic shape expressions to TOSA shape operands.

    This pass owns the when of symbolic shape lowering. It maps
    `aten.sym_size.int` to `DIM`, maps supported symbolic arithmetic to TOSA
    shape arithmetic, and finds operator arguments that contain
    shape-producing proxies. When a shape value must be materialized, it
    delegates the how to `SymbolMaterializationHelpers`, which builds and
    caches `CONST_SHAPE`, `CONCAT_SHAPE`, and other TOSA shape ops.

    """

    _passes_required_after = set()

    def __init__(self):
        super().__init__()
        self.materializer = SymbolMaterializationHelpers(self)

    def _is_shape_proxy(self, arg):
        return isinstance(arg, ProxyValue) and meta_has_shape_mark(
            getattr(arg.node, "meta", {})
        )

    def _has_shape_node_arg(self, arg):
        if isinstance(arg, torch.fx.Node):
            return meta_has_shape_mark(arg.meta)
        if isinstance(arg, (list, tuple)):
            return any(self._has_shape_node_arg(a) for a in arg)
        return False

    def _has_shape_proxy_arg(self, arg):
        if self._is_shape_proxy(arg):
            return True
        if isinstance(arg, (list, tuple)):
            return any(self._has_shape_proxy_arg(a) for a in arg)
        return False

    def _has_raw_symint_arg(self, arg):
        if isinstance(arg, torch.SymInt):
            return True
        if isinstance(arg, (list, tuple)):
            return any(self._has_raw_symint_arg(a) for a in arg)
        return False

    def _proxy_value_from_node(self, node: torch.fx.Node) -> ProxyValue:
        return ProxyValue(node.meta["val"], self.tracer.proxy(node))

    def _proxy_value_from_arg(self, arg) -> ProxyValue | int:
        if isinstance(arg, torch.fx.Node):
            return self._materialize_shape_node(arg)
        return arg

    def _meta_value_from_arg(self, arg):
        if not isinstance(arg, torch.fx.Node):
            return arg
        self._ensure_shape_node_meta(arg)
        value = arg.meta["val"]
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    def _ensure_shape_node_meta(self, node: torch.fx.Node) -> None:
        if "val" in node.meta:
            return
        target = node.target
        if not callable(target):
            return
        node.meta["val"] = target(
            *(self._meta_value_from_arg(arg) for arg in node.args),
            **{
                key: self._meta_value_from_arg(value)
                for key, value in node.kwargs.items()
            },
        )

    def _materialize_shape_node(self, node: torch.fx.Node) -> ProxyValue:
        if meta_has_shape_mark(node.meta):
            return self._proxy_value_from_node(node)
        if node.target == torch.ops.aten.sym_size.int:
            tensor_node = cast(torch.fx.Node, node.args[0])
            tensor = self._proxy_value_from_node(tensor_node)
            return self.materializer.materialize_shape_op(
                exir_ops.backend.tosa.DIM.default,
                (tensor,),
                {"axis": node.args[1]},
                NodeMetadata(node.meta),
            )
        if node.target in _SYMBOLIC_SHAPE_OPS:
            self._ensure_shape_node_meta(node)
            return self.materializer.materialize_shape_op(
                _SYMBOLIC_SHAPE_OPS[node.target],
                tuple(self._proxy_value_from_arg(arg) for arg in node.args),
                {},
                NodeMetadata(node.meta),
            )
        return self._proxy_value_from_node(node)

    def _erase_temporary_shape_expression(
        self, node: torch.fx.Node, original_nodes: set[torch.fx.Node]
    ) -> None:
        input_nodes = list(node.all_input_nodes)
        if node not in original_nodes and not node.users:
            self.tracer.graph.erase_node(node)
        for input_node in reversed(input_nodes):
            self._erase_temporary_shape_expression(input_node, original_nodes)

    def _materialize_raw_symints(self, arg):
        if isinstance(arg, torch.SymInt):
            original_nodes = set(self.tracer.graph.nodes)
            materialized = materialize_symints(self.tracer.graph, [arg])[0]
            if isinstance(materialized, torch.fx.Node):
                proxy_value = self._materialize_shape_node(materialized)
                self._erase_temporary_shape_expression(materialized, original_nodes)
                return proxy_value
            return materialized
        if isinstance(arg, list):
            return [self._materialize_raw_symints(a) for a in arg]
        if isinstance(arg, tuple):
            return tuple(self._materialize_raw_symints(a) for a in arg)
        return arg

    def should_run_pass(self, graph_module):
        visited_graph_modules = set()

        def graph_needs_shape_materialization(module):
            if id(module) in visited_graph_modules:
                return False
            visited_graph_modules.add(id(module))

            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target == torch.ops.aten.sym_size.int:
                    return True
                if meta_has_shape_mark(node.meta):
                    continue
                if any(
                    self._has_shape_node_arg(arg) or self._has_raw_symint_arg(arg)
                    for arg in node.args
                ):
                    return True

            return any(
                isinstance(child, torch.fx.GraphModule)
                and graph_needs_shape_materialization(child)
                for child in module.children()
            )

        return graph_needs_shape_materialization(graph_module)

    def call_operator(self, op, args, kwargs, meta, updated: Optional[bool] = False):
        if op == torch.ops.aten.sym_size.int:
            logger.debug("Materializing sym_size.int as TOSA DIM axis=%s", args[1])
            return self.materializer.materialize_shape_op(
                exir_ops.backend.tosa.DIM.default,
                (args[0],),
                {"axis": args[1]},
                meta,
            )

        if meta_has_shape_mark(meta.data):
            logger.debug("Forwarding already shape-marked op=%s", op)
            return super().call_operator(op, args, kwargs, meta, updated)
        new_args: list[Any] = []
        for arg in args:
            if isinstance(arg, (list, tuple)) and len(arg) > 0:
                if self._has_raw_symint_arg(arg):
                    logger.debug(
                        "Materializing raw SymInt entries for op=%s shape arg: %s",
                        op,
                        arg,
                    )
                    arg = self._materialize_raw_symints(arg)
                if self._has_shape_proxy_arg(arg):
                    logger.debug(
                        "Materializing list arg for op=%s as TOSA shape arg: %s",
                        op,
                        arg,
                    )
                    shape_op_arg = self.materializer.materialize_arglist(arg, meta)
                    new_args.append(shape_op_arg)
                else:
                    new_args.append(arg)
            else:
                new_args.append(arg)
        args = tuple(new_args)
        logger.debug("Calling rewritten op=%s args=%s", op, args)

        return super().call_operator(op, args, kwargs, meta)

    def call_sym(self, target, args, meta):
        has_shape_arg = any(self._has_shape_proxy_arg(arg) for arg in args)
        if target in _SYMBOLIC_SHAPE_OPS and has_shape_arg:
            logger.debug(
                "Materializing symbolic op target=%s as shape op=%s args=%s",
                target,
                _SYMBOLIC_SHAPE_OPS[target],
                args,
            )
            return self.materializer.materialize_shape_op(
                _SYMBOLIC_SHAPE_OPS[target], args, {}, meta
            )
        if has_shape_arg:
            raise NotImplementedError(
                f"Symbolic op target {target} not supported in symbolic to TOSA shape pass"
            )
        return super().call_sym(target, args, meta)
