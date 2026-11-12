# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import operator
from typing import Any, Optional, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    get_param_tensor,
    is_param_node,
)
from executorch.backends.arm.constants import ARM_DYNAMIC_W8A8_LINEAR_META_KEY
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node


class DecomposeChooseQParamsSymmetricPass(ArmPass):
    """Lower the supported dynamic symmetric INT8 qparam calculation.

    Only the Arm dynamic-W8A8 activation contract is handled: per-tensor INT8
    in the range [-127, 127], with a zero point of zero. The qparams are emitted
    as:

        scale = max(amax(abs(x)) / 127, eps)
        zero_point = 0

    The scale is emitted as a one-element FP32 tensor and the zero point as a
    one-element INT32 tensor. Unsupported variants are retained unchanged.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram):
        super().__init__()
        self.exported_program = exported_program

    @staticmethod
    def _target():
        return exir_ops.edge.quantized_decomposed.choose_qparams_symmetric.tensor

    @staticmethod
    def _is_int_literal(value: object, expected: int) -> bool:
        return (
            isinstance(value, int) and not isinstance(value, bool) and value == expected
        )

    @classmethod
    def _is_supported_choose(cls, node: Node) -> bool:
        if node.op != "call_function" or node.target != cls._target():
            return False
        if len(node.args) < 5:
            return False
        _, qmin, qmax, _, dtype = node.args[:5]
        return (
            cls._is_int_literal(qmin, -127)
            and cls._is_int_literal(qmax, 127)
            and dtype is torch.int8
        )

    @staticmethod
    def _number_to_float(value: Any) -> Optional[float]:
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            return float(value)
        return None

    @staticmethod
    def _tensor_to_float(value: Any) -> Optional[float]:
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            return None
        try:
            return float(value.detach().cpu().item())
        except (RuntimeError, TypeError, ValueError):
            return None

    def _resolve_param_float(self, value: Node) -> Optional[float]:
        try:
            if not is_param_node(self.exported_program, value):
                return None
            tensor = get_param_tensor(self.exported_program, value)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            return None
        return self._tensor_to_float(tensor)

    @classmethod
    def _resolve_attr_float(
        cls, graph_module: GraphModule, value: Node
    ) -> Optional[float]:
        if value.op != "get_attr":
            return None
        try:
            attr = getattr(graph_module, str(value.target))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

        tensor_value = cls._tensor_to_float(attr)
        if tensor_value is not None:
            return tensor_value
        return cls._number_to_float(attr)

    @classmethod
    def _resolve_meta_float(cls, value: Node) -> Optional[float]:
        meta_val = value.meta.get("val")
        if isinstance(meta_val, FakeTensor):
            return None
        return cls._tensor_to_float(meta_val)

    def _resolve_float(self, graph_module: GraphModule, value: Any) -> Optional[float]:
        scalar_value = self._number_to_float(value)
        if scalar_value is not None:
            return scalar_value

        tensor_value = self._tensor_to_float(value)
        if tensor_value is not None:
            return tensor_value
        if not isinstance(value, Node):
            return None

        param_value = self._resolve_param_float(value)
        if param_value is not None:
            return param_value

        attr_value = self._resolve_attr_float(graph_module, value)
        if attr_value is not None:
            return attr_value
        return self._resolve_meta_float(value)

    @classmethod
    def _collect_getitems(cls, choose: Node) -> Optional[list[Node]]:
        """Collect the canonical scale/zp getitems or reject unusual uses."""
        getitems: list[Node] = []
        indices: set[int] = set()
        for user in list(choose.users):
            if (
                user.op != "call_function"
                or user.target is not operator.getitem
                or len(user.args) < 2
                or user.args[0] is not choose
            ):
                return None

            index = user.args[1]
            if cls._is_int_literal(index, 0):
                indices.add(0)
            elif cls._is_int_literal(index, 1):
                indices.add(1)
            else:
                return None
            getitems.append(user)

        if indices != {0, 1}:
            return None
        return getitems

    def _match_choose(
        self, graph_module: GraphModule, choose: Node
    ) -> Optional[tuple[Node, float, int, list[Node]]]:
        if not self._is_supported_choose(choose):
            return None

        x, _, _, eps, _ = choose.args[:5]
        if not isinstance(x, Node):
            return None

        eps_value = self._resolve_float(graph_module, eps)
        if eps_value is None or not math.isfinite(eps_value) or eps_value <= 0.0:
            return None

        shape = getattr(x.meta.get("val"), "shape", None)
        if shape is None:
            return None
        rank = len(shape)
        if rank == 0:
            return None

        getitems = self._collect_getitems(choose)
        if getitems is None:
            return None
        return x, eps_value, rank, getitems

    @staticmethod
    def _create_qparams(
        graph: torch.fx.Graph,
        before: Node,
        x: Node,
        rank: int,
        eps_value: float,
    ) -> tuple[Node, Node]:
        with graph.inserting_before(before):
            abs_x = graph.call_function(exir_ops.edge.aten.abs.default, (x,))
            max_abs = graph.call_function(
                exir_ops.edge.aten.amax.default,
                (abs_x, list(range(rank)), True),
            )
            ones_shape = (1,) * rank
            inv_127 = graph.call_function(
                exir_ops.edge.aten.full.default,
                (ones_shape, 1.0 / 127.0),
                {"dtype": torch.float32},
            )
            raw_scale = graph.call_function(
                exir_ops.edge.aten.mul.Tensor, (max_abs, inv_127)
            )
            eps_tensor = graph.call_function(
                exir_ops.edge.aten.full.default,
                (ones_shape, eps_value),
                {"dtype": torch.float32},
            )
            scale_nd = graph.call_function(
                exir_ops.edge.aten.maximum.default, (raw_scale, eps_tensor)
            )
            scale = graph.call_function(
                exir_ops.edge.aten.view_copy.default, (scale_nd, (1,))
            )
            zero_point = graph.call_function(
                exir_ops.edge.aten.full.default,
                ((1,), 0),
                {"dtype": torch.int32},
            )
        return scale, zero_point

    @staticmethod
    def _mark_qparams(scale: Node, zero_point: Node) -> None:
        scale.meta[ARM_DYNAMIC_W8A8_LINEAR_META_KEY] = {
            "dynamic_qparam": "scale",
            "symmetric": True,
        }
        zero_point.meta[ARM_DYNAMIC_W8A8_LINEAR_META_KEY] = {
            "dynamic_qparam": "zero_point",
            "symmetric": True,
            "value": 0,
        }

    @classmethod
    def _replace_getitems(
        cls,
        graph: torch.fx.Graph,
        getitems: list[Node],
        scale: Node,
        zero_point: Node,
    ) -> None:
        for getitem in getitems:
            replacement = (
                scale if cls._is_int_literal(getitem.args[1], 0) else zero_point
            )
            getitem.replace_all_uses_with(replacement)
            graph.erase_node(getitem)

    def _decompose_choose(self, graph_module: GraphModule, choose: Node) -> bool:
        match = self._match_choose(graph_module, choose)
        if match is None:
            return False

        x, eps_value, rank, getitems = match
        graph = graph_module.graph
        scale, zero_point = self._create_qparams(graph, getitems[0], x, rank, eps_value)
        self._mark_qparams(scale, zero_point)
        self._replace_getitems(graph, getitems, scale, zero_point)
        if not choose.users:
            graph.erase_node(choose)
        return True

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for choose in list(graph_module.graph.nodes):
            modified |= self._decompose_choose(graph_module, choose)

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()
            # Populate metadata for every newly introduced helper node here.
            # Do not rely on a later pass successfully matching the Linear.
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
