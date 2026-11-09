# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    get_param_tensor,
    is_param_node,
)
from executorch.backends.arm.constants import (
    ARM_DYNAMIC_W8A8_LINEAR_META_KEY,
    DEQUANT_PER_CHANNEL_OP,
    DEQUANT_PER_TENSOR_OP,
    DEQUANT_PER_TENSOR_OP_T,
    QUANT_PER_CHANNEL_OP,
    QUANT_PER_TENSOR_OP,
    QUANT_PER_TENSOR_OP_T,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node


@dataclass(frozen=True)
class _ActivationMatch:
    q: Node
    dq: Node
    scale: Node
    zero_point: Node


@dataclass(frozen=True)
class _WeightMatch:
    quantized: Node
    scale: Any
    zero_point: Any
    per_channel: bool


class DetectDynamicW8A8LinearPass(ArmPass):
    """Detect supported FP32 dynamic-W8A8 Linear patterns for later lowering.

    The activation must use the dynamic per-tensor symmetric INT8 Q/DQ form:

        FP32 activation
          -> quantize_per_tensor.tensor(scale, zero_point, -127, 127, int8)
          -> dequantize_per_tensor.tensor(...)
          -> Linear

    The activation scale and zero point must come from the preceding dynamic
    qparam decomposition and carry the Arm dynamic-qparam metadata. The zero
    point must resolve to zero, and the Q/DQ arguments must match exactly.

    The Linear weight may use either of the supported static representations:

      * folded: an INT8 weight consumed directly by a per-tensor or
        per-channel dequantize operation;
      * unfolded: a static floating-point weight quantized to INT8 and then
        dequantized by a matching per-tensor or per-channel Q/DQ pair.

    Weight quantization must be symmetric INT8 in [-127, 127]. Per-channel
    quantization is supported only along axis 0. Weight scales must be static,
    finite, and positive, and zero points must be zero.

    The pass also validates the supported dtype and shape contract: FP32
    activation/dequantized values, INT8 quantized activation and weight,
    rank-2 weights, compatible Linear dimensions, and an optional FP32 bias
    with one value per output channel.

    This pass does not rewrite the graph. Matching Linear nodes are annotated
    with the validated activation, weight, qparam, bias, and output metadata
    consumed by LowerDynamicW8A8LinearPass. Unsupported or ambiguous patterns
    are left unchanged.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram):
        super().__init__()
        self.exported_program = exported_program

    @staticmethod
    def _is_int_literal(value: object, expected: int) -> bool:
        return (
            isinstance(value, int) and not isinstance(value, bool) and value == expected
        )

    @classmethod
    def _is_symmetric_int8_args(cls, qmin: object, qmax: object, dtype: object) -> bool:
        return (
            cls._is_int_literal(qmin, -127)
            and cls._is_int_literal(qmax, 127)
            and dtype is torch.int8
        )

    @staticmethod
    def _same_arg(lhs: Any, rhs: Any) -> bool:
        if isinstance(lhs, Node) or isinstance(rhs, Node):
            return lhs is rhs
        if isinstance(lhs, torch.dtype) or isinstance(rhs, torch.dtype):
            return lhs is rhs
        if isinstance(lhs, (bool, int, float, type(None))) and isinstance(
            rhs, (bool, int, float, type(None))
        ):
            return lhs == rhs
        return lhs is rhs

    @classmethod
    def _same_args(cls, lhs: tuple[Any, ...], rhs: tuple[Any, ...]) -> bool:
        return len(lhs) == len(rhs) and all(
            cls._same_arg(a, b) for a, b in zip(lhs, rhs)
        )

    @staticmethod
    def _node_dtype(value: Any) -> Optional[torch.dtype]:
        if not isinstance(value, Node):
            return None
        return getattr(value.meta.get("val"), "dtype", None)

    @staticmethod
    def _node_shape(value: Any) -> Optional[tuple[Any, ...]]:
        if not isinstance(value, Node):
            return None
        shape = getattr(value.meta.get("val"), "shape", None)
        return tuple(shape) if shape is not None else None

    @staticmethod
    def _literal_static(value: Any) -> Any:
        if isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, torch.Tensor) and not isinstance(value, FakeTensor):
            return value.detach().cpu()
        return None

    @staticmethod
    def _full_static(value: Node) -> Any:
        if (
            value.op != "call_function"
            or value.target != exir_ops.edge.aten.full.default
            or len(value.args) < 2
        ):
            return None
        fill_value = value.args[1]
        if isinstance(fill_value, (bool, int, float)):
            return fill_value
        return None

    def _param_static(self, value: Node) -> Any:
        try:
            if not is_param_node(self.exported_program, value):
                return None
            tensor = get_param_tensor(self.exported_program, value)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            return None
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu()
        return None

    @staticmethod
    def _attr_static(graph_module: GraphModule, value: Node) -> Any:
        if value.op != "get_attr":
            return None
        try:
            attr = getattr(graph_module, str(value.target))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        if isinstance(attr, torch.Tensor):
            return attr.detach().cpu()
        if isinstance(attr, (bool, int, float)):
            return attr
        return None

    @staticmethod
    def _meta_static(value: Node) -> Any:
        meta_val = value.meta.get("val")
        if isinstance(meta_val, torch.Tensor) and not isinstance(meta_val, FakeTensor):
            return meta_val.detach().cpu()
        return None

    def _resolve_static(self, graph_module: GraphModule, value: Any) -> Any:
        literal = self._literal_static(value)
        if literal is not None:
            return literal
        if not isinstance(value, Node):
            return None

        full_value = self._full_static(value)
        if full_value is not None:
            return full_value

        param_value = self._param_static(value)
        if param_value is not None:
            return param_value

        attr_value = self._attr_static(graph_module, value)
        if attr_value is not None:
            return attr_value
        return self._meta_static(value)

    def _is_all_zero(self, graph_module: GraphModule, value: Any) -> bool:
        resolved = self._resolve_static(graph_module, value)
        if isinstance(resolved, torch.Tensor):
            return bool(torch.all(resolved == 0).item())
        if isinstance(resolved, (bool, int, float)):
            return resolved == 0
        return False

    @staticmethod
    def _has_dynamic_marker(value: Node, kind: str) -> bool:
        marker = value.meta.get(ARM_DYNAMIC_W8A8_LINEAR_META_KEY)
        return (
            isinstance(marker, dict)
            and marker.get("dynamic_qparam") == kind
            and marker.get("symmetric") is True
        )

    @classmethod
    def _match_activation_qdq(cls, value: Any) -> Optional[tuple[Node, Node]]:
        if (
            not isinstance(value, Node)
            or value.op != "call_function"
            or value.target != DEQUANT_PER_TENSOR_OP_T
            or len(value.args) < 6
        ):
            return None

        q = value.args[0]
        if (
            not isinstance(q, Node)
            or q.op != "call_function"
            or q.target != QUANT_PER_TENSOR_OP_T
            or len(q.args) < 6
        ):
            return None
        if not cls._same_args(tuple(q.args[1:6]), tuple(value.args[1:6])):
            return None
        return q, value

    def _match_activation_qparams(
        self, graph_module: GraphModule, q: Node
    ) -> Optional[tuple[Node, Node]]:
        _, scale, zero_point, qmin, qmax, dtype = q.args[:6]
        if not self._is_symmetric_int8_args(qmin, qmax, dtype):
            return None
        if not isinstance(scale, Node) or not isinstance(zero_point, Node):
            return None
        if not self._has_dynamic_marker(scale, "scale"):
            return None
        if not self._has_dynamic_marker(zero_point, "zero_point"):
            return None
        if not self._is_all_zero(graph_module, zero_point):
            return None
        return scale, zero_point

    def _activation_types_are_supported(self, q: Node, dq: Node) -> bool:
        x = q.args[0]
        return (
            self._node_dtype(x) is torch.float32
            and self._node_dtype(q) is torch.int8
            and self._node_dtype(dq) is torch.float32
            and dq.kwargs.get("out_dtype") in (None, torch.float32)
        )

    def _match_activation(
        self, graph_module: GraphModule, value: Any
    ) -> Optional[_ActivationMatch]:
        qdq = self._match_activation_qdq(value)
        if qdq is None:
            return None
        q, dq = qdq

        qparams = self._match_activation_qparams(graph_module, q)
        if qparams is None:
            return None
        scale, zero_point = qparams

        if not self._activation_types_are_supported(q, dq):
            return None
        return _ActivationMatch(q=q, dq=dq, scale=scale, zero_point=zero_point)

    @staticmethod
    def _positive_finite_tensor(value: torch.Tensor) -> bool:
        if not value.is_floating_point() or value.numel() == 0:
            return False
        return bool(torch.all(torch.isfinite(value) & (value > 0)).item())

    @staticmethod
    def _static_qparam_shape_is_valid(
        value: torch.Tensor, expected_numel: int, per_channel: bool
    ) -> bool:
        if per_channel:
            return tuple(value.shape) == (expected_numel,)
        return value.numel() == 1

    def _valid_static_scale(
        self,
        graph_module: GraphModule,
        scale: Any,
        expected_numel: int,
        per_channel: bool,
    ) -> bool:
        resolved = self._resolve_static(graph_module, scale)
        if isinstance(resolved, torch.Tensor):
            return self._static_qparam_shape_is_valid(
                resolved, expected_numel, per_channel
            ) and self._positive_finite_tensor(resolved)
        if per_channel or expected_numel != 1:
            return False
        return (
            isinstance(resolved, (float, int))
            and not isinstance(resolved, bool)
            and math.isfinite(float(resolved))
            and float(resolved) > 0.0
        )

    def _valid_static_zero_point(
        self,
        graph_module: GraphModule,
        zero_point: Any,
        expected_numel: int,
        per_channel: bool,
        allow_none: bool,
    ) -> bool:
        if zero_point is None:
            return allow_none
        resolved = self._resolve_static(graph_module, zero_point)
        if isinstance(resolved, torch.Tensor) and not (
            self._static_qparam_shape_is_valid(resolved, expected_numel, per_channel)
        ):
            return False
        return self._is_all_zero(graph_module, zero_point)

    def _static_int8_weight(
        self, graph_module: GraphModule, value: Any
    ) -> Optional[torch.Tensor]:
        if not isinstance(value, Node):
            return None
        resolved = self._resolve_static(graph_module, value)
        if (
            isinstance(resolved, torch.Tensor)
            and resolved.dtype is torch.int8
            and resolved.ndim == 2
            and self._node_shape(value) == tuple(resolved.shape)
        ):
            return resolved
        return None

    def _make_weight_match(
        self,
        graph_module: GraphModule,
        quantized: Any,
        scale: Any,
        zero_point: Any,
        per_channel: bool,
    ) -> Optional[_WeightMatch]:
        if not isinstance(quantized, Node):
            return None
        if self._node_dtype(quantized) is not torch.int8:
            return None
        weight_shape = self._node_shape(quantized)
        if weight_shape is None or len(weight_shape) != 2:
            return None

        try:
            out_channels = int(weight_shape[0])
        except (TypeError, ValueError):
            return None
        expected_numel = out_channels if per_channel else 1
        if not self._valid_static_scale(
            graph_module, scale, expected_numel, per_channel
        ):
            return None
        if not self._valid_static_zero_point(
            graph_module,
            zero_point,
            expected_numel,
            per_channel,
            allow_none=per_channel,
        ):
            return None
        if not isinstance(scale, (Node, float, int)):
            return None
        if isinstance(scale, Node) and self._node_dtype(scale) is not torch.float32:
            return None

        return _WeightMatch(
            quantized=quantized,
            scale=scale,
            zero_point=zero_point,
            per_channel=per_channel,
        )

    def _match_folded_per_channel(
        self, graph_module: GraphModule, value: Node
    ) -> Optional[_WeightMatch]:
        if len(value.args) < 7:
            return None
        weight, scale, zero_point, axis, qmin, qmax, dtype = value.args[:7]
        if not self._is_int_literal(axis, 0):
            return None
        if not self._is_symmetric_int8_args(qmin, qmax, dtype):
            return None
        if self._static_int8_weight(graph_module, weight) is None:
            return None
        return self._make_weight_match(
            graph_module, weight, scale, zero_point, per_channel=True
        )

    def _match_folded_per_tensor(
        self, graph_module: GraphModule, value: Node
    ) -> Optional[_WeightMatch]:
        if len(value.args) < 6:
            return None
        weight, scale, zero_point, qmin, qmax, dtype = value.args[:6]
        if not self._is_symmetric_int8_args(qmin, qmax, dtype):
            return None
        if self._static_int8_weight(graph_module, weight) is None:
            return None
        return self._make_weight_match(
            graph_module, weight, scale, zero_point, per_channel=False
        )

    def _match_folded_weight(
        self, graph_module: GraphModule, value: Any
    ) -> Optional[_WeightMatch]:
        if not isinstance(value, Node) or value.op != "call_function":
            return None
        if self._node_dtype(value) is not torch.float32:
            return None
        if value.target == DEQUANT_PER_CHANNEL_OP:
            return self._match_folded_per_channel(graph_module, value)
        if value.target == DEQUANT_PER_TENSOR_OP:
            return self._match_folded_per_tensor(graph_module, value)
        return None

    def _static_source_weight_is_valid(
        self, graph_module: GraphModule, source: Any, quantized: Node
    ) -> bool:
        resolved = self._resolve_static(graph_module, source)
        return (
            isinstance(resolved, torch.Tensor)
            and resolved.ndim == 2
            and resolved.is_floating_point()
            and self._node_shape(quantized) == tuple(resolved.shape)
        )

    def _match_unfolded_per_channel(
        self, graph_module: GraphModule, dq: Node, q: Node
    ) -> Optional[_WeightMatch]:
        if len(q.args) < 7 or len(dq.args) < 7:
            return None
        if not self._same_args(tuple(q.args[1:7]), tuple(dq.args[1:7])):
            return None

        weight, scale, zero_point, axis, qmin, qmax, dtype = q.args[:7]
        if not self._is_int_literal(axis, 0):
            return None
        if not self._is_symmetric_int8_args(qmin, qmax, dtype):
            return None
        if not self._static_source_weight_is_valid(graph_module, weight, q):
            return None
        return self._make_weight_match(
            graph_module, q, scale, zero_point, per_channel=True
        )

    def _match_unfolded_per_tensor(
        self, graph_module: GraphModule, dq: Node, q: Node
    ) -> Optional[_WeightMatch]:
        if len(q.args) < 6 or len(dq.args) < 6:
            return None
        if not self._same_args(tuple(q.args[1:6]), tuple(dq.args[1:6])):
            return None

        weight, scale, zero_point, qmin, qmax, dtype = q.args[:6]
        if not self._is_symmetric_int8_args(qmin, qmax, dtype):
            return None
        if not self._static_source_weight_is_valid(graph_module, weight, q):
            return None
        return self._make_weight_match(
            graph_module, q, scale, zero_point, per_channel=False
        )

    def _match_unfolded_weight(
        self, graph_module: GraphModule, value: Any
    ) -> Optional[_WeightMatch]:
        if not isinstance(value, Node) or value.op != "call_function":
            return None
        if self._node_dtype(value) is not torch.float32 or not value.args:
            return None

        q = value.args[0]
        if not isinstance(q, Node) or q.op != "call_function":
            return None
        if self._node_dtype(q) is not torch.int8:
            return None

        if value.target == DEQUANT_PER_CHANNEL_OP and q.target == QUANT_PER_CHANNEL_OP:
            return self._match_unfolded_per_channel(graph_module, value, q)
        if value.target == DEQUANT_PER_TENSOR_OP and q.target == QUANT_PER_TENSOR_OP:
            return self._match_unfolded_per_tensor(graph_module, value, q)
        return None

    def _match_weight(
        self, graph_module: GraphModule, value: Any
    ) -> Optional[_WeightMatch]:
        folded = self._match_folded_weight(graph_module, value)
        if folded is not None:
            return folded
        return self._match_unfolded_weight(graph_module, value)

    def _linear_types_are_supported(
        self, node: Node, activation: _ActivationMatch, bias: Any
    ) -> bool:
        if self._node_dtype(node) is not torch.float32:
            return False
        if self._node_dtype(activation.dq) is not torch.float32:
            return False
        if bias is None:
            return True
        return isinstance(bias, Node) and self._node_dtype(bias) is torch.float32

    def _linear_shapes_are_supported(
        self,
        node: Node,
        activation: _ActivationMatch,
        weight: _WeightMatch,
        bias: Any,
    ) -> bool:
        x_shape = self._node_shape(activation.q)
        w_shape = self._node_shape(weight.quantized)
        out_shape = self._node_shape(node)
        if x_shape is None or w_shape is None or out_shape is None:
            return False
        if len(x_shape) < 1 or len(w_shape) != 2 or len(out_shape) != len(x_shape):
            return False
        if x_shape[-1] != w_shape[1] or out_shape[-1] != w_shape[0]:
            return False
        if bias is not None and self._node_shape(bias) != (w_shape[0],):
            return False
        return True

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.linear.default
                or len(node.args) < 2
            ):
                continue

            activation = self._match_activation(graph_module, node.args[0])
            weight = self._match_weight(graph_module, node.args[1])
            if activation is None or weight is None:
                continue

            bias = node.args[2] if len(node.args) > 2 else None
            if not self._linear_types_are_supported(node, activation, bias):
                continue
            if not self._linear_shapes_are_supported(node, activation, weight, bias):
                continue

            node.meta[ARM_DYNAMIC_W8A8_LINEAR_META_KEY] = {
                "x_q": activation.q,
                "x_scale": activation.scale,
                "x_zero_point": activation.zero_point,
                "w_q": weight.quantized,
                "w_scale": weight.scale,
                "w_zero_point": weight.zero_point,
                "w_per_channel": weight.per_channel,
                "bias": bias,
                "symmetric_zero_points": True,
                "output_dtype": torch.float32,
            }
            modified = True

        return PassResult(graph_module, modified)
