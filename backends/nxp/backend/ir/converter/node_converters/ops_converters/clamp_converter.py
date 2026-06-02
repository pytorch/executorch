# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
from executorch.backends.nxp.backend.edge_helper import try_get_arg
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    torch_type_to_numpy_type,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    _is_dequant_node,
    _is_quant_node,
    CustomDelegationOptions,
    is_not_qdq_node,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.converter.quantization_utils import (
    propagate_quantization,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    maximum_options,
    minimum_options,
)
from executorch.backends.nxp.backend.neutron_operator_support import (
    activation_supported_on_target,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.fx.passes.infra.partitioner import Partition
from torch.nn import Parameter


def _is_convertible_to_relu(node):
    bounds = ClampConverter._get_clamp_bounds(node)
    bounds = tuple(v if v is not None and math.isfinite(v) else None for v in bounds)

    # Some specific bounds can be replaced with single op ReLU.
    if bounds not in ClampConverter.RELU_COMPATIBLE_BOUNDS.values():
        return False

    return True


class ClampConverter(NodeConverter):
    RELU_COMPATIBLE_BOUNDS = {
        "ReluN1To1": (-1, 1),
        "Relu0To1": (0, 1),
        "Relu6": (0, 6),
        "Relu": (0, None),
    }

    BOUNDS_TO_RELU_NEUTRON_IR_OP = {
        (-1, 1): BuiltinOperator.RELU_N1_TO_1,
        (0, 1): BuiltinOperator.RELU_0_TO_1,
        (0, 6): BuiltinOperator.RELU6,
        (0, None): BuiltinOperator.RELU,
    }

    # noinspection PyShadowingBuiltins
    @staticmethod
    def _get_clamp_bounds(clamp_node: Node) -> tuple[float | None, float | None]:
        """Extract min and max bounds from `aten.clamp.default` node."""
        min = try_get_arg(clamp_node, 1)
        max = try_get_arg(clamp_node, 2)
        return min, max

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        # No NeutronIR-specific restrictions.
        return True

    @staticmethod
    def _io_quant_is_same(node: Node):
        quant = next(iter(node.users.keys()))
        dequant = node.args[0]

        if not _is_dequant_node(dequant):
            return False

        if not _is_quant_node(quant):
            return False

        q_params = quant.args[1:]
        dq_params = dequant.args[1:]
        return all(q == dq for q, dq in zip(q_params, dq_params))

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        relu_compatible = _is_convertible_to_relu(node)
        bounds = ClampConverter._get_clamp_bounds(node)

        if all(b is None or math.isinf(b) for b in bounds):
            return False

        if neutron_target_spec.use_new_flow_neutron_c:
            io_quant_consistent = ClampConverter._io_quant_is_same(node)
            quant_supported = NodeConverter.uses_quantization_type_for_io(
                node,
                supported_types=[torch.int8, torch.uint8],
                input_indices=[0],
                output_indices=[0],
            )

            # We either convert to ReLU -> SingleInputQuantization pattern
            # or we convert to Min/Max, which requires same quantization on
            # both input and output.
            return (relu_compatible | io_quant_consistent) and quant_supported

        return relu_compatible

    @classmethod
    def supports_partitioning_result(
        cls,
        node: Node,
        partition_list: list[Partition],
        _: CustomDelegationOptions,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        bounds = cls._get_clamp_bounds(node)

        # Neutron cannot delegate a partition where ReLU or ReLU6 is the only operator
        # and at the same time the node does not satisfy delegation requirements.
        # In contrast, ReLUN1To1 and ReLU0To1 are supported and delegated successfuly.
        if bounds in [
            cls.RELU_COMPATIBLE_BOUNDS["Relu"],
            cls.RELU_COMPATIBLE_BOUNDS["Relu6"],
        ]:
            is_alone_in_partition = cls.is_node_alone_in_partition(
                node, partition_list, filter_fn=is_not_qdq_node
            )
            if is_alone_in_partition:
                return activation_supported_on_target(node, neutron_target_spec)

        return True

    @staticmethod
    def _quantize_value(
        value: int,
        zp: int,
        scale: float,
        quant_min: int,
        quant_max: int,
        dtype: type = np.int8,
    ) -> np.integer:
        rescaled_value = round(value / scale) + zp
        return dtype(np.clip(rescaled_value, quant_min, quant_max))

    def convert(self, node: Node):
        """Convert the `aten.clamp.default` operator to either
        Neutron IR `Relu*` operator or combination of `Min` and `Max`.
        The schema is:
            aten::clamp(
                Tensor self,
                Scalar? min=None,
                Scalar? max=None
            ) -> Tensor
        """
        self.assert_convertible(node)
        to_relu = _is_convertible_to_relu(node)

        bounds = self._get_clamp_bounds(node)
        bounds = tuple(
            v if v is not None and math.isfinite(v) else None for v in bounds
        )
        t_op = self._create_tflite_op_with_io_tensors(node)

        # Clamp convertible to some variant of ReLU
        if not self.neutron_target_spec.use_new_flow_neutron_c or to_relu:
            # noinspection PyTypeChecker,PyUnboundLocalVariable
            t_op.opcode_index = self.builder.op_code_index_for_op_type(
                self.BOUNDS_TO_RELU_NEUTRON_IR_OP[bounds]
            )
            self.builder.append_operators([t_op])
            return

        q_node = node.args[0]
        assert _is_dequant_node(q_node)
        _, scale, zp, quant_min, quant_max, q_type = q_node.args
        q_type = torch_type_to_numpy_type(q_type).type

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        if x.quantization is not None and y.quantization is None:
            propagate_quantization(x, y)

        min_value, max_value = bounds

        if min_value is not None:
            min_value = self._quantize_value(
                value=min_value,
                zp=zp,
                scale=scale,
                quant_min=quant_min,
                quant_max=quant_max,
                dtype=q_type,
            )
            min_tensor = self.builder.create_tensor_for_data(
                np.array([min_value], q_type), "min"
            )
            propagate_quantization(x, min_tensor)

        if max_value is not None:
            max_value = self._quantize_value(
                value=max_value,
                zp=zp,
                scale=scale,
                quant_min=quant_min,
                quant_max=quant_max,
                dtype=q_type,
            )
            max_tensor = self.builder.create_tensor_for_data(
                np.array([max_value], q_type), "max"
            )
            propagate_quantization(x, max_tensor)

        if None not in bounds:
            tmp_y = self.builder.duplicate_tensor(x)
            tmp_x = tmp_y
            propagate_quantization(x, tmp_y)
        else:
            tmp_y = y
            tmp_x = x

        ops_to_add = []
        if max_value is not None:
            min_op = tflite_model.Operator(builtin_options=minimum_options.Minimum())
            min_op.tmp_inputs = [x, max_tensor]
            min_op.tmp_outputs = [tmp_y]
            ops_to_add.append(min_op)

        if min_value is not None:
            max_op = tflite_model.Operator(builtin_options=maximum_options.Maximum())
            max_op.tmp_inputs = [tmp_x, min_tensor]
            max_op.tmp_outputs = [y]
            ops_to_add.append(max_op)

        self.builder.append_operators(ops_to_add)
