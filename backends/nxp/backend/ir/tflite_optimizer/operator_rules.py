# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass

import executorch.backends.nxp.backend.ir.converter.builder.model_builder as model_builder
from executorch.backends.nxp.backend.ir.lib.tflite.ActivationFunctionType import (
    ActivationFunctionType,
)
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_optimizer.graph_utils import (
    NameToTensorMap,
    operator_is_type,
)
from executorch.backends.nxp.backend.ir.tflite_optimizer.optimizations.base_optimization import (
    InputTensorToOpsMap,
    OutputTensorToOpMap,
)


class OpRule(ABC):
    @abstractmethod
    def __call__(
        self,
        op: tflite_model.Operator,
        tensor_map: NameToTensorMap,
        input_to_ops_map: InputTensorToOpsMap,
        output_to_op_map: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        pass


class NoFusedActivationFunction(OpRule):

    def __call__(
        self,
        op: tflite_model.Operator,
        tensor_map: NameToTensorMap,
        input_to_ops_map: InputTensorToOpsMap,
        output_to_op_map: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        if not hasattr(op, "builtin_options"):
            return False

        if not hasattr(op.builtin_options, "fused_activation_function"):
            return False

        # noinspection PyUnresolvedReferences
        return (
            op.builtin_options.fused_activation_function == ActivationFunctionType.NONE
        )


class HasFusedActivationFunction(OpRule):

    def __call__(
        self,
        op: tflite_model.Operator,
        tensor_map: NameToTensorMap,
        input_to_ops_map: InputTensorToOpsMap,
        output_to_op_map: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        if not hasattr(op, "builtin_options"):
            return True

        if not hasattr(op.builtin_options, "fused_activation_function"):
            return True

        # noinspection PyUnresolvedReferences
        return (
            op.builtin_options.fused_activation_function != ActivationFunctionType.NONE
        )


@dataclass
class AllInputsComeFrom(OpRule):
    """Assures that all input tensors of this operator are produced by operators with op type
    `single_preceding_op_type`.
    """

    single_preceding_op_type: str

    def __call__(
        self,
        op: tflite_model.Operator,
        tensor_map: NameToTensorMap,
        input_to_ops_map: InputTensorToOpsMap,
        output_to_op_map: OutputTensorToOpMap,
        builder: "model_builder.ModelBuilder",
    ) -> bool:
        preceding_ops = [output_to_op_map[inpt.name] for inpt in op.tmp_inputs]

        return all(
            operator_is_type(preceding_op, self.single_preceding_op_type, builder)
            for preceding_op in preceding_ops
        )
