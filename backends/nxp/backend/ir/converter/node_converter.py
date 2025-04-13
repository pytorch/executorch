# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from enum import Enum
from typing import Collection

import torch

from executorch.backends.nxp.backend.ir.conversion_context import ConversionContext
from executorch.backends.nxp.backend.ir.converter.builder.aten_model_builder_director import (
    AtenModelBuilderDirector,
)
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node
from torch.nn import Parameter


def _is_quant_node(node: torch.fx.Node) -> bool:
    return node.target in [
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
    ]


def _is_dequant_node(node: torch.fx.Node) -> bool:
    return node.target in [
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
    ]


class Target(Enum):
    IGNORE = "ignore"  # No target platform. Any target specific restrictions will be ignored.

    RT700 = "imxrt700"
    IMX95 = "imx95"

    @classmethod
    def values(cls) -> list[str]:
        return [elt.value for elt in cls]


class NodeConverter(ABC):
    """
    Classes which implement conversion of torch.Node to TFLite should inherit from this class and overwrite the
     'convert()' method.
    """

    context: ConversionContext
    supported_targets: Collection

    def __init__(self, context: ConversionContext):
        self.context = context

    @abstractmethod
    def convert(self, node: Node):
        """Convert the torch.Node in 'node' to TFLite and append changes to ModelBuilder.

            Classes which implement conversion for individual operators must overwrite this method.

        :param node: torch.Node to convert.
        """
        pass

    # noinspection PyPep8Naming
    @staticmethod
    @abstractmethod
    def _is_supported_in_IR(
        node: Node, parameters_mapping: dict[str, Parameter]
    ) -> bool:
        """Check if the `node` can be converted to the intermediate representation.
            Classes which implement conversion for individual operators must overwrite this method.

        :param node: torch.Node to check.
        """
        pass

    @classmethod
    def _is_supported_on_target(cls, target: Target) -> bool:
        """Check if the node is supported on the target platform. It uses the 'supported_platform' attribute, which is
             a list of supported target platforms, and it must be defined by the specific `NodeConverter`.

        :param target: Value of the `Target` enum representing the target platform to check for.
        """
        if not (
            hasattr(cls, "supported_targets")
            and isinstance(cls.supported_targets, Collection)
        ):
            raise NotImplementedError(
                f"The NodeConverter `{cls}` does not define its `supported_targets` collection."
            )

        return target == Target.IGNORE or target in cls.supported_targets

    @classmethod
    def is_supported(
        cls, node: Node, target: Target, parameters_mapping: dict[str, Parameter]
    ) -> bool:
        """Check if the given `node` is supported in the IR and on the given `target` platform.

        :param node: torch.Node to check.
        :param target: Value of the `Target` enum representing the target platform to check for.
        :param parameters_mapping: Dict mapping tensor names to their data.
        """
        return cls._is_supported_in_IR(
            node, parameters_mapping
        ) and cls._is_supported_on_target(target)

    @staticmethod
    def _has_shared_q_params_if_quantized(node: Node) -> bool:
        """Check if node has shared quantization parameters if it's quantized."""
        if len(node.users) < 1 or len(node.all_input_nodes) < 1:
            # Some exotic operator (only consumer or only produces)
            return True

        pre_node = node.prev
        post_node = node.next

        if pre_node.name == node.all_input_nodes[0] and post_node.name == node.users[0]:
            raise RuntimeError(
                "Prev & next nodes are not the same as inputs and outputs."
            )

        if _is_dequant_node(pre_node) and _is_quant_node(post_node):
            # Node is quantized
            pre_zp = pre_node.args[1]
            pre_scale = pre_node.args[2]
            pre_type = pre_node.args[5]

            post_zp = post_node.args[1]
            post_scale = post_node.args[2]
            post_type = pre_node.args[5]

            # Q-params match?
            return (
                pre_zp == post_zp and pre_scale == post_scale and pre_type == post_type
            )

        # Node not quantized
        return True

    def assert_convertible(self, node):
        """Assert that the call `_is_supported_in_IR()` returns `True`. Otherwise, raise an exception and print an
        error message.
        """
        assert self._is_supported_in_IR(node, self.context.parameters_mapping), (
            f"Node `{node}` is not convertible to the intermediate representation. "
            "There is an error in the partitioner."
        )

    @property
    def builder(self) -> AtenModelBuilderDirector:
        """
        Get instance of TFLite ModelBuilder from conversion context.
        :return: AtenModelBuilderDirector instance.
        """
        return self.context.tflite_builder

    def _create_tflite_op_with_io_tensors(self, node: Node) -> tflite_model.Operator:
        """
        Create TFLite op wrapper with input/output tensors added into 'tmp_inputs' and 'tmp_outputs'.

        :param node: Node instance.
        :return: TFLite operator with assigned input/output tensors.
        """
        t_operator = tflite_model.Operator()

        # Initialize node's inputs
        t_operator.inputs = tflite_model.OperatorInputs()
        for ancestor_node in node.all_input_nodes:
            assert self.context.tflite_builder.tensor_exists(ancestor_node.name)
            t_operator.tmp_inputs.append(
                self.context.tflite_builder.tensor_for_name(ancestor_node.name)
            )

        # Add node's output as a new tensor
        assert self.context.tflite_builder.tensor_exists(node.name)
        t_operator.outputs = tflite_model.OperatorOutputs()
        t_operator.tmp_outputs.append(
            self.context.tflite_builder.tensor_for_name(node.name)
        )

        return t_operator
