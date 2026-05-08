# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from abc import ABC, abstractmethod
from typing import Callable

import torch

from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.edge_helper import (
    input_quantization_type,
    output_quantization_type,
)
from executorch.backends.nxp.backend.ir.conversion_context import ConversionContext
from executorch.backends.nxp.backend.ir.converter.builder.aten_model_builder_director import (
    AtenModelBuilderDirector,
)

from executorch.backends.nxp.backend.ir.converter.tensor_utils import (
    get_name_of_node_output,
)
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node
from torch.fx.passes.infra.partitioner import Partition
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


def is_not_qdq_node(node: torch.fx.Node) -> bool:
    return not (_is_quant_node(node) or _is_dequant_node(node))


class NodeConverter(ABC):
    """
    Classes which implement conversion of torch.Node to TFLite should inherit from this class and overwrite the
     'convert()' method.
    """

    context: ConversionContext

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
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        """Check if the `node` can be converted to the intermediate representation.
            Classes which implement conversion for individual operators must overwrite this method.

        :param node: torch.Node to check.
        :param parameters_mapping: Dictionary mapping static parameter names to Parameter objects containing their data
                                    (if they have any). During partitioning, this data is extracted from the model right
                                    after quantization and before edge dialect passes. Therefore, it could potentially
                                    be outdated.
        :param custom_delegation_options: Custom options which affect delegation.
        """
        pass

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        """Check if the node is supported on the target platform.
            Child classes should overwrite this method to implement specific target checks. The default implementation
            can be used by operators with no target specific requirements.

        :param node: The node (edge operator) to check.
        :param neutron_target_spec: Object for querying the target platform to retrieve its properties.
        :param parameters_mapping: Dictionary mapping static parameter names to Parameter objects containing their data
                                    (if they have any). During partitioning, this data is extracted from the model right
                                    after quantization and before edge dialect passes. Therefore, it could potentially
                                    be outdated.
        :param custom_delegation_options: Custom options which affect delegation.
        """
        return True

    @classmethod
    def is_supported(
        cls,
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        """Check if the given `node` is supported in the IR and on the given `target` platform.

        :param node: torch.Node to check.
        :param neutron_target_spec: Object for querying the target platform to retrieve its properties.
        :param parameters_mapping: Dictionary mapping static parameter names to Parameter objects containing their data
                                    (if they have any). During partitioning, this data is extracted from the model right
                                    after quantization and before edge dialect passes. Therefore, it could potentially
                                    be outdated.
        :param custom_delegation_options: Custom user options which affect node delegation.
        """
        return cls._is_supported_in_IR(
            node, parameters_mapping, custom_delegation_options
        ) and cls._is_supported_on_target(
            node, neutron_target_spec, parameters_mapping, custom_delegation_options
        )

    @classmethod
    def supports_partitioning_result(
        cls,
        node: Node,
        partition_list: list[Partition],
        custom_delegation_options: CustomDelegationOptions,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        """Check if the given `node` supports the assigned partitioning, which is stored  the `partition_list`. Child
            classes can overwrite this method in case they have delegation restrictions based on the context defined by
            the partitioning result.

        :param node: torch.Node to check.
        :param partition_list: List of proposed partitions.
        :param custom_delegation_options: Custom user options which affect node delegation.
        :param neutron_target_spec: NeutronTargetSpec instance.
        :param parameters_mapping: Dictionary mapping static parameter names to Parameter objects containing their data
                                    (if they have any). During partitioning, this data is extracted from the model right
                                    after quantization and before edge dialect passes. Therefore, it could potentially
                                    be outdated.
        :return: Boolean indicating whether the node supports the current partitioning.
        """
        return True

    @staticmethod
    def _has_shared_q_params_if_quantized(node: Node) -> bool:
        """Check if node has shared quantization parameters if it's quantized."""
        if len(node.users) < 1 or len(node.all_input_nodes) < 1:
            # Some exotic operator (only consumer or only produces)
            return True

        pre_node = node.all_input_nodes[0]
        post_node = list(node.users)[0]

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

    @staticmethod
    def is_node_alone_in_partition(
        node: Node, partition_list: list[Partition], filter_fn: Callable[[Node], bool]
    ) -> bool:
        """Return True if `node` is the only node in its partition for which `filter_fn`
        returns True.

        The function finds the unique partition containing `node` and applies
        `filter_fn` to all nodes in that partition. If only one node passes the
        predicate — and that node is `node` — the function returns True.

        :param node: The torch.fx.Node to check.
        :param partition_list: List of proposed partitions.
        :param filter_fn: Predicate applied to nodes in the partition.
                        `node` is considered alone if it is the only node
                        for which this predicate returns True.
        """
        partitions = [p for p in partition_list if node in p.nodes]
        if len(partitions) != 1:
            raise ValueError(
                "Cannot find a partition of a node in graph. This should not occur."
            )

        partition = partitions[0]
        filtered_partition_nodes = list(filter(filter_fn, partition.nodes))
        return (
            len(filtered_partition_nodes) == 1 and filtered_partition_nodes[0] == node
        )

    def assert_convertible(self, node):
        """Assert that the call `is_supported()` returns `True`. Otherwise, raise an exception and print an
        error message.
        """
        supported_in_ir = self._is_supported_in_IR(
            node,
            self.context.parameters_mapping,
            self.context.custom_delegation_options,
        )

        supported_on_target = self._is_supported_on_target(
            node,
            self.neutron_target_spec,
            self.context.parameters_mapping,
            self.context.custom_delegation_options,
        )

        assert supported_in_ir and supported_on_target, (
            f"Node `{node}` was selected for delegation to Neutron, but it is not convertible to the intermediate "
            "representation. There is an error in the Neutron partitioner. Please report this."
        )

    @property
    def builder(self) -> AtenModelBuilderDirector:
        """
        Get instance of TFLite ModelBuilder from conversion context.
        :return: AtenModelBuilderDirector instance.
        """
        return self.context.tflite_builder

    @property
    def neutron_target_spec(self) -> NeutronTargetSpec:
        """
        Get an instance of NeutronTargetSpec from the conversion context.
        :return: NeutronTargetSpec instance.
        """
        return self.builder.neutron_target_spec

    def _create_tflite_op_with_io_tensors(self, node: Node) -> tflite_model.Operator:
        """
        Create TFLite op wrapper with input/output tensors added into 'tmp_inputs' and 'tmp_outputs'.

        :param node: Node instance.
        :return: TFLite operator with assigned input/output tensors.
        """
        t_operator = tflite_model.Operator()

        # Initialize node's inputs
        t_operator.inputs = tflite_model.OperatorInputs()

        if node.target == operator.getitem:
            # Special case of a builtin function, which can extract a specific output tensor from the previous node.
            previous_node = node.args[0]
            output_index = node.args[1]
            input_name = get_name_of_node_output(previous_node, output_index)
            assert self.builder.tensor_exists(input_name)
            t_operator.tmp_inputs.append(self.builder.tensor_for_name(input_name))

        else:
            # Regular operator.
            input_nodes = []
            for arg in node.args:
                match arg:
                    case Node():
                        input_nodes.append(arg)
                    case list() if all(isinstance(node_, Node) for node_ in arg):
                        input_nodes.extend(arg)

            for ancestor_node in input_nodes:
                assert self.builder.tensor_exists(ancestor_node.name)
                t_operator.tmp_inputs.append(
                    self.builder.tensor_for_name(ancestor_node.name)
                )

        # Add node's outputs as a new tensors
        num_outputs = (
            len(node.meta["val"]) if isinstance(node.meta["val"], tuple) else 1
        )
        if num_outputs == 1:
            # Single output node.
            assert self.builder.tensor_exists(node.name)
            t_operator.outputs = tflite_model.OperatorOutputs()
            t_operator.tmp_outputs.append(self.builder.tensor_for_name(node.name))
        else:
            # The node has multiple outputs.
            t_operator.outputs = tflite_model.OperatorOutputs()
            for output_index in range(num_outputs):
                tensor_name = get_name_of_node_output(node, output_index)
                assert self.builder.tensor_exists(tensor_name)
                t_operator.tmp_outputs.append(self.builder.tensor_for_name(tensor_name))

        return t_operator

    @staticmethod
    def uses_quantization_type_for_inputs(
        node: Node,
        supported_types: list[torch.dtype],
        input_indices: list[int | tuple[int, int]],
    ) -> bool:
        """Check if `node` uses the QDQ quantization schema and inputs on the provided indices use a quantization type
            that is in `supported_types`.

        :param node: The compute node.
        :param supported_types: List of supported quantization types.
        :param input_indices: List of indices into the `node.args`, or tuples of 2 indices into `node.args[idx1][idx2]`.
        :return: True, if the `node` is QDQ quantized and has quantization input types in `supported_types`.
        """
        return all(
            input_quantization_type(node, input_index) in supported_types
            for input_index in input_indices
        )

    @staticmethod
    def uses_quantization_type_for_outputs(
        node: Node,
        supported_types: list[torch.dtype],
        output_indices: list[int] | None = None,
    ):
        """Check if `node` uses the QDQ quantization schema and outputs on the provided indices use a quantization type
            that is in `supported_types`.

        :param node: The compute node.
        :param supported_types: List of supported quantization types.
        :param output_indices: If the `node` has multiple outputs and therefore multiple `getitem` nodes follow it, the
                                indices select the outputs to be checked.
        :return: True, if the `node` is QDQ quantized and has quantization output types in `supported_types`.
        """
        if output_indices is None:
            return output_quantization_type(node) in supported_types
        else:
            return all(
                output_quantization_type(node, output_index) in supported_types
                for output_index in output_indices
            )

    @staticmethod
    def uses_quantization_type_for_io(
        node: Node,
        supported_types: list[torch.dtype],
        input_indices: list[int | tuple[int, int]],
        output_indices: list[int] | None = None,
    ):
        """Check if `node` uses the QDQ quantization schema and inputs and outputs on the provided indices use a
            quantization type that is in `supported_types`.

        :param node: The compute node.
        :param supported_types: List of supported quantization types.
        :param input_indices: List of indices into the `node.args`, or tuples of 2 indices into `node.args[idx1][idx2]`.
        :param output_indices: If the `node` has multiple outputs and therefore multiple `getitem` nodes follow it, the
                                indices select the outputs to be checked.
        :return: True, if the `node` is QDQ quantized and has quantization input types in `supported_types`.
        """
        return NodeConverter.uses_quantization_type_for_inputs(
            node, supported_types, input_indices
        ) and NodeConverter.uses_quantization_type_for_outputs(
            node, supported_types, output_indices
        )
