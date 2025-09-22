# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir.converter.builder.model_builder import (
    ModelBuilder,
)
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.tensor_formatting import TensorFormat
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.node_format import NodeFormat
from torch.fx import Node
from torch.nn import Parameter


class AtenModelBuilderDirector(ModelBuilder):
    """
    ModelBuilder's extension that simplifies some actions during build process. It also
    contains methods related to Edge program nodes conversion.
    """

    def append_as_fake_tensor(self, node: Node, node_format: NodeFormat):
        """
        Append node into ModelBuilder as tensor without data (FakeTensor). Can be used
        for activations and output tensors.

        :param node: Node instance.
        :param node_format: NodeFormat definition.
        """
        if self.tensor_exists(node.name):
            return

        tensor = node.meta["val"]
        if isinstance(tensor, tuple):
            tensor = tensor[0]  # Fake tensor
        _type = translator.convert_data_type(tensor.dtype)
        shape = list(tensor.shape)

        if node_format.is_channels_first():
            shape = translator.dims_to_channels_last(shape)

        tensor = self.create_empty_tensor(node.name, _type, shape)
        tensor.tensor_format = TensorFormat.from_node_format(node_format)

    def append_as_static_tensor(
        self, node: Node, node_format: NodeFormat, tensor: Parameter
    ):
        """
        Append node into ModelBuilder as tensor with data (static). Can be used for weights,
        permutations etc.

        :param node: Node instance.
        :param node_format: NodeFormat definition.
        :param tensor: Torch Tensor (Parameter) that holds tensor data.
        """
        assert not self.tensor_exists(node.name), f"Tensor '{node.name}' already added!"

        if self.tensor_exists(node.name):
            return

        data = tensor.data.numpy()

        if node_format.is_channels_first():
            data = translator.convert_data_to_channels_last(data)

        tensor = self.create_tensor_for_data(data, node.name)
        tensor.tensor_format = TensorFormat.from_node_format(node_format)

    def append_operators(self, ops_to_add: list[tflite_model.Operator]):
        """
        Append list of TFLite operators to created model via ModelBuilder.

        :param ops_to_add: List of operators to be added.
        """
        for op in ops_to_add:
            if op.builtin_options is not None:
                op.opcode_index = self.op_code_index_for_op_type(
                    op.builtin_options.operator_type, op.tmp_version
                )

            elif op.custom_options is not None:
                op.opcode_index = self.op_code_index_for_op_type(
                    op.custom_options.operator_type,
                    op.tmp_version,
                    op.custom_options.custom_code,
                )

            self.check_and_append_operator(op)

    def assign_model_io_to_subgraph_and_get_io_formats(
        self, graph_signature
    ) -> dict[str, dict]:
        """
        Assign model's inputs/outputs to SubGraph.

        :param graph_signature: Instance of GraphSignature.
        :returns: Mapping between IO tensors' names and their formats.
        """
        io_formats = {
            "inputs": {},
            "outputs": {},
        }

        self.get_sub_graph().inputs = tflite_model.SubGraphInputs()
        for input_name in graph_signature.user_inputs:
            tensor = self.tensor_for_name(input_name)
            assert input_name == tensor.name, (
                "Program's input name doesn't match with tensor name in TFLite. "
                "Input was probably redirected."
            )
            self.get_sub_graph().inputs.tmp_inputs.append(tensor)
            io_formats["inputs"][tensor.name] = tensor.tensor_format

        self.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
        for output_name in graph_signature.user_outputs:
            tensor = self.tensor_for_name(output_name)
            assert output_name == tensor.name, (
                "Program's output name doesn't match with tensor name in TFLite. "
                "Output was probably redirected."
            )
            self.get_sub_graph().outputs.tmp_outputs.append(tensor)

            io_formats["outputs"][tensor.name] = tensor.tensor_format

        return io_formats
