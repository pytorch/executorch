# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.logger as logger
import flatbuffers
from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig
from executorch.backends.nxp.backend.ir.conversion_context import ConversionContext
from executorch.backends.nxp.backend.ir.converter.builder.aten_model_builder_director import (
    AtenModelBuilderDirector,
)
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx import Node
from torch.nn.parameter import Parameter
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import *  # noqa F403
from executorch.backends.nxp.backend.node_format_inference import (
    NodeFormat,
    NodeFormatInference,
)
from executorch.exir.dialects._ops import ops as exir_ops

# noinspection PyProtectedMember
functions_converters = {
    exir_ops.edge.aten.addmm.default: AddMMConverter,  # noqa F405
    exir_ops.edge.aten.avg_pool2d.default: AvgPool2dConverter,  # noqa F405
    exir_ops.edge.aten.constant_pad_nd.default: ConstantPadNDConverter,  # noqa F405
    exir_ops.edge.aten.convolution.default: ConvolutionConverter,  # noqa F405
    exir_ops.edge.aten.max_pool2d.default: MaxPool2dConverter,  # noqa F405
    exir_ops.edge.aten.mm.default: MMConverter,  # noqa F405
    exir_ops.edge.aten.permute_copy.default: PermuteCopyConverter,  # noqa F405
    exir_ops.edge.aten.relu.default: ReLUConverter,  # noqa F405
    exir_ops.edge.aten._softmax.default: SoftmaxConverter,  # noqa F405
    exir_ops.edge.aten.view_copy.default: ViewCopyConverter,  # noqa F405
}


class EdgeProgramToIRConverter:
    """
    Converter from convertion of ExportedProgram in Edge dialect to IR (TFLite Flatbuffers).
    """

    _default_conversion_config = ConversionConfig()

    def convert_program(
        self,
        edge_program: ExportedProgram,
        conversion_config=_default_conversion_config,
    ) -> (bytes, dict):
        """
        Convert ExportedProgram in Edge dialect to IR (TFLite flatbuffers) as bytes.

        :param edge_program: Converter ExportedProgram.
        :param conversion_config: ConversionConfig instance.
        :return: TFLite flatbuffers as bytes.
        """
        node_formats = NodeFormatInference(edge_program).identify_node_formats()
        parameters_mapping = self.map_inputs_to_parameters(edge_program)

        cc = self.build_conversion_context(
            parameters_mapping, node_formats, conversion_config
        )

        # Program conversion
        self.append_placeholders_and_tensors(edge_program.graph.nodes, cc)
        self._convert_qdq_cluster_q_dq_nodes(edge_program.graph.nodes, cc)
        self._process_nodes(edge_program.graph.nodes, cc)

        # Assign output
        io_formats = cc.tflite_builder.assign_model_io_to_subgraph_and_get_io_formats(
            edge_program.graph_signature
        )

        # TFLite model generation
        internal_tflite_model = cc.tflite_builder.finish()
        flatbuffers_builder = flatbuffers.Builder()
        internal_tflite_model.gen_tflite(flatbuffers_builder)

        return bytes(flatbuffers_builder.Output()), io_formats

    @staticmethod
    def append_placeholders_and_tensors(nodes: list[Node], context: ConversionContext):
        for node in nodes:
            if node.op == "placeholder":
                node_format = context.node_formats[node]

                if node.name in context.parameters_mapping:
                    # Node is placeholder and has data -> append as static tensor with data
                    tensor = context.parameters_mapping[node.name]
                    context.tflite_builder.append_as_static_tensor(
                        node, node_format, tensor
                    )
                else:
                    # Node is placeholder and doesn't have data (user input) -> append as fake tensor
                    context.tflite_builder.append_as_fake_tensor(node, node_format)
            elif node.op == "call_function":
                # Node is call function -> append only output as a tensor
                node_format = context.node_formats[node]
                context.tflite_builder.append_as_fake_tensor(node, node_format)
            elif node.op == "output":
                # Nothing to do
                pass
            else:
                logger.e(
                    logger.Code.INTERNAL_ERROR, f"Unexpected node op type: '{node.op}'!"
                )

    def _process_nodes(self, nodes: list[Node], conversion_context: ConversionContext):
        """
        Go through program nodes and append their TFLite siblings into ModelBuilder.

        :param nodes: Program's nodes.
        :param conversion_context: ConversionContext instance.
        """

        qdq_related_functions = [
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        ]

        for node in nodes:
            if node.op == "call_function":
                if node.target in qdq_related_functions and "cluster" in node.meta:
                    # Skip (De)Quantize nodes that were already processed
                    pass
                elif node.target in functions_converters:
                    functions_converters[node.target](conversion_context).convert(node)
                else:
                    logger.e(
                        logger.Code.NOT_IMPLEMENTED,
                        f"Converter for '{node.target.__name__}' not implemented!",
                    )

    @staticmethod
    def map_inputs_to_parameters(edge_program: ExportedProgram) -> dict[str, Parameter]:
        """
        Create mapping between program parameters (input nodes & static data nodes) and their names.

        :param edge_program: EdgeProgram instance.
        :return: Mapping from parameter name to parameter instance.
        """
        result_map = {}

        for input_spec in edge_program.graph_signature.input_specs:
            if input_spec.kind in [InputKind.PARAMETER, InputKind.BUFFER]:
                result_map[input_spec.arg.name] = edge_program.state_dict[
                    input_spec.target
                ]

        return result_map

    @staticmethod
    def build_conversion_context(
        parameters_mapping: dict,
        node_formats: dict[Node, NodeFormat],
        conversion_config: ConversionConfig = _default_conversion_config,
    ) -> ConversionContext:
        tflite_builder = AtenModelBuilderDirector(
            3, "TFLite from EdgeProgram", conversion_config
        )

        # Add "sentinel" buffer (defined in schema.fbs)
        tflite_builder.build_empty_buffer()

        context = ConversionContext(
            tflite_builder, conversion_config, parameters_mapping, node_formats
        )

        return context

    def _convert_qdq_cluster_q_dq_nodes(
        self, nodes: list[Node], conversion_context: ConversionContext
    ):
        """
        Go through program and convert De(Quantize) nodes that are part of the QDQ cluster into
        tensors.

        :param nodes: Program's nodes.
        :param conversion_context: ConversionContext instance.
        """
        qdq_q_ops_converters = {
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: QDQDequantizeConverter,  # noqa F405
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: QDQQuantizeConverter,  # noqa F405
        }

        for node in nodes:
            part_of_qdq_cluster = "cluster" in node.meta
            if (
                node.op == "call_function"
                and node.target in qdq_q_ops_converters
                and part_of_qdq_cluster
            ):
                qdq_q_ops_converters[node.target](conversion_context).convert(node)
