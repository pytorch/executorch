# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from executorch.backends.nxp.backend.edge_helper import (
    node_is_effectively_static_tensor,
)
from executorch.backends.nxp.backend.ir.conversion_context import ConversionContext
from executorch.backends.nxp.backend.ir.converter import quantization_utils
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NeutronTargetSpec,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tensor_formatting import TensorFormat
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    transpose_options,
)
from executorch.backends.nxp.backend.neutron_operator_support import (
    is_tensor_invariant_permutation,
    transposition_is_supported_on_neutron,
)
from executorch.backends.nxp.backend.node_format import NXP_NODE_FORMAT
from torch.fx import Node
from torch.nn import Parameter

Permutation = list[int]
PermutationSupportDict = dict[str, dict[str, bool | Permutation]]


def _get_shape(node: torch.fx.Node) -> list[int]:
    return list(node.meta["val"].shape)


def get_supported_transpositions(
    node: Node, neutron_target_spec: NeutronTargetSpec
) -> PermutationSupportDict:
    """Since ExecuTorch and NeutronIR use different tensor formats, we must consider the different possible cases
     which may occur. The main permutation is always done on channels_first/formatless data, and the output is
     channels_first/formatless as well. If this is not the case, a `Transpose` is inserted before and/or
     after the main `Transpose`, to make the input/output channels_first. These additional `Transpose`
     ops must be supported by Neutron as well. Alternatively, consecutive `Transpose` ops can be fused
     together. It is possible for a pair of unsupported permutation to result in a supported one.
     Therefore, the merged permutations must also be considered.

     This function identifies which of these permutations are supported on neutron, and returns a dictionary with the
      support summary and the corresponding permutations.

    :param node: The `permute_copy` node to base the support analysis from/
    :param neutron_target_spec: NeutronTagetSpec instance.
    :return: A dictionary containing the support status and permutation, for all the possible permutations which may be
              used during the conversion of the `node`.
    """

    input_shape = node.args[0].meta["val"].shape
    output_shape = node.meta["val"].shape
    perm = list(node.args[1])

    to_nchw_perm = translator.create_channels_last_to_channels_first_permutation(
        len(input_shape), True
    )
    to_nhwc_perm = translator.create_channels_first_to_channels_last_permutation(
        len(input_shape), True
    )
    channels_last_input_shape = translator.apply_permutation_to(
        input_shape, to_nhwc_perm
    )

    main_perm_supported = transposition_is_supported_on_neutron(
        input_shape, perm, neutron_target_spec
    )

    # "To NCHW" permutation, in case the input is channels last.
    separate_pre_transpose_supported = transposition_is_supported_on_neutron(
        channels_last_input_shape, to_nchw_perm, neutron_target_spec
    )
    # The main permutation and the previous one merged.
    merged_pre_transpose_supported = transposition_is_supported_on_neutron(
        channels_last_input_shape,
        merged_pre_transpose_permutation := translator.combine_permutations(
            to_nchw_perm, perm
        ),
        neutron_target_spec,
    )

    # "To NHWC" permutation after the main `Transpose`.
    separate_post_transpose_supported = transposition_is_supported_on_neutron(
        output_shape, to_nhwc_perm, neutron_target_spec
    )

    # The main permutation and the previous one merged.
    merged_post_transpose_supported = transposition_is_supported_on_neutron(
        input_shape,
        merged_post_transpose_permutation := translator.combine_permutations(
            perm, to_nhwc_perm
        ),
        neutron_target_spec,
    )

    # "To NCHW", main permutation, and "to NHWC" all merged.
    everything_merged_supported = transposition_is_supported_on_neutron(
        input_shape,
        everything_merged_permutation := translator.combine_permutations(
            translator.combine_permutations(to_nchw_perm, perm), to_nhwc_perm
        ),
        neutron_target_spec,
    )

    return {
        "main": {"supported": main_perm_supported, "perm": perm},
        "separate_pre": {
            "supported": separate_pre_transpose_supported,
            "perm": to_nchw_perm,
        },
        "merged_pre": {
            "supported": merged_pre_transpose_supported,
            "perm": merged_pre_transpose_permutation,
        },
        "separate_post": {
            "supported": separate_post_transpose_supported,
            "perm": to_nhwc_perm,
        },
        "merged_post": {
            "supported": merged_post_transpose_supported,
            "perm": merged_post_transpose_permutation,
        },
        "everything_merged": {
            "supported": everything_merged_supported,
            "perm": everything_merged_permutation,
        },
    }


class PermuteCopyFormatHandler:
    def __init__(self, context: ConversionContext):
        self.context = context

    @property
    def neutron_target_spec(self):
        return self.context.tflite_builder.neutron_target_spec

    @property
    def builder(self):
        return self.context.tflite_builder

    def _handle_channels_first_input_and_formatless_output(
        self, perm_dict, node, t_op, ops
    ) -> Permutation:
        # The input must be permuted.
        # Either combine the permutations, or prepend a `Transpose` operator.

        if node_is_effectively_static_tensor(
            node.args[0], self.context.parameters_mapping
        ):
            # The input is static, so the operator will be removed by an optimization.
            perm = perm_dict["main"]["perm"]

        elif perm_dict["merged_pre"]["supported"]:
            # Use the combined permutation.
            perm = perm_dict["merged_pre"]["perm"]

        elif perm_dict["separate_pre"]["supported"] and perm_dict["main"]["supported"]:
            # Prepend a `Transpose` operator to make the input channels first.
            ops.add_pre(
                self.builder.create_transpose_operator_before(
                    t_op, 0, perm_dict["separate_pre"]["perm"]
                )
            )
            perm = perm_dict["main"]["perm"]

        else:
            # The `permute_copy` cannot be represented in Neutron. This should never happen.
            raise RuntimeError(
                "A `permute_copy` node was incorrectly selected for delegation. Please report this."
            )

        t_op.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        return perm

    def _handle_formatless_input_and_channels_first_output(
        self, perm_dict, node, t_op, ops
    ) -> Permutation:
        # The output must be permuted.
        # Either combine the permutations, or append a `Transpose` operator.

        if node_is_effectively_static_tensor(
            node.args[0], self.context.parameters_mapping
        ):
            # The input is static, so the operator will be removed by an optimization.
            perm = perm_dict["main"]["perm"]

        elif perm_dict["merged_post"]["supported"]:
            # Use the combined permutation.
            perm = perm_dict["merged_post"]["perm"]

        elif perm_dict["main"]["supported"] and perm_dict["separate_post"]["supported"]:
            # Append a `Transpose` operator to make the output channels first.
            perm = perm_dict["main"]["perm"]
            ops.add_post(
                self.builder.create_transpose_operator_after(
                    t_op, 0, perm_dict["separate_post"]["perm"]
                )
            )

        else:
            # The `permute_copy` cannot be represented in Neutron. This should never happen.
            raise RuntimeError(
                "A `permute_copy` node was incorrectly selected for delegation. Please report this."
            )

        t_op.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        return perm

    def _handle_channels_first_input_and_output(
        self, perm_dict, node, t_op, ops
    ) -> Permutation:
        # Both input and output must be permuted, or some merged permutations must be supported.
        if perm_dict["everything_merged"]["supported"]:
            # Combine all 3 permutations into 1.
            perm = perm_dict["everything_merged"]["perm"]

        elif (
            perm_dict["merged_pre"]["supported"]
            and perm_dict["separate_post"]["supported"]
        ):
            # Combine the input and main permutations, and append a `Transpose` to handle the output permutation.
            perm = perm_dict["merged_pre"]["perm"]
            ops.add_post(
                self.builder.create_transpose_operator_after(
                    t_op, 0, perm_dict["separate_post"]["perm"]
                )
            )

        elif (
            perm_dict["separate_pre"]["supported"]
            and perm_dict["merged_post"]["supported"]
        ):
            # Prepend a `Transpose` to handle the input permutation, and combine the main and output permutations.
            ops.add_pre(
                self.builder.create_transpose_operator_before(
                    t_op, 0, perm_dict["separate_pre"]["perm"]
                )
            )
            perm = perm_dict["everything_merged"]["supported"]

        elif (
            perm_dict["separate_pre"]["supported"]
            and perm_dict["main"]["supported"]
            and perm_dict["separate_post"]["supported"]
        ):
            # Handle each permutation separately.
            ops.add_pre(
                self.builder.create_transpose_operator_before(
                    t_op, 0, perm_dict["separate_pre"]["perm"]
                )
            )
            perm = perm_dict["main"]["perm"]
            ops.add_post(
                self.builder.create_transpose_operator_after(
                    t_op, 0, perm_dict["separate_post"]["perm"]
                )
            )

        elif node_is_effectively_static_tensor(
            node.args[0], self.context.parameters_mapping
        ):
            perm = perm_dict["main"]["perm"]

        else:
            # The `permute_copy` cannot be represented in Neutron. This should never happen.
            raise RuntimeError(
                "A `permute_copy` node was incorrectly selected for delegation. Please report this."
            )

        t_op.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST
        t_op.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        return perm

    def _handle_formatless_input_and_output(
        self, perm_dict, node, t_op, ops
    ) -> Permutation:
        # Neither the input nor the output have to be permuted.
        if perm_dict["main"]["supported"]:
            perm = perm_dict["main"]["perm"]

        elif node_is_effectively_static_tensor(
            node.args[0], self.context.parameters_mapping
        ):
            perm = perm_dict["main"]["perm"]

        else:
            # The `permute_copy` cannot be represented in Neutron. This should never happen.
            raise RuntimeError(
                "A `permute_copy` node was incorrectly selected for delegation. Please report this."
            )

        return perm

    def handle_tensor_formats(self, t_op: tflite_model.Operator, node: Node) -> OpsList:
        """Due to the different tensor formats used by ExecuTorch and NeutronIR, it may be necessary to modify the
         permutation, or insert extra permutations to equalize the tensor formats.
        This method identifies the four possible cases of input/output formats, and finds the conversion solution
         which minimizes the number of necessary `Transpose` operators.
        """
        perm_dict = get_supported_transpositions(node, self.neutron_target_spec)

        ops = OpsList(middle_op=t_op)
        input_format, output_format = (
            node.args[0].meta[NXP_NODE_FORMAT],
            node.meta[NXP_NODE_FORMAT],
        )
        if input_format.is_channels_first() and (not output_format.is_channels_first()):
            perm = self._handle_channels_first_input_and_formatless_output(
                perm_dict, node, t_op, ops
            )

        elif (
            not input_format.is_channels_first()
        ) and output_format.is_channels_first():
            perm = self._handle_formatless_input_and_channels_first_output(
                perm_dict, node, t_op, ops
            )

        elif input_format.is_channels_first() and output_format.is_channels_first():
            perm = self._handle_channels_first_input_and_output(
                perm_dict, node, t_op, ops
            )

        else:
            perm = self._handle_formatless_input_and_output(perm_dict, node, t_op, ops)

        perm_tensor = self.builder.create_tensor_for_data(
            np.array(perm, "int32"), "perm"
        )

        # Use the final permutation as the operator's second input.
        t_op.tmp_inputs = [t_op.tmp_inputs[0], perm_tensor]

        return ops


class PermuteCopyConverter(NodeConverter):

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if node_is_effectively_static_tensor(node.args[0], parameters_mapping):
            return (
                True  # The operator computes on static data. It will be removed later.
            )

        input_shape = _get_shape(node.args[0])
        perm = list(node.args[1])

        to_nhwc_perm = translator.create_channels_first_to_channels_last_permutation(
            len(input_shape), True
        )
        channels_last_input_shape = translator.apply_permutation_to(
            input_shape, to_nhwc_perm
        )

        if is_tensor_invariant_permutation(
            input_shape, perm
        ) and is_tensor_invariant_permutation(channels_last_input_shape, perm):
            # The `permute_copy` can always be represented as a Reshape.
            return True

        perm_dict = get_supported_transpositions(node, neutron_target_spec)

        input_format, output_format = (
            node.args[0].meta[NXP_NODE_FORMAT],
            node.meta[NXP_NODE_FORMAT],
        )
        if input_format.is_channels_first() and (not output_format.is_channels_first()):
            # Just the input must be permuted.
            return (
                perm_dict["separate_pre"]["supported"]
                and perm_dict["main"]["supported"]
            ) or perm_dict["merged_pre"]["supported"]

        elif (
            not input_format.is_channels_first()
        ) and output_format.is_channels_first():
            # Just the output must be permuted.
            return (
                perm_dict["separate_post"]["supported"]
                and perm_dict["main"]["supported"]
            ) or perm_dict["merged_post"]["supported"]

        elif input_format.is_channels_first() and output_format.is_channels_first():
            # Both input and output must be permuted.
            return (
                # Separate IO transpositions.
                (
                    perm_dict["separate_pre"]["supported"]
                    and perm_dict["main"]["supported"]
                    and perm_dict["separate_post"]["supported"]
                )
                # Separate input, merged output.
                or (
                    perm_dict["separate_pre"]["supported"]
                    and perm_dict["merged_post"]["supported"]
                )
                # Merged input, separate output.
                or (
                    perm_dict["merged_pre"]["supported"]
                    and perm_dict["separate_post"]["supported"]
                )
                # Merged input and output.
                or perm_dict["everything_merged"]["supported"]
            )
        else:
            # Simplest case. No format changes required.
            return perm_dict["main"]["supported"]

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if not NodeConverter._has_shared_q_params_if_quantized(node):
            return False

        return True

    def convert(self, node: Node):
        """Convert the `aten.permute_copy` operator to TFLite `Transpose`."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = transpose_options.Transpose()

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        if (
            x.quantization is not None
            and y.quantization is None
            and "cluster" in node.meta
        ):
            # We know this node is part of QDQ cluster, so we can propagate quantization to inputs of "call_function"
            # node of this cluster.
            quantization_utils.propagate_quantization(x, y)

            y.type = x.type
            assert x.quantization == y.quantization, (
                "PermuteCopyConverter: Q-params of input and output doesn't "
                "match. This indicates error in quantizer."
            )

        ops = PermuteCopyFormatHandler(self.context).handle_tensor_formats(t_op, node)

        self.builder.append_operators(ops.flatten())
