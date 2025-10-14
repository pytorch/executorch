# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from executorch.backends.nxp.backend.edge_helper import (
    node_is_effectively_static_tensor,
)
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


def _get_shape(node: torch.fx.Node) -> list[int]:
    return list(node.meta["val"].shape)


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
        output_shape = _get_shape(node)

        # Since ExecuTorch and NeutronIR use different tensor formats, we must consider the different possible cases
        #  which may occur. The main permutation is always done on channels_first/formatless data, and the output is
        #  channels_first/formatless as well. If this is not the case, a `Transpose` is inserted before and/or
        #  after the main `Transpose`, to make the input/output channels_first. These additional `Transpose`
        #  ops must be supported by Neutron as well. Alternatively, consecutive `Transpose` ops can be fused
        #  together. It is possible for a pair of unsupported permutation to result in a supported one.
        #  Therefore, the merged permutations must also be considered.
        to_nchw_perm = translator.create_channels_last_to_channels_first_permutation(
            len(input_shape), True
        )
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
            translator.combine_permutations(to_nchw_perm, perm),
            neutron_target_spec,
        )
        # "To NHWC" permutation after the main `Transpose`.
        separate_post_transpose_supported = transposition_is_supported_on_neutron(
            output_shape, to_nhwc_perm, neutron_target_spec
        )
        # The main permutation and the previous one merged.
        merged_post_transpose_supported = transposition_is_supported_on_neutron(
            input_shape,
            translator.combine_permutations(perm, to_nhwc_perm),
            neutron_target_spec,
        )
        # "To NCHW", main permutation, and "to NHWC" all merged.
        everything_merged_supported = transposition_is_supported_on_neutron(
            input_shape,
            translator.combine_permutations(
                translator.combine_permutations(to_nchw_perm, perm), to_nhwc_perm
            ),
            neutron_target_spec,
        )

        input_format, output_format = (
            node.args[0].meta[NXP_NODE_FORMAT],
            node.meta[NXP_NODE_FORMAT],
        )
        if input_format.is_channels_first() and (not output_format.is_channels_first()):
            # Just the input must be permuted.
            return (
                separate_pre_transpose_supported and main_perm_supported
            ) or merged_pre_transpose_supported

        elif (
            not input_format.is_channels_first()
        ) and output_format.is_channels_first():
            # Just the output must be permuted.
            return (
                separate_post_transpose_supported and main_perm_supported
            ) or merged_post_transpose_supported

        elif input_format.is_channels_first() and output_format.is_channels_first():
            # Both input and output must be permuted.
            return (
                # Separate IO transpositions.
                (
                    separate_pre_transpose_supported
                    and main_perm_supported
                    and separate_post_transpose_supported
                )
                # Separate input, merged output.
                or (
                    separate_pre_transpose_supported and merged_post_transpose_supported
                )
                # Merged input, separate output.
                or (
                    merged_pre_transpose_supported and separate_post_transpose_supported
                )
                # Merged input and output.
                or everything_merged_supported
            )
        else:
            # Simplest case. No format changes required.
            return main_perm_supported

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if not NodeConverter._has_shared_q_params_if_quantized(node):
            return False

        return True

    def handle_tensor_formats(self, t_op: tflite_model.Operator, node: Node) -> OpsList:
        """Due to the different tensor formats used by ExecuTorch and NeutronIR, it may be necessary to modify the
         permutation, or insert extra permutations to equalize the tensor formats.
        This method identifies the four possible cases of input/output formats, and finds the conversion solution
         which minimizes the number of necessary `Transpose` operators.
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
            input_shape, perm, self.neutron_target_spec
        )

        # "To NCHW" permutation, in case the input is channels last.
        separate_pre_transpose_supported = transposition_is_supported_on_neutron(
            channels_last_input_shape, to_nchw_perm, self.neutron_target_spec
        )
        # The main permutation and the previous one merged.
        merged_pre_transpose_supported = transposition_is_supported_on_neutron(
            channels_last_input_shape,
            merged_pre_transpose_permutation := translator.combine_permutations(
                to_nchw_perm, perm
            ),
            self.neutron_target_spec,
        )

        # "To NHWC" permutation after the main `Transpose`.
        separate_post_transpose_supported = transposition_is_supported_on_neutron(
            output_shape, to_nhwc_perm, self.neutron_target_spec
        )

        # The main permutation and the previous one merged.
        merged_post_transpose_supported = transposition_is_supported_on_neutron(
            input_shape,
            merged_post_transpose_permutation := translator.combine_permutations(
                perm, to_nhwc_perm
            ),
            self.neutron_target_spec,
        )

        # "To NCHW", main permutation, and "to NHWC" all merged.
        everything_merged_supported = transposition_is_supported_on_neutron(
            input_shape,
            everything_merged_permutation := translator.combine_permutations(
                translator.combine_permutations(to_nchw_perm, perm), to_nhwc_perm
            ),
            self.neutron_target_spec,
        )

        ops = OpsList(middle_op=t_op)
        input_format, output_format = (
            node.args[0].meta[NXP_NODE_FORMAT],
            node.meta[NXP_NODE_FORMAT],
        )
        if input_format.is_channels_first() and (not output_format.is_channels_first()):
            # The input must be permuted.
            # Either combine the permutations, or prepend a `Transpose` operator.
            if merged_pre_transpose_supported:
                # Use the combined permutation.
                perm = merged_pre_transpose_permutation
            elif separate_pre_transpose_supported and main_perm_supported:
                # Prepend a `Transpose` operator to make the input channels first.
                ops.add_pre(
                    self.builder.create_transpose_operator_before(t_op, 0, to_nchw_perm)
                )
            elif not node_is_effectively_static_tensor(
                node.args[0], self.context.parameters_mapping
            ):
                # The `permute_copy` cannot be represented in Neutron. This should never happen.
                raise RuntimeError(
                    "A `permute_copy` node was incorrectly selected for delegation. Please report this."
                )

            t_op.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        elif (
            not input_format.is_channels_first()
        ) and output_format.is_channels_first():
            # The output must be permuted.
            # Either combine the permutations, or append a `Transpose` operator.
            if merged_post_transpose_supported:
                # Use the combined permutation.
                perm = merged_post_transpose_permutation
            elif main_perm_supported and separate_post_transpose_supported:
                # Append a `Transpose` operator to make the output channels first.
                ops.add_post(
                    self.builder.create_transpose_operator_after(t_op, 0, to_nhwc_perm)
                )
            elif not node_is_effectively_static_tensor(
                node.args[0], self.context.parameters_mapping
            ):
                # The `permute_copy` cannot be represented in Neutron. This should never happen.
                raise RuntimeError(
                    "A `permute_copy` node was incorrectly selected for delegation. Please report this."
                )

            t_op.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        elif input_format.is_channels_first() and output_format.is_channels_first():
            # Both input and output must be permuted, or some merged permutations must be supported.
            if everything_merged_supported:
                # Combine all 3 permutations into 1.
                perm = everything_merged_permutation
            elif merged_pre_transpose_supported and separate_post_transpose_supported:
                # Combine the input and main permutations, and append a `Transpose` to handle the output permutation.
                perm = merged_pre_transpose_permutation
                ops.add_post(
                    self.builder.create_transpose_operator_after(t_op, 0, to_nhwc_perm)
                )
            elif separate_pre_transpose_supported and merged_post_transpose_supported:
                # Prepend a `Transpose` to handle the input permutation, and combine the main and output permutations.
                ops.add_pre(
                    self.builder.create_transpose_operator_before(t_op, 0, to_nchw_perm)
                )
                perm = merged_post_transpose_permutation
            elif (
                separate_pre_transpose_supported
                and main_perm_supported
                and separate_post_transpose_supported
            ):
                # Handle each permutation separately.
                ops.add_pre(
                    self.builder.create_transpose_operator_before(t_op, 0, to_nchw_perm)
                )
                perm = perm  # The main permutation remains unchanged.
                ops.add_post(
                    self.builder.create_transpose_operator_after(t_op, 0, to_nhwc_perm)
                )
            elif not node_is_effectively_static_tensor(
                node.args[0], self.context.parameters_mapping
            ):
                # The `permute_copy` cannot be represented in Neutron. This should never happen.
                raise RuntimeError(
                    "A `permute_copy` node was incorrectly selected for delegation. Please report this."
                )

            t_op.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST
            t_op.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        else:
            # Neither the input nor the output have to be permuted.
            if main_perm_supported:
                perm = perm  # The main permutation remains unchanged.
            elif not node_is_effectively_static_tensor(
                node.args[0], self.context.parameters_mapping
            ):
                # The `permute_copy` cannot be represented in Neutron. This should never happen.
                raise RuntimeError(
                    "A `permute_copy` node was incorrectly selected for delegation. Please report this."
                )

        perm_tensor = self.builder.create_tensor_for_data(
            np.array(perm, "int32"), "perm"
        )

        # Use the final permutation as the operator's second input.
        t_op.tmp_inputs = [t_op.tmp_inputs[0], perm_tensor]

        return ops

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

        ops = self.handle_tensor_formats(t_op, node)

        self.builder.append_operators(ops.flatten())
