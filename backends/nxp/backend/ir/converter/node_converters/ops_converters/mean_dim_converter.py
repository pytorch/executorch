# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.backend.data_format import DataFormat
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    create_channels_last_to_channels_first_permutation,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.shared.reduce_utils import (
    convert_axes_from_attribute,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    mean_options,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.fx.passes.infra.partitioner import Partition
from torch.nn import Parameter


class MeanDimConverter(NodeConverter):

    @classmethod
    def supports_partitioning_result(
        cls,
        node: Node,
        partition_list: list[Partition],
        custom_delegation_options: CustomDelegationOptions,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        dim, keepdim = MeanDimConverter._get_attrs(node)
        input_shape = node.args[0].meta["val"].shape

        is_alone_in_partition = cls.is_node_alone_in_partition(node, partition_list)

        if is_alone_in_partition and keepdim and all(input_shape[d] == 1 for d in dim):
            # The operator is a no-op, so the Neutron Converter will skip it. If it's the only node in the
            #  partition, the graph would end up empty.
            return False

        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if not NodeConverter.uses_quantization_type_for_io(
            node,
            supported_types=[torch.int8, torch.uint8],
            input_indices=[0],
            output_indices=[0],
        ):
            return False

        return True

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if hasattr(node.kwargs, "dtype") and node.kwargs["dtype"] not in [
            torch.float32,
            torch.uint32,
            torch.uint8,
        ]:
            return False

        if not NodeConverter._has_shared_q_params_if_quantized(node):
            return False

        return True

    @staticmethod
    def _to_pos_dim(d: int, rank: int):
        return d + rank if d < 0 else d

    @staticmethod
    def _normalize_dim(dim: list[int], rank: int) -> list[int]:
        # convert negative index to positive
        return [MeanDimConverter._to_pos_dim(d, rank) for d in dim]

    @staticmethod
    def _normalize_and_to_channel_last_dim(dim: list[int], rank: int) -> list[int]:
        # convert negative index to positive
        dim = MeanDimConverter._normalize_dim(dim, rank)

        perm = create_channels_last_to_channels_first_permutation(rank, True)
        dim = [perm[d] for d in dim]

        # noinspection PyTypeChecker
        return dim

    @staticmethod
    def _get_attrs(node: Node) -> tuple[list[int], bool]:
        dim = node.args[1]
        keepdim = node.args[2] if len(node.args) >= 3 else False
        return dim, keepdim

    def _get_dim_and_handle_io_formats(
        self, ops: OpsList, dim: list[int], keep_dim: bool
    ):
        t_op = ops.middle_op
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        channels_last_input = x.tensor_format.is_channels_last()
        channels_last_output = y.tensor_format.is_channels_last()
        formatless_input = not channels_last_input
        formatless_output = not channels_last_output

        dim = self._normalize_dim(dim, x.rank)

        if keep_dim:
            # The rank is preserved and the io formats should always be equal.
            assert (
                x.tensor_format == y.tensor_format
            ), "NXP backend: There is a bug in `mean.dim` format inference."

            # Just adjust the dim to match the input format.
            if channels_last_input:
                dim = self._normalize_and_to_channel_last_dim(dim, x.rank)

        else:
            # `keep_dim = False`, so the output rank != input rank, and the operator changes the tensor format.

            if channels_last_input and formatless_output:
                if 1 in dim:
                    # If we are reducing over the channels, the channels dimension gets removed and the output ends up
                    #  exactly equal in channels last and channels first, regardless of which other dimensions are
                    #  removed. Therefore, we can just adjust the `dim` and we don't need to insert any `Transpose` ops.
                    dim = self._normalize_and_to_channel_last_dim(dim, x.rank)
                elif all(spatial_dim in dim for spatial_dim in range(2, x.rank)):
                    # All spatial dims are reduced, leaving only batch and channels (both optionally). So the result is
                    #  equal in channels first and channels last as long as we adjust the `dim` to match a channels last
                    #  input (similarly to the case above).
                    dim = self._normalize_and_to_channel_last_dim(dim, x.rank)
                else:
                    # If the channels dimension is preserved, we must transpose the input to channels first (to match
                    #  the edge model) and we must keep the `dim` unchanged (referencing channels first dimensions).
                    #  Otherwise, the output would not match the input.
                    to_channels_first_perm = (
                        translator.create_channels_last_to_channels_first_permutation(
                            x.rank
                        )
                    )
                    ops.add_pre(
                        self.builder.create_transpose_operator_before(
                            t_op, 0, to_channels_first_perm
                        )
                    )
                    t_op.tmp_inputs[0].tensor_format = DataFormat.CHANNELS_FIRST

            elif formatless_input and channels_last_output:
                # We need apply the `mean` with the original `dim`, which will produce a channels first output. Then,
                #  we need to append a `Transpose` operator to make the output channels last.
                to_channels_last_perm = (
                    translator.create_channels_first_to_channels_last_permutation(
                        y.rank, True
                    )
                )
                ops.add_post(
                    self.builder.create_transpose_operator_after(
                        t_op, 0, to_channels_last_perm
                    )
                )
                t_op.tmp_outputs[0].tensor_format = DataFormat.CHANNELS_FIRST

            elif formatless_input and formatless_output:
                # No action needed.
                pass

            else:  # channels_last_input and channels_last_output
                # This case cannot currently occur, as it would require the case:
                #       channels last 4D -> mean -> channels_last 3D
                #  which cannot currently happen as the 3D conv/pooling/... is supported by adding `view_copy` nodes in
                #  the edge dialect and converting the node to 4D, and the `view_copy` nodes prevent the propagation of
                #  the format to the `mean.dim` output.
                # Therefore, the implementation cannot be tested. But from experience with other operators, it should
                #  work correctly. We just need to add 2 `Transpose` ops to make the IO channels first, and keep the
                #  `dim` unchanged.
                to_channels_first_perm = (
                    translator.create_channels_last_to_channels_first_permutation(
                        x.rank
                    )
                )
                ops.add_pre(
                    self.builder.create_transpose_operator_before(
                        t_op, 0, to_channels_first_perm
                    )
                )
                t_op.tmp_inputs[0].tensor_format = DataFormat.CHANNELS_FIRST

                to_channels_last_perm = (
                    translator.create_channels_first_to_channels_last_permutation(
                        y.rank, True
                    )
                )
                ops.add_post(
                    self.builder.create_transpose_operator_after(
                        t_op, 0, to_channels_last_perm
                    )
                )
                t_op.tmp_outputs[0].tensor_format = DataFormat.CHANNELS_FIRST

        return dim

    def convert(self, node: Node):
        """Convert the 'mean.dim' operator to NeutronIR 'Mean'.
        The ExecuTorch schema is:
            mean.dim(
                Tensor self,
                int[1]? dim,
                bool keepdim=False,
                *,
                ScalarType? dtype=None
            ) -> Tensor
        """
        self.assert_convertible(node)

        dim, keepdim = self._get_attrs(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = mean_options.Mean(keepdim)

        ops = OpsList(middle_op=t_op)
        dim = self._get_dim_and_handle_io_formats(ops, dim, keepdim)

        convert_axes_from_attribute(t_op, self.builder, dim)
        self.builder.append_operators(ops.flatten())
