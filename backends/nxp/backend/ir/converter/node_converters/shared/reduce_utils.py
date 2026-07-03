# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from executorch.backends.nxp.backend.data_format import DataFormat

from executorch.backends.nxp.backend.ir.converter.builder.model_builder import (
    ModelBuilder,
)
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from torch.fx import Node


def convert_axes_from_attribute(
    t_op: tflite_model.Operator, builder: ModelBuilder, axes: list[int] | None
):
    """Create an `axes` tensor and assign it as an input to the `t_op`, which is expected to represent an ExecuTorch
    reduction operator.
    """
    x = t_op.tmp_inputs[0]
    rank = x.rank

    if axes is None:
        # Default axes -> reduce over all dimensions.
        axes = np.arange(rank).astype(np.int32)

    else:
        # Axes are initialized.
        axes = np.asarray(axes, np.int32)

    # TFLite has `axes` as input tensor -> create it.
    axes_tensor = builder.create_tensor_for_data(axes, "axes")
    t_op.tmp_inputs.append(axes_tensor)


def _to_pos_dim(d: int, rank: int) -> int:
    return d + rank if d < 0 else d


def _normalize_dim(dim: list[int], rank: int) -> list[int]:
    # convert negative index to positive
    return [_to_pos_dim(d, rank) for d in dim]


def _normalize_and_to_channel_last_dim(dim: list[int], rank: int) -> list[int]:
    # convert negative index to positive
    dim = _normalize_dim(dim, rank)

    perm = translator.create_channels_last_to_channels_first_permutation(rank, True)
    dim = [perm[d] for d in dim]

    # noinspection PyTypeChecker
    return dim


def get_reduce_node_attrs(node: Node) -> tuple[list[int], bool]:
    dim = node.args[1]
    keepdim = node.args[2] if len(node.args) >= 3 else False
    return dim, keepdim


def get_dim_and_handle_io_formats(
    builder, ops: OpsList, dim: list[int], keep_dim: bool
):
    t_op = ops.middle_op
    x = t_op.tmp_inputs[0]
    y = t_op.tmp_outputs[0]

    channels_last_input = x.tensor_format.is_channels_last()
    channels_last_output = y.tensor_format.is_channels_last()
    formatless_input = not channels_last_input
    formatless_output = not channels_last_output

    dim = _normalize_dim(dim, x.rank)

    if keep_dim:
        # The rank is preserved and the io formats should always be equal.
        assert (
            x.tensor_format == y.tensor_format
        ), "NXP backend: There is a bug in node format inference."

        # Just adjust the dim to match the input format.
        if channels_last_input:
            dim = _normalize_and_to_channel_last_dim(dim, x.rank)

    else:
        # `keep_dim = False`, so the output rank != input rank, and the operator changes the tensor format.

        if channels_last_input and formatless_output:
            if 1 in dim:
                # If we are reducing over the channels, the channels dimension gets removed and the output ends up
                #  exactly equal in channels last and channels first, regardless of which other dimensions are
                #  removed. Therefore, we can just adjust the `dim` and we don't need to insert any `Transpose` ops.
                dim = _normalize_and_to_channel_last_dim(dim, x.rank)
            elif all(spatial_dim in dim for spatial_dim in range(2, x.rank)):
                # All spatial dims are reduced, leaving only batch and channels (both optionally). So the result is
                #  equal in channels first and channels last as long as we adjust the `dim` to match a channels last
                #  input (similarly to the case above).
                dim = _normalize_and_to_channel_last_dim(dim, x.rank)
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
                    builder.create_transpose_operator_before(
                        t_op, 0, to_channels_first_perm
                    )
                )
                t_op.tmp_inputs[0].tensor_format = DataFormat.CHANNELS_FIRST

        elif formatless_input and channels_last_output:
            # We need apply the `reduce-type node` with the original `dim`, which will produce a channels first output. Then,
            #  we need to append a `Transpose` operator to make the output channels last.
            to_channels_last_perm = (
                translator.create_channels_first_to_channels_last_permutation(
                    y.rank, True
                )
            )
            ops.add_post(
                builder.create_transpose_operator_after(t_op, 0, to_channels_last_perm)
            )
            t_op.tmp_outputs[0].tensor_format = DataFormat.CHANNELS_FIRST

        elif formatless_input and formatless_output:
            # No action needed.
            pass

        else:  # channels_last_input and channels_last_output
            # This case cannot currently occur, as it would require the case:
            #       channels last 4D -> reduce-type node -> channels_last 3D
            #  which cannot currently happen as the 3D conv/pooling/... is supported by adding `view_copy` nodes in
            #  the edge dialect and converting the node to 4D, and the `view_copy` nodes prevent the propagation of
            #  the format to the `reduce-type node` output.
            # Therefore, the implementation cannot be tested. But from experience with other operators, it should
            #  work correctly. We just need to add 2 `Transpose` ops to make the IO channels first, and keep the
            #  `dim` unchanged.
            to_channels_first_perm = (
                translator.create_channels_last_to_channels_first_permutation(x.rank)
            )
            ops.add_pre(
                builder.create_transpose_operator_before(
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
                builder.create_transpose_operator_after(t_op, 0, to_channels_last_perm)
            )
            t_op.tmp_outputs[0].tensor_format = DataFormat.CHANNELS_FIRST

    return dim
