# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import typing
from typing import cast

import torch
import torch.fx as fx

from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm._passes.insert_table_ops import TableOps
from executorch.backends.arm.operators.op_permute import transform_permutation_vector
from executorch.backends.arm.tosa.utils import tosa_shape
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.operator_support import OperatorSupportBase


def _try_determine_dtype(node: fx.Node) -> torch.dtype | None:
    dtype = get_first_fake_tensor(node).dtype
    if not dtype.is_floating_point:
        return dtype
    if node.target is exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default:
        return get_first_fake_tensor(node.all_input_nodes[0]).dtype
    q_node = list(node.users)[0]
    if q_node.target is exir_ops.edge.quantized_decomposed.quantize_per_tensor.default:
        return typing.cast(torch.dtype, q_node.args[-1])
    # We can't easily figure out dtype, return None
    return None


class EthosU55DtypeSupport(OperatorSupportBase):

    def __init__(self, reporter: WhyNoPartitionReporter):
        super().__init__()
        self.reporter = reporter

    targeted_ops_i8_i16_i32 = [
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.repeat.default,
        exir_ops.edge.aten.constant_pad_nd.default,
        exir_ops.edge.aten.view.default,
        exir_ops.edge.aten.permute.default,
    ]

    target_ops_i8 = tuple(TableOps.included_ops())

    def is_node_supported(  # noqa: C901
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:

        dtype = _try_determine_dtype(node)
        if dtype is None:
            # If we couldn't determine dtype, just return ok.
            return True

        if node.target in self.targeted_ops_i8_i16_i32:
            if dtype not in (torch.int8, torch.int16, torch.int32):
                self.reporter.report_reject(
                    node, f"Unsupported dtype {dtype} (Supports i8, i16, i32)."
                )
                return False

        if node.target in self.target_ops_i8:
            if dtype not in (torch.int8,):
                self.reporter.report_reject(
                    node, f"Unsupported dtype {dtype} (Supports i8)."
                )
                return False

        if node.target == exir_ops.edge.aten.convolution.default:
            ifm, weight = node.all_input_nodes[0:2]
            ifm_dtype = _try_determine_dtype(ifm)
            if ifm_dtype is not None and ifm_dtype not in (torch.int8, torch.int16):
                self.reporter.report_reject(
                    node, f"Unsupported input dtype {dtype} (Supports i8, i16)."
                )
                return False
            weight_dtype = _try_determine_dtype(weight)
            if weight_dtype is not None and weight_dtype not in (torch.int8,):
                self.reporter.report_reject(
                    node, f"Unsupported weight dtype {dtype} (Supports i8)."
                )
                return False
            if len(node.all_input_nodes) > 2:
                bias = node.all_input_nodes[2]
                bias_dtype = _try_determine_dtype(bias)
                if bias_dtype is not None and bias_dtype not in (torch.int32,):
                    self.reporter.report_reject(
                        node, f"Unsupported bias dtype {dtype} (Supports i32)."
                    )
                    return False

        if node.target in (
            exir_ops.edge.aten.mm.default,
            exir_ops.edge.aten.bmm.default,
        ):
            for input_node in node.all_input_nodes:
                dtype = _try_determine_dtype(input_node)
                if dtype is not None and dtype != torch.int8:
                    self.reporter.report_reject(
                        input_node,
                        f"Input {input_node.name} has unsupported dtype {dtype} (Supports i8).",
                    )
                    return False

        return True


class EthosU55NotSupported(OperatorSupportBase):
    """
    Certain operators are not supported on U55. These are listed in `unsupported_ops`.
    The comment mentions the unsupported TOSA operator that the aten operator maps to where it is not obvious.
    For unimplemented operators, this is the anticipated mapping, and it might be incorrect.
    """

    unsupported_ops = [
        exir_ops.edge.aten.any.default,  # REDUCE_ANY
        exir_ops.edge.aten.any.dim,  # REDUCE_ANY
        exir_ops.edge.aten.any.dims,  # REDUCE_ANY
        exir_ops.edge.aten.bitwise_and.Tensor,
        exir_ops.edge.aten.bitwise_or.Tensor,
        exir_ops.edge.aten.bitwise_xor.Tensor,
        exir_ops.edge.aten.bitwise_and.Scalar,
        exir_ops.edge.aten.bitwise_or.Scalar,
        exir_ops.edge.aten.bitwise_xor.Scalar,
        exir_ops.edge.aten.bitwise_not.default,
        exir_ops.edge.aten.logical_and.default,
        exir_ops.edge.aten.logical_or.default,
        exir_ops.edge.aten.logical_xor.default,
        exir_ops.edge.aten.logical_not.default,
        exir_ops.edge.aten.amax.default,  # REDUCE_MAX
        exir_ops.edge.aten.amin.default,  # REDUCE_MIN
        exir_ops.edge.aten.eq.Tensor,
        exir_ops.edge.aten.eq.Scalar,
        exir_ops.edge.aten.ge.Tensor,
        exir_ops.edge.aten.ge.Scalar,
        exir_ops.edge.aten.gt.Tensor,
        exir_ops.edge.aten.gt.Scalar,
        exir_ops.edge.aten.le.Tensor,
        exir_ops.edge.aten.le.Scalar,
        exir_ops.edge.aten.lt.Tensor,
        exir_ops.edge.aten.lt.Scalar,
        exir_ops.edge.aten.ne.Tensor,
        exir_ops.edge.aten.ne.Scalar,
        exir_ops.edge.aten.flip.default,  # REVERSE
        exir_ops.edge.aten.grid_sampler_2d,  # GATHER
        exir_ops.edge.aten.index.Tensor,  # GATHER
        exir_ops.edge.aten.index_select.default,  # GATHER
        exir_ops.edge.aten.scatter.src,
        exir_ops.edge.aten.scatter.value,
        exir_ops.edge.aten.select_scatter.default,
        exir_ops.edge.aten.scatter_reduce.two,
        exir_ops.edge.aten.scatter_add.default,
        exir_ops.edge.aten.upsample_nearest2d.vec,  # RESIZE
        exir_ops.edge.aten.upsample_bilinear2d.vec,  # RESIZE
        exir_ops.edge.aten.reflection_pad1d.default,  # REVERSE
        exir_ops.edge.aten.reflection_pad2d.default,  # REVERSE
        exir_ops.edge.aten.reflection_pad3d.default,  # REVERSE
        exir_ops.edge.aten.where.self,  # SELECT
    ]

    def __init__(self, reporter: WhyNoPartitionReporter):
        self.reporter = reporter

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:

        if node.target in self.unsupported_ops:
            self.reporter.report_reject(node, "Op is not supported on U55.")
            return False

        return True


shape_t = list[int]


class EthosU55ViewCheck(OperatorSupportBase):

    def __init__(self, reporter: WhyNoPartitionReporter):
        super().__init__()
        self.reporter = reporter

    def axes_product(self, nhwc_shape: shape_t) -> int:
        product = 1
        for axes in nhwc_shape:
            product *= axes
        return product

    # TODO: Extend this check to comply with u55 restrictions
    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """
        Check whether a given view node is supported on U55.

        Currently only checks dtypes and product of axes.

        It is not the view operator itself that is not supported on U55. In order for the
        view operator to be compatible with the channels-last format of TosaBackend,
        transposes may need to be inserted before and after the view op. If that happens
        and that transpose operator does not adhere to the limitations then it will
        result in the following error:

            CPU performance estimation for "Transpose" not implemented.
            ...
            CPU operations are not supported for GraphAPI input

        Args:
            node: The FX node representing the view_copy operator.

        Returns:
            False if the operator is not support and True if it is supported.
        """
        # Select decomposes into squeeze, which in turn becomes a view. Therefore,
        # perform the same check on select operators as view operators.
        if node.target not in (
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.select.int,
            exir_ops.edge.aten.select_copy.int,
        ):
            return True

        if node.target in (
            exir_ops.edge.aten.select.int,
            exir_ops.edge.aten.select_copy.int,
        ):
            input_node, dim, index = cast(tuple[fx.Node, int, int], node.args)

            shape = input_node.meta["val"].shape
            rank = len(shape)
            if not -rank <= dim < rank:
                self.reporter.report_reject(
                    node,
                    (f"Dimension {dim} out of range for rank {rank}."),
                )
                return False
            dim = dim % rank

            size = shape[dim]
            if not -size <= index < size:
                self.reporter.report_reject(
                    node,
                    (f"Index {index} out of range for dim {dim} with size {size}."),
                )
                return False
            index = index % size

            # Shape after squeeze. This may get converted into a view which may become
            # a transpose. This is why we're checking select.
            squeezed_shape = shape[:dim] + shape[dim + 1 :]
            shape = squeezed_shape
        else:
            shape = list(get_first_fake_tensor(node).shape)

        dtype = _try_determine_dtype(node)

        rank = len(shape)
        if rank > 4:
            if dtype == torch.int32:
                self.reporter.report_reject(node, "No support for rank > 4 in int32.")
                return False

        if dtype in (torch.int8, torch.int16):
            if self.axes_product(shape) > 65536:
                self.reporter.report_reject(
                    node,
                    f"No support for {shape=}, {dtype=}. Product of axes must be <65536",
                )
                return False

        return True


class EthosU55TransposeCheck(OperatorSupportBase):

    def __init__(self, reporter: WhyNoPartitionReporter):
        super().__init__()
        self.reporter = reporter

    def _pad_to_rank_4(
        self, shape: shape_t, permutation: list[int]
    ) -> tuple[shape_t, shape_t]:
        diff = 4 - len(shape)
        padded_shape = [1] * diff + shape
        for i in range(len(permutation)):
            permutation[i] += diff
        padded_permutation = list(range(diff)) + permutation
        return padded_shape, padded_permutation

    def axes_product(self, nhwc_shape: shape_t) -> int:
        product = 1
        for axes in nhwc_shape:
            product *= axes
        return product

    def _permute_constraint_i8_i16(
        self, nhwc_shape: list[int], permutation: list[int]
    ) -> bool:
        """Returns True if the constraints are ok."""
        N, H, W, C = nhwc_shape
        match permutation:
            case (0, 1, 2, 3):  # NHWC -> NHWC
                return True
            case (0, 2, 1, 3) | (0, 1, 3, 2) | (0, 3, 1, 2):  # NHWC -> NWHC, NHCW, NCWH
                return N * H <= 65536 and W <= 65536 and C <= 65536
            case _:
                return self.axes_product(nhwc_shape) <= 65536

    def _permute_constraint_i32(
        self, nhwc_shape: list[int], permutation: list[int]
    ) -> bool:
        """Returns True if the constraints are ok."""
        N, H, W, C = nhwc_shape
        match permutation:
            case (0, 1, 2, 3):  # NHWC -> NHWC
                return C <= 32768
            case (0, 2, 1, 3):  # NHWC -> NHWC
                return N == 1 and H <= 65536 and W <= 65536 and C <= 16384
            case (0, 1, 3, 2):  # NHWC -> NHCW
                return N * H <= 65536 and W <= 65536 and C <= 65536
            case _:
                return False

    def _permute_constraint(self, shape, permutation, dtype):
        if dtype in (torch.int8, torch.int16):
            return self._permute_constraint_i8_i16(shape, permutation)
        if dtype == torch.int32:
            return not self._permute_constraint_i32(shape, permutation)
        return True

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:

        if not node.target == exir_ops.edge.aten.permute_copy.default:
            return True

        shape = list(get_first_fake_tensor(node).shape)
        dtype = _try_determine_dtype(node)
        permutation = list(typing.cast(list[int], node.args[1]))

        rank = len(shape)
        if rank > 4:
            if dtype == torch.int32:
                self.reporter.report_reject(
                    node, f"No support for {permutation=} in int32."
                )
                return False
            if dtype in (torch.int8, torch.int16):
                if self.axes_product(shape) > 65536:
                    self.reporter.report_reject(
                        node,
                        f"No support for {shape=}, {dtype=}. Product of axes must be <65536",
                    )
                    return False
            return True

        shape, permutation = self._pad_to_rank_4(shape, permutation)
        if rank == 3 or rank == 4:
            # For rank 3 and 4, we can have channels first or channels last dim order.
            # Since we don't know which at partition-time, test both.

            nhwc_shape = tosa_shape(shape, [0, 2, 3, 1])
            nhwc_permutation = transform_permutation_vector(permutation, [0, 2, 3, 1])

            if not self._permute_constraint(nhwc_shape, nhwc_permutation, dtype):
                self.reporter.report_reject(
                    node,
                    f"Unsupported NHWC {nhwc_shape=} for {nhwc_permutation=}, {dtype=}",
                )
                return False

        if not self._permute_constraint(shape, permutation, dtype):
            self.reporter.report_reject(
                node, f"Unsupported NCHW {shape=} for {permutation=}, {dtype=}"
            )
            return False

        return True
