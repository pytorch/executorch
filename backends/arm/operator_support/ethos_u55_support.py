# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide Ethos-U55 specific operator support checks.

Contains dtype validation, explicit unsupported-op filtering, and shape/
permutation constraints for view and permute operations when targeting the
Ethos-U55 subset of TOSA.

"""
import typing

import torch
import torch.fx as fx

from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm._passes.convert_permute_singleton_to_view_pass import (
    is_singleton_permutation,
)
from executorch.backends.arm._passes.insert_table_ops import TableOps
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.operator_support import OperatorSupportBase


def _try_determine_dtype(node: fx.Node) -> torch.dtype | None:
    """Return an inferred dtype for a node when possible.

    Uses fake tensor metadata and nearby quantize/dequantize nodes to infer the
    integer dtype used by the operator. Returns ``None`` when the dtype cannot
    be determined reliably.

    Args:
        node (fx.Node): FX node to inspect.

    Returns:
        torch.dtype | None: Inferred dtype or ``None`` if unknown.

    """
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
    """Validate dtypes for U55-supported operators.

    Ensures operators use a supported integer dtype according to U55
    constraints, with specific rules for convolution, matmul, and table ops.

    Attributes:
        reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

    """

    def __init__(self, reporter: WhyNoPartitionReporter):
        """Initialize the check with a reporter.

        Args:
            reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

        """
        super().__init__()
        self.reporter = reporter

    targeted_ops_i8_i16_i32 = [
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.expand_copy.default,
        exir_ops.edge.aten.repeat.default,
        exir_ops.edge.aten.constant_pad_nd.default,
        exir_ops.edge.aten.view.default,
        exir_ops.edge.aten.permute.default,
        exir_ops.edge.aten.permute_copy.default,
    ]

    target_ops_i8_i16 = (
        *TableOps.included_ops(),
        exir_ops.edge.aten.amax.default,
        exir_ops.edge.aten.amin.default,
    )

    def is_node_supported(  # noqa: C901
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True if the node uses supported dtypes.

        Applies per-operator dtype rules for U55, including specialized input
        and weight constraints for convolution and int8-only checks for table
        operations and matmul variants.

        Args:
            submodules (typing.Mapping[str, torch.nn.Module]): Exported modules.
            node (fx.Node): FX node to check.

        Returns:
            bool: True if supported; otherwise, False.

        """
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

        if node.target in self.target_ops_i8_i16:
            if dtype not in (torch.int8, torch.int16):
                self.reporter.report_reject(
                    node, f"Unsupported dtype {dtype} (Supports i8, i16)."
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
                if dtype is not None and dtype not in (torch.int8,):
                    self.reporter.report_reject(
                        node,
                        f"Input {input_node.name} has unsupported dtype {dtype} (Supports int8).",
                    )
                    return False

        return True


class EthosU55NotSupported(OperatorSupportBase):
    """Reject operators not supported by Ethos-U55.

    The ``unsupported_ops`` list contains aten ops that either map to TOSA
    operators the U55 cannot run or remain unimplemented. The mapping comments
    capture expected TOSA equivalents when not obvious.

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
        exir_ops.edge.aten.conv3d.default,  # CONV3D
        exir_ops.edge.aten.conv3d.padding,  # CONV3D (deprecated alias)
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
        exir_ops.edge.aten.gather.default,  # GATHER
        exir_ops.edge.aten.grid_sampler_2d,  # GATHER
        exir_ops.edge.aten.index.Tensor,  # GATHER
        exir_ops.edge.aten.index_select.default,  # GATHER
        exir_ops.edge.aten.index_put.default,  # SCATTER
        exir_ops.edge.aten.scatter.src,
        exir_ops.edge.aten.scatter.value,
        exir_ops.edge.aten.select_scatter.default,
        exir_ops.edge.aten.scatter_reduce.two,
        exir_ops.edge.aten.scatter_add.default,
        exir_ops.edge.aten.unfold_copy.default,  # GATHER
        exir_ops.edge.aten.upsample_nearest2d.vec,  # RESIZE
        exir_ops.edge.aten.upsample_bilinear2d.vec,  # RESIZE
        exir_ops.edge.aten.reflection_pad1d.default,  # REVERSE
        exir_ops.edge.aten.reflection_pad2d.default,  # REVERSE
        exir_ops.edge.aten.reflection_pad3d.default,  # REVERSE
        exir_ops.edge.aten.where.self,  # SELECT
    ]

    def __init__(self, reporter: WhyNoPartitionReporter):
        """Initialize the check with a reporter.

        Args:
            reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

        """
        self.reporter = reporter

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return False for nodes explicitly unsupported on U55.

        Args:
            submodules (typing.Mapping[str, torch.nn.Module]): Exported modules.
            node (fx.Node): FX node to check.

        Returns:
            bool: False if ``node.target`` is in ``unsupported_ops``; else True.

        """
        if node.target in self.unsupported_ops:
            self.reporter.report_reject(node, "Op is not supported on U55.")
            return False

        return True


shape_t = list[int]


class EthosU55ViewCheck(OperatorSupportBase):
    """Validate view/select shapes and dtypes for U55.

    Performs lightweight checks on output shape rank and product constraints,
    with awareness that transposes may be inserted around view/select during
    lowering to channels-last.

    Attributes:
        reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

    """

    def __init__(self, reporter: WhyNoPartitionReporter):
        """Initialize the check with a reporter.

        Args:
            reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

        """
        super().__init__()
        self.reporter = reporter

    _MAX_AXIS_PRODUCT = 65536

    def axes_product(self, shape: shape_t) -> int:
        """Return the product of all axes in ``shape``.

        Args:
            shape (shape_t): Shape.

        Returns:
            int: Product of the axis sizes.

        """
        product = 1
        for axes in shape:
            product *= axes
        return product

    def _check_rank_constraints(
        self,
        node: fx.Node,
        input_shape: shape_t,
        output_shape: shape_t,
        dtype: torch.dtype | None,
    ) -> bool:
        """Validate high-rank reshape scenarios against U55 restrictions. If
        either input or output rank of node is >4, figuring out whether a
        transpose is needed or not is complex. Instead, conservatively check
        that the corresponding transpose is supported on hardware.

        Args:
            node (fx.Node): Reshape node under inspection.
            input_shape (shape_t): Source tensor dimensions.
            output_shape (shape_t): Destination tensor dimensions.
            dtype (torch.dtype | None): Detected tensor dtype, if known.

        Returns:
            bool: ``True`` when rank/product constraints are satisfied.

        """
        input_rank = len(input_shape)
        output_rank = len(output_shape)

        if input_rank > 4:
            if self.axes_product(input_shape) > self._MAX_AXIS_PRODUCT:
                self.reporter.report_reject(
                    node,
                    f"Input may require transpose operator. No support for {input_shape=}, "
                    f"{dtype=}. Product of axes must be <={self._MAX_AXIS_PRODUCT}",
                )
                return False
            if dtype == torch.int32:
                self.reporter.report_reject(
                    node,
                    "Input may require transpose operator. No support for rank >= 4 transposes in int32.",
                )
                return False

        if output_rank > 4:
            if self.axes_product(output_shape) > self._MAX_AXIS_PRODUCT:
                shape = output_shape
                self.reporter.report_reject(
                    node,
                    f"Operator may require transpose operator. No support for {shape=}, "
                    f"{dtype=}. Product of axes must be <={self._MAX_AXIS_PRODUCT}",
                )
                return False
            if dtype == torch.int32:
                self.reporter.report_reject(
                    node,
                    "Output may require transpose operator. No support for rank >= 4 transposes in int32.",
                )
                return False

        return True

    # TODO: Extend this check to comply with u55 restrictions
    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Check whether a given view/select node is U55-supported.

        Currently only checks dtypes and product of axes.

        Args:
            submodules (typing.Mapping[str, torch.nn.Module]): Exported modules.
            node (fx.Node): FX node for ``view_copy`` or ``select``.

        Returns:
            bool: False if rejected by constraints; otherwise, True.

        """
        # Select decomposes into squeeze, which in turn becomes a view. Therefore,
        # perform the same check on select operators as view operators.
        if node.target not in (
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.select.int,
            exir_ops.edge.aten.unsqueeze_copy.default,
            exir_ops.edge.aten.squeeze_copy.dims,
            exir_ops.edge.aten.select_copy.int,
        ):
            return True

        shape = list(get_first_fake_tensor(node).shape)
        dtype = _try_determine_dtype(node)

        if node.target in (
            exir_ops.edge.aten.select.int,
            exir_ops.edge.aten.select_copy.int,
        ):
            input_shape = list(get_first_fake_tensor(node.all_input_nodes[0]).shape)
            dim = typing.cast(int, node.args[1])
            input_shape[dim] = 1
        else:
            input_shape = list(get_first_fake_tensor(node.all_input_nodes[0]).shape)

        if not self._check_rank_constraints(node, input_shape, shape, dtype):
            return False

        return True


class EthosU55TransposeCheck(OperatorSupportBase):
    """Validate permute nodes against U55 reshape/transpose limits.

    Applies dtype- and rank-specific constraints to permutations. Tests both
    NCHW and NHWC interpretations for rank-3/4 shapes since dim order is unknown
    at partition time.

    Attributes:
        reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

    """

    def __init__(self, reporter: WhyNoPartitionReporter):
        """Initialize the check with a reporter.

        Args:
            reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

        """
        super().__init__()
        self.reporter = reporter

    def _pad_to_rank_4(
        self, shape: shape_t, permutation: list[int]
    ) -> tuple[shape_t, shape_t]:
        """Pad shape/permutation to rank 4 by prepending ones/indices.

        Args:
            shape (list[int]): Original shape.
            permutation (list[int]): Original permutation indices.

        Returns:
            tuple[list[int], list[int]]: Padded shape and permutation.

        """
        diff = 4 - len(shape)
        padded_shape = [1] * diff + shape
        for i in range(len(permutation)):
            permutation[i] += diff
        padded_permutation = list(range(diff)) + permutation
        return padded_shape, padded_permutation

    def axes_product(self, nhwc_shape: shape_t) -> int:
        """Return the product of all axes in ``nhwc_shape``.

        Args:
            nhwc_shape (list[int]): Shape in NHWC order.

        Returns:
            int: Product of the axis sizes.

        """
        product = 1
        for axes in nhwc_shape:
            product *= axes
        return product

    def _permute_constraint_i8_i16(
        self, nhwc_shape: list[int], permutation: list[int]
    ) -> bool:
        """Return True if permutation meets i8/i16 constraints."""
        N, H, W, C = nhwc_shape

        if is_singleton_permutation(nhwc_shape, permutation):
            return True

        match permutation:
            case (0, 1, 2, 3):  # NHWC -> NHWC
                return True
            case (
                (0, 2, 1, 3) | (0, 1, 3, 2) | (0, 3, 1, 2) | (0, 2, 3, 1) | (0, 3, 2, 1)
            ):
                # NHWC -> NWHC, NHCW, NCWH, NCHW, NCHW -> NHWC
                return N * H <= 65536 and W <= 65536 and C <= 65536
            case _:
                return self.axes_product(nhwc_shape) <= 65536

    def _permute_constraint_i32(
        self, nhwc_shape: list[int], permutation: list[int]
    ) -> bool:
        """Return True if permutation meets i32 constraints."""
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
        """Return True if permutation meets dtype-specific constraints."""
        if dtype in (torch.int8, torch.int16):
            return self._permute_constraint_i8_i16(shape, permutation)
        if dtype == torch.int32:
            return not self._permute_constraint_i32(shape, permutation)
        return True

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True if a permute node satisfies U55 constraints.

        Tests both NCHW and NHWC interpretations for rank-3/4 shapes, and
        applies dtype-specific limits to shapes and permutations.

        Args:
            submodules (typing.Mapping[str, torch.nn.Module]): Exported modules.
            node (fx.Node): FX node to check.

        Returns:
            bool: True if supported; otherwise, False.

        """
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
            if not self._permute_constraint(shape, permutation, dtype):
                self.reporter.report_reject(
                    node,
                    f"Unsupported {shape=} for {permutation=}, {dtype=}",
                )
                return False

        return True


class EthosU55CastCheck(OperatorSupportBase):
    """Reject unsupported casts on U55.

    U55 does not support casting from INT32 or any casts involving BOOL. Note that
    casting from one dtype to the same dtype is a no-op and is supported.


    Attributes:
        reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

    """

    targets = [
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    ]

    def __init__(self, reporter: WhyNoPartitionReporter):
        """Initialize the check with a reporter.

        Args:
            reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

        """
        super().__init__()
        self.reporter = reporter

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True if the node satisfies the cast constraints of U55.

        Args:
            submodules (typing.Mapping[str, torch.nn.Module]): Exported modules.
            node (fx.Node): FX node to check.

        Returns:
            bool: True if supported; otherwise, False.

        """
        if node.target not in self.targets:
            return True
        input_dtype = get_first_fake_tensor(node.all_input_nodes[0]).dtype
        output_dtype = get_first_fake_tensor(node).dtype
        if input_dtype == output_dtype:
            # This is ok as this will not result in a cast
            return True
        if input_dtype in (torch.bool, torch.int32):
            self.reporter.report_reject(
                node, f"Casting from {input_dtype} is not supported on U55."
            )
            return False
        if output_dtype in (torch.bool,):
            self.reporter.report_reject(
                node, f"Casting to {output_dtype} is not supported on U55."
            )
            return False

        return True
