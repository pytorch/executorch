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
                    node, f"Unsupported input dtype {ifm_dtype} (Supports i8, i16)."
                )
                return False
            weight_dtype = _try_determine_dtype(weight)
            if weight_dtype is not None and weight_dtype not in (torch.int8,):
                self.reporter.report_reject(
                    node, f"Unsupported weight dtype {weight_dtype} (Supports i8)."
                )
                return False
            if len(node.all_input_nodes) > 2:
                bias = node.all_input_nodes[2]
                bias_dtype = _try_determine_dtype(bias)
                if bias_dtype is not None and bias_dtype not in (torch.int32,):
                    self.reporter.report_reject(
                        node, f"Unsupported bias dtype {bias_dtype} (Supports i32)."
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


class EthosU55ResizeCheck(OperatorSupportBase):
    """Accept nearest-neighbor upscales supported by Ethos-U55.

    Ethos-U55 supports nearest-neighbor TOSA RESIZE when both spatial dimensions
    have batch size 1 and use the same power-of-two scale in the range 2x
    through 8x.

    """

    def __init__(self, reporter: WhyNoPartitionReporter):
        """Initialize the check with a reporter.

        Args:
            reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

        """
        self.reporter = reporter

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True when a resize satisfies the U55 constraints.

        Args:
            submodules (typing.Mapping[str, torch.nn.Module]): Exported modules.
            node (fx.Node): FX node to check.

        Returns:
            bool: True if supported; otherwise, False.

        """
        if node.target != exir_ops.edge.aten.upsample_nearest2d.vec:
            return True

        input_shape = get_first_fake_tensor(node.all_input_nodes[0]).shape
        output_shape = get_first_fake_tensor(node).shape
        input_batch = input_shape[0]
        output_batch = output_shape[0]
        if isinstance(input_batch, torch.SymInt) or isinstance(
            output_batch, torch.SymInt
        ):
            self.reporter.report_reject(
                node,
                "U55 nearest-neighbor resize requires a static batch size of 1.",
            )
            return False
        if input_batch != 1 or output_batch != 1:
            self.reporter.report_reject(
                node, "U55 nearest-neighbor resize requires batch size 1."
            )
            return False

        scale_factors_arg = node.args[2]
        if scale_factors_arg is not None:
            scale_factors = typing.cast(typing.Sequence[float], scale_factors_arg)
            if len(scale_factors) != 2 or not (
                scale_factors[0] == scale_factors[1] and scale_factors[0] in (2, 4, 8)
            ):
                self.reporter.report_reject(
                    node,
                    "U55 nearest-neighbor resize requires equal 2x, 4x, or 8x "
                    "scale factors.",
                )
                return False
            return True

        input_height, input_width = input_shape[-2:]
        output_height, output_width = output_shape[-2:]
        if any(
            isinstance(dim, torch.SymInt)
            for dim in (input_height, input_width, output_height, output_width)
        ):
            self.reporter.report_reject(
                node,
                "U55 nearest-neighbor resize with an explicit size requires "
                "static spatial dimensions.",
            )
            return False
        if any(
            output_height == input_height * scale
            and output_width == input_width * scale
            for scale in (2, 4, 8)
        ):
            return True

        self.reporter.report_reject(
            node, "U55 nearest-neighbor resize requires a 2x, 4x, or 8x upscale."
        )
        return False


class EthosU55ReverseCheck(OperatorSupportBase):
    """Accept the REVERSE cases proven to run on Ethos-U55."""

    def __init__(self, reporter: WhyNoPartitionReporter):
        self.reporter = reporter

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        del submodules
        if node.target == exir_ops.edge.aten.flip.default:
            input_rank = len(get_first_fake_tensor(node.all_input_nodes[0]).shape)
            dims = typing.cast(typing.Sequence[int], node.args[1])
            if input_rank == 4 and len(dims) == 1 and dims[0] % input_rank in (1, 2):
                return True
            self.reporter.report_reject(
                node,
                "U55 flip support is limited to rank-4 channel or height reversal.",
            )
            return False

        return True


class EthosU55UnfoldCopyCheck(OperatorSupportBase):
    """Accept bounded static unfold_copy cases that lower to slices."""

    max_windows = 16

    def __init__(self, reporter: WhyNoPartitionReporter):
        self.reporter = reporter

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        del submodules
        if node.target != exir_ops.edge.aten.unfold_copy.default:
            return True

        input_arg, dim, size, step = node.args
        input_node = typing.cast(fx.Node, input_arg)
        input_tensor = get_first_fake_tensor(input_node)
        input_shape = input_tensor.shape
        if (
            input_tensor.dtype == torch.bool
            or not all(isinstance(arg, int) for arg in (dim, size, step))
            or any(not isinstance(value, int) for value in input_shape)
        ):
            self.reporter.report_reject(
                node, "U55 unfold_copy requires static non-BOOL input."
            )
            return False

        rank = len(input_shape)
        dim = typing.cast(int, dim) % rank
        size = typing.cast(int, size)
        step = typing.cast(int, step)
        windows = (input_shape[dim] - size) // step + 1
        if windows > self.max_windows:
            self.reporter.report_reject(
                node,
                f"U55 unfold_copy supports at most {self.max_windows} windows.",
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
