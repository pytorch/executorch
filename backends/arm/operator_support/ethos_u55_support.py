# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import typing

import torch
import torch.fx as fx
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm._passes.insert_table_ops import TableOps
from executorch.exir.backend.utils import WhyNoPartitionReporter

from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.operator_support import OperatorSupportBase


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

    def _try_determine_dtype(self, node: fx.Node) -> torch.dtype | None:
        """Attempt to figure out the quantized data type of node. On failure, return None."""

        dtype = get_first_fake_tensor(node).dtype
        if not dtype.is_floating_point:
            return dtype

        if (
            node.target
            is exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
        ):
            return get_first_fake_tensor(node.all_input_nodes[0]).dtype

        if len(node.users) == 0:
            return None

        q_node = list(node.users)[0]
        if (
            q_node.target
            is exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
        ):
            return typing.cast(torch.dtype, q_node.args[-1])

        # We can't easily figure out dtype, return None
        return None

    def is_node_supported(  # noqa: C901
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:

        dtype = self._try_determine_dtype(node)
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
            ifm_dtype = self._try_determine_dtype(ifm)
            if ifm_dtype is not None and ifm_dtype not in (torch.int8, torch.int16):
                self.reporter.report_reject(
                    node, f"Unsupported input dtype {dtype} (Supports i8, i16)."
                )
                return False
            weight_dtype = self._try_determine_dtype(weight)
            if weight_dtype is not None and weight_dtype not in (torch.int8,):
                self.reporter.report_reject(
                    node, f"Unsupported weight dtype {dtype} (Supports i8)."
                )
                return False
            if len(node.all_input_nodes) > 2:
                bias = node.all_input_nodes[2]
                bias_dtype = self._try_determine_dtype(bias)
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
                dtype = self._try_determine_dtype(input_node)
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
        exir_ops.edge.aten.bitwise_not,
        exir_ops.edge.aten.logical_and.default,
        exir_ops.edge.aten.logical_or.default,
        exir_ops.edge.aten.logical_xor.default,
        exir_ops.edge.aten.logical_not.default,
        exir_ops.edge.aten.amax.default,  # REDUCE_MAX
        exir_ops.edge.aten.amin.default,  # REDUCE_MIN
        exir_ops.edge.aten.eq.Tensor,
        exir_ops.edge.aten.eq.Scalar,
        exir_ops.edge.aten.ge.Tensor,
        exir_ops.edge.aten.gt.Tensor,
        exir_ops.edge.aten.le.Tensor,
        exir_ops.edge.aten.lt.Tensor,
        exir_ops.edge.aten.flip.default,  # REVERSE
        exir_ops.edge.aten.grid_sampler_2d,  # GATHER
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
