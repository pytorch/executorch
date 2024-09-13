# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import cast, List, Optional

import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
    XNNPartitionerConfig,
)
from executorch.backends.xnnpack.utils.quant_utils import is_dequant, is_quant
from executorch.backends.xnnpack.utils.utils import get_input_node
from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    format_target_name,
)
from executorch.exir.backend.utils import WhyNoPartition
from torch.export import ExportedProgram

logger = logging.getLogger(__name__)
why = WhyNoPartition(logger=logger)


class GenericNodePartitionerConfig(XNNPartitionerConfig):
    def __init__(self, fused_act: Optional[List[str]] = None, **kwargs):
        """
        fused_act is a list of node target names that can be fused with this
        node under quantization
        """
        self.fused_acts = fused_act or []
        super().__init__(**kwargs)

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        return self.check_common_constraints(node, ep)

    def get_node_and_deps(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        deps = [node]
        quantized_deps = []
        if ConfigPrecisionType.STATIC_QUANT in self.enabled_precision_types:
            # try to partition dequant inputs and quant outputs if static quant is enabled
            if [(is_dequant(dq_input)) for dq_input in node.all_input_nodes].count(
                False
            ):
                # if not all inputs are dequant nodes then it isn't quantized
                return deps

            quantized_deps.extend(node.all_input_nodes)

            # check if quantized pattern has fused activation
            if len(node.users) != 1:
                return deps

            node_output = list(node.users)[0]
            if (
                node_output.op == "call_function"
                and format_target_name(node_output.target.__name__) in self.fused_acts
            ):
                quantized_deps.append(node_output)
                fused_out_users = list(node_output.users.keys())
                if len(fused_out_users) == 1:
                    node_output = fused_out_users[0]

            if not is_quant(node_output):
                # Expected node --> fused_act (optional) --> dequant
                return deps

            quantized_deps.append(node_output)

        return deps + quantized_deps


class QuantizedPerTensorConfig(GenericNodePartitionerConfig):
    target_name = "quantize_per_tensor.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.STATIC_QUANT]


class DeQuantizedPerTensorConfig(GenericNodePartitionerConfig):
    target_name = "dequantize_per_tensor.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.STATIC_QUANT]


class HardtanhConfig(GenericNodePartitionerConfig):
    target_name = "hardtanh.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]


class AddConfig(GenericNodePartitionerConfig):
    target_name = "add.Tensor"

    def __init__(self, **kwargs):
        super().__init__(fused_act=["relu.default"], **kwargs)

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]


class ReLUConfig(GenericNodePartitionerConfig):
    target_name = "relu.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]


class AbsConfig(GenericNodePartitionerConfig):
    target_name = "abs.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class AvgPoolingConfig(GenericNodePartitionerConfig):
    target_name = "avg_pool2d.default"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        """
        XNNPACK does not support ceil_mode = True and count_include_pad = True
        Additionally, we only support divisor_override if divisor_override = pooling region
        """
        if not self.check_common_constraints(node, ep):
            return False

        args = node.args

        ceil_mode = False  # default is False
        if len(args) >= 5:
            ceil_mode = cast(bool, args[4])

        count_include_pad = True  # default is True
        if len(args) >= 6:
            count_include_pad = cast(bool, args[5])

        kernel_size = cast(List[int], args[1])
        pooling_region = kernel_size[0] * kernel_size[1]
        divisor_override = pooling_region  # Default divisor is pooling_region
        if len(args) >= 7:
            divisor_override = cast(int, args[6])

        if ceil_mode:
            why(node, reason="ceil mode is not supported")
            return False

        if count_include_pad:
            why(
                node,
                reason="zero-padding in the averaging calculation is not supported",
            )
            return False

        if divisor_override != pooling_region:
            why(node, reason="divisor override is not supported")
            return False

        return True

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class CatConfig(GenericNodePartitionerConfig):
    target_name = "cat.default"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        """
        Only support concatenation of 2 - 4 tensors
        """
        if not self.check_common_constraints(node, ep):
            return False

        num_tensors = len(node.all_input_nodes)

        if not (num_tensors >= 2 and num_tensors <= 4):
            why(
                node,
                reason=f"only support concatenation of 2 - 4 tensors, got {num_tensors} tensors",
            )
            return False

        return True

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]


class CeilConfig(GenericNodePartitionerConfig):
    target_name = "ceil.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class ClampConfig(GenericNodePartitionerConfig):
    target_name = "clamp.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]


class DivConfig(GenericNodePartitionerConfig):
    target_name = "div.Tensor"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class EluConfig(GenericNodePartitionerConfig):
    target_name = "elu.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        return torch.ops.aten.elu.default


class SoftmaxConfig(GenericNodePartitionerConfig):
    target_name = "_softmax.default"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        """
        Check that dim is always the last dim
        """
        if not self.check_common_constraints(node, ep):
            return False

        dim = cast(int, node.args[1])
        node_input = node.all_input_nodes[0]
        tensor_dims = node_input.meta["val"].dim()

        if not (dim == -1 or dim == tensor_dims - 1):
            why(
                node,
                reason=f"dim must be the last dim, got dim = {dim} for tensor of rank {tensor_dims}",
            )
            return False
        return True

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class PermuteConfig(GenericNodePartitionerConfig):
    target_name = "permute_copy.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]


class SigmoidConfig(GenericNodePartitionerConfig):
    target_name = "sigmoid.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class MulConfig(GenericNodePartitionerConfig):
    target_name = "mul.Tensor"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]


class MaximumConfig(GenericNodePartitionerConfig):
    target_name = "maximum.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class MaxPool2dConfig(GenericNodePartitionerConfig):
    target_name = "max_pool2d.default"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        """
        XNNPACK's maxpool2d does not support ceil mode
        """
        if not self.check_common_constraints(node, ep):
            return False

        is_ceil_mode = len(node.args) >= 6 and cast(bool, node.args[5])
        if is_ceil_mode:
            why(node, reason="ceil mode is not supported")
            return False
        return True

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        return torch.ops.aten.max_pool2d.default


class UpsampleBilinear2dConfig(GenericNodePartitionerConfig):
    target_name = "upsample_bilinear2d.vec"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        return torch.ops.aten.upsample_bilinear2d.vec


class FloorConfig(GenericNodePartitionerConfig):
    target_name = "floor.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class HardswishConfig(GenericNodePartitionerConfig):
    target_name = "hardswish.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class LeakyReLUConfig(GenericNodePartitionerConfig):
    target_name = "leaky_relu.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class MeanDimConfig(GenericNodePartitionerConfig):
    target_name = "mean.dim"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        """
        Mean Dim currently only supports averaging 4D tensors across the innermost
        dimensions
        """
        if not self.check_common_constraints(node, ep):
            return False

        dims = node.args[1]
        output_dims = node.meta["val"].dim()

        if dims not in ([-2, -1], [-1, -2]):
            why(
                node,
                reason="mean.dim only supports averaging 4D tensors across the innermost dimensions",
            )
            return False

        if output_dims != 4:
            why(
                node,
                reason=f"mean.dim only supports averaging 4D tensors, got tensor of rank {output_dims}",
            )
            return False
        return True

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]


class MinimumConfig(GenericNodePartitionerConfig):
    target_name = "minimum.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class NegConfig(GenericNodePartitionerConfig):
    target_name = "neg.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class PowConfig(GenericNodePartitionerConfig):
    target_name = "pow.Tensor_Scalar"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        """
        Only support powers of two
        """
        if not self.check_common_constraints(node, ep):
            return False

        power = node.args[1]

        if not isinstance(power, int):
            why(node, reason=f"only support int powers, got {power}")
            return False

        if power != 2:
            why(node, reason=f"only support power == 2, got {power}")
            return False
        return True

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class SliceCopyConfig(GenericNodePartitionerConfig):
    target_name = "slice_copy.Tensor"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        """
        Support slicing with stride = 1, no zero-dim tensors, Slice isn't supported
        if the input or output is dynamic
        """
        if not self.check_common_constraints(node, ep):
            return False

        stride = 1
        if len(node.args) > 4:
            stride = cast(int, node.args[4])

        if stride != 1:
            return False

        input_node = get_input_node(node, 0)
        output_node = node

        input_shape = list(input_node.meta["val"].shape)
        output_shape = list(output_node.meta["val"].shape)

        for dim in input_shape:
            if not isinstance(dim, int) or dim == 0:
                why(
                    node,
                    reason=f"input tensor has invalid shape, dim: {dim} of type {type(dim)}. Expecting non-zero, int values.",
                )
                return False

        for dim in output_shape:
            if not isinstance(dim, int) or dim == 0:
                why(
                    node,
                    reason=f"output tensor has invalid shape, dim: {dim} of type {type(dim)}. Expecting non-zero, int values.",
                )
                return False

        return True

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]


class SquareRootConfig(GenericNodePartitionerConfig):
    target_name = "sqrt.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class ConstantPadConfig(GenericNodePartitionerConfig):
    target_name = "constant_pad_nd.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class SubConfig(GenericNodePartitionerConfig):
    target_name = "sub.Tensor"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]


class BMMConfig(GenericNodePartitionerConfig):
    """
    Despite being a GEMM Kernel, BMM Can be partitioned like a single node partitioner
    because it does not perform any packing on the inputs being matrix multiplied
    """

    target_name = "bmm.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class SDPAConfig(GenericNodePartitionerConfig):
    target_name = "scaled_dot_product_attention.default"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        """
        Requires Mask to have Rank 2
        """
        if not self.check_common_constraints(node, ep):
            return False

        if len(node.all_input_nodes) < 4:
            return False
        mask_node = node.all_input_nodes[3]
        mask_rank = mask_node.meta["val"].dim()
        if mask_rank != 2:
            why(
                node,
                reason=f"mask must have rank 2, got mask of rank {mask_rank}",
            )
            return False

        return True

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        return torch.ops.aten.scaled_dot_product_attention.default

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]
