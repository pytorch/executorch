# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.quantizer.quantization_annotator import (
    _conv_ops,
    _one_to_one,
)
from executorch.backends.cortex_m.quantizer.pattern_checkers import PatternCheck
from torch._ops import OpOverload


def combo_pattern(*pattern_lists):
    "Returns the cartesian product of the given pattern lists."
    return [tuple(pattern) for pattern in product(*pattern_lists)]


class ReluFusedPatternCheck(PatternCheck):
    @classmethod
    def check_quantization_config(cls, pattern, quantization_config):
        if quantization_config is None:
            return True

        output_node = pattern[-1] if pattern else None
        output_qspec = quantization_config.get_output_act_qspec(output_node)
        if output_qspec is None:
            return False

        return output_qspec.qscheme not in (
            torch.per_tensor_symmetric,
            torch.per_channel_symmetric,
        )


class ArithmeticFloatInputsCheck(PatternCheck):
    @classmethod
    def check_pattern(cls, pattern):
        """For arithmetic ops all inputs must be quantizeable for quantization
        to make sense.
        """
        for node in pattern:
            for input_node in node.all_input_nodes:
                try:
                    tensor = get_first_fake_tensor(input_node)
                except Exception:
                    return False
                if not tensor.dtype.is_floating_point:
                    return False

        return True


BINARY_OP_PATTERNS = [
    (torch.ops.aten.add.Tensor,),
    (torch.ops.aten.add_.Tensor,),
    (torch.ops.aten.sub.Tensor,),
    (torch.ops.aten.sub_.Tensor,),
    (torch.ops.aten.matmul.default,),
    (torch.ops.aten.mm.default,),
    (torch.ops.aten.bmm.default,),
    (torch.ops.aten.mul.Tensor,),
    (torch.ops.aten.mul_.Tensor,),
]
ACTIVATION_FUNCTION_PATTERNS = [
    (torch.ops.aten.hardswish.default,),
    (torch.ops.aten.hardswish_.default,),
]

LINEAR_OPS = [torch.ops.aten.linear.default]
FUSED_ACTIVATION_OPS = [
    torch.ops.aten.relu.default,
    torch.ops.aten.relu_.default,
    torch.ops.aten.hardtanh.default,
    torch.ops.aten.hardtanh_.default,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardsigmoid_.default,
    torch.ops.aten.clamp.default,
    torch.ops.aten.clamp_.default,
]
BATCH_NORM_OPS = [torch.ops.aten.batch_norm.default]
LINEAR_OP_PATTERNS = (
    combo_pattern(LINEAR_OPS)
    + combo_pattern(LINEAR_OPS, FUSED_ACTIVATION_OPS)
    + combo_pattern(LINEAR_OPS, BATCH_NORM_OPS)
    + combo_pattern(LINEAR_OPS, BATCH_NORM_OPS, FUSED_ACTIVATION_OPS)
)
CONV_OP_PATTERNS = (
    combo_pattern(_conv_ops)
    + combo_pattern(_conv_ops, FUSED_ACTIVATION_OPS)
    + combo_pattern(_conv_ops, BATCH_NORM_OPS)
    + combo_pattern(_conv_ops, BATCH_NORM_OPS, FUSED_ACTIVATION_OPS)
)
FUSED_RELU_OP_PATTERNS = (
    combo_pattern(LINEAR_OPS, FUSED_ACTIVATION_OPS)
    + combo_pattern(LINEAR_OPS, BATCH_NORM_OPS, FUSED_ACTIVATION_OPS)
    + combo_pattern(_conv_ops, FUSED_ACTIVATION_OPS)
    + combo_pattern(_conv_ops, BATCH_NORM_OPS, FUSED_ACTIVATION_OPS)
)

ALL_QPARAM_OP_PATTERNS = (
    [(target,) for target in _one_to_one]
    + ACTIVATION_FUNCTION_PATTERNS
    + CONV_OP_PATTERNS
    + LINEAR_OP_PATTERNS
    + BINARY_OP_PATTERNS
    + [
        (torch.ops.aten.full.default,),
        (torch.ops.aten.full,),
        (torch.ops.aten.zeros.default,),
        (torch.ops.aten.ones.default,),
        (torch.ops.aten.fill_.Scalar,),
        (torch.ops.aten.scalar_tensor.default,),
        (torch.ops.aten.zeros_like.default,),
        (torch.ops.aten._softmax.default,),
        (torch.ops.aten.softmax.int,),
        (torch.ops.aten.div.Tensor,),
        (torch.ops.aten.div_.Tensor,),
        (torch.ops.aten.div.Tensor_mode,),
        (torch.ops.aten.floor,),
        (torch.ops.aten.floor_divide.default,),
        (torch.ops.aten.logit.default,),
        (torch.ops.aten.glu.default,),
        (torch.ops.aten.addmm.default,),
        (torch.ops.aten.layer_norm.default,),
        (torch.ops.aten.group_norm.default,),
        (torch.ops.aten.sqrt.default,),
        (torch.ops.aten.silu.default,),
        (torch.ops.aten.silu_.default,),
        (torch.ops.aten.logit.default,),
        (torch.ops.aten.var.dim,),
        (torch.ops.aten.var.correction,),
        (torch.ops.aten.leaky_relu.default,),
        (torch.ops.aten.leaky_relu_.default,),
        (torch.ops.aten.linalg_vector_norm.default,),
        (torch.ops.aten.log_softmax.int,),
        (torch.ops.aten.round.default,),
        (torch.ops.aten.arange.start_step,),
        (torch.ops.aten.embedding.default,),
        (torch.ops.aten.adaptive_avg_pool2d.default,),
        (torch.ops.aten.upsample_bilinear2d.vec,),
        (torch.ops.aten.upsample_nearest2d.vec,),
        (torch.ops.aten.avg_pool2d.default,),
        (torch.ops.aten.max_pool2d.default,),
        (torch.ops.aten.cosine_similarity.default,),
        (torch.ops.aten.sigmoid.default,),
        (torch.ops.aten.remainder.Tensor,),
        (torch.ops.aten.remainder.Scalar,),
        (torch.ops.aten.mean.dim,),
        (torch.ops.aten.mean.default,),
        (torch.ops.aten.neg.default,),
        (torch.ops.aten.scaled_dot_product_attention.default,),
        (torch.ops.aten.abs.default,),
        (torch.ops.aten.minimum.default,),
        (torch.ops.aten.maximum.default,),
        (torch.ops.aten.lt.Tensor,),
        (torch.ops.aten.le.Tensor,),
        (torch.ops.aten.gt.Tensor,),
        (torch.ops.aten.ge.Tensor,),
        (torch.ops.aten.eq.Tensor,),
        (torch.ops.aten.ne.Tensor,),
        (torch.ops.aten.lt.Scalar,),
        (torch.ops.aten.le.Scalar,),
        (torch.ops.aten.gt.Scalar,),
        (torch.ops.aten.ge.Scalar,),
        (torch.ops.aten.eq.Scalar,),
        (torch.ops.aten.ne.Scalar,),
    ]
)
TOSA_QUANTIZER_SUPPORT_DICT: dict[tuple[OpOverload, ...], type[PatternCheck] | None] = {
    pattern: None for pattern in ALL_QPARAM_OP_PATTERNS
}
for pattern in FUSED_RELU_OP_PATTERNS:
    TOSA_QUANTIZER_SUPPORT_DICT[pattern] = ReluFusedPatternCheck
for pattern in BINARY_OP_PATTERNS:
    TOSA_QUANTIZER_SUPPORT_DICT[pattern] = ArithmeticFloatInputsCheck
