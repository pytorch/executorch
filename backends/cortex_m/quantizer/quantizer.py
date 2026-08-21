# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import cast, List, Optional

from executorch.backends.arm.quantizer.arm_quantizer_utils import (
    _mark_node_as_quantized,
    PatternCheck,
    PatternQuantizer,
    SharedQspecQuantizer,
)
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.node_finders import (
    GlobalNodeFinder,
    NodeTargetNodeFinder,
)
from executorch.backends.cortex_m.quantizer.pattern_checkers import (
    CortexMExplicitConv1DCheck,
    CortexMExplicitConv2DCheck,
    CortexMExplicitConvTranspose2DCheck,
)
from executorch.backends.cortex_m.quantizer.pattern_matcher import PatternMatcher
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT8_PER_CHANNEL_CONFIG,
    INT8_PER_TENSOR_CONFIG,
)
from executorch.backends.cortex_m.quantizer.quantizer_support import (
    __name__ as cortex_m_quantizer_support_module,
    CONV1D_OP_PATTERNS,
    CONV_OP_PATTERNS,
    CONV_TRANSPOSE_OP_PATTERNS,
    CORTEX_M_QUANTIZER_SUPPORT_DICT,
)
from executorch.backends.cortex_m.quantizer_reporter import QuantizerReporter
from torch._ops import OpOverload
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantizer import ComposableQuantizer, Quantizer


def mark_node_as_annotated(
    node,
    input_qspec_map,
    output_qspec,
    is_quantized,
) -> None:
    _mark_node_as_quantized(node, input_qspec_map, output_qspec, is_quantized)


class CortexMQuantizer(ComposableQuantizer):

    def __init__(
        self,
        per_tensor_config: Optional[QuantizationConfig] = None,
        use_explicit_layout: bool = False,
    ) -> None:
        """Cortex-M PT2E quantizer.

        Args:
            per_tensor_config: Per-tensor activation config applied to the
                non-conv elementwise ops (div/mul/add/...) that
                ``GlobalNodeFinder`` matches anywhere in the graph. Convolutions
                are always quantized with the per-channel config. Defaults to
                ``INT8_PER_TENSOR_CONFIG``; pass ``INT16_PER_TENSOR_CONFIG`` to
                quantize the ops that support it (e.g. ``quantized_div``) with
                int16 activations.
            use_explicit_layout: Allow contiguous NCHW Conv2d and
                ConvTranspose2d patterns. Legacy mode still requires
                channels-last tensors during quantization.
        """
        per_tensor_config = per_tensor_config or INT8_PER_TENSOR_CONFIG

        conv_targets: set[OpOverload] = set()
        conv_patterns = CONV_OP_PATTERNS.keys() | CONV_TRANSPOSE_OP_PATTERNS.keys()
        if use_explicit_layout:
            conv_patterns |= CONV1D_OP_PATTERNS.keys()
        for key in conv_patterns:
            conv_targets.update(key)

        support_dict_name = (
            cortex_m_quantizer_support_module + ".CORTEX_M_QUANTIZER_SUPPORT_DICT"
        )
        support_dict = dict(CORTEX_M_QUANTIZER_SUPPORT_DICT)
        if use_explicit_layout:
            for pattern in CONV1D_OP_PATTERNS:
                support_dict[pattern] = CortexMExplicitConv1DCheck
            for pattern in CONV_OP_PATTERNS:
                support_dict[pattern] = CortexMExplicitConv2DCheck
            for pattern in CONV_TRANSPOSE_OP_PATTERNS:
                support_dict[pattern] = CortexMExplicitConvTranspose2DCheck

        pattern_matcher = PatternMatcher(
            cast(
                dict[tuple[OpOverload, ...], Optional[type[PatternCheck]]],
                support_dict,
            ),
            support_dict_name=support_dict_name,
        )
        quantizers: List[Quantizer] = [
            PatternQuantizer(
                INT8_PER_CHANNEL_CONFIG,
                node_finder=NodeTargetNodeFinder(list(conv_targets)),
                pattern_matcher=pattern_matcher,
            ),
            PatternQuantizer(
                per_tensor_config,
                node_finder=GlobalNodeFinder(),
                pattern_matcher=pattern_matcher,
            ),
            SharedQspecQuantizer(),
        ]
        super().__init__(quantizers)

    def annotate(self, model):
        reporter = QuantizerReporter(self.quantizers)
        model = super().annotate(model)
        reporter.log_quantizer_report(model)
        return model

    def validate(self, model: GraphModule) -> None:
        return None

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        pass_manager = CortexMPassManager(None)
        return pass_manager.transform_for_annotation(model)
