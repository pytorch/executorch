# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

from executorch.backends.arm.quantizer.arm_quantizer_utils import (
    _mark_node_as_quantized,
    PatternQuantizer,
    SharedQspecQuantizer,
)
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.node_finders import (
    GlobalNodeFinder,
    NodeTargetNodeFinder,
)
from executorch.backends.cortex_m.quantizer.pattern_matcher import PatternMatcher
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT8_PER_CHANNEL_CONFIG,
    INT8_PER_TENSOR_CONFIG,
)
from executorch.backends.cortex_m.quantizer.quantizer_reporter import QuantizerReporter
from executorch.backends.cortex_m.quantizer.quantizer_support import (
    __name__ as cortex_m_quantizer_support_module,
    CONV_OP_PATTERNS,
    CONV_TRANSPOSE_OP_PATTERNS,
    CORTEX_M_QUANTIZER_SUPPORT_DICT,
)
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

    def __init__(self) -> None:
        conv_targets = set()
        for key in CONV_OP_PATTERNS.keys() | CONV_TRANSPOSE_OP_PATTERNS.keys():
            conv_targets.update(key)

        support_dict_name = (
            cortex_m_quantizer_support_module + ".CORTEX_M_QUANTIZER_SUPPORT_DICT"
        )
        pattern_matcher = PatternMatcher(
            CORTEX_M_QUANTIZER_SUPPORT_DICT,
            support_dict_name=support_dict_name,
        )
        quantizers: List[Quantizer] = [
            PatternQuantizer(
                INT8_PER_CHANNEL_CONFIG,
                node_finder=NodeTargetNodeFinder(conv_targets),
                pattern_matcher=pattern_matcher,
            ),
            PatternQuantizer(
                INT8_PER_TENSOR_CONFIG,
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

    def validate(self, model: GraphModule) -> bool:
        return True

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        pass_manager = CortexMPassManager(None)
        return pass_manager.transform_for_annotation(model)
