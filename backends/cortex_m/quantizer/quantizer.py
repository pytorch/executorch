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
    GlobalNoOutputNodeFinder,
    NodeTargetActivationNodeFinder,
    NodeTargetNodeFinder,
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
    CORTEX_M_EXPLICIT_LAYOUT_QUANTIZER_SUPPORT_DICT,
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
    """Convolutions use ``INT8_PER_CHANNEL_CONFIG`` and every other supported op
    ``per_tensor_config``. ``set_node_target`` overrides the config of an op,
    and each call overrides those before it.
    """

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
                default to the per-channel config. Defaults to
                ``INT8_PER_TENSOR_CONFIG``; pass ``INT16_PER_TENSOR_CONFIG`` to
                quantize the ops that support it (e.g. ``quantized_div``) with
                int16 activations.
            use_explicit_layout: Select the support checks for the experimental
                explicit-layout pipeline. Legacy mode continues to require
                channels-last convolution inputs.
        """
        per_tensor_config = per_tensor_config or INT8_PER_TENSOR_CONFIG

        support_dict = CORTEX_M_QUANTIZER_SUPPORT_DICT
        support_dict_name = (
            cortex_m_quantizer_support_module + ".CORTEX_M_QUANTIZER_SUPPORT_DICT"
        )
        conv_patterns = CONV_OP_PATTERNS.keys() | CONV_TRANSPOSE_OP_PATTERNS.keys()
        if use_explicit_layout:
            support_dict = CORTEX_M_EXPLICIT_LAYOUT_QUANTIZER_SUPPORT_DICT
            support_dict_name = (
                cortex_m_quantizer_support_module
                + ".CORTEX_M_EXPLICIT_LAYOUT_QUANTIZER_SUPPORT_DICT"
            )
            conv_patterns |= CONV1D_OP_PATTERNS.keys()

        conv_targets: set[OpOverload] = set()
        for key in conv_patterns:
            conv_targets.update(key)

        self.pattern_matcher = PatternMatcher(
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
                pattern_matcher=self.pattern_matcher,
            ),
            PatternQuantizer(
                per_tensor_config,
                node_finder=GlobalNoOutputNodeFinder(),
                pattern_matcher=self.pattern_matcher,
            ),
            SharedQspecQuantizer(),
        ]
        super().__init__(quantizers)

    def set_node_target(
        self, node_target: OpOverload, quantization_config: Optional[QuantizationConfig]
    ) -> "CortexMQuantizer":
        """Quantize every node calling `node_target`, and the activations its
        patterns fuse after it, with `quantization_config`, e.g.
        ``torch.ops.aten.conv2d.default`` with ``INT8_PER_TENSOR_CONFIG`` for
        per-tensor convolution weights. ``None`` leaves them unquantized, and
        works for any op.
        """
        if not isinstance(node_target, OpOverload):
            raise TypeError(
                "set_node_target: expected an OpOverload such as "
                f"torch.ops.aten.conv2d.default, got {node_target}"
            )
        patterns = [p for p in self.pattern_matcher.support_dict if p[0] == node_target]
        if quantization_config is not None and not patterns:
            raise ValueError(
                f"Quantization of {node_target} is not supported by the "
                "cortex-m backend"
            )
        activations = list(dict.fromkeys(op for p in patterns for op in p[1:]))
        self.quantizers.insert(
            0,
            PatternQuantizer(
                quantization_config,
                node_finder=NodeTargetActivationNodeFinder(node_target, activations),
                pattern_matcher=self.pattern_matcher,
            ),
        )
        return self

    def annotate(self, model):
        reporter = QuantizerReporter(self.quantizers)
        model = super().annotate(model)
        reporter.log_quantizer_report(model)
        return model

    def validate(self, model: GraphModule) -> None:
        return None

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        pass_manager = CortexMPassManager()
        return pass_manager.transform_for_annotation(model)
