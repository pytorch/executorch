# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import operator
from typing import cast, List, Optional

import torch
from executorch.backends.arm.quantizer.arm_quantizer_utils import (
    _mark_node_as_quantized,
    is_annotated,
    PatternCheck,
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
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_PER_CHANNEL_CONFIG,
    INT8_PER_TENSOR_CONFIG,
)
from executorch.backends.cortex_m.quantizer.quantizer_support import (
    __name__ as cortex_m_quantizer_support_module,
    CONV_OP_PATTERNS,
    CONV_TRANSPOSE_OP_PATTERNS,
    CORTEX_M_QUANTIZER_SUPPORT_DICT,
)
from executorch.backends.cortex_m.quantizer_reporter import (
    QuantizerInfo,
    QuantizerReporter,
    QuantizerReporterUser,
)
from torch._ops import OpOverload
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantizer import ComposableQuantizer, Quantizer


def _is_zero_init(node) -> bool:
    """True if an initial hidden/cell state node is a constant zero tensor."""
    if not isinstance(node, torch.fx.Node):
        return False
    if node.target == torch.ops.aten.zeros.default:
        return True
    if node.target == torch.ops.aten.full.default:
        return bool(len(node.args) > 1 and node.args[1] == 0)
    return False


class LstmBoundaryQuantizer(Quantizer, QuantizerReporterUser):
    """Annotate a single-layer, unidirectional, biased, zero-initial-state,
    output-only ``aten.lstm.input`` at its activation boundaries only: int8
    per-tensor on the input activation and on the primary output
    (``getitem(0)``). The weight/bias params are left in float so the lowering
    pass can quantize each gate per-tensor ahead of time (a single per-tensor
    scale over the 4*hidden weight stack would be too coarse for the CMSIS
    kernel).

    Configurations outside that scope are left unannotated. Because the flow
    preserves ``aten.lstm.input`` through to_edge and there is no portable LSTM
    kernel, an unannotated LSTM cannot be lowered; ``AtenToCortexMPass`` raises a
    clear error if one survives rather than failing later with a missing kernel.
    """

    _TARGET = torch.ops.aten.lstm.input

    def __init__(self) -> None:
        super().__init__()
        QuantizerReporterUser.__init__(self)

    def get_quantizer_info(self) -> QuantizerInfo:
        return QuantizerInfo(
            self.__class__.__name__,
            str(self._TARGET),
            "int8 per-tensor activations (weights kept float)",
            __name__ + ".LstmBoundaryQuantizer",
        )

    @staticmethod
    def _rejection_reason(node) -> Optional[str]:
        # args: (input, hx, params, has_biases, num_layers, dropout, train,
        #        bidirectional, batch_first)
        hx, params = node.args[1], node.args[2]
        has_biases, num_layers, bidirectional = (
            node.args[3],
            node.args[4],
            node.args[7],
        )
        if not has_biases:
            return "bias is required"
        if num_layers != 1:
            return "only single-layer is supported"
        if bidirectional:
            return "only unidirectional is supported"
        # A projection LSTM appends weight_hr, so params has 5 (biased) entries.
        if len(params) != 4:
            return "projection LSTM is not supported"
        if not all(_is_zero_init(h) for h in hx):
            return "only a zero initial state is supported"
        # The fused op emits only the sequence output; h_n/c_n (getitem 1/2)
        # would leave the original LSTM live and unlowerable.
        for user in node.users:
            if (
                user.target == operator.getitem
                and user.args[1] != 0
                and len(user.users) > 0
            ):
                return "h_n / c_n outputs are not supported"
        return None

    def annotate(self, model: GraphModule) -> None:  # type: ignore[override]
        for node in model.graph.nodes:
            if node.target != self._TARGET or is_annotated(node):
                continue
            reason = self._rejection_reason(node)
            if reason is not None:
                self.report_reject([node], f"cortex_m LSTM: {reason}.")
                continue
            _mark_node_as_quantized(
                node,
                {node.args[0]: INT8_ACTIVATION_PER_TENSOR_QSPEC},
                None,
                is_quantized=True,
            )
            for user in node.users:
                if (
                    user.target == operator.getitem
                    and user.args[1] == 0
                    and not is_annotated(user)
                ):
                    _mark_node_as_quantized(
                        user, {}, INT8_ACTIVATION_PER_TENSOR_QSPEC, is_quantized=True
                    )
            self.report_accept([node])

    def validate(self, model: GraphModule) -> bool:  # type: ignore[override]
        return True


def mark_node_as_annotated(
    node,
    input_qspec_map,
    output_qspec,
    is_quantized,
) -> None:
    _mark_node_as_quantized(node, input_qspec_map, output_qspec, is_quantized)


class CortexMQuantizer(ComposableQuantizer):

    def __init__(self) -> None:
        conv_targets: set[OpOverload] = set()
        for key in CONV_OP_PATTERNS.keys() | CONV_TRANSPOSE_OP_PATTERNS.keys():
            conv_targets.update(key)

        support_dict_name = (
            cortex_m_quantizer_support_module + ".CORTEX_M_QUANTIZER_SUPPORT_DICT"
        )
        pattern_matcher = PatternMatcher(
            cast(
                dict[tuple[OpOverload, ...], Optional[type[PatternCheck]]],
                CORTEX_M_QUANTIZER_SUPPORT_DICT,
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
                INT8_PER_TENSOR_CONFIG,
                node_finder=GlobalNodeFinder(),
                pattern_matcher=pattern_matcher,
            ),
            LstmBoundaryQuantizer(),
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
