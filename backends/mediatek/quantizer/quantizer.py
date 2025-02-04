# Copyright (c) 2024 MediaTek Inc.
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

from torch.ao.quantization.quantizer import Quantizer
from torch.fx import GraphModule

from .._passes.decompose_scaled_dot_product_attention import (
    DecomposeScaledDotProductAttention,
)
from .annotator import annotate
from .qconfig import get_quant_config, Precision


class NeuropilotQuantizer(Quantizer):

    def __init__(self):
        super().__init__()

        # TODO: Provide setter functions for those attributes
        self._precision = Precision.A8W8
        self._is_per_channel = True
        self._is_qat = False

    def setup_precision(self, precision: Precision) -> None:
        self._precision = precision

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        model = DecomposeScaledDotProductAttention()(model).graph_module
        return model

    def annotate(self, model: GraphModule) -> GraphModule:
        self._annotate(model)
        return model

    def validate(self, model: GraphModule) -> None:
        pass

    def _annotate(self, gm: GraphModule) -> None:
        quant_config = get_quant_config(
            self._precision, self._is_per_channel, self._is_qat
        )
        annotate(gm.graph, quant_config)
