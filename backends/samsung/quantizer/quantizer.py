# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Sequence

import torch
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantizer import Quantizer

from .annotator import annotate
from .qconfig import get_quant_config, Precision, QuantInfoManager


global_quant_info = QuantInfoManager()


class EnnQuantizer(Quantizer):

    def __init__(self):
        super().__init__()

        self._precision = Precision.A8W8
        global_quant_info.set_precision(self._precision)
        self._is_per_channel = True
        self._is_qat = False
        self.custom_quant_annotations: Sequence[Callable] = []

    def setup_precision(self, quant_dtype: Precision) -> None:
        assert quant_dtype in Precision, f"No support for Precision {quant_dtype}."
        self._precision = quant_dtype
        global_quant_info.set_precision(self._precision)

    def setup_quant_params(
        self, quant_dtype: Precision, is_per_channel=True, is_qat=False
    ) -> None:
        assert quant_dtype in Precision, f"No support for Precision {quant_dtype}."
        self._precision = quant_dtype
        self._is_per_channel = is_per_channel
        self._is_qat = is_qat

    def annotate(self, model: GraphModule) -> GraphModule:
        self._annotate(model)
        self._annotate_custom_annotation(model)
        return model

    def _annotate(self, gm: GraphModule) -> None:
        quant_config = get_quant_config(
            self._precision, self._is_per_channel, self._is_qat
        )
        annotate(gm.graph, quant_config)

    def add_custom_quant_annotations(
        self, custom_quant_annotations: Sequence[Callable]
    ) -> None:
        self.custom_quant_annotations = custom_quant_annotations

    def _annotate_custom_annotation(self, gm: GraphModule) -> None:
        for annotation_func in self.custom_quant_annotations:
            annotation_func(gm)

    def validate(self, model: torch.fx.GraphModule) -> None:
        return
