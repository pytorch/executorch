from typing import Any, Optional, Sequence, Tuple

import torch

from executorch.backends.test.harness.stages.stage import Stage, StageType
from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)

from torch.export import export_for_training

from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torchao.quantization.pt2e.quantizer import Quantizer


class Quantize(Stage):
    def __init__(
        self,
        quantizer: Optional[Quantizer] = None,
        quantization_config: Optional[Any] = None,
        calibrate: bool = True,
        calibration_samples: Optional[Sequence[Any]] = None,
        is_qat: Optional[bool] = False,
        set_global: bool = True,
    ):
        self.quantizer = quantizer
        self.quantization_config = quantization_config
        self.calibrate = calibrate
        self.calibration_samples = calibration_samples

        if self.quantization_config is not None and set_global:
            self.quantizer.set_global(self.quantization_config)

        self.converted_graph = None
        self.is_qat = is_qat

    def stage_type(self) -> str:
        return StageType.QUANTIZE

    def run(
        self, artifact: torch.nn.Module, inputs: Optional[Tuple[torch.Tensor]]
    ) -> None:
        assert inputs is not None
        if self.is_qat:
            artifact.train()
        captured_graph = export_for_training(artifact, inputs, strict=True).module()

        assert isinstance(captured_graph, torch.fx.GraphModule)

        if self.is_qat:
            prepared = prepare_qat_pt2e(captured_graph, self.quantizer)
        else:
            prepared = prepare_pt2e(captured_graph, self.quantizer)

        if self.calibrate:
            # Calibrate prepared model to provide data to quantization observers.
            if self.calibration_samples is not None:
                for inp in self.calibration_samples:
                    prepared(*inp)
            else:
                prepared(*inputs)

        converted = convert_pt2e(prepared)
        DuplicateDynamicQuantChainPass()(converted)

        self.converted_graph = converted

    @property
    def artifact(self) -> torch.fx.GraphModule:
        return self.converted_graph

    @property
    def graph_module(self) -> str:
        return self.converted_graph

    def run_artifact(self, inputs):
        return self.converted_graph.forward(*inputs)
