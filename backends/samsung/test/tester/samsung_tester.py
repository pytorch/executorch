# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.
import copy
from typing import Any, List, Optional, Sequence, Tuple, Union

import executorch.backends.test.harness.stages as BaseStages
import torch
from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
from executorch.backends.samsung.quantizer.quantizer import EnnQuantizer, Precision
from executorch.backends.samsung.test.utils import RuntimeExecutor
from executorch.backends.samsung.utils.export_utils import (
    get_edge_compile_config,
    get_enn_pass_list,
)
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.backend.backend_details import CompileSpec

from executorch.exir.pass_manager import PassType
from torch.export import ExportedProgram

from torchao.quantization.pt2e.quantizer import Quantizer


class Export(BaseStages.Export):
    pass


class Quantize(BaseStages.Quantize):
    def __init__(
        self,
        quantizer: Optional[Quantizer] = None,
        quantization_config: Optional[Any] = None,
        calibrate: bool = True,
        calibration_samples: Optional[Sequence[Any]] = None,
        is_qat: Optional[bool] = False,
    ):
        super().__init__(
            quantizer=quantizer,
            quantization_config=quantization_config,
            calibrate=calibrate,
            calibration_samples=calibration_samples,
            is_qat=is_qat,
        )


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    def __init__(
        self,
        compile_specs: Optional[List[CompileSpec]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
        transform_passes: Optional[Union[Sequence[PassType]]] = None,
    ):
        compile_specs = compile_specs or []
        self.partitioners = [EnnPartitioner(compile_specs=compile_specs)]
        self.edge_compile_config = edge_compile_config or get_edge_compile_config()
        self.transform_passes = transform_passes or get_enn_pass_list()
        self.edge_dialect_program = None

    def run(
        self, artifact: ExportedProgram, inputs=None, generate_etrecord: bool = False
    ) -> None:
        artifact_copy = copy.deepcopy(artifact)
        self.edge_dialect_program = to_edge_transform_and_lower(
            artifact_copy,
            transform_passes=self.transform_passes,
            partitioner=self.partitioners,
            compile_config=self.edge_compile_config,
        )


class ToExecutorch(BaseStages.ToExecutorch):
    def run_artifact(self, inputs):
        runtime_executor = RuntimeExecutor(self.artifact, inputs)
        return runtime_executor.run_on_device()


class SamsungTester(TesterBase):
    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        compile_specs: Optional[List[CompileSpec]] = None,
    ):
        module.eval()

        stage_classes = TesterBase.default_stage_classes() | {
            StageType.EXPORT: Export,
            StageType.TO_EXECUTORCH: ToExecutorch,
        }

        super().__init__(
            module=module,
            stage_classes=stage_classes,
            example_inputs=example_inputs,
            dynamic_shapes=None,
        )

        self.original_module = module
        self.exported_module = module
        self.example_inputs = example_inputs
        self.compile_specs = compile_specs

    def quantize(self, quantize_stage: Optional[Quantize] = None):
        if quantize_stage is None:
            quantizer = EnnQuantizer()
            quantizer.setup_quant_params(Precision.A8W8)
            quantize_stage = Quantize(quantizer)

        return super().quantize(quantize_stage)

    def to_edge_transform_and_lower(
        self,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
    ):
        to_edge_transform_and_lower_stage = ToEdgeTransformAndLower(
            self.compile_specs, edge_compile_config
        )

        return super().to_edge_transform_and_lower(to_edge_transform_and_lower_stage)
