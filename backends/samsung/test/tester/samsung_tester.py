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
from executorch.backends.samsung.test.utils.quant_checkers import get_checker
from executorch.backends.samsung.utils.export_utils import get_edge_compile_config
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.backends.transforms.decompose_sdpa import (
    DecomposeScaledDotProductAttention,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.backend.backend_details import CompileSpec

from executorch.exir.pass_manager import PassType
from torch.export import export, ExportedProgram

from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)

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
        checker_config=None,
    ):
        super().__init__(
            quantizer=quantizer,
            quantization_config=quantization_config,
            calibrate=calibrate,
            calibration_samples=calibration_samples,
            is_qat=is_qat,
        )
        self.checker_config = checker_config

    def run(
        self, artifact: torch.nn.Module, inputs: Optional[Tuple[torch.Tensor]]
    ) -> None:
        assert inputs is not None
        if self.is_qat:
            artifact.train()
        captured_graph = export(artifact, inputs, strict=True).module()

        assert isinstance(captured_graph, torch.fx.GraphModule)

        DecomposeScaledDotProductAttention()(captured_graph)

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

        converted = convert_pt2e(prepared, fold_quantize=False)

        self.converted_graph = converted
        if self.checker_config:
            checker = get_checker(artifact, converted, self.checker_config)
            checker.check()


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
        self.transform_passes = transform_passes
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

    def quantize(
        self,
        quantize_stage: Optional[Quantize] = None,
        cali_dataset=None,
        checker_config=None,
    ):
        if quantize_stage is None:
            quantizer = EnnQuantizer()
            quantizer.setup_quant_params(Precision.A8W8)
            quantize_stage = Quantize(
                quantizer,
                calibration_samples=cali_dataset,
                checker_config=checker_config,
            )

        return super().quantize(quantize_stage)

    def to_edge_transform_and_lower(
        self,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
    ):
        to_edge_transform_and_lower_stage = ToEdgeTransformAndLower(
            self.compile_specs, edge_compile_config
        )

        return super().to_edge_transform_and_lower(to_edge_transform_and_lower_stage)
