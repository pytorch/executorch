# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Sequence, Tuple

import executorch
import executorch.backends.test.harness.stages as BaseStages

import torch
from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
)
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.backend.partitioner import Partitioner
from torch.export import ExportedProgram


class Quantize(BaseStages.Quantize):
    def __init__(
        self,
        quantizer: QnnQuantizer,
        quantization_config: Optional[Any] = None,
        calibrate: bool = True,
        calibration_samples: Optional[Sequence[Any]] = None,
        is_qat: Optional[bool] = False,
    ):
        super().__init__(
            quantizer=quantizer,
            calibrate=calibrate,
            calibration_samples=calibration_samples,
            is_qat=is_qat,
            set_global=False,
        )


class Partition(BaseStages.Partition):
    def __init__(self, partitioner: Optional[Partitioner] = None):
        super().__init__(
            partitioner=partitioner or QnnPartitioner,
        )


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    def __init__(
        self,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
        soc_model: str = "SM8650",
        use_fp16: bool = True,
    ):
        backend_options = generate_htp_compiler_spec(use_fp16=use_fp16)
        self.chipset = get_soc_to_chipset_map()[soc_model]
        self.compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=self.chipset,
            backend_options=backend_options,
        )

        super().__init__(
            partitioners=partitioners or [QnnPartitioner(self.compiler_specs)],
            edge_compile_config=edge_compile_config
            or EdgeCompileConfig(_check_ir_validity=False),
            default_partitioner_cls=QnnPartitioner,
        )

    def run(
        self, artifact: ExportedProgram, inputs=None, generate_etrecord: bool = False
    ) -> None:
        ep = QnnPassManager().transform_for_export_pipeline(artifact)
        transform_passes = QnnPassManager().get_to_edge_transform_passes(ep)

        self.edge_dialect_program = to_edge_transform_and_lower(
            ep,
            transform_passes=transform_passes,
            partitioner=self.partitioners,
            compile_config=self.edge_compile_conf,
            generate_etrecord=generate_etrecord,
        )


class QualcommTester(TesterBase):
    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        dynamic_shapes: Optional[Tuple[Any]] = None,
        use_fp16: bool = True,
    ):
        def create_to_edge_transform_and_lower(*args, **kwargs):
            kwargs["use_fp16"] = use_fp16
            return ToEdgeTransformAndLower(*args, **kwargs)

        # Specialize for Qualcomm
        stage_classes = executorch.backends.test.harness.Tester.default_stage_classes() | {
            StageType.PARTITION: Partition,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER: create_to_edge_transform_and_lower,
        }

        super().__init__(
            module=module,
            stage_classes=stage_classes,
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
