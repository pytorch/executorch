# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, List, Optional, Sequence, Tuple

import executorch
import executorch.backends.test.harness.stages as BaseStages
import torch

from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.backends.openvino.quantizer.quantizer import OpenVINOQuantizer
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.partitioner import Partitioner


class Quantize(BaseStages.Quantize):
    def __init__(
        self,
        calibrate: bool = True,
        calibration_samples: Optional[Sequence[Any]] = None,
        is_qat=False,
    ):
        super().__init__(
            quantizer=OpenVINOQuantizer(),
            calibrate=calibrate,
            calibration_samples=calibration_samples,
            is_qat=is_qat,
            fold_quantize=False,
        )


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    def __init__(
        self,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
        compile_specs: Optional[List[CompileSpec]] = None,
    ):
        compile_specs = compile_specs or [CompileSpec("device", b"CPU")]
        super().__init__(
            default_partitioner_cls=lambda: OpenvinoPartitioner(compile_specs),  # type: ignore[arg-type]
            partitioners=partitioners,
            edge_compile_config=edge_compile_config
            or EdgeCompileConfig(_check_ir_validity=False),
        )


class Partition(BaseStages.Partition):
    def __init__(
        self,
        partitioner: Optional[Partitioner] = None,
        compile_specs: Optional[List[CompileSpec]] = None,
    ):
        super().__init__(
            partitioner=partitioner or OpenvinoPartitioner(compile_specs or []),
        )


class OpenVINOTester(TesterBase):
    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        dynamic_shapes: Optional[Tuple[Any]] = None,
        compile_specs: Optional[List[CompileSpec]] = None,
    ):
        compile_specs = compile_specs or [CompileSpec("device", b"CPU")]
        stage_classes = (
            executorch.backends.test.harness.Tester.default_stage_classes()
            | {
                StageType.PARTITION: functools.partial(
                    Partition, compile_specs=compile_specs
                ),
                StageType.QUANTIZE: Quantize,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER: functools.partial(
                    ToEdgeTransformAndLower,
                    compile_specs=compile_specs,
                ),
            }
        )

        super().__init__(
            module=module,
            stage_classes=stage_classes,
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
