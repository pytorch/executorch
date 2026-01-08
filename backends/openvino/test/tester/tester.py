# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Sequence, Tuple
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import functools
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

logger.info("26 partition")


class Quantize(BaseStages.Quantize):
    def __init__(
        self,
        calibrate: bool = True,
        calibration_samples: Optional[Sequence[Any]] = None,
        is_qat=False,
    ):
        logger.info("IN OPENVINO QUANTIZE STAGE")
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
        compile_specs: Optional[List[CompileSpec]] = [CompileSpec("device", b"CPU")],
    ):
        logger.info("IN TO EDGE TRANSFORM AND LOWER ____")

        super().__init__(
            default_partitioner_cls=lambda: OpenvinoPartitioner(compile_specs),
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
        # If no compile specs provided, default to CPU
        logger.info("IN PARTITION ____")

        super().__init__(
            partitioner=partitioner or OpenvinoPartitioner(compile_specs),
        )


class OpenVINOTester(TesterBase):
    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        dynamic_shapes: Optional[Tuple[Any]] = None,
        compile_specs: Optional[List[CompileSpec]] = [CompileSpec("device", b"CPU")],
    ):
        # Specialize for OpenVINO
        logger.info("IN TESTER")
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

        logger.info(f"Stage classes: {stage_classes}")

        super().__init__(
            module=module,
            stage_classes=stage_classes,
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
