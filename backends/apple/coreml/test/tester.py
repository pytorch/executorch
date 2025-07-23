# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, List, Optional, Sequence, Tuple

import coremltools as ct
import executorch
import executorch.backends.test.harness.stages as BaseStages
import torch

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.apple.coreml.quantizer import CoreMLQuantizer
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.partitioner import Partitioner


def _create_default_partitioner(
    minimum_deployment_target: Any = ct.target.iOS15,
) -> CoreMLPartitioner:
    return CoreMLPartitioner(
        compile_specs=CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=minimum_deployment_target
        )
    )


def _get_static_int8_linear_qconfig():
    return ct.optimize.torch.quantization.LinearQuantizerConfig(
        global_config=ct.optimize.torch.quantization.ModuleLinearQuantizerConfig(
            quantization_scheme="symmetric",
            activation_dtype=torch.quint8,
            weight_dtype=torch.qint8,
            weight_per_channel=True,
        )
    )


class Quantize(BaseStages.Quantize):
    def __init__(
        self,
        quantizer: Optional[CoreMLQuantizer] = None,
        quantization_config: Optional[Any] = None,
        calibrate: bool = True,
        calibration_samples: Optional[Sequence[Any]] = None,
        is_qat: Optional[bool] = False,
    ):
        super().__init__(
            quantizer=quantizer
            or CoreMLQuantizer(
                quantization_config or _get_static_int8_linear_qconfig()
            ),
            calibrate=calibrate,
            calibration_samples=calibration_samples,
            is_qat=is_qat,
        )


class Partition(BaseStages.Partition):
    def __init__(
        self,
        partitioner: Optional[Partitioner] = None,
        minimum_deployment_target: Optional[Any] = ct.target.iOS15,
    ):
        super().__init__(
            partitioner=partitioner
            or _create_default_partitioner(minimum_deployment_target),
        )


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    def __init__(
        self,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
        minimum_deployment_target: Optional[Any] = ct.target.iOS15,
    ):
        super().__init__(
            default_partitioner_cls=lambda: _create_default_partitioner(
                minimum_deployment_target
            ),
            partitioners=partitioners,
            edge_compile_config=edge_compile_config,
        )


class CoreMLTester(TesterBase):
    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        dynamic_shapes: Optional[Tuple[Any]] = None,
        minimum_deployment_target: Optional[Any] = ct.target.iOS15,
    ):
        # Specialize for XNNPACK
        stage_classes = (
            executorch.backends.test.harness.Tester.default_stage_classes()
            | {
                StageType.QUANTIZE: Quantize,
                StageType.PARTITION: functools.partial(
                    Partition, minimum_deployment_target=minimum_deployment_target
                ),
                StageType.TO_EDGE_TRANSFORM_AND_LOWER: functools.partial(
                    ToEdgeTransformAndLower,
                    minimum_deployment_target=minimum_deployment_target,
                ),
            }
        )

        super().__init__(
            module=module,
            stage_classes=stage_classes,
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
