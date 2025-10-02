# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Sequence, Tuple

import executorch
import executorch.backends.test.harness.stages as BaseStages

import torch
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
    get_symmetric_quantization_config as get_symmetric_quantization_config_vulkan,
    VulkanQuantizer,
)
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.partitioner import Partitioner
from torchao.quantization.pt2e.quantizer import Quantizer


class Quantize(BaseStages.Quantize):
    def __init__(
        self,
        quantizer: Optional[Quantizer] = None,
        quantization_config: Any | None = None,
        calibrate: bool = True,
        calibration_samples: Optional[Sequence[Any]] = None,
        is_qat: Optional[bool] = False,
    ):
        super().__init__(
            quantizer=quantizer or VulkanQuantizer(),
            quantization_config=(
                quantization_config or get_symmetric_quantization_config_vulkan()
            ),
            calibrate=calibrate,
            calibration_samples=calibration_samples,
            is_qat=is_qat,
        )


class Partition(BaseStages.Partition):
    def __init__(self, partitioner: Optional[Partitioner] = None):
        super().__init__(
            partitioner=partitioner or VulkanPartitioner(),
        )


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    def __init__(
        self,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
    ):
        super().__init__(
            default_partitioner_cls=VulkanPartitioner,
            partitioners=partitioners,
            edge_compile_config=edge_compile_config
            or EdgeCompileConfig(_check_ir_validity=False),
        )


class VulkanTester(TesterBase):
    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        dynamic_shapes: Optional[Tuple[Any]] = None,
    ):
        stage_classes = (
            executorch.backends.test.harness.Tester.default_stage_classes()
            | {
                StageType.PARTITION: Partition,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER: ToEdgeTransformAndLower,
            }
        )

        super().__init__(
            module=module,
            stage_classes=stage_classes,
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
