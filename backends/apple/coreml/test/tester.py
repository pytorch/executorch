# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Tuple

import executorch
import executorch.backends.test.harness.stages as BaseStages

import torch
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.partitioner import Partitioner


class Partition(BaseStages.Partition):
    def __init__(self, partitioner: Optional[Partitioner] = None):
        super().__init__(
            partitioner=partitioner or CoreMLPartitioner,
        )


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    def __init__(
        self,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
    ):
        super().__init__(
            default_partitioner_cls=CoreMLPartitioner,
            partitioners=partitioners,
            edge_compile_config=edge_compile_config,
        )


class CoreMLTester(TesterBase):
    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        dynamic_shapes: Optional[Tuple[Any]] = None,
    ):
        # Specialize for XNNPACK
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
