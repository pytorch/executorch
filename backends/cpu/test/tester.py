# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from executorch.backends.cpu.partitioner import CPUPartitioner
from executorch.backends.test.harness import Tester
from executorch.backends.test.harness.stages import (
    Partition,
    StageType,
    ToEdgeTransformAndLower,
)
from executorch.exir import EdgeCompileConfig


class CPUTester(Tester):
    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: tuple[Any, ...],
        dynamic_shapes: Any = None,
    ) -> None:
        super().__init__(
            module,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            stage_classes=Tester.default_stage_classes()
            | {
                StageType.PARTITION: partial(Partition, partitioner=CPUPartitioner()),
                StageType.TO_EDGE_TRANSFORM_AND_LOWER: partial(
                    ToEdgeTransformAndLower,
                    default_partitioner_cls=CPUPartitioner,
                    edge_compile_config=EdgeCompileConfig(),
                ),
            },
        )
