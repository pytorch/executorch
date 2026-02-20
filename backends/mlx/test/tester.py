# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, List, Optional, Tuple

import executorch
import executorch.backends.test.harness.stages as BaseStages
import torch

from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.partitioner import Partitioner


def _create_default_partitioner(
    compile_specs: List[CompileSpec] | None = None,
) -> MLXPartitioner:
    return MLXPartitioner(compile_specs=compile_specs)


class Partition(BaseStages.Partition):
    def __init__(
        self,
        partitioner: Optional[Partitioner] = None,
        compile_specs: Optional[List[CompileSpec]] = None,
    ):
        super().__init__(
            partitioner=partitioner or _create_default_partitioner(compile_specs),
        )


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    def __init__(
        self,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
        compile_specs: Optional[List[CompileSpec]] = None,
    ):
        super().__init__(
            default_partitioner_cls=lambda: _create_default_partitioner(compile_specs),
            partitioners=partitioners,
            edge_compile_config=edge_compile_config,
        )


class MLXTester(TesterBase):
    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        dynamic_shapes: Optional[Tuple[Any]] = None,
        compile_specs: Optional[List[CompileSpec]] = None,
    ):
        stage_classes = (
            executorch.backends.test.harness.Tester.default_stage_classes()
            | {
                StageType.PARTITION: functools.partial(
                    Partition, compile_specs=compile_specs
                ),
                StageType.TO_EDGE_TRANSFORM_AND_LOWER: functools.partial(
                    ToEdgeTransformAndLower, compile_specs=compile_specs
                ),
            }
        )

        super().__init__(
            module=module,
            stage_classes=stage_classes,
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
