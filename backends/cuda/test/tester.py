# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Tuple

import executorch
import executorch.backends.test.harness.stages as BaseStages
import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.partitioner import Partitioner


def _create_default_partitioner() -> CudaPartitioner:
    """Create a CudaPartitioner with default compile specs."""
    compile_specs = [CudaBackend.generate_method_name_compile_spec("forward")]
    return CudaPartitioner(compile_specs)


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    """CUDA-specific ToEdgeTransformAndLower stage."""

    def __init__(
        self,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
    ):
        if partitioners is None:
            partitioners = [_create_default_partitioner()]

        super().__init__(
            default_partitioner_cls=_create_default_partitioner,
            partitioners=partitioners,
            edge_compile_config=edge_compile_config
            or EdgeCompileConfig(_check_ir_validity=False),
        )


class CudaTester(TesterBase):
    """
    Tester subclass for CUDA backend.

    This tester defines the recipe for lowering models to the CUDA backend
    using AOTInductor compilation.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        dynamic_shapes: Optional[Tuple[Any]] = None,
    ):
        stage_classes = (
            executorch.backends.test.harness.Tester.default_stage_classes()
            | {
                StageType.TO_EDGE_TRANSFORM_AND_LOWER: ToEdgeTransformAndLower,
            }
        )

        super().__init__(
            module=module,
            stage_classes=stage_classes,
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
