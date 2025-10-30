# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final, List

from executorch.backends.aoti.aoti_partitioner import AotiPartitioner
from executorch.backends.cuda.cuda_backend import CudaBackend  # usort: skip
from executorch.exir._warnings import experimental
from executorch.exir.backend.compile_spec_schema import CompileSpec


@final
@experimental(
    "This API and all of cuda backend related functionality are experimental."
)
class CudaPartitioner(AotiPartitioner):
    """
    CUDA partitioner driven by AOTInductor backend.
    """

    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        super().__init__(CudaBackend.__name__, compile_spec)
