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
from executorch.exir.passes.propagate_device_pass import TARGET_DEVICE_COMPILE_SPEC_KEY


@final
@experimental(
    "This API and all of cuda backend related functionality are experimental."
)
class CudaPartitioner(AotiPartitioner):
    """
    CUDA partitioner driven by AOTInductor backend.

    This partitioner adds a target_device compile spec to enable device info
    propagation. The PropagateDevicePass will read this spec and mark delegate
    output tensors with CUDA device type, which flows through to serialization.
    """

    def __init__(
        self,
        compile_spec: List[CompileSpec],
    ) -> None:
        """
        Initialize the CUDA partitioner.

        Args:
            compile_spec: List of compile specs for the backend. To specify a
                         target CUDA device, include a CompileSpec with key
                         "target_device" (e.g., value "cuda:1"). If not
                         provided, defaults to "cuda:0".
        """
        # Add target_device compile spec for device propagation if not already present
        has_target_device = any(
            spec.key == TARGET_DEVICE_COMPILE_SPEC_KEY for spec in compile_spec
        )
        if not has_target_device:
            compile_spec = list(compile_spec) + [
                CompileSpec(
                    TARGET_DEVICE_COMPILE_SPEC_KEY,
                    b"cuda:0",
                )
            ]
        super().__init__(CudaBackend.__name__, compile_spec)
