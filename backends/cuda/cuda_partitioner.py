# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
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

        The delegate runs on that one device at runtime, whichever device is
        current when the program loads or runs; to use another GPU, export
        for it. A method whose delegates target different devices is not
        supported.
        """
        target_devices = [
            spec.value
            for spec in compile_spec
            if spec.key == TARGET_DEVICE_COMPILE_SPEC_KEY
        ]
        # The runtime reads the same spec to pick the device it runs on, and
        # accepts only these forms, so refuse anything else here rather than
        # when the program is loaded.
        for value in target_devices:
            if not re.fullmatch(rb"cuda(:[0-9]+)?", value):
                raise ValueError(
                    f"target_device must be 'cuda' or 'cuda:N', got {value!r}"
                )
        # Add target_device compile spec for device propagation if not already present
        if not target_devices:
            compile_spec = list(compile_spec) + [
                CompileSpec(
                    TARGET_DEVICE_COMPILE_SPEC_KEY,
                    b"cuda:0",
                )
            ]
        super().__init__(CudaBackend.__name__, compile_spec)
