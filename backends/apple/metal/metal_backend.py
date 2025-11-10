# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing
from typing import Any, Dict, final

from executorch.backends.aoti.aoti_backend import AotiBackend
from executorch.exir._warnings import experimental
from executorch.exir.backend.backend_details import BackendDetails


@final
@experimental(
    "This API and all of Metal backend related functionality are experimental."
)
class MetalBackend(AotiBackend, BackendDetails):
    """
    MetalBackend is a backend that compiles a model to run on Metal/MPS devices. It uses the AOTInductor compiler to generate
    optimized Metal kernels for the model's operators with libtorch-free. The compiled model can be executed on Metal devices
    using the Executorch runtime.
    """

    @staticmethod
    def get_device_name() -> str:
        return "metal"

    @staticmethod
    def get_supported_fallback_kernels() -> Dict[str, Any]:
        return {
            "aoti_torch_mps_addmm_out": None,
            "aoti_torch_mps_convolution": None,
            "aoti_torch_mps_mm_out": None,
            "at::_ops::_scaled_dot_product_attention_math_for_mps::call": None,
        }

    @staticmethod
    def get_decomposition_table() -> Dict[Any, Any]:
        return {}

    @staticmethod
    def get_aoti_compile_options() -> Dict[str, typing.Any]:
        return {
            # Do not link against the full PyTorch/libtorch library
            "aot_inductor.link_libtorch": False,
            # Separate weight constants from the .so file
            "aot_inductor.package": True,
            "aot_inductor.package_constants_in_so": False,
            # Store weight constants on disk in a binary blob
            "aot_inductor.package_constants_on_disk_format": "binary_blob",
            # Enable maximum automatic tuning for optimal performance
            "max_autotune": True,
            # "aot_inductor.debug_compile": True,
            # "aot_inductor.force_mmap_weights": False,
        }
