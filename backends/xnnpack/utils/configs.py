# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import executorch.exir as exir
from executorch.exir import CaptureConfig
from executorch.exir.backend.canonical_partitioners.duplicate_dequant_node_pass import (
    DuplicateDequantNodePass,
)
from executorch.exir.pass_manager import PassType

### XNNPACK Configs ###
def get_xnnpack_edge_compile_config() -> exir.EdgeCompileConfig:
    return exir.EdgeCompileConfig(
        _check_ir_validity=False,
    )


def get_transform_passes(additional_passes=None) -> List[PassType]:
    additional_passes = additional_passes if additional_passes else []
    passes = additional_passes + [DuplicateDequantNodePass()]
    return passes


def get_xnnpack_executorch_backend_config(
    additional_passes=None,
) -> exir.ExecutorchBackendConfig:
    additional_passes = additional_passes if additional_passes else []
    return exir.ExecutorchBackendConfig(
        passes=additional_passes,
        extract_segments=True,
    )


def get_xnnpack_capture_config(
    dynamic_shape=False,
    enable_aot: Optional[bool] = None,
    unlift: Optional[bool] = None,
):
    if enable_aot is None:
        return CaptureConfig(enable_dynamic_shape=dynamic_shape)
    else:
        unlift = unlift if unlift is not None else enable_aot
        return CaptureConfig(
            enable_dynamic_shape=dynamic_shape, enable_aot=enable_aot, _unlift=unlift
        )
