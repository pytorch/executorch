# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import executorch.exir as exir
from executorch.exir.pass_manager import PassType


### XNNPACK Configs ###
def get_xnnpack_edge_compile_config(
    skip_dim_order: bool = False,
) -> exir.EdgeCompileConfig:
    return exir.EdgeCompileConfig(
        _check_ir_validity=False, _skip_dim_order=skip_dim_order
    )


def get_transform_passes(additional_passes=None) -> List[PassType]:
    passes = additional_passes if additional_passes else []
    return passes


def get_xnnpack_executorch_backend_config(
    additional_passes=None,
) -> exir.ExecutorchBackendConfig:
    additional_passes = additional_passes if additional_passes else []
    return exir.ExecutorchBackendConfig(
        passes=additional_passes,
        extract_delegate_segments=True,
    )
