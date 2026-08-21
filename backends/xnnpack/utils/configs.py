# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import executorch.exir as exir

from executorch.backends.xnnpack._passes.lift_constant_scalar_operands_pass import (
    LiftConstantScalarOperandsPass,
)
from executorch.backends.xnnpack._passes.remove_noop_expand_copy_pass import (
    RemoveNoopExpandCopyPass,
)
from executorch.exir.pass_manager import PassType


### XNNPACK Configs ###
def get_xnnpack_edge_compile_config(
    skip_dim_order: bool = False,
) -> exir.EdgeCompileConfig:
    return exir.EdgeCompileConfig(
        _check_ir_validity=False, _skip_dim_order=skip_dim_order
    )


def get_transform_passes(additional_passes=None) -> List[PassType]:
    passes = [RemoveNoopExpandCopyPass(), LiftConstantScalarOperandsPass()]
    if additional_passes:
        passes.extend(additional_passes)
    return passes


def get_xnnpack_executorch_backend_config(
    additional_passes=None,
) -> exir.ExecutorchBackendConfig:
    additional_passes = additional_passes if additional_passes else []
    return exir.ExecutorchBackendConfig(
        passes=additional_passes,
        extract_delegate_segments=True,
    )
