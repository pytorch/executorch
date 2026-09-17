# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from executorch.backends.apple.coreai.compiler.preprocess import (
    AOTCompileConfig,
    coreai_sidecar_dir,
    CoreAIBackend,
)
from executorch.backends.apple.coreai.partition.partitioner import CoreAIPartitioner
from executorch.exir import EdgeCompileConfig
from executorch.exir.pass_base import PassType


def get_default_passes() -> List[PassType]:
    """Default edge transform passes for Core AI lowering.

    ``NarrowToCoreAIDtypesPass`` casts int64/float64 graph inputs to 32-bit at
    the boundary (preserving the model's external I/O dtype) so index-style
    inputs can be delegated. Core AI supports only up-to-32-bit dtypes.
    """
    from executorch.backends.apple.coreai.passes import NarrowToCoreAIDtypesPass

    return [NarrowToCoreAIDtypesPass()]


def get_default_compile_config() -> EdgeCompileConfig:
    """Default ``EdgeCompileConfig`` for Core AI lowering.

    ``_skip_dim_order=True`` keeps ExecuTorch on ``aten._to_copy`` instead of
    emitting ``dim_order_ops._to_dim_order_copy``, which coreai-torch cannot
    lower (its validator requires dim-order ops be decomposed).
    """
    return EdgeCompileConfig(_skip_dim_order=True)


__all__ = [
    "AOTCompileConfig",
    "CoreAIBackend",
    "CoreAIPartitioner",
    "coreai_sidecar_dir",
    "get_default_compile_config",
    "get_default_passes",
]
