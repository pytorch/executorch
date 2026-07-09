#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF Q5_K format lowering for the MLX backend.

Re-exports the lightweight constants/header from :mod:`.common` so they can be
imported without pulling in the MLX builder. The ``emit_*`` lowerings live in
:mod:`.linear` / :mod:`.embedding` (called by ``custom_kernel_ops.gguf.patterns``)
and are not imported here.

By default the legacy export-time repack path
(:mod:`.linear_mlx_native` / :mod:`.embedding_mlx_native`) is used. Set
``ET_MLX_EMIT_DIRECT_GGUF=1`` to emit the fused Metal kernels that read raw
GGUF bytes instead.
"""

import os

from executorch.backends.mlx.custom_kernel_ops.gguf.q5k.common import (  # noqa: F401
    _Q5K_HEADER,
    Q5K_BLOCK_BYTES,
    QK_K,
)


def emit_direct_gguf() -> bool:
    """Return True to emit fused kernels that read raw GGUF bytes.

    Defaults to False (the legacy MLX-native repack path); set
    ``ET_MLX_EMIT_DIRECT_GGUF=1`` to enable the fused kernels.
    """
    return os.environ.get("ET_MLX_EMIT_DIRECT_GGUF", "0") != "0"
