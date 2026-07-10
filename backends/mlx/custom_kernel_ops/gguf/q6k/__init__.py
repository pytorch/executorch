#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF Q6_K format lowering for the MLX backend.

Re-exports the lightweight constants/header from :mod:`.common` so they can be
imported without pulling in the MLX builder. The ``emit_*`` lowerings live in
:mod:`.linear` / :mod:`.embedding` (called by ``custom_kernel_ops.gguf.patterns``)
and are not imported here.

Q6_K's native group size is 16, which MLX's affine kernels do not support (only
32/64/128). The MLX-native repack path (:mod:`.linear_mlx_native` /
:mod:`.embedding_mlx_native`) is therefore only usable when adjacent sub-blocks
merge losslessly into a group size >= 32; otherwise lowering falls back to the
fused Metal kernels. Set ``ET_MLX_EMIT_DIRECT_GGUF=1`` to always use the fused
kernels.
"""

import os

from executorch.backends.mlx.custom_kernel_ops.gguf.q6k.common import (  # noqa: F401
    _Q6K_HEADER,
    Q6K_BLOCK_BYTES,
    QK_K,
)


def emit_direct_gguf() -> bool:
    """Return True to emit fused kernels that read raw GGUF bytes.

    Defaults to False (attempt the MLX-native repack path, which itself falls
    back to fused kernels when the weight does not merge to an MLX-supported
    group size); set ``ET_MLX_EMIT_DIRECT_GGUF=1`` to force the fused kernels.
    """
    return os.environ.get("ET_MLX_EMIT_DIRECT_GGUF", "0") != "0"
