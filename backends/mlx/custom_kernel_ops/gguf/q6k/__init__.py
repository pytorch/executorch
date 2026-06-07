#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF Q6_K format implementation (fused custom Metal kernels).

Re-exports the lightweight constants/header from :mod:`.common` so they can be
imported without pulling in the MLX builder. The ``emit_*`` lowerings live in
:mod:`.linear` / :mod:`.embedding` (called by ``custom_kernel_ops.gguf.patterns``)
and are not imported here.
"""

from executorch.backends.mlx.custom_kernel_ops.gguf.q6k.common import (  # noqa: F401
    _Q6K_HEADER,
    Q6K_BLOCK_BYTES,
    QK_K,
)
