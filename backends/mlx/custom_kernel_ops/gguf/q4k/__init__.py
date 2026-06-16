#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF Q4_K format lowering for the MLX backend (fused Metal kernels).

See :mod:`.linear` / :mod:`.embedding` for the ``emit_*`` lowerings (called by
``custom_kernel_ops.gguf.patterns``); they are not imported here to keep the
package import light.

Set ``ET_MLX_EMIT_DIRECT_GGUF=0`` to use the legacy export-time repack path
(:mod:`.linear_mlx_native` / :mod:`.embedding_mlx_native`) instead.
"""

from __future__ import annotations

import os


def emit_direct_gguf() -> bool:
    """Return True to emit fused kernels that read raw GGUF bytes (default)."""
    return os.environ.get("ET_MLX_EMIT_DIRECT_GGUF", "0") != "0"
