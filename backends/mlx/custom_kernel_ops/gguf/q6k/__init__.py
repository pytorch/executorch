#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF **Q6_K** format implementation.

* :mod:`.common`    -- shared primitives (constants + Metal header). Re-exported
  here so ``from ...gguf.q6k import Q6K_BLOCK_BYTES`` stays lightweight (no MLX
  builder import).
* :mod:`.linear`    -- Q6_K mat-vec/mat-mat kernels + ``emit_linear`` lowering.
* :mod:`.embedding` -- Q6_K gather kernel + ``emit_embedding`` lowering.

The pattern handlers that match ``torchao::gguf_dequantize -> linear/embedding``
and call these ``emit_*`` functions live one level up in
``custom_kernel_ops.gguf.patterns``. ``.linear`` / ``.embedding`` are
intentionally NOT imported here so importing :mod:`.common` for the pure-torch
dequant does not pull in the builder.
"""

from executorch.backends.mlx.custom_kernel_ops.gguf.q6k.common import (  # noqa: F401
    _Q6K_HEADER,
    Q6K_BLOCK_BYTES,
    QK_K,
)
