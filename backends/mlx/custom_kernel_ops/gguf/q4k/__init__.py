#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF **Q4_K** format lowering for the MLX backend.

Q4_K maps onto MLX's native affine 4-bit kernels (no custom Metal):

* :mod:`.common`    -- repack a raw Q4_K blob into MLX qparams.
* :mod:`.linear`    -- ``emit_linear`` (``QuantizedMatmulNode``).
* :mod:`.embedding` -- ``emit_embedding`` (gather + ``DequantizeNode``).

The pattern handlers in ``custom_kernel_ops.gguf.patterns`` call these ``emit_*``
functions. ``.linear`` / ``.embedding`` are intentionally NOT imported here so
the package import stays light.
"""
