#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF-quantized weight lowering for the MLX backend.

Submodules:

* :mod:`.q6k`      -- shared Q6_K primitives (constants, pure-torch dequant,
  Metal header) and the fused mat-vec / mat-mat / gather kernels. Importing
  ``.q6k`` (or ``.q6k.common``) is lightweight and does not touch the registry.
* :mod:`.patterns` -- registers MLX pattern handlers that match
  ``torchao::dequantize_gguf -> linear/embedding`` (what ``ExportableGGUFTensor``
  exports) and lower them to the Q6_K kernels.

To enable GGUF lowering, import :mod:`.patterns` for its side effect::

    import executorch.backends.mlx.custom_kernel_ops.gguf.patterns  # noqa: F401

This package ``__init__`` is intentionally side-effect free so importing
``.q6k`` for the pure-torch dequant does not pull in the MLX builder/registry.
"""
