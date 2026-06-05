#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF-quantized custom ops for the MLX backend.

Submodules:

* :mod:`.q6k`       -- shared Q6_K primitives (constants, pure-torch dequant,
  Metal header). Import this for symbols; it does not register any op.
* :mod:`.linear`    -- registers ``mlx::gguf_linear``.
* :mod:`.embedding` -- registers ``mlx::gguf_embedding``.

To register an op, import the corresponding submodule for its side effect, e.g.
``import executorch.backends.mlx.custom_kernel_ops.gguf.linear  # noqa: F401``.

This package ``__init__`` is intentionally side-effect free so importing
``.q6k`` for the pure-torch dequant does not pull in the MLX builder/registry.
"""
