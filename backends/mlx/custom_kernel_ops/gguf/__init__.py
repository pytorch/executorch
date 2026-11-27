#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF-quantized weight lowering for the MLX backend.

Import :mod:`.patterns` for its side effect to enable lowering of
``torchao::dequantize_gguf -> linear/embedding`` to the Q6_K / Q4_K kernels::

    import executorch.backends.mlx.custom_kernel_ops.gguf.patterns  # noqa: F401

This ``__init__`` is side-effect free, so importing ``.q6k`` for the pure-torch
dequant does not pull in the MLX builder/registry.
"""
