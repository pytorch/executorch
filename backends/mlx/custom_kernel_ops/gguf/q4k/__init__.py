#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF Q4_K format lowering for the MLX backend (native affine 4-bit).

See :mod:`.linear` / :mod:`.embedding` for the ``emit_*`` lowerings (called by
``custom_kernel_ops.gguf.patterns``); they are not imported here to keep the
package import light.
"""
