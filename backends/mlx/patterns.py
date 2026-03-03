#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
MLX Pattern Handlers - pattern-based op lowering for fused operations.

This module contains pattern handlers that match multi-node subgraphs and lower
them to optimized MLX operations.
"""
