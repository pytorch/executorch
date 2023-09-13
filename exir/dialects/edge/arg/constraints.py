# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def scatter_add_index_size_max(x, dim, src, d):
    max_d = src.size(d)
    if d != dim:
        max_d = min(max_d, x.size(d))
    return max_d
