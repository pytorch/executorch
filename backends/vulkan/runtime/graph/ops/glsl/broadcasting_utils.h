/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

ivec4 broadcast_indices(const ivec4 out_idx, const ivec4 in_sizes) {
  ivec4 in_idx = out_idx;
  for (int i = 0; i < 4; ++i) {
    if (out_idx[i] >= in_sizes[i]) {
      in_idx[i] = 0;
    }
  }
  return in_idx;
}
