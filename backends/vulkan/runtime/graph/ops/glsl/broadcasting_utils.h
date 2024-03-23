/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

ivec4 out_coord_to_in_coord(const ivec4 out_coord, const ivec4 in_sizes) {
  ivec4 in_coord = out_coord;
  for (int i = 0; i < 4; ++i) {
    if (out_coord[i] >= in_sizes[i]) {
      in_coord[i] = 0;
    }
  }
  return in_coord;
}
