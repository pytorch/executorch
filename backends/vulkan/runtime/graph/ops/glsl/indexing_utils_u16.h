/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef INDEXING_UTILS_U16_H
#define INDEXING_UTILS_U16_H

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

u16vec3 idx_to_u16pos_x_wise(uint idx, int size_x, int size_y) {
  const uint div_by_x = idx / size_x;
  return u16vec3(
      idx % size_x,
      div_by_x % size_y,
      div_by_x / size_y);
}

#endif // INDEXING_UTILS_U16_H
