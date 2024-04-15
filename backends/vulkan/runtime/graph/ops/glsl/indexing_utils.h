/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define divup4(x) ((x + 3) / 4)

#define to_buffer_i(idx, sizes)                      \
  idx.x + idx.y* sizes.x + idx.z* sizes.y* sizes.x + \
      idx.w* sizes.z* sizes.y* sizes.x;

#define get_packed_dim_C_packed(vec) vec.z
#define get_packed_dim_W_packed(vec) vec.x
#define get_packed_dim_H_packed(vec) vec.y

#define get_packed_stride_C_packed(vec) (vec.x * vec.y)
#define get_packed_stride_W_packed(vec) (1)
#define get_packed_stride_H_packed(vec) (vec.x)

#define to_tensor_idx_C_packed(pos, sizes) \
  ivec4(pos.x, pos.y, (pos.z * 4) % sizes.z, (pos.z * 4) / sizes.z)

#define to_tensor_idx_W_packed(pos, sizes) \
  ivec4((pos.x * 4), pos.y, pos.z % sizes.z, pos.z / sizes.z)

#define to_tensor_idx_H_packed(pos, sizes) \
  ivec4(pos.x, (pos.y * 4), pos.z % sizes.z, pos.z / sizes.z)

#define to_texture_pos_C_packed(idx, sizes) \
  ivec3(idx.x, idx.y, (idx.z + idx.w * sizes.z) / 4)

#define to_texture_pos_W_packed(idx, sizes) \
  ivec3(idx.x / 4, idx.y, (idx.z + idx.w * sizes.z))

#define to_texture_pos_H_packed(idx, sizes) \
  ivec3(idx.x, idx.y / 4, (idx.z + idx.w * sizes.z))

// Given a buffer(1-D) index cur, compute a new index where the corresponding
// tensor(N-D)'s adjacent dimensions are swapped. The parameters x,y and plane
// describe sizes. As an example, let's say we want to swap dimensions 0,1 for a
// tensor of shape {4,3,2,24} to obtain {3,4,2,24}. Then, x=4, y=3 and
// plane=2*24=48.
#define swap_adj_dims(cur, x, y, plane)                        \
  cur +                                                        \
      plane *                                                  \
          ((1 - y) * ((cur % (x * y * plane)) / (y * plane)) + \
           (x - 1) * ((cur % (y * plane)) / plane))
