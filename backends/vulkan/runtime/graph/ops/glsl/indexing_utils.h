/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define DIVUP4(x) ((x + 3) / 4)

#define PACKED_DIM_CHANNELS_PACKED(vec) vec.z

#define PACKED_DIM_WIDTH_PACKED(vec) vec.x

#define PACKED_DIM_HEIGHT_PACKED(vec) vec.y

#define POS_TO_COORD_CHANNELS_PACKED(pos, sizes) \
  ivec4(pos.x, pos.y, (pos.z * 4) % sizes.z, (pos.z * 4) / sizes.z)

#define POS_TO_COORD_WIDTH_PACKED(pos, sizes) \
  ivec4((pos.x * 4), pos.y, pos.z % sizes.z, pos.z / sizes.z)

#define POS_TO_COORD_HEIGHT_PACKED(pos, sizes) \
  ivec4(pos.x, (pos.y * 4), pos.z % sizes.z, pos.z / sizes.z)

#define COORD_TO_POS_CHANNELS_PACKED(coord, sizes) \
  ivec3(coord.x, coord.y, (coord.z + coord.w * sizes.z) / 4)

#define COORD_TO_POS_WIDTH_PACKED(coord, sizes) \
  ivec3(coord.x / 4, coord.y, (coord.z + coord.w * sizes.z))

#define COORD_TO_POS_HEIGHT_PACKED(coord, sizes) \
  ivec3(coord.x, coord.y / 4, (coord.z + coord.w * sizes.z))

#define COORD_TO_POS_CHANNELS_PACKED(coord, sizes) \
  ivec3(coord.x, coord.y, (coord.z + coord.w * sizes.z) / 4)

#define COORD_TO_BUFFER_IDX(coord, sizes)                  \
  coord.x + coord.y* sizes.x + coord.z* sizes.y* sizes.x + \
      coord.w* sizes.z* sizes.y* sizes.x;

#define STRIDE_CHANNELS_PACKED(vec) (vec.x * vec.y)

#define STRIDE_WIDTH_PACKED(vec) (1)

#define STRIDE_HEIGHT_PACKED(vec) (vec.x)

// Given a buffer(1-D) index cur, compute a new index where the corresponding
// tensor(N-D)'s adjacent dimensions are swapped. The parameters x,y and plane
// describe sizes. As an example, let's say we want to swap dimensions 0,1 for a
// tensor of shape {4,3,2,24} to obtain {3,4,2,24}. Then, x=4, y=3 and
// plane=2*24=48.
#define SWAP_ADJ_DIMS(cur, x, y, plane)                       \
  cur +                                                       \
      plane*(                                                 \
          (1 - y) * ((cur % (x * y * plane)) / (y * plane)) + \
          (x - 1) * ((cur % (y * plane)) / plane))
