/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

#define COORD_TO_BUFFER_IDX(coord, sizes)                  \
  coord.x + coord.y* sizes.x + coord.z* sizes.y* sizes.x + \
      coord.w* sizes.z* sizes.y* sizes.x;

#define STRIDE_CHANNELS_PACKED(vec) (vec.x * vec.y)

#define STRIDE_WIDTH_PACKED(vec) (1)

#define STRIDE_HEIGHT_PACKED(vec) (vec.x)
