/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#include "indexing_utils.h"

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[2][DTYPE]} image_out;
layout(set = 0, binding = 1) buffer  PRECISION restrict readonly Buffer {
  ${T[DTYPE]} data[];
}
buffer_in;

// Corresponds to {1,4,9,24} in the example below.
layout(set = 0, binding = 2) uniform PRECISION restrict GpuSizes {
  ivec4 data;
}
gpu_sizes;

// Corresponds to {3,3,7,10} in the example below.
layout(set = 0, binding = 3) uniform PRECISION restrict OriginalSizes {
  ivec4 data;
}
original_sizes;

// Corresponds to {3,3,8,12} in the example below.
layout(set = 0, binding = 4) uniform PRECISION restrict AlignedSizes {
  ivec4 data;
}
padded_sizes;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Computes special prepacking for a 2D convolution. Each shader invocation
 * calculates the input buffer location to read into the desired texel. This
 * packing was originally developed on CPU and that approach is described in the
 * rest of this comment. Refer to the code-level comments, for how we translate
 * it to GPU by reversing the steps.
 *
 * Consider example weight tensor of size {10,7,3,3}. The following
 * transformations will be applied.
 *
 * 1. Pad the N and C dims so that both are a multiple of 4. In this case, 2
 * batches and 1 channel of padding are added, producing a tensor of size
 * {12,8,3,3}.
 *      at::pad(x, {0,0,0,0,0,2,0,1}, "constant", 0);
 *
 * 2. Split the tensor along the C dim so that each split has 4 channels.
 *      x.reshape({12,2,4,3,3});
 *
 * 3. For each split, "fold" the C dim into the W dim. Suppose the first rows
 * at H=0 of the split have values
 *    0,1,2 | 10,11,12 | 20,21,22 | 30,31,32
 *
 * where | denotes a channel boundary. Then, the goal is to combine those rows
 * into one row with the values
 *    0, 10, 20, 30, 1, 11, 21, 31, 2, 12, 22, 32
 *
 *      x.permute({0,1,3,4,2}).reshape({12,2,3,12});
 *
 * 4. Stack the splits belonging to the same batch horizontally by swapping the
 * C and H dims.
 *      x.permute({0,2,1,3}).reshape({12,3,24});
 *
 * 5. Repeat a similar process to "fold" the N dim into the C dim. Split along
 * the N dim so that each split has 4 batches.
 *      x.reshape({3,4,3,24});
 *
 * 6. Stack the batches on each other vertically by swapping the N and C dims.
 *      x.permute({1,0,2,3}).reshape({4,9,24});
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 coord = POS_TO_COORD_CHANNELS_PACKED(pos, gpu_sizes.data);

  if (any(greaterThanEqual(coord, gpu_sizes.data))) {
    return;
  }

  // As in usual staging shaders, map from GPU texel position to normal CPU
  // buffer indices: (24,9) -> (4,9,24)
  const int base_index = COORD_TO_BUFFER_IDX(coord, gpu_sizes.data);
  const ivec4 p0 =
      base_index + ivec4(0, 1, 2, 3) * STRIDE_CHANNELS_PACKED(gpu_sizes.data);

  // Re-map the normal CPU buffer indices to special indices, through a series
  // of permutations: reshape is a no-op to the underlying indices, and permute
  // is one of the hardest math problems I've ever solved.
  //
  // Undo step 6 premute: (4,3,3,24) -> (3,4,3,24)
  // Undo step 4 permute: (12,3,2,12) -> (12,2,3,12)
  // Undo step 3 permute, part 1: (12,2,3h,3w,4) -> (12,2,3h,4,3w)
  // Undo step 3 permute, part 2: (12,2,3h,4,3w) -> (12,2,4,3h,3w)
  const ivec4 p1 = SWAP_DIMS(
      p0,
      4,
      (padded_sizes.data.w / 4),
      (padded_sizes.data.y * padded_sizes.data.z * padded_sizes.data.x));
  const ivec4 p2 = SWAP_DIMS(
      p1,
      padded_sizes.data.y,
      (padded_sizes.data.z / 4),
      (padded_sizes.data.x * 4));
  const ivec4 p3 = SWAP_DIMS(p2, padded_sizes.data.x, 4, 1);
  const ivec4 p4 = SWAP_DIMS(p3, padded_sizes.data.y, 4, padded_sizes.data.x);

  // For values in the padded region, write zero instead of buffer data.
  //
  // Undo step 1 pad: (12,8,3,3) -> (10,7,3,3)
  const ivec4 c = p4 %
      (padded_sizes.data.z * padded_sizes.data.y * padded_sizes.data.x) /
      (padded_sizes.data.y * padded_sizes.data.x);
  const ivec4 n =
      p4 / (padded_sizes.data.z * padded_sizes.data.y * padded_sizes.data.x);
  const ivec4 p5 = p4 -
      n * (padded_sizes.data.z - original_sizes.data.z) * padded_sizes.data.y *
          padded_sizes.data.x;
  const ivec4 mask = ivec4(greaterThanEqual(c, original_sizes.data.zzzz)) |
      ivec4(greaterThanEqual(n, original_sizes.data.wwww));

  ${T[DTYPE]} val_x = mix(buffer_in.data[p5.x], 0, mask.x);
  ${T[DTYPE]} val_y = mix(buffer_in.data[p5.y], 0, mask.y);
  ${T[DTYPE]} val_z = mix(buffer_in.data[p5.z], 0, mask.z);
  ${T[DTYPE]} val_w = mix(buffer_in.data[p5.w], 0, mask.w);

  ${VEC4_T[DTYPE]} texel = ${VEC4_T[DTYPE]}(val_x, val_y, val_z, val_w);

  imageStore(image_out, pos.xy, texel);
}
