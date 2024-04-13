/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define BUF_T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${texel_type(DTYPE)}
#define SCALAR_T ${texel_component_type(DTYPE)}

#define to_tensor_idx to_tensor_idx_${PACKING}
#define get_packed_stride get_packed_stride_${PACKING}

#include "indexing_utils.h"

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[2][DTYPE]} image_out;
layout(set = 0, binding = 1) buffer  PRECISION restrict readonly Buffer {
  BUF_T data[];
}
buffer_in;

// Corresponds to {1,4,6,36} in the example below.
layout(set = 0, binding = 2) uniform PRECISION restrict GpuSizes {
  ivec4 data;
}
gpu_sizes;

// Corresponds to {3,3,7,10} in the example below.
layout(set = 0, binding = 3) uniform PRECISION restrict OriginalSizes {
  ivec4 data;
}
original_sizes;

// Corresponds to {8,12} in the example below.
layout(set = 0, binding = 4) uniform PRECISION restrict PaddedSizes {
  ivec2 data;
}
padded_sizes;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Computes special prepacking for a 2D transpose convolution. Each shader
 * invocation calculates the input buffer location to read into the desired
 * texel.
 *
 * For details, refer to conv2d_prepack_weights.glsl which uses a similar
 * approach. For transpose, there are slight differences to reflect the data
 * access pattern in the shader. First, the weight tensor is flipped along the H
 * and W dims. Second, steps 3 and 4 are slightly different so that the splits
 * are interleaved.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(pos, gpu_sizes.data);

  if (any(greaterThanEqual(idx, gpu_sizes.data))) {
    return;
  }

  // As in usual staging shaders, map from GPU texel position to normal CPU
  // buffer indices: (36,6) -> (4,6,36)
  const int base_index = to_buffer_i(idx, gpu_sizes.data);
  const ivec4 p0 =
      base_index + ivec4(0, 1, 2, 3) * get_packed_stride(gpu_sizes.data);

  // Re-map the normal CPU buffer indices to special indices, through a series
  // of mappings: reshape is a no-op to the underlying indices, so we only map
  // for flip, pad, and permute.
  const int Np = padded_sizes.data.y;
  const int Cp = padded_sizes.data.x;
  const int N = original_sizes.data.w;
  const int C = original_sizes.data.z;
  const int H = original_sizes.data.y;
  const int W = original_sizes.data.x;

  // Undo step 6 premute: (4,2,3,36) -> (2,4,3,36)
  // In the following comments, a=b=c=3.
  // Undo step 3 permute, part 1: (8,a,b,c,4) -> (8,a,c,b,4)
  // Undo step 3 permute, part 2: (8,a,c,b,4) -> (8,c,a,b,4)
  // Undo step 3 permute, part 3: (8,c,a,b,4) -> (8,c,a,4,b)
  // Undo step 3 permute, part 4: (8,c,a,4,b) -> (8,c,4,a,b)
  const ivec4 p1 = swap_adj_dims(p0, 4, (Cp / 4), (H * Np * W));
  const ivec4 p2 = swap_adj_dims(p1, W, (Np / 4), 4);
  const ivec4 p3 = swap_adj_dims(p2, H, (Np / 4), (W * 4));
  const ivec4 p4 = swap_adj_dims(p3, W, 4, 1);
  const ivec4 p5 = swap_adj_dims(p4, H, 4, W);

  // Undo step 0 permute: (8,12,3,3) -> (12,8,3,3)
  const ivec4 p6 = swap_adj_dims(p5, Cp, Np, (W * H));
  // Undo step 0 flip: (2,3)
  const ivec4 w = p6 % W;
  const ivec4 h = p6 % (H * W) / W;
  const ivec4 p7 = p6 + W - 1 - 2 * w + W * (H - 1 - 2 * h);

  // Undo step 1 pad: (12,8,3,3) -> (10,7,3,3)
  // For values in the padded region, write zero instead of buffer data.
  const ivec4 c = p7 % (Cp * H * W) / (H * W);
  const ivec4 n = p7 / (Cp * H * W);
  const ivec4 p8 = p7 - n * (Cp - C) * H * W;
  const ivec4 mask = ivec4(greaterThanEqual(c, ivec4(C))) |
      ivec4(greaterThanEqual(n, ivec4(N)));

  SCALAR_T val_x = mix(SCALAR_T(buffer_in.data[p8.x]), 0, mask.x);
  SCALAR_T val_y = mix(SCALAR_T(buffer_in.data[p8.y]), 0, mask.y);
  SCALAR_T val_z = mix(SCALAR_T(buffer_in.data[p8.z]), 0, mask.z);
  SCALAR_T val_w = mix(SCALAR_T(buffer_in.data[p8.w]), 0, mask.w);

  VEC4_T texel = VEC4_T(val_x, val_y, val_z, val_w);

  imageStore(image_out, pos.xy, texel);
}
